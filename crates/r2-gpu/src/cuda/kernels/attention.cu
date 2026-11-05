// Fused causal grouped-query attention, flash form: nothing of size seq x seq
// is ever written. Layouts and the backward's decomposition are the wgpu
// kernels' (`src/attention.rs`): q and the output are [nseq*seq][nh*hd], k and
// v are [nseq*seq][nkv*hd], and nh/nkv = group query heads share one kv head.
//
// One thread owns one output row and sums its terms in a fixed order, so every
// result is the same bits on every run. No staging: these read global memory
// directly, unlike the wgpu kernels, whose compiler cost it more. MAX_HD bounds
// the per-thread row arrays; the host refuses a larger hd and keeps its CPU path.
#include "r2_math.h"

#define MAX_HD 256u
#define BLOCK 64u

#define THREAD_INDEX (blockIdx.x * BLOCK + threadIdx.x)

__device__ __forceinline__ float dot(const float *a, const float *b, unsigned n) {
    float s = 0.0f;
    for (unsigned c = 0; c < n; c++) s = fmaf(a[c], b[c], s);
    return s;
}

// O and the per-query log-sum-exp L the backward reuses, by the online softmax:
// running max m, running sum l, accumulator rescaled when the max rises.
extern "C" __global__ void attn_forward(const float *Q, const float *K, const float *V, float *O, float *L,
                                        unsigned nseq, unsigned seq, unsigned nh, unsigned nkv, unsigned hd,
                                        unsigned group, float scale) {
    const unsigned idx = THREAD_INDEX;                  // row * nh + head
    if (idx >= nseq * seq * nh) return;
    const unsigned row = idx / nh, h = idx % nh, qi = row % seq, base = row - qi;
    const unsigned kvh = h / group, kw = nkv * hd;
    float q[MAX_HD], acc[MAX_HD];
    const float *qp = Q + row * nh * hd + h * hd;
    for (unsigned c = 0; c < hd; c++) { q[c] = qp[c]; acc[c] = 0.0f; }
    float m = -3.0e38f, l = 0.0f;
    for (unsigned j = 0; j <= qi; j++) {
        const unsigned off = (base + j) * kw + kvh * hd;
        const float sc = dot(q, K + off, hd) * scale;
        if (sc > m) {
            const float corr = exp_r2(m - sc);
            l = l * corr;
            for (unsigned c = 0; c < hd; c++) acc[c] = acc[c] * corr;
            m = sc;
        }
        const float p = exp_r2(sc - m);
        l = l + p;
        const float *vp = V + off;
        for (unsigned c = 0; c < hd; c++) acc[c] = fmaf(p, vp[c], acc[c]);
    }
    const float inv = 1.0f / l;
    float *op = O + row * nh * hd + h * hd;
    for (unsigned c = 0; c < hd; c++) op[c] = acc[c] * inv;
    L[idx] = m + logf(l);
}

// D_i = dO_i . O_i, one thread per (row, head).
extern "C" __global__ void attn_delta(const float *O, const float *G, float *D, unsigned rows, unsigned hd) {
    const unsigned idx = THREAD_INDEX;
    if (idx >= rows) return;
    D[idx] = dot(O + idx * hd, G + idx * hd, hd);
}

// dV_j = sum_i P_ij dO_i and dK_j = scale sum_i dS_ij Q_i, together: one thread
// per (row, kv head), walking every query head of the group and every query at
// or after the key, with P_ij recomputed from L and never stored.
extern "C" __global__ void attn_dkv(const float *Q, const float *K, const float *V, const float *G,
                                    const float *L, const float *D, float *dK, float *dV,
                                    unsigned nseq, unsigned seq, unsigned nh, unsigned nkv, unsigned hd,
                                    unsigned group, float scale) {
    const unsigned idx = THREAD_INDEX;                  // row * nkv + kv head
    if (idx >= nseq * seq * nkv) return;
    const unsigned row = idx / nkv, kvh = idx % nkv, kj = row % seq, base = row - kj;
    const unsigned koff = row * nkv * hd + kvh * hd;
    float k[MAX_HD], v[MAX_HD], ak[MAX_HD], av[MAX_HD];
    for (unsigned c = 0; c < hd; c++) { k[c] = K[koff + c]; v[c] = V[koff + c]; ak[c] = 0.0f; av[c] = 0.0f; }
    for (unsigned qh = kvh * group; qh < (kvh + 1u) * group; qh++) {
        for (unsigned qi = kj; qi < seq; qi++) {
            const unsigned lidx = (base + qi) * nh + qh;
            const float *q = Q + lidx * hd;
            const float *g = G + lidx * hd;
            const float p = exp_r2(dot(q, k, hd) * scale - L[lidx]);
            const float ds = p * (dot(g, v, hd) - D[lidx]) * scale;
            for (unsigned c = 0; c < hd; c++) {
                av[c] = fmaf(p, g[c], av[c]);
                ak[c] = fmaf(ds, q[c], ak[c]);
            }
        }
    }
    for (unsigned c = 0; c < hd; c++) { dK[koff + c] = ak[c]; dV[koff + c] = av[c]; }
}

// dQ_i = scale sum_j dS_ij K_j: one thread per (row, query head).
extern "C" __global__ void attn_dq(const float *Q, const float *K, const float *V, const float *G,
                                   const float *L, const float *D, float *dQ,
                                   unsigned nseq, unsigned seq, unsigned nh, unsigned nkv, unsigned hd,
                                   unsigned group, float scale) {
    const unsigned idx = THREAD_INDEX;                  // row * nh + head
    if (idx >= nseq * seq * nh) return;
    const unsigned row = idx / nh, h = idx % nh, qi = row % seq, base = row - qi;
    const unsigned kvh = h / group, kw = nkv * hd;
    const float li = L[idx], di = D[idx];
    float q[MAX_HD], g[MAX_HD], acc[MAX_HD];
    for (unsigned c = 0; c < hd; c++) { q[c] = Q[idx * hd + c]; g[c] = G[idx * hd + c]; acc[c] = 0.0f; }
    for (unsigned j = 0; j <= qi; j++) {
        const unsigned off = (base + j) * kw + kvh * hd;
        const float *kp = K + off;
        const float p = exp_r2(dot(q, kp, hd) * scale - li);
        const float ds = p * (dot(g, V + off, hd) - di) * scale;
        for (unsigned c = 0; c < hd; c++) acc[c] = fmaf(ds, kp[c], acc[c]);
    }
    for (unsigned c = 0; c < hd; c++) dQ[idx * hd + c] = acc[c];
}
