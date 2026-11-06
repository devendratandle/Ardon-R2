// Fused causal grouped-query attention as batched GEMMs. The wgpu path never
// writes anything of size seq x seq, because an integrated GPU has no bandwidth
// to spare. This device has it: the scores and their gradient are stored, and
// nearly all the arithmetic goes through one register-tiled kernel, attn_gemm,
// at the speed of a GEMM. The row-at-a-time form this replaced ran 30x slower
// than torch's scaled_dot_product_attention on a Tesla T4.
//
// With S_ij = scale Q_i.K_j over j <= i and P the softmax of S over each row:
//
//   forward    O_i  = sum_j P_ij V_j           L_i = logsumexp(S_i)
//   backward   dV_j = sum_i P_ij dO_i
//              dS_ij = P_ij (dO_i.V_j - D_i)   D_i = dO_i.O_i
//              dK_j = scale sum_i dS_ij Q_i
//              dQ_i = scale sum_j dS_ij K_j
//
// Every element comes from one thread summing in a fixed order, so results are
// the same bits on every run. Heads of a GQA group write separate partials, and
// attn_group_sum adds them in head order.
#include "r2_math.h"

#define BM 64u
#define BN 64u
#define BK 16u
#define TM 8u
#define TN 8u
#define THREADS ((BM / TM) * (BN / TN))
#define LOADS_A ((BM * BK) / THREADS)
#define LOADS_B ((BK * BN) / THREADS)
#define BAND_TO_ROW 1u
#define BAND_FROM_ROW 2u

// One (sequence, head) product per grid z: C(i, j) = alpha sum_k A(i, k) B(k, j),
// where each operand is addressed as base + first * index + second * index, and
// base = sequence * ss + head * hs (the kv head of the group when *_kv is set).
// With tri set, only the part with j <= i is promised, and whole tiles above the
// diagonal are skipped. band says A is triangular, so a tile of rows only sums the
// k that can be nonzero: BAND_TO_ROW for A(i, k) = 0 past k > i, BAND_FROM_ROW for
// k < i. cuBLAS cannot be told this, and half of these products is that zero.
extern "C" __global__ void attn_gemm(const float *A, const float *B, float *C,
                                     unsigned m, unsigned k, unsigned n, unsigned nh, unsigned group, unsigned tri, unsigned band,
                                     unsigned a_ss, unsigned a_hs, unsigned a_is, unsigned a_ks, unsigned a_kv,
                                     unsigned b_ss, unsigned b_hs, unsigned b_ks, unsigned b_ns, unsigned b_kv,
                                     unsigned c_ss, unsigned c_hs, unsigned c_is, unsigned c_ns, float alpha) {
    static __shared__ float As[BK * BM];
    static __shared__ float Bs[BK * BN];
    const unsigned m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    if (tri && n0 >= m0 + BM) return;
    const unsigned s = blockIdx.z / nh, h = blockIdx.z % nh;
    const float *a = A + s * a_ss + (a_kv ? h / group : h) * a_hs;
    const float *b = B + s * b_ss + (b_kv ? h / group : h) * b_hs;
    float *c_out = C + s * c_ss + h * c_hs;
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    const unsigned tid = ty * (BN / TN) + tx;

    float c[TM][TN];
    for (unsigned i = 0; i < TM; i++)
        for (unsigned j = 0; j < TN; j++) c[i][j] = 0.0f;

    const unsigned k_lo = band == BAND_FROM_ROW ? m0 : 0u;
    const unsigned k_hi = band == BAND_TO_ROW && m0 + BM < k ? m0 + BM : k;
    for (unsigned k0 = k_lo; k0 < k_hi; k0 += BK) {
        for (unsigned t = 0; t < LOADS_A; t++) {
            const unsigned idx = tid + t * THREADS;
            const unsigned ii = m0 + idx % BM, kk = k0 + idx / BM;
            As[idx] = (ii < m && kk < k) ? a[ii * a_is + kk * a_ks] : 0.0f;
        }
        for (unsigned t = 0; t < LOADS_B; t++) {
            const unsigned idx = tid + t * THREADS;
            const unsigned jj = n0 + idx % BN, kk = k0 + idx / BN;
            Bs[idx] = (jj < n && kk < k) ? b[kk * b_ks + jj * b_ns] : 0.0f;
        }
        __syncthreads();
        for (unsigned kk = 0; kk < BK; kk++) {
            float x[TM], y[TN];
            for (unsigned i = 0; i < TM; i++) x[i] = As[kk * BM + ty * TM + i];
            for (unsigned j = 0; j < TN; j++) y[j] = Bs[kk * BN + tx * TN + j];
            for (unsigned i = 0; i < TM; i++)
                for (unsigned j = 0; j < TN; j++) c[i][j] = fmaf(x[i], y[j], c[i][j]);
        }
        __syncthreads();
    }

    const unsigned mbase = m0 + ty * TM, nbase = n0 + tx * TN;
    for (unsigned i = 0; i < TM; i++) {
        if (mbase + i >= m) break;
        for (unsigned j = 0; j < TN; j++) {
            if (nbase + j >= n) break;
            c_out[(mbase + i) * c_is + (nbase + j) * c_ns] = c[i][j] * alpha;
        }
    }
}

// The row kernels below run ROWS rows to a block, one warp to a row: a warp's
// LANES threads take consecutive elements, so its 32 loads or stores fall in one
// cache line. One thread per row, which this replaced, sends every warp to 32
// lines at once, and took more time than all the GEMMs on a Tesla T4. A reduction
// over a row goes through shared memory in lane order, so its bits repeat.
#define LANES 32u
#define ROWS 8u

#define ROW_THREAD \
    const unsigned lane = threadIdx.x % LANES, warp = threadIdx.x / LANES; \
    const unsigned idx = blockIdx.x * ROWS + warp;

// In place on S, one row per (sequence, head, query): the causal softmax of the
// row, with L its log-sum-exp; entries past the diagonal become 0.
extern "C" __global__ void attn_softmax(float *S, float *L, unsigned nseq, unsigned seq, unsigned nh) {
    static __shared__ float red[ROWS * LANES];
    ROW_THREAD
    if (idx >= nseq * nh * seq) return;
    const unsigned z = idx / seq, i = idx % seq;
    float *row = S + idx * seq;
    const float *mine = red + warp * LANES;
    float m = -3.0e38f, l = 0.0f;
    for (unsigned c = lane; c <= i; c += LANES) m = fmaxf(m, row[c]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (unsigned u = 0; u < LANES; u++) m = fmaxf(m, mine[u]);
    __syncthreads();
    for (unsigned c = lane; c < seq; c += LANES) {
        if (c <= i) {
            const float p = exp_r2(row[c] - m);
            row[c] = p;
            l += p;
        } else {
            row[c] = 0.0f;
        }
    }
    red[threadIdx.x] = l;
    __syncthreads();
    l = 0.0f;
    for (unsigned u = 0; u < LANES; u++) l += mine[u];
    const float inv = 1.0f / l;
    for (unsigned c = lane; c <= i; c += LANES) row[c] = row[c] * inv;
    if (lane == 0) {
        const unsigned sq = z / nh, h = z % nh;
        L[(sq * seq + i) * nh + h] = m + logf(l);
    }
}

// In place on S: P_ij = exp(S_ij - L_i), the forward's P from its L.
extern "C" __global__ void attn_probs(float *S, const float *L, unsigned nseq, unsigned seq, unsigned nh) {
    ROW_THREAD
    if (idx >= nseq * nh * seq) return;
    const unsigned z = idx / seq, i = idx % seq;
    const float li = L[(z / nh * seq + i) * nh + z % nh];
    float *row = S + idx * seq;
    for (unsigned c = lane; c < seq; c += LANES) row[c] = c <= i ? exp_r2(row[c] - li) : 0.0f;
}

// In place on GV: dS_ij = P_ij (GV_ij - D_i); entries past the diagonal become 0.
extern "C" __global__ void attn_dscore(float *GV, const float *P, const float *D, unsigned nseq, unsigned seq, unsigned nh) {
    ROW_THREAD
    if (idx >= nseq * nh * seq) return;
    const unsigned z = idx / seq, i = idx % seq;
    const float di = D[(z / nh * seq + i) * nh + z % nh];
    float *row = GV + idx * seq;
    const float *p = P + idx * seq;
    for (unsigned c = lane; c < seq; c += LANES) row[c] = c <= i ? p[c] * (row[c] - di) : 0.0f;
}

// D_i = dO_i . O_i, one row per (row, head).
extern "C" __global__ void attn_delta(const float *O, const float *G, float *D, unsigned rows, unsigned hd) {
    static __shared__ float red[ROWS * LANES];
    ROW_THREAD
    if (idx >= rows) return;
    float s = 0.0f;
    for (unsigned c = lane; c < hd; c += LANES) s = fmaf(O[idx * hd + c], G[idx * hd + c], s);
    red[threadIdx.x] = s;
    __syncthreads();
    if (lane == 0) {
        s = 0.0f;
        for (unsigned u = 0; u < LANES; u++) s += red[warp * LANES + u];
        D[idx] = s;
    }
}

// OUT[row][kv head][c] = sum over the group's heads of PART[row][head][c], in head order.
extern "C" __global__ void attn_group_sum(const float *PART, float *OUT, unsigned rows, unsigned nkv, unsigned hd, unsigned group) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nkv * hd) return;
    const unsigned c = i % hd, kvh = (i / hd) % nkv, row = i / (hd * nkv);
    const unsigned base = row * nkv * group * hd + kvh * group * hd + c;
    float acc = 0.0f;
    for (unsigned g = 0; g < group; g++) acc += PART[base + g * hd];
    OUT[i] = acc;
}
