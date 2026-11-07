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
#define TN 4u
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
        // Lanes walk the index that is contiguous in memory, so a warp's loads share
        // a cache line: the other way this load ran at a third of the speed on a T4.
        for (unsigned t = 0; t < LOADS_A; t++) {
            const unsigned idx = tid + t * THREADS;
            const unsigned r = a_ks < a_is ? idx / BK : idx % BM, s = a_ks < a_is ? idx % BK : idx / BM;
            const unsigned ii = m0 + r, kk = k0 + s;
            As[s * BM + r] = (ii < m && kk < k) ? a[ii * a_is + kk * a_ks] : 0.0f;
        }
        for (unsigned t = 0; t < LOADS_B; t++) {
            const unsigned idx = tid + t * THREADS;
            const unsigned r = b_ks < b_ns ? idx / BK : idx % BN, s = b_ks < b_ns ? idx % BK : idx / BN;
            const unsigned jj = n0 + r, kk = k0 + s;
            Bs[s * BN + r] = (jj < n && kk < k) ? b[kk * b_ks + jj * b_ns] : 0.0f;
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

// The forward as one kernel, for hd <= BN: scores stay in registers and the softmax
// is kept online, so nothing of size seq x seq goes through global memory, and the
// keys past a query tile's diagonal are never read. One block per (query tile,
// sequence x head). For each key tile: S in registers; each row's max and sum over
// the threads that share it; P to shared memory; O += P V, in registers, rescaled
// whenever a row's max moves. O and L are written at the end.
// Four warps to a block, 8 x 4 values to a thread, as PyTorch's kernel on this
// device: at 8 x 8 and two warps the same work ran 3x slower than that kernel.
#define FM 8u
#define FN 4u
#define FTHREADS ((BM / FM) * (BN / FN))
#define FLOADS_B ((BK * BN) / FTHREADS)

extern "C" __global__ void attn_fused(const float *Q, const float *K, const float *V, float *O, float *L,
                                      unsigned nseq, unsigned seq, unsigned nh, unsigned nkv, unsigned hd, float scale) {
    static __shared__ float As[BM * BM];
    static __shared__ float Bs[BK * BN];
    static __shared__ float Ps[BM * BN];
    static __shared__ float Red[BM * (BN / FN)];
    const unsigned m0 = blockIdx.x * BM;
    const unsigned s = blockIdx.y / nh, h = blockIdx.y % nh, group = nh / nkv;
    const unsigned qrow = nh * hd, kvrow = nkv * hd;
    const float *q = Q + s * seq * qrow + h * hd;
    const float *kk_base = K + s * seq * kvrow + (h / group) * hd;
    const float *v = V + s * seq * kvrow + (h / group) * hd;
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    const unsigned tid = ty * (BN / FN) + tx;
    const unsigned mbase = m0 + ty * FM;

    float o[FM][FN], mi[FM], si[FM];
    for (unsigned i = 0; i < FM; i++) {
        mi[i] = -3.0e38f;
        si[i] = 0.0f;
        for (unsigned j = 0; j < FN; j++) o[i][j] = 0.0f;
    }

    // The query tile, once: every key tile below reads the same one. hd <= BM.
    for (unsigned t = 0; t < (BM * BM) / FTHREADS; t++) {
        const unsigned idx = tid + t * FTHREADS;
        const unsigned ii = m0 + idx / BM, kk = idx % BM;
        As[kk * BM + idx / BM] = (ii < seq && kk < hd) ? q[ii * qrow + kk] : 0.0f;
    }

    for (unsigned n0 = 0; n0 < seq && n0 < m0 + BM; n0 += BN) {
        const unsigned nbase = n0 + tx * FN;
        // S = scale Q K^T for this tile
        float c[FM][FN];
        for (unsigned i = 0; i < FM; i++)
            for (unsigned j = 0; j < FN; j++) c[i][j] = 0.0f;
        for (unsigned k0 = 0; k0 < hd; k0 += BK) {
            for (unsigned t = 0; t < FLOADS_B; t++) {
                const unsigned idx = tid + t * FTHREADS;
                const unsigned r = idx / BK, s = idx % BK;
                const unsigned jj = n0 + r, kk = k0 + s;
                Bs[s * BN + r] = (jj < seq && kk < hd) ? kk_base[jj * kvrow + kk] : 0.0f;
            }
            __syncthreads();
            for (unsigned kk = 0; kk < BK; kk++) {
                float x[FM], y[FN];
                for (unsigned i = 0; i < FM; i++) x[i] = As[(k0 + kk) * BM + ty * FM + i];
                for (unsigned j = 0; j < FN; j++) y[j] = Bs[kk * BN + tx * FN + j];
                for (unsigned i = 0; i < FM; i++)
                    for (unsigned j = 0; j < FN; j++) c[i][j] = fmaf(x[i], y[j], c[i][j]);
            }
            __syncthreads();
        }
        // each row's new max, over the threads sharing the row
        for (unsigned i = 0; i < FM; i++) {
            const unsigned row = mbase + i;
            float m = -3.0e38f;
            for (unsigned j = 0; j < FN; j++) {
                c[i][j] *= scale;
                if (row < seq && nbase + j < seq && nbase + j <= row) m = fmaxf(m, c[i][j]);
            }
            Red[(ty * FM + i) * (BN / FN) + tx] = m;
        }
        __syncthreads();
        float rescale[FM];
        for (unsigned i = 0; i < FM; i++) {
            float m = mi[i];
            for (unsigned u = 0; u < BN / FN; u++) m = fmaxf(m, Red[(ty * FM + i) * (BN / FN) + u]);
            rescale[i] = n0 == 0 ? 1.0f : exp_r2(mi[i] - m);
            mi[i] = m;
        }
        __syncthreads();
        // P, unnormalized, against the new max; the row's sum
        for (unsigned i = 0; i < FM; i++) {
            const unsigned row = mbase + i;
            float l = 0.0f;
            for (unsigned j = 0; j < FN; j++) {
                float p = 0.0f;
                if (row < seq && nbase + j < seq && nbase + j <= row) {
                    p = exp_r2(c[i][j] - mi[i]);
                    l += p;
                }
                Ps[(ty * FM + i) * BN + tx * FN + j] = p;
                o[i][j] *= rescale[i];
            }
            Red[(ty * FM + i) * (BN / FN) + tx] = l;
        }
        __syncthreads();
        for (unsigned i = 0; i < FM; i++) {
            float l = 0.0f;
            for (unsigned u = 0; u < BN / FN; u++) l += Red[(ty * FM + i) * (BN / FN) + u];
            si[i] = si[i] * rescale[i] + l;
        }
        __syncthreads();
        // O += P V: keys are the summed index, V's columns the output's
        for (unsigned k0 = 0; k0 < BN; k0 += BK) {
            for (unsigned t = 0; t < FLOADS_B; t++) {
                const unsigned idx = tid + t * FTHREADS;
                const unsigned cc = idx % BN, key = n0 + k0 + idx / BN;
                Bs[idx] = (key < seq && cc < hd) ? v[key * kvrow + cc] : 0.0f;
            }
            __syncthreads();
            for (unsigned kk = 0; kk < BK; kk++) {
                float x[FM], y[FN];
                for (unsigned i = 0; i < FM; i++) x[i] = Ps[(ty * FM + i) * BN + k0 + kk];
                for (unsigned j = 0; j < FN; j++) y[j] = Bs[kk * BN + tx * FN + j];
                for (unsigned i = 0; i < FM; i++)
                    for (unsigned j = 0; j < FN; j++) o[i][j] = fmaf(x[i], y[j], o[i][j]);
            }
            __syncthreads();
        }
    }

    float *out = O + s * seq * qrow + h * hd;
    for (unsigned i = 0; i < FM; i++) {
        const unsigned row = mbase + i;
        if (row >= seq) break;
        const float inv = 1.0f / si[i];
        for (unsigned j = 0; j < FN; j++) {
            const unsigned col = tx * FN + j;
            if (col >= hd) break;
            out[row * qrow + col] = o[i][j] * inv;
        }
        if (tx == 0) L[(s * seq + row) * nh + h] = mi[i] + logf(si[i]);
    }
}

// The start of the backward as one kernel, for hd <= BM, as attn_fused is the forward:
// P_ij = exp(S_ij - L_i) and dS_ij = P_ij (dO_i.V_j - D_i), both products in registers,
// so neither score matrix goes through global memory to reach its row kernel. K and V
// are read together, and each query tile and its gradient once. Only tiles up to a
// row's diagonal tile are written, which is all the band products read. The sums run
// in the order attn_gemm's do, so the bits are the same as the stored form's.
extern "C" __global__ void attn_scores(const float *Q, const float *K, const float *G, const float *V,
                                       const float *L, const float *D, float *P, float *DS,
                                       unsigned nseq, unsigned seq, unsigned nh, unsigned nkv, unsigned hd, float scale) {
    static __shared__ float As[BM * BM];
    static __shared__ float Gs[BM * BM];
    static __shared__ float Bs[BK * BN];
    static __shared__ float Vs[BK * BN];
    const unsigned m0 = blockIdx.x * BM;
    const unsigned s = blockIdx.y / nh, h = blockIdx.y % nh, group = nh / nkv;
    const unsigned qrow = nh * hd, kvrow = nkv * hd;
    const float *q = Q + s * seq * qrow + h * hd;
    const float *g = G + s * seq * qrow + h * hd;
    const float *kk_base = K + s * seq * kvrow + (h / group) * hd;
    const float *v = V + s * seq * kvrow + (h / group) * hd;
    float *p_out = P + (s * nh + h) * seq * seq;
    float *ds_out = DS + (s * nh + h) * seq * seq;
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    const unsigned tid = ty * (BN / FN) + tx;
    const unsigned mbase = m0 + ty * FM;

    float li[FM], di[FM];
    for (unsigned i = 0; i < FM; i++) {
        const unsigned row = mbase + i;
        li[i] = row < seq ? L[(s * seq + row) * nh + h] : 0.0f;
        di[i] = row < seq ? D[(s * seq + row) * nh + h] : 0.0f;
    }

    for (unsigned t = 0; t < (BM * BM) / FTHREADS; t++) {
        const unsigned idx = tid + t * FTHREADS;
        const unsigned ii = m0 + idx / BM, kk = idx % BM;
        const bool in = ii < seq && kk < hd;
        As[kk * BM + idx / BM] = in ? q[ii * qrow + kk] : 0.0f;
        Gs[kk * BM + idx / BM] = in ? g[ii * qrow + kk] : 0.0f;
    }

    for (unsigned n0 = 0; n0 < seq && n0 < m0 + BM; n0 += BN) {
        const unsigned nbase = n0 + tx * FN;
        float c[FM][FN], d[FM][FN];
        for (unsigned i = 0; i < FM; i++)
            for (unsigned j = 0; j < FN; j++) { c[i][j] = 0.0f; d[i][j] = 0.0f; }
        for (unsigned k0 = 0; k0 < hd; k0 += BK) {
            for (unsigned t = 0; t < FLOADS_B; t++) {
                const unsigned idx = tid + t * FTHREADS;
                const unsigned r = idx / BK, sk = idx % BK;
                const unsigned jj = n0 + r, kk = k0 + sk;
                const bool in = jj < seq && kk < hd;
                Bs[sk * BN + r] = in ? kk_base[jj * kvrow + kk] : 0.0f;
                Vs[sk * BN + r] = in ? v[jj * kvrow + kk] : 0.0f;
            }
            __syncthreads();
            for (unsigned kk = 0; kk < BK; kk++) {
                float x[FM], w[FM], y[FN], z[FN];
                for (unsigned i = 0; i < FM; i++) {
                    x[i] = As[(k0 + kk) * BM + ty * FM + i];
                    w[i] = Gs[(k0 + kk) * BM + ty * FM + i];
                }
                for (unsigned j = 0; j < FN; j++) {
                    y[j] = Bs[kk * BN + tx * FN + j];
                    z[j] = Vs[kk * BN + tx * FN + j];
                }
                for (unsigned i = 0; i < FM; i++)
                    for (unsigned j = 0; j < FN; j++) {
                        c[i][j] = fmaf(x[i], y[j], c[i][j]);
                        d[i][j] = fmaf(w[i], z[j], d[i][j]);
                    }
            }
            __syncthreads();
        }
        for (unsigned i = 0; i < FM; i++) {
            const unsigned row = mbase + i;
            if (row >= seq) break;
            for (unsigned j = 0; j < FN; j++) {
                const unsigned col = nbase + j;
                if (col >= seq) break;
                float pv = 0.0f, dv = 0.0f;
                if (col <= row) {
                    pv = exp_r2(c[i][j] * scale - li[i]);
                    dv = pv * (d[i][j] - di[i]);
                }
                p_out[row * seq + col] = pv;
                ds_out[row * seq + col] = dv;
            }
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

// Where a row's zeros stop mattering: the band products read none past the end of
// the row's diagonal tile. tile 0 asks for the whole row, for a reader with no band.
static inline __device__ unsigned row_end(unsigned i, unsigned seq, unsigned tile) {
    if (tile == 0u) return seq;
    const unsigned end = (i / tile + 1u) * tile;
    return end < seq ? end : seq;
}

// In place on S: P_ij = exp(S_ij - L_i), the forward's P from its L.
extern "C" __global__ void attn_probs(float *S, const float *L, unsigned nseq, unsigned seq, unsigned nh, unsigned tile) {
    ROW_THREAD
    if (idx >= nseq * nh * seq) return;
    const unsigned z = idx / seq, i = idx % seq, end = row_end(i, seq, tile);
    const float li = L[(z / nh * seq + i) * nh + z % nh];
    float *row = S + idx * seq;
    for (unsigned c = lane; c < end; c += LANES) row[c] = c <= i ? exp_r2(row[c] - li) : 0.0f;
}

// In place on GV: dS_ij = P_ij (GV_ij - D_i); entries past the diagonal become 0.
extern "C" __global__ void attn_dscore(float *GV, const float *P, const float *D, unsigned nseq, unsigned seq, unsigned nh, unsigned tile) {
    ROW_THREAD
    if (idx >= nseq * nh * seq) return;
    const unsigned z = idx / seq, i = idx % seq, end = row_end(i, seq, tile);
    const float di = D[(z / nh * seq + i) * nh + z % nh];
    float *row = GV + idx * seq;
    const float *p = P + idx * seq;
    for (unsigned c = lane; c < end; c += LANES) row[c] = c <= i ? p[c] * (row[c] - di) : 0.0f;
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
