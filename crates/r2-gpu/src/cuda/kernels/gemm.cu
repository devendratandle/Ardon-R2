// sgemm, the tiled kernel: the WGSL one's structure (`src/gemm.rs`, which
// explains it) with the same constants, so the two stay comparable.
//
// A block owns a BM x BN tile of C and walks K in slabs of BK. Each slab of A
// and B is staged in shared memory once, whichever way the operand is stored,
// and every thread then accumulates a TM x TN register tile of C over the
// whole depth. Each element of C comes from one thread summing in a fixed
// order, so the result is the same bits on every run.
//
// When the output has few tiles the depth is split (grid z): each split writes
// a partial P[z][m][n], and sgemm_reduce sums the partials in split order.
#include <math.h>

#define BM 128u
#define BN 128u
#define BK 16u
#define TM 8u
#define TN 8u
#define THREADS ((BM / TM) * (BN / TN))
#define LOADS_A ((BM * BK) / THREADS)
#define LOADS_B ((BK * BN) / THREADS)

extern "C" __global__ void sgemm_tiled(const float *A, const float *B, float *C, float *P,
                                       unsigned m, unsigned k, unsigned n,
                                       unsigned lda, unsigned ldb, unsigned ldc,
                                       unsigned ta, unsigned tb, unsigned acc,
                                       unsigned ksplit, unsigned kchunk) {
    static __shared__ float As[BK * BM];
    static __shared__ float Bs[BK * BN];
    const unsigned m0 = blockIdx.y * BM;
    const unsigned n0 = blockIdx.x * BN;
    const unsigned kbeg = blockIdx.z * kchunk;
    const unsigned kend = min(kbeg + kchunk, k);
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    const unsigned tid = ty * (BN / TN) + tx;

    float c[TM][TN];
    for (unsigned i = 0; i < TM; i++)
        for (unsigned j = 0; j < TN; j++) c[i][j] = 0.0f;

    for (unsigned k0 = kbeg; k0 < kend; k0 += BK) {
        for (unsigned t = 0; t < LOADS_A; t++) {
            const unsigned idx = tid + t * THREADS;
            const unsigned r = ta ? idx % BM : idx / BK;
            const unsigned s = ta ? idx / BM : idx % BK;
            const unsigned mm = m0 + r, kk = k0 + s;
            float v = 0.0f;
            if (mm < m && kk < kend) v = ta ? A[kk * lda + mm] : A[mm * lda + kk];
            As[s * BM + r] = v;
        }
        for (unsigned t = 0; t < LOADS_B; t++) {
            const unsigned idx = tid + t * THREADS;
            const unsigned s = tb ? idx % BK : idx / BN;
            const unsigned r = tb ? idx / BK : idx % BN;
            const unsigned kk = k0 + s, nn = n0 + r;
            float v = 0.0f;
            if (kk < kend && nn < n) v = tb ? B[nn * ldb + kk] : B[kk * ldb + nn];
            Bs[s * BN + r] = v;
        }
        __syncthreads();
        for (unsigned kk = 0; kk < BK; kk++) {
            float a[TM], b[TN];
            for (unsigned i = 0; i < TM; i++) a[i] = As[kk * BM + ty * TM + i];
            for (unsigned j = 0; j < TN; j++) b[j] = Bs[kk * BN + tx * TN + j];
            for (unsigned i = 0; i < TM; i++)
                for (unsigned j = 0; j < TN; j++) c[i][j] = fmaf(a[i], b[j], c[i][j]);
        }
        __syncthreads();
    }

    const unsigned mbase = m0 + ty * TM, nbase = n0 + tx * TN;
    for (unsigned i = 0; i < TM; i++) {
        if (mbase + i >= m) break;
        for (unsigned j = 0; j < TN; j++) {
            if (nbase + j >= n) break;
            const unsigned o = (mbase + i) * ldc + nbase + j;
            if (ksplit > 1u) P[blockIdx.z * m * n + o] = c[i][j];
            else if (acc) C[o] += c[i][j];
            else C[o] = c[i][j];
        }
    }
}

// C (=|+=) sum over z of P[z], splits in order: one thread per element.
extern "C" __global__ void sgemm_reduce(const float *P, float *C, unsigned m, unsigned n,
                                        unsigned ksplit, unsigned acc) {
    const unsigned o = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned mn = m * n;
    if (o >= mn) return;
    float s = 0.0f;
    for (unsigned z = 0; z < ksplit; z++) s += P[z * mn + o];
    if (acc) C[o] += s; else C[o] = s;
}
