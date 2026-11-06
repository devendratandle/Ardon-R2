// What the card itself can do, with none of R2's work in the way: registers only,
// then memory in the patterns the kernels use. These kernels are never part of
// training. The figures are what an operation's time is held against.
#include <math.h>

#define LANES 32u
#define BLOCK 256u
#define WARPS (BLOCK / LANES)

#define WARP_THREAD \
    const unsigned lane = threadIdx.x % LANES; \
    const unsigned warp = blockIdx.x * WARPS + threadIdx.x / LANES;

// Eight fused multiply-adds an iteration, in registers, touching memory once at the end.
extern "C" __global__ void ceil_fma(float *out, unsigned iters, float x, float y) {
    float c[8];
    for (unsigned j = 0; j < 8u; j++) c[j] = (float)(threadIdx.x + j) * 1.0e-3f;
    for (unsigned it = 0; it < iters; it++)
        for (unsigned j = 0; j < 8u; j++) c[j] = fmaf(c[j], x, y);
    float s = 0.0f;
    for (unsigned j = 0; j < 8u; j++) s += c[j];
    out[blockIdx.x * BLOCK + threadIdx.x] = s;
}

// One warp to `len` consecutive elements, its lanes side by side, as the row kernels.
extern "C" __global__ void ceil_read(const float *A, float *out, unsigned len) {
    WARP_THREAD
    float s = 0.0f;
    for (unsigned j = lane; j < len; j += LANES) s += A[warp * len + j];
    out[blockIdx.x * BLOCK + threadIdx.x] = s;
}

extern "C" __global__ void ceil_write(float *Y, unsigned len, float v) {
    WARP_THREAD
    for (unsigned j = lane; j < len; j += LANES) Y[warp * len + j] = v;
}

extern "C" __global__ void ceil_copy(const float *A, float *Y, unsigned len) {
    WARP_THREAD
    for (unsigned j = lane; j < len; j += LANES) Y[warp * len + j] = A[warp * len + j];
}

// A tile's loads and nothing else: 64 rows, 16 columns at a time, into shared memory,
// with four warps to a block as the tiled products. a(i, k) = A[i * is + k * ks].
#define BM 64u
#define BK 16u
#define THREADS 128u
extern "C" __global__ void ceil_tile(const float *A, float *out, unsigned m, unsigned k, unsigned is, unsigned ks) {
    static __shared__ float As[BK * BM];
    const unsigned tid = threadIdx.y * 16u + threadIdx.x, m0 = blockIdx.x * BM;
    float s = 0.0f;
    for (unsigned k0 = 0; k0 < k; k0 += BK) {
        for (unsigned t = 0; t < (BM * BK) / THREADS; t++) {
            const unsigned idx = tid + t * THREADS;
            const unsigned ii = m0 + idx % BM, kk = k0 + idx / BM;
            As[idx] = (ii < m && kk < k) ? A[ii * is + kk * ks] : 0.0f;
        }
        __syncthreads();
        s += As[tid];
        __syncthreads();
    }
    out[blockIdx.x * THREADS + tid] = s;
}
