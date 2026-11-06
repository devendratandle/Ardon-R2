// Adam, operation for operation the CPU kernel (`r2_train::optim::adam_scalar`)
// in the same association. CUDA's float / and sqrtf are correctly rounded by
// default, and -fmad=false keeps the products and sums separate as the CPU's
// are, so this is bit-identical without the WGSL path's div_cr and sqrt_cr.
#include <math.h>

extern "C" __global__ void adam_step(float *W, const float *G, float *M, float *V, unsigned n,
                                     float b1, float b2, float bc1, float bc2,
                                     float lr, float eps, float scale, float c) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = G[i] / scale;
    const float m = b1 * M[i] + (1.0f - b1) * gi;
    const float v = b2 * V[i] + (1.0f - b2) * gi * gi;
    M[i] = m;
    V[i] = v;
    const float mh = m / bc1;
    const float vh = v / bc2;
    float w = W[i];
    if (c != 0.0f) w = fmaf(-c, w, w);
    W[i] = w - lr * mh / (sqrtf(vh) + eps);
}

// G = (G + PEER[base..]) / 2. Addition commutes bit for bit, so two ranks each
// doing this with the other's values hold the same result.
extern "C" __global__ void grad_mean(float *G, const float *PEER, unsigned base, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    G[i] = (G[i] + PEER[base + i]) * 0.5f;
}

#define NORM_CHUNK 256u

// Sum of g^2 per chunk with its rounding errors kept (TwoSum, and fmaf for each
// square): one thread per chunk, the host adds the chunks in f64. As in
// `src/optim.rs`, whose chunk size this is.
extern "C" __global__ void grad_sq_chunk(const float *G, float *OUT, unsigned n, unsigned base) {
    const unsigned chunk = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned start = chunk * NORM_CHUNK;
    if (start >= n) return;
    const unsigned end = min(start + NORM_CHUNK, n);
    float hi = 0.0f, lo = 0.0f;
    for (unsigned i = start; i < end; i++) {
        const float x = G[i];
        const float a = x * x;
        const float s = hi + a;
        const float b = s - hi;
        lo = lo + ((hi - (s - b)) + (a - b)) + fmaf(x, x, -a);
        hi = s;
    }
    OUT[2u * (base + chunk)] = hi;
    OUT[2u * (base + chunk) + 1u] = lo;
}
