// The elementwise ops of `r2_gpu::Op`, in place, one thread per element.
// `op` is the code in `cuda::mod.rs`: 0 Identity, 1 Relu, 2 Sigmoid, 3 Tanh,
// 4 Exp, 5 Scale(a).
#include <math.h>

extern "C" __global__ void map_op(float *x, unsigned n, unsigned op, float a) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    switch (op) {
        case 1: v = v > 0.0f ? v : 0.0f; break;
        case 2: v = 1.0f / (1.0f + expf(-v)); break;
        case 3: v = tanhf(v); break;
        case 4: v = expf(v); break;
        case 5: v = a * v; break;
        default: break;
    }
    x[i] = v;
}
