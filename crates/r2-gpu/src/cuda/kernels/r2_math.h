// R2's exp, `r2_tensor::ops::exp_r2`, instruction for instruction: the same
// clamp, ties-to-even rounding, split-ln2 reduction, Horner chain in fmaf and
// 2^n from the exponent bits. Every step is exactly rounded, so the device
// gives the CPU's bits. rintf rounds ties to even in the default mode.
#pragma once
#include <math.h>

__device__ __forceinline__ float exp_r2(float x0) {
    if (x0 != x0) return x0;
    if (x0 < -87.33654f) return 0.0f;
    const float x = fminf(x0, 88.37626f);
    const float n = rintf(x * 1.44269504f);
    float r = fmaf(-n, 0.693359375f, x);
    r = fmaf(-n, -2.1219444e-4f, r);
    float y = 1.9875691e-4f;
    y = fmaf(y, r, 1.3981999e-3f);
    y = fmaf(y, r, 8.333452e-3f);
    y = fmaf(y, r, 4.1665796e-2f);
    y = fmaf(y, r, 1.6666665e-1f);
    y = fmaf(y, r, 5.0000001e-1f);
    y = fmaf(y, r * r, r);
    y = y + 1.0f;
    union { unsigned u; float f; } p;
    p.u = (unsigned)((int)n + 127) << 23;
    return y * p.f;
}
