// The elementwise tail of a transformer step, forward and backward: rmsnorm,
// SiLU, products and sums, RoPE, softmax cross-entropy, the embedding gather.
// The operations and the meaning of each code are `src/elementwise.rs`'s.
//
// One thread owns one output element or row and sums in a fixed order, so the
// results are the same bits on every run. The backward ops that scatter on a
// CPU are written as gathers, and no kernel uses atomics. Where the wgpu kernels
// reduce a row across a workgroup, one thread here walks the row: this device
// has lanes to spare, and the row reductions need no barriers this way.
#include "r2_math.h"

#define IDX (blockIdx.x * blockDim.x + threadIdx.x)

// y[r] = x[r] / rms(x[r]) * w, one thread per row.
extern "C" __global__ void rmsnorm_fwd(const float *X, const float *W, float *Y, unsigned rows, unsigned d, float eps) {
    const unsigned r = IDX;
    if (r >= rows) return;
    const float *x = X + r * d;
    float ss = 0.0f;
    for (unsigned j = 0; j < d; j++) ss = fmaf(x[j], x[j], ss);
    const float scale = 1.0f / sqrtf(ss / (float)d + eps);
    for (unsigned j = 0; j < d; j++) Y[r * d + j] = x[j] * scale * W[j];
}

// dX (assign or accumulate), and the per-row rinv the dW pass reads.
extern "C" __global__ void rmsnorm_bwd_x(const float *X, const float *W, const float *G, float *GX, float *RI,
                                         unsigned rows, unsigned d, float eps, unsigned acc) {
    const unsigned r = IDX;
    if (r >= rows) return;
    const float *x = X + r * d;
    const float *g = G + r * d;
    float ss = 0.0f, sg = 0.0f;
    for (unsigned j = 0; j < d; j++) {
        ss = fmaf(x[j], x[j], ss);
        sg = fmaf(g[j] * W[j], x[j], sg);
    }
    const float ri = 1.0f / sqrtf(ss / (float)d + eps);
    const float coef = ri * ri * ri / (float)d * sg;
    RI[r] = ri;
    for (unsigned j = 0; j < d; j++) {
        const float dx = g[j] * W[j] * ri - coef * x[j];
        if (acc) GX[r * d + j] += dx; else GX[r * d + j] = dx;
    }
}

// dW, stage one: PART[b][j] = sum over the rows of block b of G X rinv, one
// thread per (block, column), rows in order. S holds rinv from 0 and the
// partials from part_off. One thread per column over ALL rows would walk the
// whole matrix at a stride with few lanes busy.
extern "C" __global__ void rmsnorm_bwd_w_part(const float *X, const float *G, float *S, unsigned part_off,
                                              unsigned rows, unsigned d, unsigned parts) {
    const unsigned i = IDX;
    if (i >= parts * d) return;
    const unsigned b = i / d, j = i % d;
    const unsigned chunk = (rows + parts - 1u) / parts;
    const unsigned r0 = b * chunk, r1 = min(r0 + chunk, rows);
    float acc = 0.0f;
    for (unsigned r = r0; r < r1; r++) acc = fmaf(G[r * d + j] * X[r * d + j], S[r], acc);
    S[part_off + b * d + j] = acc;
}

// dW, stage two: GW[j] (=|+=) sum over b of PART[b][j], blocks in order.
extern "C" __global__ void rmsnorm_bwd_w_sum(const float *S, unsigned part_off, float *GW, unsigned d, unsigned parts, unsigned add) {
    const unsigned j = IDX;
    if (j >= d) return;
    float acc = 0.0f;
    for (unsigned b = 0; b < parts; b++) acc += S[part_off + b * d + j];
    if (add) GW[j] += acc; else GW[j] = acc;
}

__device__ __forceinline__ float silu(float a) { return a / (1.0f + exp_r2(-a)); }

// Y = f(A, B). op: 0 silu(A), 1 A*B, 2 A+B, 3 silu(A)*B, 4 Y + A, 5 A.
extern "C" __global__ void flat_fwd(const float *A, const float *B, float *Y, unsigned n, unsigned op) {
    const unsigned i = IDX;
    if (i >= n) return;
    const float a = A[i];
    float y;
    switch (op) {
        case 0: y = silu(a); break;
        case 1: y = a * B[i]; break;
        case 2: y = a + B[i]; break;
        case 3: y = silu(a) * B[i]; break;
        case 4: y = Y[i] + a; break;
        default: y = a; break;
    }
    Y[i] = y;
}

// Backward of the flat ops. op: 0 silu, 1 mul, 2 add. ga_add / gb_add choose
// accumulate over assign; ga_on / gb_on say whether each is wanted at all.
extern "C" __global__ void flat_bwd(const float *A, const float *B, const float *G, float *GA, float *GB, unsigned n,
                                    unsigned op, unsigned ga_add, unsigned gb_add, unsigned ga_on, unsigned gb_on) {
    const unsigned i = IDX;
    if (i >= n) return;
    const float g = G[i];
    float da, db = 0.0f;
    if (op == 0) {
        const float a = A[i];
        const float s = 1.0f / (1.0f + exp_r2(-a));
        da = g * (s + a * s * (1.0f - s));
    } else if (op == 1) {
        da = g * B[i];
        db = g * A[i];
    } else {
        da = g;
        db = g;
    }
    if (ga_on) { if (ga_add) GA[i] += da; else GA[i] = da; }
    if (gb_on) { if (gb_add) GB[i] += db; else GB[i] = db; }
}

// RoPE, one thread per (row, head, pair), angles from the host's table. bwd = 1
// is the inverse rotation, with add choosing accumulate over assign.
extern "C" __global__ void rope(const float *X, const float *TAB, float *Y, unsigned rows, unsigned hd,
                                unsigned period, unsigned heads, unsigned bwd, unsigned add) {
    const unsigned half = hd / 2u, i = IDX;
    if (i >= rows * heads * half) return;
    const unsigned pr = i % half, h = (i / half) % heads, r = i / (half * heads);
    const unsigned t = ((r % period) * half + pr) * 2u;
    const float c = TAB[t], s = TAB[t + 1u];
    const unsigned o = r * heads * hd + h * hd + 2u * pr;
    const float a = X[o], b = X[o + 1u];
    if (!bwd) {
        Y[o] = a * c - b * s;
        Y[o + 1u] = a * s + b * c;
    } else {
        const float ra = a * c + b * s, rb = -a * s + b * c;
        if (add) { Y[o] += ra; Y[o + 1u] += rb; } else { Y[o] = ra; Y[o + 1u] = rb; }
    }
}

#define IGNORED 0xFFFFFFFFu

// Per row: LSE[r] = logsumexp(row), LOSS[r] = LSE[r] - x[target]; an ignored
// row has no loss at all.
extern "C" __global__ void softmax_ce_fwd(const float *X, const unsigned *ids, float *LSE, float *LOSS, unsigned rows, unsigned d) {
    const unsigned r = IDX;
    if (r >= rows) return;
    if (ids[r] == IGNORED) { LSE[r] = 0.0f; LOSS[r] = 0.0f; return; }
    const float *x = X + r * d;
    float m = -3.0e38f, sum = 0.0f;
    for (unsigned j = 0; j < d; j++) m = fmaxf(m, x[j]);
    for (unsigned j = 0; j < d; j++) sum += exp_r2(x[j] - m);
    const float lse = m + logf(sum);
    LSE[r] = lse;
    LOSS[r] = lse - x[ids[r]];
}

// GX (=|+=) inv * (exp(x - lse) - [j == target]).
extern "C" __global__ void softmax_ce_bwd(const float *X, const unsigned *ids, const float *LSE, float *GX,
                                          unsigned n, unsigned d, float inv, unsigned add) {
    const unsigned i = IDX;
    if (i >= n) return;
    const unsigned r = i / d, j = i % d;
    if (ids[r] == IGNORED) {
        if (!add) GX[i] = 0.0f;
        return;
    }
    float v = exp_r2(X[i] - LSE[r]);
    if (j == ids[r]) v -= 1.0f;
    v = v * inv;
    if (add) GX[i] += v; else GX[i] = v;
}

// FLAG[at] = 1 if any of X[0..n] is inf or nan. Racing writes of the same 1 are benign.
extern "C" __global__ void nonfinite(const float *X, float *FLAG, unsigned n, unsigned at) {
    const unsigned i = IDX;
    if (i >= n) return;
    if (!isfinite(X[i])) FLAG[at] = 1.0f;
}

// Y[i][:] = T[ids[i]][:].
extern "C" __global__ void embed_fwd(const float *T, const unsigned *ids, float *Y, unsigned n, unsigned d) {
    const unsigned i = IDX;
    if (i >= n) return;
    Y[i] = T[ids[i / d] * d + i % d];
}

// GT[v][j] (=|+=) sum over the positions of token v of G[pos][j], with those
// positions given as CSR. One thread per (v, j), positions in order.
extern "C" __global__ void embed_bwd(const float *G, const unsigned *offs, const unsigned *pos, float *GT,
                                     unsigned n, unsigned d, unsigned add) {
    const unsigned i = IDX;
    if (i >= n) return;
    const unsigned v = i / d, j = i % d;
    float acc = 0.0f;
    for (unsigned k = offs[v]; k < offs[v + 1u]; k++) acc += G[pos[k] * d + j];
    if (add) GT[i] += acc; else GT[i] = acc;
}
