#!/bin/sh
# Regenerate the committed .ptx from the .cu kernels. Run where nvcc is.
#
# -fmad=false: nvcc otherwise contracts a*b+c into one fused operation, and
# the kernels promise the CPU's bits. They ask for a fused operation by
# calling fmaf(), exactly where the CPU code calls mul_add().
set -e
cd "$(dirname "$0")"
for f in *.cu; do
    nvcc -ptx -arch=sm_75 -fmad=false -o "${f%.cu}.ptx" "$f"
done
