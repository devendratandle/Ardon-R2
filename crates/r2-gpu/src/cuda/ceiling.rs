//! Launches for `kernels/ceiling.cu`, which measure the card and no part of training:
//! what `examples/cuda_ceiling.rs` holds each operation against.

use super::ffi::{Arg, Kernel};
use super::{loaded, Loaded, Tensor};
use std::sync::OnceLock;

const BLOCK: u32 = 256;
const WARPS: usize = 8;

const PTX: &str = include_str!("kernels/ceiling.ptx");

struct Kernels { fma: Kernel, read: Kernel, write: Kernel, copy: Kernel, tile: Kernel }
static KERNELS: Loaded<Kernels> = OnceLock::new();

fn kernels() -> Option<&'static Kernels> {
    loaded(&KERNELS, PTX, |m| Ok(Kernels {
        fma: m.kernel("ceil_fma", 4)?,
        read: m.kernel("ceil_read", 3)?,
        write: m.kernel("ceil_write", 3)?,
        copy: m.kernel("ceil_copy", 3)?,
        tile: m.kernel("ceil_tile", 6)?,
    })).ok()
}

/// `blocks` blocks of 256 threads, each thread `iters` x 8 multiply-adds in registers.
pub fn fma(out: &Tensor, blocks: u32, iters: u32) -> bool {
    kernels().is_some_and(|k| k.fma.launch([blocks, 1, 1], [BLOCK, 1, 1], &[Arg::F(out), Arg::Int(iters), Arg::Real(0.999), Arg::Real(1.0e-3)]).is_ok())
}

/// Warps and elements per warp that cover `n` elements in `len`-element runs.
fn split(n: usize, len: usize) -> (u32, u32) { ((n / len / WARPS).max(1) as u32, len as u32) }

pub fn read(a: &Tensor, out: &Tensor, n: usize, len: usize) -> bool {
    let (blocks, len) = split(n, len);
    kernels().is_some_and(|k| k.read.launch([blocks, 1, 1], [BLOCK, 1, 1], &[Arg::F(a), Arg::F(out), Arg::Int(len)]).is_ok())
}

pub fn write(y: &Tensor, n: usize, len: usize) -> bool {
    let (blocks, len) = split(n, len);
    kernels().is_some_and(|k| k.write.launch([blocks, 1, 1], [BLOCK, 1, 1], &[Arg::F(y), Arg::Int(len), Arg::Real(1.0)]).is_ok())
}

pub fn copy(a: &Tensor, y: &Tensor, n: usize, len: usize) -> bool {
    let (blocks, len) = split(n, len);
    kernels().is_some_and(|k| k.copy.launch([blocks, 1, 1], [BLOCK, 1, 1], &[Arg::F(a), Arg::F(y), Arg::Int(len)]).is_ok())
}

/// The loads of every tile of an `m x k` matrix, addressed `a(i, j) = A[i * is + j * ks]`.
pub fn tile(a: &Tensor, out: &Tensor, m: usize, k: usize, is: usize, ks: usize) -> bool {
    let blocks = m.div_ceil(64) as u32;
    kernels().is_some_and(|ks_| ks_.tile.launch([blocks, 1, 1], [16, 8, 1], &[
        Arg::F(a), Arg::F(out), Arg::Int(m as u32), Arg::Int(k as u32), Arg::Int(is as u32), Arg::Int(ks as u32),
    ]).is_ok())
}
