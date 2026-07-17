//! r2-mesh — hardware topology, collectives, and sharding: the
//! trillion-scale training foundation (docs/LLM_TRILLION_ARCHITECTURE.md).
//!
//! THE PRINCIPLE: training code never names a device count. It speaks to
//! three abstractions — a declared hardware TOPOLOGY (cluster → node →
//! device slot, with bandwidth tiers), a COLLECTIVE trait (all_reduce &
//! friends), and SHARD descriptors (how each tensor is split). Swap the
//! in-process reference transport for NCCL/RCCL/MPI behind the same trait
//! and the identical training loop runs on one GPU or a hundred thousand.
//! That property — interfaces written for the mesh from day one — is the
//! industrial-grade "no ceiling" guarantee.
//!
//! This crate ships the trait definitions plus `ThreadCollective`, a real
//! multi-worker reference implementation whose collectives are
//! mathematically correct. It proves the seams on one machine; hardware
//! transports are a bring-up (Opus), not a redesign.

use std::sync::{Arc, Barrier, Mutex};

// ── Layer 1: hardware topology ("indexing child slots") ────────────────

/// A device slot's address in the cluster tree: (node index, slot index).
/// Nodes are hosts; slots are accelerators (GPU/NPU) or CPU workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId {
    pub node: u32,
    pub slot: u32,
}

/// Link speed class between devices. The sharding planner places
/// high-traffic groups (tensor-parallel) on Fast links and low-traffic
/// splits (pipeline stages, data-parallel) across Slow links — placing
/// the communication on the right wires is the whole game at scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkTier {
    /// Intra-node: NVLink / PCIe / shared memory.
    Fast,
    /// Inter-node: InfiniBand / Ethernet.
    Slow,
}

/// Declared hardware topology — pure description, no compute. Built from
/// config (mesh.toml, Opus) or synthesized for the in-process reference.
#[derive(Debug, Clone)]
pub struct Topology {
    /// nodes[i] = number of device slots on node i.
    pub nodes: Vec<u32>,
}

impl Topology {
    /// A single node with `slots` devices (the reference/laptop shape).
    pub fn single_node(slots: u32) -> Self {
        Topology { nodes: vec![slots] }
    }
    pub fn device_count(&self) -> usize {
        self.nodes.iter().map(|&s| s as usize).sum()
    }
    /// Every device id in deterministic order (node-major).
    pub fn devices(&self) -> Vec<DeviceId> {
        let mut out = Vec::new();
        for (n, &slots) in self.nodes.iter().enumerate() {
            for s in 0..slots {
                out.push(DeviceId { node: n as u32, slot: s });
            }
        }
        out
    }
    /// Link tier between two devices: same node ⇒ Fast, else Slow.
    pub fn link(&self, a: DeviceId, b: DeviceId) -> LinkTier {
        if a.node == b.node { LinkTier::Fast } else { LinkTier::Slow }
    }
    /// Fast-tier groups: the slots of each node (where tensor-parallel
    /// groups belong).
    pub fn fast_groups(&self) -> Vec<Vec<DeviceId>> {
        self.nodes.iter().enumerate()
            .map(|(n, &slots)| (0..slots)
                .map(|s| DeviceId { node: n as u32, slot: s }).collect())
            .collect()
    }
}

// ── Layer 2: collectives (the transport seam) ──────────────────────────

/// Reduction operator for collectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp { Sum, Max, Min }

/// A communication group (e.g. one tensor-parallel row, one data-parallel
/// replica set). Groups are registered against the topology so the
/// planner controls which wires they use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(pub u32);

/// THE transport seam. Training code calls this trait and nothing else.
/// `rank` is the caller's position within the group (0..group_size).
///
/// Implementations: `ThreadCollective` (this crate — in-process reference,
/// mathematically correct); NCCL/RCCL/MPI (feature-gated, Opus) behind
/// the SAME signatures. The training loop cannot tell them apart — that
/// is the scaling guarantee.
pub trait Collective: Send + Sync {
    /// Element-wise reduce across the group; every rank ends with the result.
    fn all_reduce(&self, rank: usize, buf: &mut [f32], op: ReduceOp);
    /// Concatenate each rank's `src` into `dst` (rank-major), on every rank.
    fn all_gather(&self, rank: usize, src: &[f32], dst: &mut [f32]);
    /// Reduce then scatter: rank r receives the r-th shard of the reduction.
    fn reduce_scatter(&self, rank: usize, src: &[f32], dst: &mut [f32], op: ReduceOp);
    /// Copy root's buffer to every rank.
    fn broadcast(&self, rank: usize, buf: &mut [f32], root: usize);
    /// Synchronize the group.
    fn barrier(&self, rank: usize);
    /// Number of ranks in the group.
    fn group_size(&self) -> usize;
}

// ── Reference implementation: ThreadCollective ─────────────────────────

/// In-process collective over worker threads — the correctness reference.
/// Shared slots guarded by a mutex + barrier phases. Not tuned for speed
/// (hardware impls are); tuned to be OBVIOUSLY correct so the training
/// math above it can be trusted, and so every seam is exercised on one
/// machine with no cluster.
pub struct ThreadCollective {
    n: usize,
    slots: Mutex<Vec<Vec<f32>>>,
    gate: Barrier,
    /// Second barrier phase so readers finish before the next op reuses slots.
    gate2: Barrier,
}

impl ThreadCollective {
    pub fn new(group_size: usize) -> Arc<Self> {
        Arc::new(ThreadCollective {
            n: group_size,
            slots: Mutex::new(vec![Vec::new(); group_size]),
            gate: Barrier::new(group_size),
            gate2: Barrier::new(group_size),
        })
    }

    /// Deposit this rank's contribution, wait for all, then read combined.
    fn exchange(&self, rank: usize, data: &[f32]) -> Vec<Vec<f32>> {
        {
            let mut s = self.slots.lock().unwrap();
            s[rank] = data.to_vec();
        }
        self.gate.wait();
        let all = { self.slots.lock().unwrap().clone() };
        self.gate2.wait(); // everyone has read; safe to reuse slots
        all
    }
}

impl Collective for ThreadCollective {
    fn all_reduce(&self, rank: usize, buf: &mut [f32], op: ReduceOp) {
        let all = self.exchange(rank, buf);
        for (i, x) in buf.iter_mut().enumerate() {
            let mut acc = all[0][i];
            for contrib in &all[1..] {
                acc = match op {
                    ReduceOp::Sum => acc + contrib[i],
                    ReduceOp::Max => acc.max(contrib[i]),
                    ReduceOp::Min => acc.min(contrib[i]),
                };
            }
            *x = acc;
        }
    }

    fn all_gather(&self, rank: usize, src: &[f32], dst: &mut [f32]) {
        let all = self.exchange(rank, src);
        let k = src.len();
        for (r, part) in all.iter().enumerate() {
            dst[r * k..(r + 1) * k].copy_from_slice(part);
        }
    }

    fn reduce_scatter(&self, rank: usize, src: &[f32], dst: &mut [f32], op: ReduceOp) {
        let all = self.exchange(rank, src);
        let k = dst.len();
        let off = rank * k;
        for i in 0..k {
            let mut acc = all[0][off + i];
            for contrib in &all[1..] {
                acc = match op {
                    ReduceOp::Sum => acc + contrib[off + i],
                    ReduceOp::Max => acc.max(contrib[off + i]),
                    ReduceOp::Min => acc.min(contrib[off + i]),
                };
            }
            dst[i] = acc;
        }
    }

    fn broadcast(&self, rank: usize, buf: &mut [f32], root: usize) {
        let all = self.exchange(rank, buf);
        buf.copy_from_slice(&all[root]);
    }

    fn barrier(&self, _rank: usize) {
        self.gate.wait();
        self.gate2.wait();
    }

    fn group_size(&self) -> usize { self.n }
}

// ── Layer 3: sharding descriptors ──────────────────────────────────────

/// How a parameter/activation tensor is split across the mesh — the
/// 3D-parallelism core (tensor × pipeline × data, the Megatron/ZeRO model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard {
    /// Full copy on every device (data-parallel parameters).
    Replicated,
    /// Split along `dim` across a FAST-tier group (Megatron column/row
    /// parallel matmul; partial products + all_reduce).
    TensorParallel { dim: u8, group: GroupId },
    /// Whole layers assigned to pipeline stage `stage` (activations cross
    /// the SLOW tier between stages, micro-batched).
    PipelineStage { stage: u16 },
    /// ZeRO-3: parameters/grads/optimizer state sharded across `group`;
    /// gathered on demand. Why 1T of optimizer state fits a cluster.
    ZeroSharded { group: GroupId },
}

/// Tensor-parallel matmul reference: each rank holds a COLUMN slice of B
/// (n × k_local) and computes its slice of A·B locally; all_gather
/// assembles the full result on every rank. Proves the Shard +
/// Collective seams compose. `a` is m×n row-major, `b_local` is the
/// rank's n×k_local column block; returns full m×(k_local·ranks).
pub fn tp_matmul(
    coll: &dyn Collective, rank: usize,
    a: &[f32], m: usize, n: usize,
    b_local: &[f32], k_local: usize,
) -> Vec<f32> {
    // Local product: m × k_local.
    let mut local = vec![0.0f32; m * k_local];
    for i in 0..m {
        for j in 0..k_local {
            let mut acc = 0.0f32;
            for t in 0..n {
                acc += a[i * n + t] * b_local[t * k_local + j];
            }
            local[i * k_local + j] = acc;
        }
    }
    // Gather every rank's column block, then interleave to row-major full.
    let ranks = coll.group_size();
    let mut gathered = vec![0.0f32; m * k_local * ranks];
    coll.all_gather(rank, &local, &mut gathered);
    let k_full = k_local * ranks;
    let mut full = vec![0.0f32; m * k_full];
    for r in 0..ranks {
        let block = &gathered[r * m * k_local..(r + 1) * m * k_local];
        for i in 0..m {
            for j in 0..k_local {
                full[i * k_full + r * k_local + j] = block[i * k_local + j];
            }
        }
    }
    full
}

/// ZeRO-sharded SGD step reference: each rank owns shard `rank` of the
/// parameters; gradients are reduce-scattered (each rank receives the
/// summed grad for ITS shard only), the rank updates its shard, and
/// all_gather reassembles full parameters. This is the ZeRO pattern that
/// lets optimizer state exceed any single device's memory. (AdamW state
/// follows the same ownership — Opus.)
pub fn zero_sgd_step(
    coll: &dyn Collective, rank: usize,
    params_full: &mut [f32], grad_local_full: &[f32], lr: f32,
) {
    let ranks = coll.group_size();
    let shard = params_full.len() / ranks;
    let mut grad_shard = vec![0.0f32; shard];
    coll.reduce_scatter(rank, grad_local_full, &mut grad_shard, ReduceOp::Sum);
    // Update ONLY the owned shard.
    let own = &mut params_full[rank * shard..(rank + 1) * shard].to_vec();
    for (p, g) in own.iter_mut().zip(&grad_shard) {
        *p -= lr * *g / ranks as f32; // mean-reduce semantics
    }
    // Reassemble full parameters everywhere.
    let mut gathered = vec![0.0f32; shard * ranks];
    coll.all_gather(rank, own, &mut gathered);
    params_full.copy_from_slice(&gathered);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn run_ranks<F>(n: usize, f: F) -> Vec<Vec<f32>>
    where F: Fn(usize, Arc<ThreadCollective>) -> Vec<f32> + Send + Sync + 'static {
        let coll = ThreadCollective::new(n);
        let f = Arc::new(f);
        let mut handles = Vec::new();
        for r in 0..n {
            let c = coll.clone();
            let f = f.clone();
            handles.push(thread::spawn(move || f(r, c)));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }

    #[test]
    fn topology_child_slots_and_tiers() {
        let t = Topology { nodes: vec![4, 4] }; // 2 nodes × 4 slots
        assert_eq!(t.device_count(), 8);
        let d = t.devices();
        assert_eq!(d[0], DeviceId { node: 0, slot: 0 });
        assert_eq!(d[7], DeviceId { node: 1, slot: 3 });
        assert_eq!(t.link(d[0], d[1]), LinkTier::Fast);  // same node
        assert_eq!(t.link(d[0], d[7]), LinkTier::Slow);  // cross node
        assert_eq!(t.fast_groups().len(), 2);
        assert_eq!(t.fast_groups()[0].len(), 4);
    }

    #[test]
    fn all_reduce_sums_across_ranks() {
        let out = run_ranks(4, |rank, c| {
            let mut buf = vec![rank as f32 + 1.0; 3]; // ranks contribute 1,2,3,4
            c.all_reduce(rank, &mut buf, ReduceOp::Sum);
            buf
        });
        for buf in out { assert_eq!(buf, vec![10.0, 10.0, 10.0]); }
    }

    #[test]
    fn all_gather_concatenates_rank_major() {
        let out = run_ranks(3, |rank, c| {
            let src = vec![rank as f32; 2];
            let mut dst = vec![0.0; 6];
            c.all_gather(rank, &src, &mut dst);
            dst
        });
        for dst in out { assert_eq!(dst, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]); }
    }

    #[test]
    fn reduce_scatter_gives_each_rank_its_shard() {
        let out = run_ranks(2, |rank, c| {
            // Each rank contributes [1,2,3,4]; sum = [2,4,6,8];
            // rank0 gets [2,4], rank1 gets [6,8].
            let src = vec![1.0, 2.0, 3.0, 4.0];
            let mut dst = vec![0.0; 2];
            c.reduce_scatter(rank, &src, &mut dst, ReduceOp::Sum);
            dst
        });
        assert_eq!(out[0], vec![2.0, 4.0]);
        assert_eq!(out[1], vec![6.0, 8.0]);
    }

    #[test]
    fn tensor_parallel_matmul_matches_single_device() {
        // A (2×3) · B (3×4), B split column-wise across 2 ranks.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![ // 3×4 row-major
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        // Single-device reference.
        let mut want = vec![0.0f32; 8];
        for i in 0..2 { for j in 0..4 { for t in 0..3 {
            want[i * 4 + j] += a[i * 3 + t] * b[t * 4 + j];
        }}}
        let a2 = a.clone(); let b2 = b.clone(); let want2 = want.clone();
        let out = run_ranks(2, move |rank, c| {
            // rank's column block: columns [rank*2, rank*2+2) of B → 3×2.
            let mut b_local = vec![0.0f32; 6];
            for t in 0..3 { for j in 0..2 {
                b_local[t * 2 + j] = b2[t * 4 + rank * 2 + j];
            }}
            tp_matmul(c.as_ref(), rank, &a2, 2, 3, &b_local, 2)
        });
        for full in out { assert_eq!(full, want2); }
        let _ = (a, b, want);
    }

    #[test]
    fn zero_sharded_step_matches_single_device_sgd() {
        // 2 ranks, 4 params. Each rank's local grad = [1,2,3,4] → summed
        // grad [2,4,6,8], mean [1,2,3,4]; lr=0.1 from params [10,10,10,10]
        // ⇒ [9.9, 9.8, 9.7, 9.6] on EVERY rank.
        let out = run_ranks(2, |rank, c| {
            let mut params = vec![10.0f32; 4];
            let grad = vec![1.0, 2.0, 3.0, 4.0];
            zero_sgd_step(c.as_ref(), rank, &mut params, &grad, 0.1);
            params
        });
        for p in out {
            for (got, want) in p.iter().zip([9.9f32, 9.8, 9.7, 9.6]) {
                assert!((got - want).abs() < 1e-6, "{} vs {}", got, want);
            }
        }
    }
}
