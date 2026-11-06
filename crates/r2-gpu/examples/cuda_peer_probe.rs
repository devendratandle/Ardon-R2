//! Can two processes copy directly between two cards, and how fast? Each process
//! sees both GPUs and picks one with `R2_CUDA_DEVICE`. The two meet through files
//! in the directory given as the second argument.
//!
//!   R2_CUDA_DEVICE=0 cuda_peer_probe 0 /tmp/probe &
//!   R2_CUDA_DEVICE=1 cuda_peer_probe 1 /tmp/probe

use r2_gpu::cuda::ffi::{peer_access, PeerBuffer};
use r2_gpu::cuda::{device_line, sync, Tensor};
use std::time::{Duration, Instant};

const SIZES: [usize; 3] = [7_000_000, 40_000_000, 125_000_000];

fn wait_for(path: &str) -> Vec<u8> {
    let start = Instant::now();
    loop {
        if let Ok(b) = std::fs::read(path) { if !b.is_empty() { return b; } }
        if start.elapsed() > Duration::from_secs(120) { panic!("no {path} after 2 minutes"); }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn put(path: &str, bytes: &[u8]) {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, bytes).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (rank, dir): (usize, &str) = (args[1].parse().unwrap(), &args[2]);
    std::fs::create_dir_all(dir).unwrap();
    println!("rank {rank}: {}", device_line());
    let other = 1 - rank;
    match peer_access(other) {
        Ok((can, me)) => println!("rank {rank}: device {me} can access device {other} directly: {can}"),
        Err(e) => println!("rank {rank}: peer query failed: {e}"),
    }
    let name = |n: usize| format!("{dir}/handle_{n}");
    let n = *SIZES.iter().max().unwrap();
    let fill: Vec<f32> = (0..n).map(|i| (rank + 1) as f32 + (i % 1024) as f32).collect();
    let mine = Tensor::upload(&fill).unwrap();
    if rank == 0 {
        put(&name(n), &mine.ipc_handle().unwrap());
        // kept alive until the other side has finished
        wait_for(&format!("{dir}/done"));
        println!("rank 0: the other side finished");
        return;
    }
    let mut h = [0u8; 64];
    h.copy_from_slice(&wait_for(&name(n)));
    let peer = match PeerBuffer::open(h) {
        Ok(p) => p,
        Err(e) => { println!("rank 1: FAILED to map the peer buffer: {e}"); put(&format!("{dir}/done"), b"x"); return; }
    };
    println!("rank 1: mapped {} elements of the other card", peer.len());
    let there = Tensor::zeros(n).unwrap();
    for &m in &SIZES {
        let (mut rd, mut wr) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            let t = Instant::now();
            if let Err(e) = peer.read(0, &there, 0, m) { println!("read failed: {e}"); put(&format!("{dir}/done"), b"x"); return; }
            sync();
            rd.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            if let Err(e) = peer.write(0, &mine, 0, m) { println!("write failed: {e}"); put(&format!("{dir}/done"), b"x"); return; }
            sync();
            wr.push(t.elapsed().as_secs_f64());
        }
        rd.sort_by(f64::total_cmp);
        wr.sort_by(f64::total_cmp);
        let gb = m as f64 * 4.0 / 1e9;
        println!("{:>4} MB: read {:.1} ms ({:.1} GB/s), write {:.1} ms ({:.1} GB/s)",
                 m * 4 / 1_000_000, rd[2] * 1e3, gb / rd[2], wr[2] * 1e3, gb / wr[2]);
    }
    let back = there.to_vec().unwrap();
    println!("bits arrived: {}", back[..SIZES[2]].iter().enumerate().all(|(i, x)| *x == 1.0 + (i % 1024) as f32));
    put(&format!("{dir}/done"), b"x");
}
