//! Does restructuring for CPU actually change the answer at long context?
use r2_tensor::segment::{full_attention_pairs, segment_attention_pairs, SegmentCache};
// (KvCache not needed here — SegmentCache owns it.)

fn main() {
    let (seg, mem) = (256usize, 64usize);
    // A 6-core CPU sustains ~10 GFLOP/s here; an A100 ~150,000.
    let cpu_gflops = 10.0f64;
    let layers_heads = 32.0 * 32.0;      // 32 layers x 32 heads, ~7B shape
    let flops_per_pair = 2.0 * 128.0;    // dot product over head_dim=128

    println!("{:>10} {:>16} {:>14} {:>9} {:>12} {:>12}",
             "tokens", "full pairs", "segment pairs", "saving", "full CPU", "segment CPU");
    println!("{}", "-".repeat(80));
    for &n in &[16_000usize, 32_000, 160_000, 1_000_000] {
        let f = full_attention_pairs(n);
        let s = segment_attention_pairs(n, seg, mem);
        let secs = |p: u64| p as f64 * flops_per_pair * layers_heads / (cpu_gflops * 1e9);
        let fmt = |x: f64| if x > 86400.0 { format!("{:.1} days", x/86400.0) }
                           else if x > 3600.0 { format!("{:.1} hours", x/3600.0) }
                           else if x > 60.0 { format!("{:.1} min", x/60.0) }
                           else { format!("{:.1} s", x) };
        println!("{:>10} {:>16} {:>14} {:>8}x {:>12} {:>12}",
                 n, f, s, f / s.max(1), fmt(secs(f)), fmt(secs(s)));
    }

    // Memory: the KV cache is what actually stops a CPU serving long context.
    println!("\nKV cache at 32 layers x 8 kv-heads x 128 dim, fp16:");
    let per_token_kb = 2.0 * 32.0 * 8.0 * 128.0 * 2.0 / 1024.0;
    for &n in &[16_000usize, 160_000, 1_000_000] {
        let full_gb = per_token_kb * n as f64 / 1024.0 / 1024.0;
        let seg_gb = per_token_kb * (seg + mem) as f64 / 1024.0 / 1024.0;
        println!("  {:>9} tokens: full {:>8.2} GB   segment {:>6.3} GB (bounded)",
                 n, full_gb, seg_gb);
    }

    // Prove the bound holds in the running implementation, not just on paper.
    let mut c = SegmentCache::new(8, 128, seg, mem);
    let k = vec![0.1f32; 8 * 128];
    for _ in 0..50_000 { c.append(&k, &k).unwrap(); }
    println!("\nafter {} tokens streamed: cache holds {} positions (cap {})",
             c.seen, c.len(), c.capacity());
}
