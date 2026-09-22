//! This adapter's float arithmetic against the correctly rounded answer:
//! which operations are exact, how far the rest are off, and whether the
//! driver fuses `a * b + c`. Run it on every device R2's determinism
//! guarantee is meant to cover — the rows that are not "exact" on some
//! device are exactly the operations R2 has to implement itself.
//!
//!     cargo run --release -p r2-gpu --features gpu --example precision_probe

fn main() {
    let n = 1 << 20;
    println!("adapter: {}", r2_gpu::adapter_info());
    let Some(st) = r2_gpu::numerics::probe(n) else { println!("no GPU"); return; };
    println!("{n} inputs per operation, against the correctly rounded f32\n");
    println!("{:<14} {:>12} {:>10} {:>9}", "operation", "exact", "% exact", "max ULP");
    println!("{}", "-".repeat(48));
    for (name, s) in r2_gpu::numerics::OPS.iter().zip(&st) {
        println!("{:<14} {:>12} {:>9.3}% {:>9}", name, s.exact, 100.0 * s.exact as f64 / s.n as f64, s.max_ulp);
    }
    let mad = &st[3];
    println!("\na * b + c written plainly: {} of {} results equal the FUSED answer where fused and", mad.fused, mad.n);
    println!("unfused differ — {}", if mad.fused > 0 { "the driver contracts a*b+c into an FMA" } else { "no evidence of contraction" });
}
