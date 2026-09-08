//! LMO-13 and LMO-14 — `grad_A = g·Bᵀ` and `grad_B = Aᵀ·g`, against
//! PyTorch and JAX, on the matmul shapes the model actually runs.
//!
//! Neither stage has ever had a reference number. The queue records
//! LMO-13 as "DONE" and LMO-14 as "OPEN — marked serial in the profile",
//! both judged against R2's own past rather than against anything else —
//! the identical hole that left LMO-15 unmeasured for so long.
//!
//! They are worth measuring because after the attention fusion they are
//! what is left: at the shipping shape, of forward + grad_A + grad_B,
//! grad_A is 52%, grad_B 31%, forward 18%.
//!
//! METHOD. One operand requires grad, the other does not, so the tape
//! computes exactly one of the two gradients — the `requires` guards added
//! in this same series make that a real saving rather than a label. Each
//! gradient is reported as (fwd+bwd) − (fwd), measured separately, which
//! is the same subtraction the PyTorch and JAX halves do.
//!
//!     cargo run --release -p r2-train --example lmo14_grad_ab
//!     python benchmarks/llm/lmo14_grad_ab.py

use r2_autograd::Tape;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn t_us(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    median((0..9).map(|_| {
        let s = std::time::Instant::now();
        for _ in 0..reps { f(); }
        s.elapsed().as_secs_f64() / reps as f64 * 1e6
    }).collect())
}

fn main() {
    // The shapes of the shipping step: dim 256, 4 layers, ffn 768,
    // vocab 8,000, 32 x 64 = 2,048 tokens. `count` is how many times a
    // step runs that shape, so a stage's true weight is visible.
    let shapes: Vec<(usize, usize, usize, usize, &str)> = vec![
        (2048, 256, 8000, 1, "output head"),
        (2048, 256, 768, 8, "ffn w1/w3"),
        (2048, 768, 256, 4, "ffn w2"),
        (2048, 256, 256, 8, "q/o proj"),
        (2048, 256, 128, 8, "k/v proj"),
    ];

    println!("LMO-13/14 — grad_A and grad_B, R2");
    println!("  microseconds per CALL, median of 9\n");
    println!("{:>14} {:>18} {:>5} {:>11} {:>11} {:>11}",
             "block", "m x k x n", "x/step", "fwd", "grad_A", "grad_B");
    println!("{}", "-".repeat(76));

    let mut rows = Vec::new();
    for &(m, k, n, cnt, label) in &shapes {
        let mk = |sz: usize, ph: f32| -> Vec<f32> {
            (0..sz).map(|i| ((i as f32) * 0.0011 + ph).sin()).collect()
        };
        let a = mk(m * k, 0.0);
        let b = mk(k * n, 1.0);
        let g = mk(m * n, 2.0);
        let reps = ((4e9 / (m as f64 * k as f64 * n as f64)).ceil() as usize).clamp(2, 20);

        let fwd = t_us(reps, || {
            let mut tp = Tape::new();
            let av = tp.leaf(a.clone(), false);
            let bv = tp.leaf(b.clone(), false);
            std::hint::black_box(tp.matmul(av, bv, m, k, n));
        });
        // Only A requires grad -> only grad_A is computed.
        let fb_a = t_us(reps, || {
            let mut tp = Tape::new();
            let av = tp.leaf(a.clone(), true);
            let bv = tp.leaf(b.clone(), false);
            let o = tp.matmul(av, bv, m, k, n);
            tp.backward_from(o, &g);
            std::hint::black_box(tp.grad(av).len());
        });
        // Only B requires grad -> only grad_B is computed.
        let fb_b = t_us(reps, || {
            let mut tp = Tape::new();
            let av = tp.leaf(a.clone(), false);
            let bv = tp.leaf(b.clone(), true);
            let o = tp.matmul(av, bv, m, k, n);
            tp.backward_from(o, &g);
            std::hint::black_box(tp.grad(bv).len());
        });
        let (ga, gb) = ((fb_a - fwd).max(0.0), (fb_b - fwd).max(0.0));
        println!("{label:>14} {:>6}x{k:<4}x{n:<6} {cnt:>5} {fwd:>11.1} {ga:>11.1} {gb:>11.1}",
                 m);
        rows.push((label.to_string(), m, k, n, cnt, fwd, ga, gb));
    }

    let tot = |f: fn(&(String, usize, usize, usize, usize, f64, f64, f64)) -> f64| -> f64 {
        rows.iter().map(|r| f(r) * r.4 as f64).sum()
    };
    let (tf, ta, tb) = (tot(|r| r.5), tot(|r| r.6), tot(|r| r.7));
    println!("{}", "-".repeat(76));
    println!("  per step, weighted by count:  fwd {:.0} us   grad_A {:.0} us ({:.0}%)   \
              grad_B {:.0} us ({:.0}%)",
             tf, ta, ta / (tf + ta + tb) * 100.0, tb, tb / (tf + ta + tb) * 100.0);

    let mut js = String::from("{\n  \"rows\": [\n");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 { js.push_str(",\n"); }
        js.push_str(&format!(
            "    {{\"label\": \"{}\", \"m\": {}, \"k\": {}, \"n\": {}, \"count\": {}, \
             \"fwd\": {:.1}, \"grad_a\": {:.1}, \"grad_b\": {:.1}}}",
            r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7));
    }
    js.push_str("\n  ]\n}\n");
    match std::fs::write("lmo14_r2.json", js) {
        Ok(()) => println!("\nwrote lmo14_r2.json — run `python benchmarks/llm/lmo14_grad_ab.py`\n\
                            for the joint R2 / PyTorch / JAX table and the verdict."),
        Err(e) => eprintln!("could not write lmo14_r2.json: {e}"),
    }
}
