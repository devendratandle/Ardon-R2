//! Where does the embedding BACKWARD spend 2.7 ms when the scatter-add is
//! ~0.5 ms? Splits `backward_from` on an embed node into its pieces at the
//! shipping shape (2,048 tokens, dim 256, vocab 8,000):
//!
//!   scatter, warm buffer     the scatter-add into a table gradient whose
//!                            pages are already resident
//!   scatter, fresh buffer    the same into a freshly calloc'd 8 MB buffer
//!                            (what a fresh tape gives it): page faults
//!   backward_from, fresh     the whole thing through the tape
//!   backward_from, pooled    the same with the tape drawing its VALUE
//!                            buffers from a pool (what training runs)
//!
//!     cargo run --release -p r2-train --example embed_probe

use r2_autograd::{BufPool, Tape};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let (t, d, vocab) = (2048usize, 256usize, 8000usize);
    let table: Vec<f32> = (0..vocab * d).map(|i| ((i as f32) * 0.001).sin()).collect();
    let tokens: Vec<usize> = (0..t).map(|i| (i * 7919 + 13) % vocab).collect();
    let g: Vec<f32> = (0..t * d).map(|i| ((i as f32) * 0.002).cos()).collect();
    let reps = 9;

    let scatter = |gt: &mut [f32]| {
        for (i, &tok) in tokens.iter().enumerate() {
            let dst = &mut gt[tok * d..tok * d + d];
            for (o, s) in dst.iter_mut().zip(&g[i * d..i * d + d]) { *o += s; }
        }
    };

    // serial scatter, warm
    let mut warm = vec![0.0f32; vocab * d];
    for x in warm.iter_mut() { *x = 1.0; }
    let s_warm = median((0..reps).map(|_| {
        let s = std::time::Instant::now(); scatter(&mut warm); s.elapsed().as_secs_f64() * 1e6
    }).collect());
    // serial scatter, fresh calloc each time
    let s_fresh = median((0..reps).map(|_| {
        let mut fresh = vec![0.0f32; vocab * d];
        let s = std::time::Instant::now(); scatter(&mut fresh); let e = s.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(&fresh); e
    }).collect());

    // whole backward through the tape, fresh tape each rep
    let bf_fresh = median((0..reps).map(|_| {
        let mut tp = Tape::new();
        let w = tp.leaf(table.clone(), true);
        let x = tp.embed(w, &tokens, d);
        let s = std::time::Instant::now();
        tp.backward_from(x, &g);
        let e = s.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(tp.grad(w).len()); e
    }).collect());
    // forward through the tape, fresh vs pooled
    let fw_fresh = median((0..reps).map(|_| {
        let mut tp = Tape::new();
        let w = tp.leaf(table.clone(), true);
        let s = std::time::Instant::now();
        let x = tp.embed(w, &tokens, d);
        let e = s.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(tp.value(x).len()); e
    }).collect());
    let mut pool = BufPool::new();
    let fw_pool = median((0..reps).map(|_| {
        let mut tp = Tape::with_pool(std::mem::take(&mut pool));
        let w = tp.leaf(table.clone(), true);
        let s = std::time::Instant::now();
        let x = tp.embed(w, &tokens, d);
        let e = s.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(tp.value(x).len());
        let _ = tp.take_value(w);
        pool = tp.into_pool(); e
    }).collect());
    // the leaf itself: pushing an 8 MB parameter allocates its 8 MB gradient
    let leaf = median((0..reps).map(|_| {
        let mut tp = Tape::new();
        let s = std::time::Instant::now();
        let w = tp.leaf(table.clone(), true);
        let e = s.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(tp.grad(w).len()); e
    }).collect());

    println!("embedding backward pieces, {t} tokens x dim {d}, vocab {vocab}, microseconds (median of {reps})\n");
    println!("{:<52} {:>9}", "scatter-add, serial, warm 8 MB gradient", s_warm);
    println!("{:<52} {:>9}", "scatter-add, serial, fresh calloc gradient", s_fresh);
    println!("{:<52} {:>9}", "backward_from through the tape (fresh tape)", bf_fresh);
    println!("{:<52} {:>9}", "forward embed, fresh tape", fw_fresh);
    println!("{:<52} {:>9}", "forward embed, pooled tape (training)", fw_pool);
    println!("{:<52} {:>9}", "leaf(table): 8 MB copy + 8 MB gradient calloc", leaf);
    println!("\n  fresh - warm on the scatter is the page-fault cost of a calloc'd");
    println!("  gradient buffer; backward_from - scatter is the tape's own overhead.");
}
