//! Rolling-window kernel — Phase K.9.

// ════════════════════════════════════════════════════════════════════
// Phase K.9 — Rolling window operations
// ════════════════════════════════════════════════════════════════════
//
// Fixed-width window reductions: for window size `w`, output position
// `i` (with i in `w-1..n`) holds the reduction over `data[i-w+1..=i]`.
// Output is shorter than input by `w-1` (matches R's `zoo::rollapply`
// with align="right", no padding).
//
//   rollsum:  sliding sum
//   rollmean: sliding mean
//   rollmax:  sliding maximum (deque-based, O(n))
//   rollmin:  sliding minimum (deque-based, O(n))
//   rollsd:   sliding sample standard deviation (two-pass per window)
//
// Common in time-series stats: moving averages, rolling volatility,
// running extremes.
//
// NA semantics: if any element in the window is None, that output is None.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollingOp {
    Sum,
    Mean,
    Max,
    Min,
    Sd,
}

/// Rolling-window reduction. Output length = `data.len() - window + 1`
/// if `window <= data.len()`, otherwise empty. Window of 0 returns empty.
pub fn rolling(op: RollingOp, data: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    if window == 0 || data.len() < window { return Vec::new(); }
    let out_len = data.len() - window + 1;
    let mut out = Vec::with_capacity(out_len);

    match op {
        RollingOp::Sum | RollingOp::Mean => {
            // Sliding sum with incremental update. NA tracking: keep
            // `na_count` for the current window; when ≥1, emit None.
            let mut sum = 0.0_f64;
            let mut na_count: usize = 0;
            // Initial window [0..window].
            for v in &data[..window] {
                match v { Some(x) => sum += x, None => na_count += 1 }
            }
            let denom = window as f64;
            let push = |out: &mut Vec<Option<f64>>, sum: f64, na: usize| {
                if na > 0 { out.push(None); }
                else if matches!(op, RollingOp::Mean) { out.push(Some(sum / denom)); }
                else { out.push(Some(sum)); }
            };
            push(&mut out, sum, na_count);
            for i in window..data.len() {
                // Drop leftmost, add rightmost.
                match data[i - window] {
                    Some(x) => sum -= x,
                    None => na_count -= 1,
                }
                match data[i] {
                    Some(x) => sum += x,
                    None => na_count += 1,
                }
                push(&mut out, sum, na_count);
            }
        }
        RollingOp::Max | RollingOp::Min => {
            // Deque-based O(n) sliding extremum. Each element enters
            // and leaves the deque at most once.
            // Index-only deque; values fetched from data[].
            // NA: when window contains any None, emit None and reset.
            use std::collections::VecDeque;
            let mut deque: VecDeque<usize> = VecDeque::new();
            let cmp = |a: f64, b: f64| -> bool {
                if matches!(op, RollingOp::Max) { a >= b } else { a <= b }
            };
            // Walk index, building deque of "candidates": each new index
            // pops back items with worse-or-equal value (they can never
            // be the answer once a better/equal-newer one is in).
            // NA-handling: track most-recent NA index.
            let mut last_na: Option<usize> = None;
            for i in 0..data.len() {
                match data[i] {
                    None => { last_na = Some(i); deque.clear(); }
                    Some(x) => {
                        while let Some(&back) = deque.back() {
                            if cmp(x, data[back].unwrap()) { deque.pop_back(); }
                            else { break; }
                        }
                        deque.push_back(i);
                    }
                }
                // Once we have i >= window-1, emit the answer.
                if i + 1 >= window {
                    let win_start = i + 1 - window;
                    // Drop fronts that are no longer in window.
                    while let Some(&front) = deque.front() {
                        if front < win_start { deque.pop_front(); }
                        else { break; }
                    }
                    let window_has_na = last_na.map_or(false, |na_i| na_i >= win_start);
                    if window_has_na || deque.is_empty() {
                        out.push(None);
                    } else {
                        out.push(Some(data[*deque.front().unwrap()].unwrap()));
                    }
                }
            }
        }
        RollingOp::Sd => {
            // Two-pass sample SD per window. Simpler than incremental
            // (Welford-style) variance and avoids numerical drift on
            // long windows.
            for i in 0..out_len {
                let win = &data[i..i + window];
                let mut na_count = 0usize;
                let mut sum = 0.0_f64;
                for v in win {
                    match v { Some(x) => sum += x, None => { na_count += 1; } }
                }
                if na_count > 0 || window < 2 { out.push(None); continue; }
                let mean = sum / window as f64;
                let mut ss = 0.0_f64;
                for v in win {
                    let x = v.unwrap();
                    let d = x - mean;
                    ss += d * d;
                }
                out.push(Some((ss / (window - 1) as f64).sqrt()));
            }
        }
    }
    out
}
