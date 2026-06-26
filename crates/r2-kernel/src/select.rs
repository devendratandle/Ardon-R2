//! Select / find kernel — Phase K.8.

// ════════════════════════════════════════════════════════════════════
// Phase K.8 — Select / find operations
// ════════════════════════════════════════════════════════════════════
//
// Reductions that return positions (indices) or partial orderings
// rather than aggregates:
//
//   which_max / which_min: 0-based index of the first max/min
//   nth_smallest(k):       value of the kth smallest (quickselect)
//   top_k(k) / bottom_k:   indices of the k largest / smallest
//
// NA handling: any None in the input propagates the appropriate
// "no answer" (returns None for index-returning ops; skips None for
// quickselect-style ops to match R's na.rm=TRUE default on these).

/// Index of the first maximum. `None` if input has any NA or is empty.
/// (R's `which.max(c(1, NA, 3))` returns `integer(0)`; we return None
/// to match the broader NA-propagation pattern of the kernel layer.)
pub fn which_max(data: &[Option<f64>]) -> Option<usize> {
    if data.is_empty() { return None; }
    let mut best_idx = 0usize;
    let mut best_val = match data[0] { Some(v) => v, None => return None };
    for (i, v) in data.iter().enumerate().skip(1) {
        match v {
            Some(x) => { if *x > best_val { best_val = *x; best_idx = i; } }
            None => return None,
        }
    }
    Some(best_idx)
}

/// Index of the first minimum. Same semantics as `which_max`.
pub fn which_min(data: &[Option<f64>]) -> Option<usize> {
    if data.is_empty() { return None; }
    let mut best_idx = 0usize;
    let mut best_val = match data[0] { Some(v) => v, None => return None };
    for (i, v) in data.iter().enumerate().skip(1) {
        match v {
            Some(x) => { if *x < best_val { best_val = *x; best_idx = i; } }
            None => return None,
        }
    }
    Some(best_idx)
}

/// Quickselect: returns the value of the `k`-th smallest element (0-indexed).
/// Skips NAs. Returns `None` if all NA or `k >= count of non-NA`.
/// O(n) average, O(n²) worst case but in practice fine since stdlib's
/// `select_nth_unstable_by` uses a hardened pivot strategy.
///
/// Uses the scratch pool for the unwrapped buffer.
pub fn nth_smallest(data: &[Option<f64>], k: usize) -> Option<f64> {
    let mut buf = r2_memory::scratch_acquire(data.len());
    for x in data { if let Some(v) = x { buf.push(*v); } }
    let result = if k >= buf.len() {
        None
    } else {
        let (_, mid, _) = buf.select_nth_unstable_by(k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(*mid)
    };
    r2_memory::scratch_release(buf);
    result
}

/// Indices of the `k` largest elements, in descending order of value.
/// Skips NAs. Uses a binary heap of size `k` — O(n log k) regardless of k.
/// Returns at most `min(k, len-na_count)` indices.
pub fn top_k(data: &[Option<f64>], k: usize) -> Vec<usize> {
    if k == 0 { return Vec::new(); }
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    // Min-heap of (value, idx) — keeps the k largest.
    // Wrap f64 in a NaN-safe ord helper.
    #[derive(PartialEq)]
    struct OrdF64(f64, usize);
    impl Eq for OrdF64 {}
    impl Ord for OrdF64 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.1.cmp(&other.1))
        }
    }
    impl PartialOrd for OrdF64 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
    }
    let mut heap: BinaryHeap<Reverse<OrdF64>> = BinaryHeap::with_capacity(k + 1);
    for (i, v) in data.iter().enumerate() {
        if let Some(x) = v {
            if heap.len() < k {
                heap.push(Reverse(OrdF64(*x, i)));
            } else if let Some(Reverse(OrdF64(top, _))) = heap.peek() {
                if x > top {
                    heap.pop();
                    heap.push(Reverse(OrdF64(*x, i)));
                }
            }
        }
    }
    // Extract in descending order.
    let mut result: Vec<(f64, usize)> = heap.into_iter()
        .map(|Reverse(OrdF64(v, i))| (v, i))
        .collect();
    result.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    result.into_iter().map(|(_, i)| i).collect()
}

/// Indices of the `k` smallest elements, in ascending order of value.
/// Mirror of `top_k`.
pub fn bottom_k(data: &[Option<f64>], k: usize) -> Vec<usize> {
    if k == 0 { return Vec::new(); }
    use std::collections::BinaryHeap;
    #[derive(PartialEq)]
    struct OrdF64(f64, usize);
    impl Eq for OrdF64 {}
    impl Ord for OrdF64 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.1.cmp(&other.1))
        }
    }
    impl PartialOrd for OrdF64 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
    }
    // Max-heap — keeps the k smallest.
    let mut heap: BinaryHeap<OrdF64> = BinaryHeap::with_capacity(k + 1);
    for (i, v) in data.iter().enumerate() {
        if let Some(x) = v {
            if heap.len() < k {
                heap.push(OrdF64(*x, i));
            } else if let Some(OrdF64(top, _)) = heap.peek() {
                if x < top {
                    heap.pop();
                    heap.push(OrdF64(*x, i));
                }
            }
        }
    }
    let mut result: Vec<(f64, usize)> = heap.into_iter()
        .map(|OrdF64(v, i)| (v, i))
        .collect();
    result.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    result.into_iter().map(|(_, i)| i).collect()
}
