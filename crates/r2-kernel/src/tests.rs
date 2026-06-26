    use super::*;
    use r2_oracle::Op;

    fn data() -> Vec<Option<f64>> {
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]
    }

    fn data_with_na() -> Vec<Option<f64>> {
        vec![Some(1.0), None, Some(3.0), Some(4.0), Some(5.0)]
    }

    #[test]
    fn serial_sum_correct() {
        assert_eq!(SerialBackend.reduce(ReduceOp::Sum, &data()), Some(15.0));
    }

    #[test]
    fn rayon_sum_correct() {
        assert_eq!(RayonBackend.reduce(ReduceOp::Sum, &data()), Some(15.0));
    }

    #[test]
    fn serial_mean_correct() {
        assert_eq!(SerialBackend.reduce(ReduceOp::Mean, &data()), Some(3.0));
    }

    #[test]
    fn rayon_mean_correct() {
        assert_eq!(RayonBackend.reduce(ReduceOp::Mean, &data()), Some(3.0));
    }

    #[test]
    fn na_propagates_serial() {
        assert_eq!(SerialBackend.reduce(ReduceOp::Sum, &data_with_na()), None);
        assert_eq!(SerialBackend.reduce(ReduceOp::Mean, &data_with_na()), None);
    }

    #[test]
    fn na_propagates_rayon() {
        assert_eq!(RayonBackend.reduce(ReduceOp::Sum, &data_with_na()), None);
        assert_eq!(RayonBackend.reduce(ReduceOp::Mean, &data_with_na()), None);
    }

    #[test]
    fn min_max_prod() {
        let d = data();
        assert_eq!(SerialBackend.reduce(ReduceOp::Min, &d), Some(1.0));
        assert_eq!(SerialBackend.reduce(ReduceOp::Max, &d), Some(5.0));
        assert_eq!(SerialBackend.reduce(ReduceOp::Prod, &d), Some(120.0));
        assert_eq!(RayonBackend.reduce(ReduceOp::Min, &d), Some(1.0));
        assert_eq!(RayonBackend.reduce(ReduceOp::Max, &d), Some(5.0));
        assert_eq!(RayonBackend.reduce(ReduceOp::Prod, &d), Some(120.0));
    }

    #[test]
    fn dispatcher_picks_backend() {
        // Small input → Serial, returns same answer.
        assert_eq!(reduce(ReduceOp::Sum, &data()), Some(15.0));
    }

    #[test]
    fn empty_input() {
        let empty: Vec<Option<f64>> = vec![];
        assert_eq!(SerialBackend.reduce(ReduceOp::Sum, &empty), Some(0.0));
        assert_eq!(SerialBackend.reduce(ReduceOp::Mean, &empty), None);
        assert_eq!(SerialBackend.reduce(ReduceOp::Min, &empty), None);
    }

    // ── Map kernel tests (Phase K.2) ─────────────────────────────────

    fn map_data() -> Vec<Option<f64>> {
        vec![Some(1.0), Some(4.0), None, Some(9.0), Some(16.0)]
    }

    #[test]
    fn serial_sqrt() {
        let r = SerialBackend.map(MapOp::Sqrt, &map_data());
        assert_eq!(r, vec![Some(1.0), Some(2.0), None, Some(3.0), Some(4.0)]);
    }

    #[test]
    fn rayon_sqrt() {
        let r = RayonBackend.map(MapOp::Sqrt, &map_data());
        assert_eq!(r, vec![Some(1.0), Some(2.0), None, Some(3.0), Some(4.0)]);
    }

    #[test]
    fn map_abs_neg_log() {
        let d: Vec<Option<f64>> = vec![Some(-2.0), Some(0.0), Some(2.0)];
        assert_eq!(SerialBackend.map(MapOp::Abs, &d),
            vec![Some(2.0), Some(0.0), Some(2.0)]);
        assert_eq!(SerialBackend.map(MapOp::Neg, &d),
            vec![Some(2.0), Some(-0.0), Some(-2.0)]);
        let lns = SerialBackend.map(MapOp::Ln, &vec![Some(1.0), Some(std::f64::consts::E)]);
        assert!((lns[0].unwrap() - 0.0).abs() < 1e-12);
        assert!((lns[1].unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn map_dispatcher_picks_backend() {
        let r = map(MapOp::Sqrt, &map_data());
        assert_eq!(r, vec![Some(1.0), Some(2.0), None, Some(3.0), Some(4.0)]);
    }

    #[test]
    fn na_preserved_in_map() {
        let d: Vec<Option<f64>> = vec![Some(4.0), None, Some(9.0)];
        let r = SerialBackend.map(MapOp::Sqrt, &d);
        assert_eq!(r[1], None);
    }

    // ── Binary kernel tests (Phase K.3) ──────────────────────────────

    #[test]
    fn binary_add_serial_and_rayon_agree() {
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        let b: Vec<Option<f64>> = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)];
        let s = SerialBackend.binary(BinaryOp::Add, &a, &b);
        let r = RayonBackend.binary(BinaryOp::Add, &a, &b);
        assert_eq!(s, r);
        assert_eq!(s, vec![Some(11.0), Some(22.0), Some(33.0), Some(44.0)]);
    }

    #[test]
    fn binary_div_with_na() {
        let a: Vec<Option<f64>> = vec![Some(10.0), None, Some(9.0)];
        let b: Vec<Option<f64>> = vec![Some(2.0), Some(3.0), Some(3.0)];
        let r = SerialBackend.binary(BinaryOp::Div, &a, &b);
        assert_eq!(r, vec![Some(5.0), None, Some(3.0)]);
    }

    #[test]
    fn binary_dispatcher_picks_backend() {
        let a: Vec<Option<f64>> = vec![Some(2.0); 5];
        let b: Vec<Option<f64>> = vec![Some(3.0); 5];
        let r = binary(BinaryOp::Mul, &a, &b);
        assert_eq!(r, vec![Some(6.0); 5]);
    }

    // ── par_for tests (Phase K.4) ────────────────────────────────────

    #[test]
    fn par_for_serial_path() {
        // Small n → Oracle returns Serial → in-order indexed result.
        let r: Vec<usize> = par_for(Op::PerElementMap, 5, |i| i * 2);
        assert_eq!(r, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn par_for_rayon_path() {
        // Use TreeBuild op (threshold=1 → always Rayon-eligible) to force the
        // parallel branch. Result must still be in stable index order.
        let r: Vec<usize> = par_for(Op::TreeBuild, 100, |i| i * i);
        for (i, v) in r.iter().enumerate() {
            assert_eq!(*v, i * i);
        }
    }

    #[test]
    fn par_for_empty() {
        let r: Vec<usize> = par_for(Op::PerElementMap, 0, |i| i);
        assert!(r.is_empty());
    }

    // ── Ternary kernel tests (Phase K.5) ─────────────────────────────

    #[test]
    fn ternary_muladd_serial_and_rayon_agree() {
        // a*b + c
        let a: Vec<Option<f64>> = vec![Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
        let b: Vec<Option<f64>> = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)];
        let c: Vec<Option<f64>> = vec![Some(1.0), Some(1.0), Some(1.0), Some(1.0)];
        let s = SerialBackend.ternary(TernaryOp::MulAdd, &a, &b, &c);
        let r = RayonBackend.ternary(TernaryOp::MulAdd, &a, &b, &c);
        assert_eq!(s, r);
        assert_eq!(s, vec![Some(21.0), Some(61.0), Some(121.0), Some(201.0)]);
    }

    #[test]
    fn ternary_muladd_na_propagates_from_any_input() {
        let a: Vec<Option<f64>> = vec![Some(2.0), None, Some(4.0), Some(5.0)];
        let b: Vec<Option<f64>> = vec![Some(10.0), Some(20.0), None, Some(40.0)];
        let c: Vec<Option<f64>> = vec![Some(1.0), Some(1.0), Some(1.0), None];
        let r = SerialBackend.ternary(TernaryOp::MulAdd, &a, &b, &c);
        assert_eq!(r, vec![Some(21.0), None, None, None]);
    }

    #[test]
    fn ternary_dispatcher_routes() {
        // For a small N, dispatcher should pick Serial — but either is correct.
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        let b: Vec<Option<f64>> = vec![Some(4.0), Some(5.0), Some(6.0)];
        let c: Vec<Option<f64>> = vec![Some(7.0), Some(8.0), Some(9.0)];
        let r = ternary(TernaryOp::MulAdd, &a, &b, &c);
        // 1*4+7=11, 2*5+8=18, 3*6+9=27
        assert_eq!(r, vec![Some(11.0), Some(18.0), Some(27.0)]);
    }

    // ── Strided reduction tests (Phase K.6) ──────────────────────────

    fn strided_data() -> Vec<Option<f64>> {
        // 5x3 column-major: cols are [1..5], [10..50 step 10], [100..500 step 100].
        // data[row + col*5] for column-major access.
        let mut d = Vec::with_capacity(15);
        for col_factor in [1.0, 10.0, 100.0] {
            for row in 1..=5 { d.push(Some(row as f64 * col_factor)); }
        }
        d
    }

    #[test]
    fn strided_reduce_sum_row_of_5x3() {
        let d = strided_data();
        // Row 0 (zero-indexed): elements at offset 0, stride 5 (nrow), count 3.
        // Values: 1.0, 10.0, 100.0 → sum = 111.0.
        let s = SerialBackend.reduce_strided(ReduceOp::Sum, &d, 0, 5, 3);
        assert_eq!(s, Some(111.0));
        // Row 2: offset 2, stride 5, count 3 → 3.0, 30.0, 300.0 = 333.0.
        let s = SerialBackend.reduce_strided(ReduceOp::Sum, &d, 2, 5, 3);
        assert_eq!(s, Some(333.0));
    }

    #[test]
    fn strided_reduce_serial_and_rayon_agree() {
        let d = strided_data();
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Min, ReduceOp::Max, ReduceOp::Prod] {
            for offset in 0..5 {
                let s = SerialBackend.reduce_strided(op, &d, offset, 5, 3);
                let r = RayonBackend.reduce_strided(op, &d, offset, 5, 3);
                assert_eq!(s, r, "op={:?} offset={}", op, offset);
            }
        }
    }

    #[test]
    fn strided_reduce_na_propagates() {
        let mut d = strided_data();
        d[5] = None; // row 0 col 1 = NA → walking row 0 hits this.
        let s = SerialBackend.reduce_strided(ReduceOp::Sum, &d, 0, 5, 3);
        assert_eq!(s, None);
        // Row 1 walks indices 1, 6, 11 — does not include index 5.
        let s = SerialBackend.reduce_strided(ReduceOp::Sum, &d, 1, 5, 3);
        assert_eq!(s, Some(2.0 + 20.0 + 200.0));
    }

    #[test]
    fn strided_reduce_matches_naive_copy() {
        // Sanity: strided result == reduce over copy
        let d = strided_data();
        for offset in 0..5 {
            let copied: Vec<Option<f64>> = (0..3).map(|k| d[offset + k * 5]).collect();
            let strided = reduce_strided(ReduceOp::Mean, &d, offset, 5, 3);
            let naive = SerialBackend.reduce(ReduceOp::Mean, &copied);
            assert_eq!(strided, naive, "offset {}", offset);
        }
    }

    #[test]
    fn strided_reduce_var_and_sd() {
        let d = strided_data();
        // Row 0: values 1, 10, 100. mean=37, var=sum((x-37)^2)/(3-1).
        let var = SerialBackend.reduce_strided(ReduceOp::Var, &d, 0, 5, 3).unwrap();
        let sd  = SerialBackend.reduce_strided(ReduceOp::Sd,  &d, 0, 5, 3).unwrap();
        let mean: f64 = 37.0;
        let expected_var = ((1.0_f64 - mean).powi(2) + (10.0_f64 - mean).powi(2) + (100.0_f64 - mean).powi(2)) / 2.0;
        assert!((var - expected_var).abs() < 1e-9);
        assert!((sd - expected_var.sqrt()).abs() < 1e-9);
    }

    // ── Scan kernel tests (Phase K.7) ────────────────────────────────

    #[test]
    fn scan_cumsum_basic() {
        let d: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        let r = SerialBackend.scan(ScanOp::Cumsum, &d);
        assert_eq!(r, vec![Some(1.0), Some(3.0), Some(6.0), Some(10.0)]);
    }

    #[test]
    fn scan_cumprod_basic() {
        let d: Vec<Option<f64>> = vec![Some(2.0), Some(3.0), Some(4.0)];
        let r = SerialBackend.scan(ScanOp::Cumprod, &d);
        assert_eq!(r, vec![Some(2.0), Some(6.0), Some(24.0)]);
    }

    #[test]
    fn scan_cummax_basic() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0)];
        let r = SerialBackend.scan(ScanOp::Cummax, &d);
        assert_eq!(r, vec![Some(3.0), Some(3.0), Some(4.0), Some(4.0), Some(5.0)]);
    }

    #[test]
    fn scan_cummin_basic() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0)];
        let r = SerialBackend.scan(ScanOp::Cummin, &d);
        assert_eq!(r, vec![Some(3.0), Some(1.0), Some(1.0), Some(1.0), Some(1.0)]);
    }

    #[test]
    fn scan_na_propagates_forward() {
        // R semantics: cumsum(c(1, 2, NA, 4)) → c(1, 3, NA, NA)
        let d: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), None, Some(4.0)];
        let r = SerialBackend.scan(ScanOp::Cumsum, &d);
        assert_eq!(r, vec![Some(1.0), Some(3.0), None, None]);
    }

    #[test]
    fn scan_serial_and_rayon_agree_on_large_input() {
        // n ≥ 4096 triggers Rayon's chunked path; verify it matches serial.
        // Use relative tolerance: cumprod accumulates to ~1e100 over 10K
        // elements, and floating-point addition / multiplication is
        // non-associative, so the chunked order produces ~ULP-level
        // differences from the sequential order. That's correct
        // behavior, not a bug.
        let n = 10_000;
        let d: Vec<Option<f64>> = (0..n).map(|i| Some(((i % 17) as f64) * 0.01 + 1.0)).collect();
        for op in [ScanOp::Cumsum, ScanOp::Cumprod, ScanOp::Cummax, ScanOp::Cummin] {
            let s = SerialBackend.scan(op, &d);
            let r = RayonBackend.scan(op, &d);
            assert_eq!(s.len(), r.len(), "op={:?}", op);
            for (i, (a, b)) in s.iter().zip(r.iter()).enumerate() {
                match (a, b) {
                    (Some(x), Some(y)) => {
                        // Skip when both overflowed to ±inf (cumprod on
                        // ≥ 1.x values past ~9K elements crosses f64::MAX).
                        if x.is_infinite() && y.is_infinite() && x.signum() == y.signum() {
                            continue;
                        }
                        let mag = x.abs().max(y.abs()).max(1.0);
                        let rel = (x - y).abs() / mag;
                        assert!(rel < 1e-12, "op={:?} i={} rel={} ({} vs {})",
                            op, i, rel, x, y);
                    }
                    (None, None) => {}
                    _ => panic!("NA mismatch op={:?} i={}: {:?} vs {:?}", op, i, a, b),
                }
            }
        }
    }

    // ── Distance kernel tests (Phase K.11) ───────────────────────────

    #[test]
    fn distance_euclidean_basic() {
        let a: Vec<Option<f64>> = vec![Some(0.0), Some(0.0), Some(0.0)];
        let b: Vec<Option<f64>> = vec![Some(3.0), Some(4.0), Some(0.0)];
        // 3-4-5 triangle in xy plane
        assert!((distance(DistanceOp::Euclidean, &a, &b).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn distance_manhattan_basic() {
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        let b: Vec<Option<f64>> = vec![Some(4.0), Some(6.0), Some(3.0)];
        // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        assert_eq!(distance(DistanceOp::Manhattan, &a, &b), Some(7.0));
    }

    #[test]
    fn distance_cosine_basic() {
        // Parallel vectors have cosine distance 0.
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        let b: Vec<Option<f64>> = vec![Some(2.0), Some(4.0), Some(6.0)];
        let d = distance(DistanceOp::Cosine, &a, &b).unwrap();
        assert!(d.abs() < 1e-12, "cosine of parallel = {}", d);
        // Orthogonal vectors have cosine distance 1.
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(0.0)];
        let b: Vec<Option<f64>> = vec![Some(0.0), Some(1.0)];
        let d = distance(DistanceOp::Cosine, &a, &b).unwrap();
        assert!((d - 1.0).abs() < 1e-12, "cosine of orthogonal = {}", d);
    }

    #[test]
    fn distance_skips_na_positions() {
        let a: Vec<Option<f64>> = vec![Some(1.0), None, Some(3.0)];
        let b: Vec<Option<f64>> = vec![Some(4.0), Some(5.0), Some(3.0)];
        // Manhattan: |1-4|=3 at idx 0; idx 1 skipped; |3-3|=0 at idx 2. Total = 3.
        assert_eq!(distance(DistanceOp::Manhattan, &a, &b), Some(3.0));
    }

    #[test]
    fn pairwise_distance_diagonal_and_symmetric() {
        // 3 points in 2D, row-major.
        let data: Vec<Option<f64>> = vec![
            Some(0.0), Some(0.0),     // point 0
            Some(3.0), Some(4.0),     // point 1
            Some(6.0), Some(8.0),     // point 2
        ];
        let d = pairwise_distance(DistanceOp::Euclidean, &data, 3, 2);
        // Diagonal should be 0.
        for i in 0..3 { assert_eq!(d[i * 3 + i], Some(0.0)); }
        // (0,1) and (1,0) should both be 5 (3-4-5 triangle).
        assert!((d[0 * 3 + 1].unwrap() - 5.0).abs() < 1e-12);
        assert!((d[1 * 3 + 0].unwrap() - 5.0).abs() < 1e-12);
        // (0,2): sqrt(36+64) = 10
        assert!((d[0 * 3 + 2].unwrap() - 10.0).abs() < 1e-12);
    }

    // ── Hash-agg kernel tests (Phase K.10) ───────────────────────────

    #[test]
    fn hash_agg_sum_basic() {
        let keys: Vec<Option<u64>>   = vec![Some(1), Some(2), Some(1), Some(2), Some(3)].into_iter().collect();
        let vals: Vec<Option<f64>>   = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0), Some(50.0)];
        let r = hash_agg(AggOp::Sum, &keys, &vals);
        // Groups: key 1 → 10+30=40; key 2 → 20+40=60; key 3 → 50.
        // Insertion order: 1, 2, 3.
        assert_eq!(r.keys, vec![1, 2, 3]);
        assert_eq!(r.values, vec![Some(40.0), Some(60.0), Some(50.0)]);
    }

    #[test]
    fn hash_agg_mean_basic() {
        let keys: Vec<Option<u64>>   = vec![Some(1), Some(2), Some(1), Some(2)].into_iter().collect();
        let vals: Vec<Option<f64>>   = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)];
        let r = hash_agg(AggOp::Mean, &keys, &vals);
        assert_eq!(r.values, vec![Some(20.0), Some(30.0)]);
    }

    #[test]
    fn hash_agg_count_skips_na() {
        let keys: Vec<Option<u64>>   = vec![Some(1), Some(2), None, Some(1), Some(2)].into_iter().collect();
        let vals: Vec<Option<f64>>   = vec![Some(10.0), None, Some(30.0), Some(40.0), Some(50.0)];
        let r = hash_agg(AggOp::Count, &keys, &vals);
        // NA-keyed value (30.0) skipped; NA value for key=2 skipped.
        // Key 1: 2 values, Key 2: 1 value.
        assert_eq!(r.keys, vec![1, 2]);
        assert_eq!(r.values, vec![Some(2.0), Some(1.0)]);
    }

    #[test]
    fn hash_agg_min_max() {
        let keys: Vec<Option<u64>>   = vec![Some(1), Some(1), Some(1), Some(2)].into_iter().collect();
        let vals: Vec<Option<f64>>   = vec![Some(5.0), Some(2.0), Some(8.0), Some(7.0)];
        let r_min = hash_agg(AggOp::Min, &keys, &vals);
        let r_max = hash_agg(AggOp::Max, &keys, &vals);
        assert_eq!(r_min.values, vec![Some(2.0), Some(7.0)]);
        assert_eq!(r_max.values, vec![Some(8.0), Some(7.0)]);
    }

    #[test]
    fn hash_tabulate_basic() {
        let keys: Vec<Option<u64>> = vec![Some(1), Some(2), Some(1), Some(3), Some(1), Some(2)].into_iter().collect();
        let r = hash_tabulate(&keys);
        // Insertion order: 1, 2, 3 with counts 3, 2, 1.
        assert_eq!(r.keys, vec![1, 2, 3]);
        assert_eq!(r.values, vec![Some(3.0), Some(2.0), Some(1.0)]);
    }

    // ── Rolling kernel tests (Phase K.9) ─────────────────────────────

    #[test]
    fn rolling_sum_and_mean_basic() {
        // rollsum([1,2,3,4,5], 3) = [6, 9, 12]
        let d: Vec<Option<f64>> = (1..=5).map(|i| Some(i as f64)).collect();
        let r = rolling(RollingOp::Sum, &d, 3);
        assert_eq!(r, vec![Some(6.0), Some(9.0), Some(12.0)]);
        let m = rolling(RollingOp::Mean, &d, 3);
        assert_eq!(m, vec![Some(2.0), Some(3.0), Some(4.0)]);
    }

    #[test]
    fn rolling_max_min_basic() {
        let d: Vec<Option<f64>> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
            .into_iter().map(Some).collect();
        // window=3, rollmax: max(3,1,4)=4, max(1,4,1)=4, max(4,1,5)=5,
        //                    max(1,5,9)=9, max(5,9,2)=9, max(9,2,6)=9
        assert_eq!(rolling(RollingOp::Max, &d, 3),
            vec![Some(4.0), Some(4.0), Some(5.0), Some(9.0), Some(9.0), Some(9.0)]);
        assert_eq!(rolling(RollingOp::Min, &d, 3),
            vec![Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(2.0), Some(2.0)]);
    }

    #[test]
    fn rolling_na_propagates_within_window() {
        let d: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), None, Some(4.0), Some(5.0)];
        // window=2: [1,2]=3, [2,NA]=NA, [NA,4]=NA, [4,5]=9
        assert_eq!(rolling(RollingOp::Sum, &d, 2),
            vec![Some(3.0), None, None, Some(9.0)]);
    }

    #[test]
    fn rolling_sd_basic() {
        // Window of 3 over [1,2,3,4,5]: each window has var = 1, sd = 1.
        let d: Vec<Option<f64>> = (1..=5).map(|i| Some(i as f64)).collect();
        let r = rolling(RollingOp::Sd, &d, 3);
        for v in &r {
            let x = v.unwrap();
            assert!((x - 1.0).abs() < 1e-12, "got {}", x);
        }
    }

    #[test]
    fn rolling_window_larger_than_data_returns_empty() {
        let d: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        assert!(rolling(RollingOp::Sum, &d, 10).is_empty());
        assert!(rolling(RollingOp::Sum, &d, 0).is_empty());
    }

    // ── Select kernel tests (Phase K.8) ──────────────────────────────

    #[test]
    fn select_which_max_min() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0), Some(9.0), Some(2.0)];
        assert_eq!(which_max(&d), Some(5)); // value 9.0 at index 5
        assert_eq!(which_min(&d), Some(1)); // value 1.0 at index 1 (first)
    }

    #[test]
    fn select_which_with_na() {
        let d: Vec<Option<f64>> = vec![Some(3.0), None, Some(4.0)];
        assert_eq!(which_max(&d), None);
        assert_eq!(which_min(&d), None);
    }

    #[test]
    fn select_nth_smallest_basic() {
        let d: Vec<Option<f64>> = vec![Some(5.0), Some(2.0), Some(8.0), Some(1.0), Some(3.0)];
        assert_eq!(nth_smallest(&d, 0), Some(1.0)); // min
        assert_eq!(nth_smallest(&d, 2), Some(3.0)); // median (of 5)
        assert_eq!(nth_smallest(&d, 4), Some(8.0)); // max
        assert_eq!(nth_smallest(&d, 5), None);      // out of range
    }

    #[test]
    fn select_nth_smallest_skips_na() {
        let d: Vec<Option<f64>> = vec![Some(5.0), None, Some(2.0), None, Some(8.0)];
        assert_eq!(nth_smallest(&d, 0), Some(2.0));
        assert_eq!(nth_smallest(&d, 2), Some(8.0));
        assert_eq!(nth_smallest(&d, 3), None);
    }

    #[test]
    fn select_top_k_basic() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0), Some(9.0), Some(2.0)];
        // Top 3 in descending order of value: 9, 5, 4 → indices 5, 4, 2
        assert_eq!(top_k(&d, 3), vec![5, 4, 2]);
    }

    #[test]
    fn select_bottom_k_basic() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0), Some(9.0), Some(2.0)];
        // Bottom 3 in ascending order: 1, 1, 2 → indices 1, 3, 6
        // (1.0 appears at both 1 and 3; tie-break by index)
        assert_eq!(bottom_k(&d, 3), vec![1, 3, 6]);
    }

    #[test]
    fn select_top_k_skips_na_and_handles_k_larger_than_data() {
        let d: Vec<Option<f64>> = vec![Some(3.0), None, Some(5.0)];
        assert_eq!(top_k(&d, 5), vec![2, 0]); // only 2 non-NA values
        assert_eq!(top_k(&d, 0), Vec::<usize>::new());
    }

    #[test]
    fn scan_rayon_handles_na_correctly() {
        // NA in the middle of a Rayon-sized chunk should poison from
        // that index forward across all subsequent chunks.
        let n = 10_000;
        let mut d: Vec<Option<f64>> = (0..n).map(|i| Some((i + 1) as f64)).collect();
        d[5000] = None;
        let s = SerialBackend.scan(ScanOp::Cumsum, &d);
        let r = RayonBackend.scan(ScanOp::Cumsum, &d);
        for i in 0..n {
            assert_eq!(s[i].is_some(), r[i].is_some(),
                "NA presence mismatch at i={}", i);
        }
        // Everything at and after i=5000 should be None.
        for i in 5000..n {
            assert!(s[i].is_none() && r[i].is_none(), "i={}", i);
        }
    }
