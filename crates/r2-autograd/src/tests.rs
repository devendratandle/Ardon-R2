//! Every backward is checked against finite differences here: an analytic
//! gradient must match the numeric one, or it does not ship.

use super::*;


/// A small transformer-shaped graph run three times through ONE pool:
/// step 1 has fresh buffers, steps 2 and 3 get recycled ones full of
/// stale values and stale gradients. Every value and every gradient
/// must be bit-identical to a tape with no pool at all.
#[test]
fn pooled_tapes_reproduce_fresh_tapes_bit_for_bit() {
    let (t, d, ffn, vocab) = (16usize, 8usize, 24usize, 40usize);
    let table: Vec<f32> = (0..vocab * d).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
    let w1: Vec<f32> = (0..d * ffn).map(|i| ((i as f32) * 0.11).cos() * 0.3).collect();
    let w3: Vec<f32> = (0..d * ffn).map(|i| ((i as f32) * 0.23).sin() * 0.3).collect();
    let w2: Vec<f32> = (0..ffn * d).map(|i| ((i as f32) * 0.17).cos() * 0.3).collect();
    let nw: Vec<f32> = vec![1.0; d];
    let tokens: Vec<usize> = (0..t).map(|i| (i * 7 + 3) % vocab).collect();
    let targets: Vec<usize> = (0..t).map(|i| (i * 11 + 1) % vocab).collect();

    let run = |tape: &mut Tape| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let lt = tape.leaf(table.clone(), true);
        let l1 = tape.leaf(w1.clone(), true);
        let l3 = tape.leaf(w3.clone(), true);
        let l2 = tape.leaf(w2.clone(), true);
        let ln = tape.leaf(nw.clone(), true);
        let x = tape.embed(lt, &tokens, d);
        let h = tape.rmsnorm(x, ln, d, 1e-5);
        let gate = tape.matmul(h, l1, t, d, ffn);
        let up = tape.matmul(h, l3, t, d, ffn);
        let act = tape.silu(gate);
        let gated = tape.mul(act, up);
        let down = tape.matmul(gated, l2, t, ffn, d);
        let y = tape.add(x, down);
        // logits via the table as an output head: t x vocab
        let tt = tape.transpose(lt, vocab, d);
        let logits = tape.matmul(y, tt, t, d, vocab);
        let loss = tape.softmax_ce(logits, vocab, targets.clone());
        tape.backward(loss);
        let vals = [x, h, gate, up, act, gated, down, y, logits, loss].iter().map(|v| tape.value(*v).to_vec()).collect();
        let grads = [lt, l1, l3, l2, ln, x, h, gate, y].iter().map(|v| tape.grad(*v).to_vec()).collect();
        (vals, grads)
    };

    let mut fresh = Tape::new();
    let want = run(&mut fresh);

    let mut pool = BufPool::new();
    for step in 0..3 {
        let mut tape = Tape::with_pool(std::mem::take(&mut pool));
        let got = run(&mut tape);
        assert_eq!(got.0, want.0, "values differ on pooled step {step}");
        assert_eq!(got.1, want.1, "gradients differ on pooled step {step}");
        let (n, _) = tape.pool_stats();
        pool = tape.into_pool();
        assert!(pool.stats().0 > n, "step {step}: into_pool did not return the buffers");
    }
}

/// Build attention the OLD way — slice, transpose, matmul, mask,
/// softmax, matmul, concat — exactly as `llm.rs::forward_fused` did
/// before `Op::Attention` existed. The fused op has to agree with this
/// or it is not a refactor, it is a different model.
#[allow(clippy::too_many_arguments)]
fn attention_decomposed(tape: &mut Tape, q: Var, k: Var, v: Var, nseq: usize,
                        seq: usize, nh: usize, nkv: usize, hd: usize) -> Var {
    let group = nh / nkv;
    let mut seq_ctx: Vec<Var> = Vec::with_capacity(nseq);
    for s in 0..nseq {
        let (q_s, k_s, v_s) = if nseq == 1 {
            (q, k, v)
        } else {
            (tape.slice_rows(q, nh * hd, s * seq, seq),
             tape.slice_rows(k, nkv * hd, s * seq, seq),
             tape.slice_rows(v, nkv * hd, s * seq, seq))
        };
        let mut heads: Vec<Var> = Vec::with_capacity(nh);
        for qh in 0..nh {
            let kvh = qh / group;
            let qs = tape.slice_cols(q_s, seq, nh * hd, qh * hd, hd);
            let ks = tape.slice_cols(k_s, seq, nkv * hd, kvh * hd, hd);
            let vs = tape.slice_cols(v_s, seq, nkv * hd, kvh * hd, hd);
            let kt = tape.transpose(ks, seq, hd);
            let sc = tape.matmul(qs, kt, seq, hd, seq);
            let sc = tape.scale_mask_causal(sc, seq, 1.0 / (hd as f32).sqrt());
            let at = tape.softmax_rows(sc, seq);
            heads.push(tape.matmul(at, vs, seq, seq, hd));
        }
        seq_ctx.push(tape.concat_cols(&heads, seq, hd));
    }
    if nseq == 1 { seq_ctx[0] } else { tape.concat_rows(&seq_ctx) }
}

/// The fused op must compute the SAME function as the decomposition it
/// replaces — value and all three gradients — or the 1,156 tape nodes
/// it deletes were carrying meaning.
///
/// Tolerance is f32 rounding, not equality: the fused form sums over
/// `j <= i` while the decomposition softmaxes a full row containing
/// `-inf`, so the two do the same arithmetic in a different order.
#[test]
fn attention_matches_the_decomposition() {
    for &(nseq, seq, nh, nkv, hd) in &[
        (1usize, 4usize, 2usize, 1usize, 4usize),   // MQA, one sequence
        (3, 5, 4, 2, 6),                            // GQA, ragged-ish
        (2, 6, 3, 3, 4),                            // MHA, no grouping
        // Kernel widths and blocks: hd 8 / 16 / 64 pick the 8-, 16-
        // lane accumulators; seq 21 / 33 / 40 span several 16-query
        // blocks with a ragged last one; the last case is large enough
        // to take the parallel (sequence, head) path.
        (2, 21, 2, 1, 8),
        (1, 33, 3, 3, 16),
        (4, 40, 4, 2, 64),
    ] {
        let rows = nseq * seq;
        let mk = |n: usize, ph: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * 0.37 + ph).sin() * 0.8).collect()
        };
        let (qv, kv_, vv) = (mk(rows * nh * hd, 0.0), mk(rows * nkv * hd, 1.3),
                             mk(rows * nkv * hd, 2.7));
        let g = mk(rows * nh * hd, 0.9);
        let scale = 1.0 / (hd as f32).sqrt();

        let mut ta = Tape::new();
        let (qa, ka, va) = (ta.leaf(qv.clone(), true), ta.leaf(kv_.clone(), true),
                            ta.leaf(vv.clone(), true));
        let oa = ta.attention(qa, ka, va, nseq, seq, nh, nkv, hd, scale);
        ta.backward_from(oa, &g);

        let mut tb = Tape::new();
        let (qb, kb, vb) = (tb.leaf(qv.clone(), true), tb.leaf(kv_.clone(), true),
                            tb.leaf(vv.clone(), true));
        let ob = attention_decomposed(&mut tb, qb, kb, vb, nseq, seq, nh, nkv, hd);
        tb.backward_from(ob, &g);

        let close = |a: &[f32], b: &[f32], what: &str| {
            assert_eq!(a.len(), b.len(), "{what}: length differs");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert!((x - y).abs() <= 2e-5 * (1.0 + y.abs()),
                        "{what}[{i}] fused {x} vs decomposed {y} \
                         (nseq {nseq} seq {seq} nh {nh} nkv {nkv} hd {hd})");
            }
        };
        close(ta.value(oa), tb.value(ob), "value");
        close(ta.grad(qa), tb.grad(qb), "grad_q");
        close(ta.grad(ka), tb.grad(kb), "grad_k");
        close(ta.grad(va), tb.grad(vb), "grad_v");
    }
}

/// And it must pass the same finite-difference gate as every other op,
/// independently of the decomposition — if both were wrong the test
/// above would still pass.
#[test]
fn attention_gradient_matches_finite_difference() {
    let (nseq, seq, nh, nkv, hd) = (2usize, 4usize, 2usize, 1usize, 3usize);
    let rows = nseq * seq;
    let scale = 1.0 / (hd as f32).sqrt();
    let mk = |n: usize, ph: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.41 + ph).sin() * 0.7).collect()
    };
    let kv_ = mk(rows * nkv * hd, 1.1);
    let vv = mk(rows * nkv * hd, 2.2);
    // Differentiate w.r.t. q, with k and v fixed: a scalar loss so
    // finite differences apply.
    let qv = mk(rows * nh * hd, 0.0);
    check_grad(&qv, |t: &mut Tape, p: &[f32]| {
        let q = t.leaf(p.to_vec(), true);
        let k = t.leaf(kv_.clone(), false);
        let v = t.leaf(vv.clone(), false);
        let o = t.attention(q, k, v, nseq, seq, nh, nkv, hd, scale);
        let l = t.sum_all(o);
        (q, l)
    });
    // And w.r.t. v, which reaches the loss by a different path.
    check_grad(&vv, |t: &mut Tape, p: &[f32]| {
        let q = t.leaf(qv.clone(), false);
        let k = t.leaf(kv_.clone(), false);
        let v = t.leaf(p.to_vec(), true);
        let o = t.attention(q, k, v, nseq, seq, nh, nkv, hd, scale);
        let l = t.sum_all(o);
        (v, l)
    });
}

/// A token must not see its future. Perturbing position `i` of k or v
/// may change outputs at positions >= i and must leave every earlier
/// position bit-identical — the causal mask is the one property whose
/// failure trains a model that cheats and still looks healthy.
#[test]
fn attention_is_causal() {
    let (nseq, seq, nh, nkv, hd) = (1usize, 6usize, 2usize, 1usize, 4usize);
    let rows = nseq * seq;
    let scale = 1.0 / (hd as f32).sqrt();
    let mk = |n: usize, ph: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.29 + ph).sin()).collect()
    };
    let qv = mk(rows * nh * hd, 0.0);
    let kv_ = mk(rows * nkv * hd, 1.0);
    let vv = mk(rows * nkv * hd, 2.0);
    let run = |k: &[f32], v: &[f32]| -> Vec<f32> {
        let mut t = Tape::new();
        let (a, b, c) = (t.leaf(qv.clone(), false), t.leaf(k.to_vec(), false),
                         t.leaf(v.to_vec(), false));
        let o = t.attention(a, b, c, nseq, seq, nh, nkv, hd, scale);
        t.value(o).to_vec()
    };
    let base = run(&kv_, &vv);
    for pos in 1..seq {
        let mut k2 = kv_.clone();
        let mut v2 = vv.clone();
        for c in 0..nkv * hd {
            k2[pos * nkv * hd + c] += 3.0;
            v2[pos * nkv * hd + c] += 3.0;
        }
        let got = run(&k2, &v2);
        for i in 0..pos {
            for c in 0..nh * hd {
                let (a, b) = (base[i * nh * hd + c], got[i * nh * hd + c]);
                assert_eq!(a, b,
                    "changing position {pos} changed output at EARLIER position {i} \
                     (col {c}): {a} -> {b}. Attention is not causal.");
            }
        }
    }
}

/// Assert analytic grad (from a fresh tape built by `build`) matches
/// the finite-difference grad of the same scalar function.
fn check_grad<B>(params: &[f32], build: B)
where B: Fn(&mut Tape, &[f32]) -> (Var, Var) {
    // Analytic: build tape, backward, read leaf grad.
    let mut t = Tape::new();
    let (leaf, loss) = build(&mut t, params);
    t.backward(loss);
    let analytic = t.grad(leaf).to_vec();
    // Numeric: scalar loss as a function of the leaf's params.
    let numeric = finite_diff(params, |p| {
        let mut t = Tape::new();
        let (_, loss) = build(&mut t, p);
        t.value(loss)[0]
    });
    let maxerr = analytic.iter().zip(&numeric)
        .map(|(a, n)| (a - n).abs()).fold(0.0f32, f32::max);
    assert!(maxerr < 2e-2, "grad mismatch: analytic {:?} numeric {:?}", analytic, numeric);
}

#[test]
fn add_mul_chain_grad() {
    check_grad(&[1.5, -2.0, 0.5], |t, p| {
        let x = t.leaf(p.to_vec(), true);
        let c = t.leaf(vec![2.0, 3.0, -1.0], false);
        let y = t.mul(x, c);       // x*c
        let z = t.add(y, x);       // x*c + x
        let loss = t.sum_all(z);
        (x, loss)
    });
}

#[test]
fn matmul_grad() {
    check_grad(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |t, p| {
        let a = t.leaf(p.to_vec(), true);          // 2×3
        let b = t.leaf(vec![1.0, 0.5, -1.0, 2.0, 0.0, 1.5], false); // 3×2
        let c = t.matmul(a, b, 2, 3, 2);           // 2×2
        let loss = t.sum_all(c);
        (a, loss)
    });
}

#[test]
fn silu_grad() {
    check_grad(&[-1.0, 0.3, 2.0, -0.5], |t, p| {
        let x = t.leaf(p.to_vec(), true);
        let y = t.silu(x);
        let loss = t.sum_all(y);
        (x, loss)
    });
}

#[test]
fn rmsnorm_grad_wrt_x() {
    check_grad(&[0.5, -1.5, 2.0, 0.25], |t, p| {
        let x = t.leaf(p.to_vec(), true);
        let w = t.leaf(vec![1.0, 0.5, 1.5, 2.0], false);
        let y = t.rmsnorm(x, w, 4, 1e-5);
        let loss = t.sum_all(y);
        (x, loss)
    });
}

#[test]
fn rmsnorm_grad_wrt_w() {
    check_grad(&[1.0, 0.5, 1.5, 2.0], |t, p| {
        let x = t.leaf(vec![0.5, -1.5, 2.0, 0.25], false);
        let w = t.leaf(p.to_vec(), true);
        let y = t.rmsnorm(x, w, 4, 1e-5);
        let loss = t.sum_all(y);
        (w, loss)
    });
}

#[test]
fn mse_grad() {
    check_grad(&[0.2, 0.8, -0.4], |t, p| {
        let pred = t.leaf(p.to_vec(), true);
        let loss = t.mse(pred, vec![1.0, 0.0, -1.0]);
        (pred, loss)
    });
}

#[test]
fn transpose_grad() {
    check_grad(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |t, p| {
        let x = t.leaf(p.to_vec(), true);       // 2×3
        let xt = t.transpose(x, 2, 3);          // 3×2
        let c = t.leaf(vec![1.0, -2.0, 0.5, 1.5, -1.0, 2.0], false);
        let prod = t.mul(xt, c);
        let loss = t.sum_all(prod);
        (x, loss)
    });
}

#[test]
fn softmax_rows_grad() {
    check_grad(&[1.0, 2.0, 0.5, -1.0, 0.3, 2.0], |t, p| {
        let x = t.leaf(p.to_vec(), true);       // 2 rows × 3
        let sm = t.softmax_rows(x, 3);
        let c = t.leaf(vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.5], false);
        let prod = t.mul(sm, c);
        let loss = t.sum_all(prod);
        (x, loss)
    });
}

#[test]
fn softmax_cross_entropy_grad() {
    // 2 rows × 3 classes; targets [0, 2].
    check_grad(&[2.0, 1.0, 0.1, -1.0, 0.5, 3.0], |t, p| {
        let logits = t.leaf(p.to_vec(), true);
        let loss = t.softmax_ce(logits, 3, vec![0, 2]);
        (logits, loss)
    });
}

#[test]
fn tiny_mlp_trains() {
    // A 1-layer net loss must DECREASE under gradient steps — proves
    // forward+backward compose into real learning.
    let mut w = vec![0.1f32; 6]; // 3→2
    let x = vec![1.0, 2.0, -1.0]; // 1×3
    let target = vec![1.0, -1.0];
    let mut prev = f32::INFINITY;
    for _ in 0..50 {
        let mut t = Tape::new();
        let wv = t.leaf(w.clone(), true);
        let xv = t.leaf(x.clone(), false);
        let y = t.matmul(xv, wv, 1, 3, 2);
        let a = t.silu(y);
        let loss = t.mse(a, target.clone());
        t.backward(loss);
        let g = t.grad(wv).to_vec();
        for (wi, gi) in w.iter_mut().zip(&g) { *wi -= 0.1 * gi; }
        let l = t.value(loss)[0];
        assert!(l <= prev + 1e-5, "loss went up: {} -> {}", prev, l);
        prev = l;
    }
    assert!(prev < 1.0, "final loss {}", prev);
}

// ── RoPE ─────────────────────────────────────────────────────────────


/// RoPE must pass the same finite-difference gate as every other op:
/// the analytic backward has to match a numeric derivative, or a model
/// trains toward the wrong thing while still appearing to converge.
#[test]
fn rope_gradient_matches_finite_difference() {
    let (rows, nh, hd, base) = (4usize, 2usize, 4usize, 10000.0f32);
    let n = rows * nh * hd;
    let x0: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin()).collect();
    // Scalar objective: sum of squares after rotation.
    let loss_of = |p: &[f32]| -> f32 {
        let mut t = Tape::new();
        let x = t.leaf(p.to_vec(), true);
        let r = t.rope(x, rows, nh, hd, base);
        let sq = t.mul(r, r);
        let l = t.sum_all(sq);
        t.vals[l.0][0]
    };
    let mut t = Tape::new();
    let x = t.leaf(x0.clone(), true);
    let r = t.rope(x, rows, nh, hd, base);
    let sq = t.mul(r, r);
    let l = t.sum_all(sq);
    t.backward(l);
    let analytic = t.grad(x).to_vec();
    let numeric = finite_diff(&x0, loss_of);
    for (i, (a, b)) in analytic.iter().zip(&numeric).enumerate() {
        assert!((a - b).abs() < 2e-2, "elem {i}: analytic {a} vs numeric {b}");
    }
}

/// A rotation preserves length — the property that lets RoPE encode
/// position without changing the scale of what flows through it.
#[test]
fn rope_preserves_norm_and_matches_the_inference_kernel() {
    let (rows, nh, hd) = (3usize, 2usize, 4usize);
    let n = rows * nh * hd;
    let x0: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.21).cos()).collect();
    let mut t = Tape::new();
    let x = t.leaf(x0.clone(), false);
    let r = t.rope(x, rows, nh, hd, 10000.0);
    let out = t.vals[r.0].clone();

    let norm = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>().sqrt();
    assert!((norm(&out) - norm(&x0)).abs() < 1e-4, "RoPE must preserve norm");

    // And it must agree bit-for-bit with the serving kernel, so a
    // trained model behaves identically when served by r2-tensor.
    let mut want = x0.clone();
    for r_i in 0..rows {
        for h in 0..nh {
            let off = r_i * nh * hd + h * hd;
            r2_tensor::ops::rope_inplace(&mut want[off..off + hd], r_i, 10000.0);
        }
    }
    assert_eq!(out, want, "training RoPE must equal the inference RoPE exactly");
}
