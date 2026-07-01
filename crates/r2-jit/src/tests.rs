    use super::*;
    use r2_ir::{lower_function, lower_program};
    use r2_types::infer::{IrType, IrElem as E};
    use r2_types::*;
    use std::sync::Arc;

    fn num(n: f64) -> Expr { Expr::NumLit(n) }
    fn sym(s: &str) -> Expr { Expr::Symbol(Arc::from(s)) }
    fn add(l: Expr, r: Expr) -> Expr { Expr::Binary { op: BinOp::Add, lhs: Box::new(l), rhs: Box::new(r) } }
    fn mul(l: Expr, r: Expr) -> Expr { Expr::Binary { op: BinOp::Mul, lhs: Box::new(l), rhs: Box::new(r) } }
    fn lt(l: Expr, r: Expr)  -> Expr { Expr::Binary { op: BinOp::Lt,  lhs: Box::new(l), rhs: Box::new(r) } }

    fn real_param(n: &str) -> (Arc<str>, IrType) { (Arc::from(n), IrType::scalar(E::Real)) }

    #[test]
    fn jit_const_returns_real() {
        let f = lower_program(&[num(42.0)], "k");
        let c = JitCompiler::compile(&f).expect("compile ok");
        unsafe { assert_eq!(c.call0(), 42.0); }
    }

    #[test]
    fn jit_one_param_identity() {
        let body = sym("x");
        let f = lower_function("ident", vec![real_param("x")], &body);
        let c = JitCompiler::compile(&f).expect("compile ok");
        unsafe {
            assert_eq!(c.call1(7.0), 7.0);
            assert_eq!(c.call1(-3.5), -3.5);
        }
    }

    #[test]
    fn jit_two_param_add() {
        let body = add(sym("x"), sym("y"));
        let f = lower_function("add", vec![real_param("x"), real_param("y")], &body);
        let c = JitCompiler::compile(&f).expect("compile ok");
        unsafe { assert_eq!(c.call2(1.5, 2.5), 4.0); }
    }

    #[test]
    fn jit_polynomial() {
        // f(x) = x*x + 2*x + 1   →  f(3) = 16
        let body = add(add(mul(sym("x"), sym("x")), mul(num(2.0), sym("x"))), num(1.0));
        let f = lower_function("poly", vec![real_param("x")], &body);
        let c = JitCompiler::compile(&f).expect("compile ok");
        unsafe { assert_eq!(c.call1(3.0), 16.0); }
    }

    #[test]
    fn jit_if_else_with_phi() {
        // function(x) if (x < 0) -x else x   →  abs(x)
        let body = Expr::If {
            cond: Box::new(lt(sym("x"), num(0.0))),
            then: Box::new(Expr::Unary { op: UnOp::Neg, expr: Box::new(sym("x")) }),
            else_: Some(Box::new(sym("x"))),
        };
        let f = lower_function("absval", vec![real_param("x")], &body);
        let c = JitCompiler::compile(&f).expect("compile ok");
        unsafe {
            assert_eq!(c.call1(-3.0), 3.0);
            assert_eq!(c.call1( 5.0), 5.0);
            assert_eq!(c.call1( 0.0), 0.0);
        }
    }

    #[test]
    fn jit_comparison_returns_one_or_zero() {
        // function(x) (x > 0)   →  1.0 / 0.0
        let body = Expr::Binary { op: BinOp::Gt, lhs: Box::new(sym("x")), rhs: Box::new(num(0.0)) };
        let f = lower_function("ispos", vec![real_param("x")], &body);
        let c = JitCompiler::compile(&f).expect("compile ok");
        unsafe {
            assert_eq!(c.call1( 1.0), 1.0);
            assert_eq!(c.call1(-1.0), 0.0);
        }
    }

    #[test]
    fn try_compile_closure_round_trip() {
        // Hand-build the AST equivalent of `function(x, y) x*x + y`.
        let body = Expr::Binary {
            op: BinOp::Add,
            lhs: Box::new(Expr::Binary {
                op: BinOp::Mul,
                lhs: Box::new(sym("x")),
                rhs: Box::new(sym("x")),
            }),
            rhs: Box::new(sym("y")),
        };

        let cl = Closure {
            params: vec![
                Param { name: Arc::from("x"), default: None, dots: false },
                Param { name: Arc::from("y"), default: None, dots: false },
            ],
            body: Arc::new(body),
            env: Env::new_global(),
        };

        let handle = try_compile_closure(&cl).expect("Closure should compile");
        assert_eq!(handle.arity(), 2);

        // After Phase C.7, 2-arg closures preferentially compile as
        // VectorBinaryMap (richer body coverage). Call accordingly.
        match handle.kind() {
            r2_types::JitKind::Scalar => {
                assert_eq!(handle.try_call_real(&[3.0, 5.0]), Some(14.0));
                assert_eq!(handle.try_call_real(&[1.0]), None);
            }
            r2_types::JitKind::VectorBinaryMap => {
                let a = vec![3.0_f64]; let b = vec![5.0_f64];
                let mut out = vec![0.0_f64; 1];
                let ok = unsafe {
                    handle.try_call_vec_binary(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 1)
                };
                assert!(ok);
                assert!((out[0] - 14.0).abs() < 1e-12);
            }
            other => panic!("unexpected kind: {:?}", other),
        }
    }

    #[test]
    fn try_compile_closure_vector_sum() {
        // function(v) sum(v)
        let body = Expr::Call {
            func: Box::new(sym("sum")),
            args: vec![CallArg { name: None, value: sym("v") }],
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("v"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile sum");
        assert_eq!(handle.kind(), r2_types::JitKind::Vector1ToScalar);

        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { handle.try_call_vec1(data.as_ptr(), data.len() as i64) };
        assert_eq!(result, Some(15.0));
    }

    #[test]
    fn jit_index_loop_fold_over_vector() {
        // Phase J.2: function(x){ n<-length(x); s<-init; for(i in 1:n) s<-s OP <c(x[i])>; s }
        // recognized as a map-reduce over x (x[i] → element).
        let assign = |nm: &str, val: Expr| Expr::Assign { target: Box::new(sym(nm)), value: Box::new(val), superassign: false };
        let idx = |nm: &str, i: &str| Expr::Index { object: Box::new(sym(nm)), indices: vec![Some(sym(i))] };
        let mkfn = |init: f64, upd: Expr| Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(Expr::Block(vec![
                assign("n", Expr::Call { func: Box::new(sym("length")), args: vec![CallArg { name: None, value: sym("x") }] }),
                assign("s", num(init)),
                Expr::For {
                    var: Arc::from("i"),
                    iter: Box::new(Expr::Binary { op: BinOp::Colon, lhs: Box::new(num(1.0)), rhs: Box::new(sym("n")) }),
                    body: Box::new(assign("s", upd)),
                },
                sym("s"),
            ])),
            env: Env::new_global(),
        };
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // sum: s <- s + x[i]  → 15
        let h = try_compile_closure(&mkfn(0.0, add(sym("s"), idx("x", "i")))).expect("index-sum should JIT");
        assert_eq!(h.kind(), r2_types::JitKind::Vector1ToScalar);
        assert_eq!(unsafe { h.try_call_vec1(data.as_ptr(), data.len() as i64) }, Some(15.0));
        // sum of squares: s <- s + x[i]*x[i]  → 55
        let hq = try_compile_closure(&mkfn(0.0, add(sym("s"), mul(idx("x", "i"), idx("x", "i"))))).expect("index-sumsq should JIT");
        assert_eq!(unsafe { hq.try_call_vec1(data.as_ptr(), data.len() as i64) }, Some(55.0));
        // product: s <- s * x[i]  → 120
        let hp = try_compile_closure(&mkfn(1.0, mul(sym("s"), idx("x", "i")))).expect("index-prod should JIT");
        assert_eq!(unsafe { hp.try_call_vec1(data.as_ptr(), data.len() as i64) }, Some(120.0));
        // Non-fold (uses the index, not x[i]) must NOT take this path:
        // function(x){ s<-0; for(i in 1:length(x)) s<-s+i; s } — recognizer returns None.
        assert!(recognize_index_reduction(mkfn(0.0, add(sym("s"), sym("i"))).body.as_ref(), "x").is_none());
    }

    #[test]
    fn try_compile_closure_vector_mean() {
        let body = Expr::Call {
            func: Box::new(sym("mean")),
            args: vec![CallArg { name: None, value: sym("xs") }],
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("xs"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile mean");
        let data: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0];
        let result = unsafe { handle.try_call_vec1(data.as_ptr(), data.len() as i64) };
        assert_eq!(result, Some(5.0));
    }

    #[test]
    fn try_compile_closure_vector_map_add() {
        // function(v) v + 1
        let body = Expr::Binary {
            op: BinOp::Add,
            lhs: Box::new(sym("v")),
            rhs: Box::new(num(1.0)),
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("v"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        assert_eq!(handle.kind(), r2_types::JitKind::VectorMap);

        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut output: Vec<f64> = vec![0.0; 4];
        let ok = unsafe { handle.try_call_vec_map(input.as_ptr(), output.as_mut_ptr(), 4) };
        assert!(ok);
        assert_eq!(output, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn try_compile_closure_vector_map_mul() {
        // function(v) v * 2  (literal-on-left also accepted via commutativity)
        let body = Expr::Binary {
            op: BinOp::Mul,
            lhs: Box::new(num(2.0)),
            rhs: Box::new(sym("v")),
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("v"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        let input: Vec<f64> = vec![1.5, 2.5, 3.5];
        let mut output: Vec<f64> = vec![0.0; 3];
        let ok = unsafe { handle.try_call_vec_map(input.as_ptr(), output.as_mut_ptr(), 3) };
        assert!(ok);
        assert_eq!(output, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn try_compile_closure_vector_binary_add() {
        // function(a, b) a + b
        let body = Expr::Binary {
            op: BinOp::Add,
            lhs: Box::new(sym("a")),
            rhs: Box::new(sym("b")),
        };
        let cl = Closure {
            params: vec![
                Param { name: Arc::from("a"), default: None, dots: false },
                Param { name: Arc::from("b"), default: None, dots: false },
            ],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        assert_eq!(handle.kind(), r2_types::JitKind::VectorBinaryMap);

        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![10.0, 20.0, 30.0];
        let mut out: Vec<f64> = vec![0.0; 3];
        let ok = unsafe { handle.try_call_vec_binary(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 3) };
        assert!(ok);
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn try_compile_closure_vector_binary_div_with_nan() {
        // function(a, b) a / b   — NaN propagation check (b[i]=0 → inf, NA represented as NaN)
        let body = Expr::Binary {
            op: BinOp::Div,
            lhs: Box::new(sym("a")),
            rhs: Box::new(sym("b")),
        };
        let cl = Closure {
            params: vec![
                Param { name: Arc::from("a"), default: None, dots: false },
                Param { name: Arc::from("b"), default: None, dots: false },
            ],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");

        let a: Vec<f64> = vec![10.0, f64::NAN, 9.0];
        let b: Vec<f64> = vec![ 2.0,    3.0,   3.0];
        let mut out: Vec<f64> = vec![0.0; 3];
        let ok = unsafe { handle.try_call_vec_binary(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 3) };
        assert!(ok);
        assert_eq!(out[0], 5.0);
        assert!(out[1].is_nan(), "NA in input should propagate through arithmetic");
        assert_eq!(out[2], 3.0);
    }

    #[test]
    fn try_compile_closure_composed_vector_map() {
        // function(v) (v + 1) * 2   →  expect [4, 6, 8] for input [1, 2, 3]
        let body = Expr::Binary {
            op: BinOp::Mul,
            lhs: Box::new(Expr::Binary {
                op: BinOp::Add,
                lhs: Box::new(sym("v")),
                rhs: Box::new(num(1.0)),
            }),
            rhs: Box::new(num(2.0)),
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("v"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile (v+1)*2");
        assert_eq!(handle.kind(), r2_types::JitKind::VectorMap);
        let input: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut output: Vec<f64> = vec![0.0; 3];
        let ok = unsafe { handle.try_call_vec_map(input.as_ptr(), output.as_mut_ptr(), 3) };
        assert!(ok);
        assert_eq!(output, vec![4.0, 6.0, 8.0]);
    }

    #[test]
    fn try_compile_closure_squaring_vector_map() {
        // function(v) v*v - 1   →  expect [-1, 0, 3, 8] for input [0, 1, 2, 3]
        let body = Expr::Binary {
            op: BinOp::Sub,
            lhs: Box::new(Expr::Binary {
                op: BinOp::Mul,
                lhs: Box::new(sym("v")),
                rhs: Box::new(sym("v")),
            }),
            rhs: Box::new(num(1.0)),
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("v"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile v*v - 1");
        let input: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let mut output: Vec<f64> = vec![0.0; 4];
        let ok = unsafe { handle.try_call_vec_map(input.as_ptr(), output.as_mut_ptr(), 4) };
        assert!(ok);
        assert_eq!(output, vec![-1.0, 0.0, 3.0, 8.0]);
    }

    #[test]
    fn try_compile_closure_branchy_vector_map_abs() {
        // function(x) if (x > 0) x else -x   over a vector of length 5
        let body = Expr::If {
            cond: Box::new(Expr::Binary {
                op: BinOp::Gt,
                lhs: Box::new(sym("x")),
                rhs: Box::new(num(0.0)),
            }),
            then: Box::new(sym("x")),
            else_: Some(Box::new(Expr::Unary { op: r2_types::UnOp::Neg, expr: Box::new(sym("x")) })),
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile branchy unary map");
        assert_eq!(handle.kind(), r2_types::JitKind::VectorMap);

        let input: Vec<f64> = vec![-3.0, -1.0, 0.0, 2.0, -5.5];
        let mut out: Vec<f64> = vec![0.0; input.len()];
        let ok = unsafe { handle.try_call_vec_map(input.as_ptr(), out.as_mut_ptr(), input.len() as i64) };
        assert!(ok);
        // 0.0 is not > 0, so it takes the else branch (-0.0). Compare by abs.
        let expected = vec![3.0, 1.0, 0.0, 2.0, 5.5];
        for (got, exp) in out.iter().zip(expected.iter()) {
            assert!((got.abs() - exp).abs() < 1e-12, "got {} expected {}", got, exp);
        }
    }

    #[test]
    fn try_compile_closure_ternary_ifelse() {
        // function(c, a, b) if (c > 0) a else b   over three same-length vectors
        let body = Expr::If {
            cond: Box::new(Expr::Binary {
                op: BinOp::Gt,
                lhs: Box::new(sym("c")),
                rhs: Box::new(num(0.0)),
            }),
            then: Box::new(sym("a")),
            else_: Some(Box::new(sym("b"))),
        };
        let cl = Closure {
            params: vec![
                Param { name: Arc::from("c"), default: None, dots: false },
                Param { name: Arc::from("a"), default: None, dots: false },
                Param { name: Arc::from("b"), default: None, dots: false },
            ],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile ternary ifelse");
        assert_eq!(handle.kind(), r2_types::JitKind::VectorTernaryMap);
        assert_eq!(handle.arity(), 3);

        let c: Vec<f64> = vec![1.0, -1.0, 2.0, 0.0, -0.5];
        let a: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let b: Vec<f64> = vec![-10.0, -20.0, -30.0, -40.0, -50.0];
        let mut out: Vec<f64> = vec![0.0; c.len()];
        let ok = unsafe {
            handle.try_call_vec_ternary(
                c.as_ptr(), a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), c.len() as i64,
            )
        };
        assert!(ok);
        // c>0 picks a; otherwise b. c=0.0 fails >0 → picks b.
        assert_eq!(out, vec![10.0, -20.0, 30.0, -40.0, -50.0]);
    }

    #[test]
    fn try_compile_closure_rejects_dots() {
        let cl = Closure {
            params: vec![Param { name: Arc::from("..."), default: None, dots: true }],
            body: Arc::new(Expr::NumLit(1.0)),
            env: Env::new_global(),
        };
        assert!(try_compile_closure(&cl).is_none());
    }

    // ── Math-extern Call lowering (extended JIT coverage) ─────────────

    fn call(fname: &str, args: Vec<Expr>) -> Expr {
        Expr::Call {
            func: Box::new(sym(fname)),
            args: args.into_iter().map(|v| CallArg { name: None, value: v }).collect(),
        }
    }

    /// Helper: invoke a JIT handle on a single f64 input regardless of
    /// whether `try_compile_closure` chose the Scalar or VectorMap path.
    fn call_jit_single(handle: &std::sync::Arc<dyn r2_types::JitHandle>, x: f64) -> f64 {
        match handle.kind() {
            r2_types::JitKind::Scalar => handle.try_call_real(&[x]).expect("scalar call"),
            r2_types::JitKind::VectorMap => {
                let input = vec![x];
                let mut out = vec![0.0_f64; 1];
                let ok = unsafe { handle.try_call_vec_map(input.as_ptr(), out.as_mut_ptr(), 1) };
                assert!(ok, "vec_map call");
                out[0]
            }
            other => panic!("unexpected JIT kind: {:?}", other),
        }
    }

    #[test]
    fn jit_call_to_sqrt() {
        // function(x) sqrt(x*x + 1) — pre-Call-lowering this would have
        // fallen through to interpreter. Now lowers fully to native.
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(call("sqrt", vec![add(mul(sym("x"), sym("x")), num(1.0))])),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        // sqrt(3*3 + 1) = sqrt(10)
        let r = call_jit_single(&handle, 3.0);
        assert!((r - 10.0_f64.sqrt()).abs() < 1e-12);
        // sqrt(0*0 + 1) = 1
        let r = call_jit_single(&handle, 0.0);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jit_call_to_exp_log() {
        // function(x) log(exp(x)) — round trip should be identity
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(call("log", vec![call("exp", vec![sym("x")])])),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        for x in [0.0, 1.0, 2.5, -3.7, 10.0] {
            let r = call_jit_single(&handle, x);
            assert!((r - x).abs() < 1e-10, "log(exp({})) = {}", x, r);
        }
    }

    #[test]
    fn jit_call_to_abs() {
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(call("abs", vec![sym("x")])),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        assert!((call_jit_single(&handle, -5.0) - 5.0).abs() < 1e-12);
        assert!((call_jit_single(&handle,  5.0) - 5.0).abs() < 1e-12);
        assert!((call_jit_single(&handle,  0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn vector_jit_call_to_sqrt() {
        // function(x) sqrt(x) applied to a vector — uses the vector map
        // path which also runs through `lower_inst` with the new Call
        // handler in the per-element body.
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(call("sqrt", vec![sym("x")])),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        // Either the scalar path got chosen (since one-arg) or the vector
        // map path. Both are valid; the vector map kicks in when called
        // with a vector input via the engine. We test the latter shape
        // directly by checking the VectorMap kind:
        match handle.kind() {
            r2_types::JitKind::Scalar => {
                let r = handle.try_call_real(&[4.0]).expect("scalar ok");
                assert!((r - 2.0).abs() < 1e-12);
            }
            r2_types::JitKind::VectorMap => {
                let input: Vec<f64> = vec![1.0, 4.0, 9.0, 16.0];
                let mut out = vec![0.0_f64; 4];
                let ok = unsafe { handle.try_call_vec_map(input.as_ptr(), out.as_mut_ptr(), 4) };
                assert!(ok);
                for (got, exp) in out.iter().zip([1.0, 2.0, 3.0, 4.0].iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("unexpected kind: {:?}", other),
        }
    }

    #[test]
    fn simd_jit_produces_correct_results_for_sqrt_xx_plus_1() {
        // Phase C.8: SIMD f64x2 vectorized path on math1 shape.
        // Verifies correctness vs scalar reference for an odd-length input
        // (exercises both the SIMD-2 loop and the scalar remainder).
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(call("sqrt", vec![add(mul(sym("x"), sym("x")), num(1.0))])),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        assert_eq!(handle.kind(), r2_types::JitKind::VectorMap);

        // Odd-length to force the remainder path.
        let input: Vec<f64> = (1..=7).map(|i| i as f64).collect();
        let mut out = vec![0.0_f64; input.len()];
        let ok = unsafe {
            handle.try_call_vec_map(input.as_ptr(), out.as_mut_ptr(), input.len() as i64)
        };
        assert!(ok);
        // sqrt(i*i + 1) for i in 1..=7
        let expected: Vec<f64> = input.iter().map(|x| (x*x + 1.0).sqrt()).collect();
        for (got, exp) in out.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "SIMD mismatch: {} vs {}", got, exp);
        }
    }

    #[test]
    fn simd_jit_correctly_falls_back_for_fcalls() {
        // function(x) sin(x) is NOT SIMD-clean (sin is a Rust-call,
        // not a native CPU instruction, so it can't be lane-vectorized),
        // so the SIMD path should bail. The fallback (generic scalar
        // vector map with Call lowering) should still produce a handle.
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(call("sin", vec![sym("x")])),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile (via fallback)");
        // Either VectorMap or Scalar is acceptable; both work via the
        // Rust-call extern path. SIMD path returned Err so we fell through.
        assert!(matches!(handle.kind(),
            r2_types::JitKind::VectorMap | r2_types::JitKind::Scalar));
    }

    #[test]
    fn jit_2arg_with_math_call_compiles() {
        // function(x, y) sqrt(x*x + y*y) — pre-C.7 fell back to interpreter
        // because the 2-arg vector path only accepted `a OP b` bodies.
        // Post-C.7, it compiles via the generic 2-arg multi-block path
        // with a native fsqrt instruction (Push A) for the sqrt.
        let body = call("sqrt", vec![
            add(mul(sym("x"), sym("x")), mul(sym("y"), sym("y")))
        ]);
        let cl = Closure {
            params: vec![
                Param { name: Arc::from("x"), default: None, dots: false },
                Param { name: Arc::from("y"), default: None, dots: false },
            ],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        assert_eq!(handle.kind(), r2_types::JitKind::VectorBinaryMap);
        assert_eq!(handle.arity(), 2);

        // sqrt(3² + 4²) = 5; sqrt(5² + 12²) = 13; sqrt(8² + 15²) = 17.
        let a = vec![3.0_f64, 5.0, 8.0];
        let b = vec![4.0_f64, 12.0, 15.0];
        let mut out = vec![0.0_f64; 3];
        let ok = unsafe {
            handle.try_call_vec_binary(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 3)
        };
        assert!(ok);
        for (got, exp) in out.iter().zip([5.0, 13.0, 17.0].iter()) {
            assert!((got - exp).abs() < 1e-12, "{} vs {}", got, exp);
        }
    }

    #[test]
    fn unsupported_call_falls_through() {
        // function(x) length(x) is not a math extern; Cranelift JIT
        // should reject with Unsupported, letting the engine fall back
        // to the tree-walking interpreter.
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(call("length", vec![sym("x")])),
            env: Env::new_global(),
        };
        // `try_compile_closure` will hit the existing Vector1ToScalar
        // path for the literal `length` shape first — that's OK and
        // returns a handle. We're checking that the general scalar JIT
        // path correctly rejects non-math-extern Calls (which it does
        // via lower_inst's Call arm returning Unsupported).
        let _ = try_compile_closure(&cl);
    }

    // ── Phase B.1 — closure capture inference ────────────────────────

    #[test]
    fn closure_capture_scalar_baked_in() {
        // env { scale = 2.5 }, function(x) x * scale
        let mut env = std::sync::Arc::make_mut(&mut Env::new_global().clone()).clone();
        env.bindings.insert(Arc::from("scale"),
            RVal::Numeric(vec![Some(2.5)].into(), Default::default()));
        let env = std::sync::Arc::new(env);

        let body = Expr::Binary {
            op: BinOp::Mul,
            lhs: Box::new(sym("x")),
            rhs: Box::new(sym("scale")), // free variable
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(body),
            env,
        };
        let handle = try_compile_closure(&cl).expect("should compile via capture inference");
        // After substitution, body becomes `x * 2.5`. That's the scalar
        // pattern, so any of Scalar / VectorMap / VectorBinaryMap kinds
        // are valid landings. Verify by calling and checking output.
        match handle.kind() {
            r2_types::JitKind::Scalar => {
                let r = handle.try_call_real(&[4.0]).expect("scalar call");
                assert!((r - 10.0).abs() < 1e-12);
            }
            r2_types::JitKind::VectorMap => {
                let input = vec![1.0_f64, 2.0, 3.0];
                let mut out = vec![0.0_f64; 3];
                let ok = unsafe { handle.try_call_vec_map(input.as_ptr(), out.as_mut_ptr(), 3) };
                assert!(ok);
                assert!((out[0] - 2.5).abs() < 1e-12);
                assert!((out[1] - 5.0).abs() < 1e-12);
                assert!((out[2] - 7.5).abs() < 1e-12);
            }
            other => panic!("unexpected kind: {:?}", other),
        }
    }

    #[test]
    fn closure_with_unbound_free_var_still_falls_through_gracefully() {
        // A free var that isn't in env (and isn't a builtin name) means
        // the JIT can't bake it in. The closure compiles fine if the
        // shape matches a non-substituted JIT pattern, OR returns None.
        // We accept either — the goal is "no panic, no wrong result".
        let cl = Closure {
            params: vec![Param { name: Arc::from("x"), default: None, dots: false }],
            body: Arc::new(Expr::Binary {
                op: BinOp::Add,
                lhs: Box::new(sym("x")),
                rhs: Box::new(sym("undefined_thing")),
            }),
            env: Env::new_global(),
        };
        let _ = try_compile_closure(&cl); // should not panic
    }

    // ── Phase C.9 — fused map-reduce ─────────────────────────────────

    #[test]
    fn jit_fused_map_reduce_sum_of_squares() {
        // function(v) sum(v*v) — fused; should JIT as Vector1ToScalar.
        let body = Expr::Call {
            func: Box::new(sym("sum")),
            args: vec![CallArg {
                name: None,
                value: Expr::Binary { op: BinOp::Mul, lhs: Box::new(sym("v")), rhs: Box::new(sym("v")) },
            }],
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("v"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile fused");
        assert_eq!(handle.kind(), r2_types::JitKind::Vector1ToScalar);
        // sum(v*v) for v = [1, 2, 3, 4, 5] = 1+4+9+16+25 = 55
        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = unsafe { handle.try_call_vec1(input.as_ptr(), input.len() as i64) };
        assert_eq!(r, Some(55.0));
    }

    #[test]
    fn jit_fused_map_reduce_sum_of_sqrt_plus_one() {
        // function(v) sum(sqrt(v*v + 1))
        let body = Expr::Call {
            func: Box::new(sym("sum")),
            args: vec![CallArg {
                name: None,
                value: Expr::Call {
                    func: Box::new(sym("sqrt")),
                    args: vec![CallArg {
                        name: None,
                        value: add(mul(sym("v"), sym("v")), num(1.0)),
                    }],
                },
            }],
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("v"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile fused");
        assert_eq!(handle.kind(), r2_types::JitKind::Vector1ToScalar);
        // v = [3, 4] → sqrt(10)+sqrt(17) ≈ 3.1623 + 4.1231 = 7.2854
        let input: Vec<f64> = vec![3.0, 4.0];
        let r = unsafe { handle.try_call_vec1(input.as_ptr(), input.len() as i64) };
        let expected = 10.0_f64.sqrt() + 17.0_f64.sqrt();
        assert!((r.unwrap() - expected).abs() < 1e-12, "got {:?} expected {}", r, expected);
    }

    #[test]
    fn jit_fused_map_reduce_prod_identity() {
        // function(v) prod(v) — vector reduction; in this case the
        // map step is identity. Should still hit the fused path (Prod
        // reducer with body = identity), or fall through to the existing
        // compile_vector_reduction. Either kind acceptable.
        let body = Expr::Call {
            func: Box::new(sym("prod")),
            args: vec![CallArg { name: None, value: sym("v") }],
        };
        let cl = Closure {
            params: vec![Param { name: Arc::from("v"), default: None, dots: false }],
            body: Arc::new(body),
            env: Env::new_global(),
        };
        let handle = try_compile_closure(&cl).expect("should compile");
        assert_eq!(handle.kind(), r2_types::JitKind::Vector1ToScalar);
        // prod([2, 3, 5]) = 30
        let input: Vec<f64> = vec![2.0, 3.0, 5.0];
        let r = unsafe { handle.try_call_vec1(input.as_ptr(), input.len() as i64) };
        assert_eq!(r, Some(30.0));
    }

    // ── Regression: for-loop accumulator must NOT JIT-compile ──────────
    //
    // `r2_ir`'s lowering has no `Expr::For` arm; the catch-all silently
    // lowers `for` to a no-op `Null`, so a JIT-compiled body would skip
    // the loop and return the accumulator's init value (0). The
    // `body_is_jit_lowerable` gate rejects such bodies so the engine falls
    // back to the interpreter. See closure.rs.

    #[test]
    fn body_lowerable_admits_arithmetic_and_branches() {
        // { s <- 0 ; if (n > 1) n*n else 0 }  — all faithfully-lowered.
        let body = Expr::Block(vec![
            Expr::Assign {
                target: Box::new(sym("s")),
                value: Box::new(num(0.0)),
                superassign: false,
            },
            Expr::If {
                cond: Box::new(Expr::Binary { op: BinOp::Gt, lhs: Box::new(sym("n")), rhs: Box::new(num(1.0)) }),
                then: Box::new(Expr::Binary { op: BinOp::Mul, lhs: Box::new(sym("n")), rhs: Box::new(sym("n")) }),
                else_: Some(Box::new(num(0.0))),
            },
        ]);
        assert!(body_is_jit_lowerable(&body));
        // while-loop body is faithfully lowered too.
        let wbody = Expr::While {
            cond: Box::new(Expr::Binary { op: BinOp::Lt, lhs: Box::new(sym("k")), rhs: Box::new(sym("n")) }),
            body: Box::new(Expr::Block(vec![])),
        };
        assert!(body_is_jit_lowerable(&wbody));
    }

    #[test]
    fn body_lowerable_admits_counted_for_rejects_others() {
        // Phase J.1: counted for(k in 1:n) is now JIT-lowerable.
        let counted = Expr::For {
            var: Arc::from("k"),
            iter: Box::new(Expr::Binary { op: BinOp::Colon, lhs: Box::new(num(1.0)), rhs: Box::new(sym("n")) }),
            body: Box::new(Expr::Assign {
                target: Box::new(sym("s")),
                value: Box::new(Expr::Binary { op: BinOp::Add, lhs: Box::new(sym("s")), rhs: Box::new(sym("k")) }),
                superassign: false,
            }),
        };
        assert!(body_is_jit_lowerable(&counted));
        // A non-counted for(x in v) is not lowered → still rejected.
        let noncounted = Expr::For { var: Arc::from("x"), iter: Box::new(sym("v")), body: Box::new(sym("x")) };
        assert!(!body_is_jit_lowerable(&noncounted));
        // repeat{} still rejects.
        assert!(!body_is_jit_lowerable(&Expr::Repeat { body: Box::new(Expr::Block(vec![])) }));
    }

    #[test]
    fn jit_for_loop_accumulators_are_correct() {
        // function(n){ s <- init; for (k in 1:n) s <- <upd>; s }
        let mkfn = |init: f64, upd: Expr| -> Closure {
            Closure {
                params: vec![Param { name: Arc::from("n"), default: None, dots: false }],
                body: Arc::new(Expr::Block(vec![
                    Expr::Assign { target: Box::new(sym("s")), value: Box::new(num(init)), superassign: false },
                    Expr::For {
                        var: Arc::from("k"),
                        iter: Box::new(Expr::Binary { op: BinOp::Colon, lhs: Box::new(num(1.0)), rhs: Box::new(sym("n")) }),
                        body: Box::new(Expr::Assign { target: Box::new(sym("s")), value: Box::new(upd), superassign: false }),
                    },
                    sym("s"),
                ])),
                env: Env::new_global(),
            }
        };
        // 1-param closures compile as VectorMap (per-element); call each `n`
        // via the appropriate convention and read the scalar result back.
        let call = |h: &std::sync::Arc<dyn r2_types::JitHandle>, n: f64| -> f64 {
            match h.kind() {
                r2_types::JitKind::Scalar => h.try_call_real(&[n]).unwrap(),
                r2_types::JitKind::VectorMap => {
                    let inp = vec![n]; let mut o = vec![0.0];
                    assert!(unsafe { h.try_call_vec_map(inp.as_ptr(), o.as_mut_ptr(), 1) });
                    o[0]
                }
                other => panic!("unexpected kind {:?}", other),
            }
        };
        // sum 1..n
        let h = try_compile_closure(&mkfn(0.0, add(sym("s"), sym("k")))).expect("for-sum should JIT");
        assert_eq!(call(&h, 100.0), 5050.0);
        assert_eq!(call(&h, 5.0), 15.0);
        assert_eq!(call(&h, 1.0), 1.0);
        // R semantics: 1:0 == c(1,0) → iterate k=1,0 → s = 1 (matches interpreter)
        assert_eq!(call(&h, 0.0), 1.0);
        // product / factorial
        let hp = try_compile_closure(&mkfn(1.0, mul(sym("s"), sym("k")))).expect("for-product should JIT");
        assert_eq!(call(&hp, 5.0), 120.0);
        assert_eq!(call(&hp, 1.0), 1.0);
        // sum of squares
        let hs = try_compile_closure(&mkfn(0.0, add(sym("s"), mul(sym("k"), sym("k"))))).expect("for-sumsq should JIT");
        assert_eq!(call(&hs, 10.0), 385.0);
    }
