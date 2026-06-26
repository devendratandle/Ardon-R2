//! Hypothesis tests — Phase R.10.
//!
//! Hit-and-get hypothesis tests: caller invokes the function, the
//! formatted result prints to stdout, the return value is a small
//! `RVal::TypeInstance` carrying `statistic`, `p.value`, and so on.
//! Unlike fitted-model functions (`lm`, `glm`, `aov`), these tests do
//! NOT need a separate `summary()` step — the `t.test`/`chisq.test`
//! call is the one-shot interaction.
//!
//! Therefore they migrate as plain pure builtins
//! (`fn(&[EvalArg]) -> Result<RVal, R2Err>`); no split-handler pattern,
//! no EngineCtx, no engine state.
//!
//! Hosted: `t.test`, `chisq.test`, `cor.test`, `shapiro.test`,
//! `wilcox.test`, `fisher.test`. The numerical primitives they share
//! (`t_cdf`, `chi_sq_cdf`, `ln_gamma`, `gamma_approx`, `incomplete_beta`,
//! `fmt_pval`, `signif_stars`) live in this module too and are
//! re-exported for engine-internal callers (`lm` summary print uses
//! `fmt_pval`/`signif_stars`).
//!
//! **t.test status (v0.1.0):** R-style output (data:/CI/alt-hypothesis/
//! sample-estimates), Welch–Satterthwaite df for unequal-variance two-
//! sample, formula syntax `t.test(x ~ y)`, paired test with Pearson r,
//! `id =` named arg for within-subject auto-pairing. p-value uses a
//! trapezoidal-rule incomplete-beta integration (~1e-4 accuracy);
//! closure path to LAPACK-grade is Lentz CF (tracked in KNOWN_LIMITATIONS).
//!
//! **fisher.test status (v0.1.0):** exact hypergeometric (via `lchoose`
//! / `hypergeom_pmf`) replacing the earlier χ² approximation.

use r2_types::{Attrs, ErrKind, EvalArg, R2Err, RVal};
use std::sync::Arc;

#[inline]
fn first(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn nth(a: &[EvalArg], i: usize) -> RVal { a.get(i).map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

#[inline]
fn rnum(n: f64) -> RVal { RVal::Numeric(vec![Some(n)].into(), Attrs::default()) }

#[inline]
fn rnums(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) }

#[inline]
fn rstr(s: &str) -> RVal { RVal::Character(vec![Some(Arc::from(s))], Attrs::default()) }

#[inline]
fn runtime_err(msg: String) -> R2Err { R2Err { msg, kind: ErrKind::Runtime } }


#[inline]
fn fmt_n(n: f64) -> String { r2_types::fmt_num(n) }

// Split into the numeric primitives and the test builtins. Both reach the
// shared arg/format helpers above (first/nth/rnum/fmt_n/…) via `use
// super::*` — child modules see ancestor items. Re-exported flat so every
// `r2_stats::htest::*` path (t_cdf, f_sf, bi_t_test, …) is unchanged.
mod probability;
mod tests_impl;
pub use probability::*;
pub use tests_impl::*;

#[cfg(test)]
mod test_suite {
    use super::*;

    fn nums(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    fn chs(items: &[&str]) -> RVal {
        RVal::Character(items.iter().map(|s| Some(Arc::from(*s))).collect(), Attrs::default())
    }

    fn formula(lhs: RVal, rhs: RVal) -> RVal {
        RVal::List(vec![
            (Some(Arc::from("~lhs")), lhs),
            (Some(Arc::from("~rhs")), rhs),
            (Some(Arc::from("~class")), RVal::Character(vec![Some(Arc::from("formula"))], Attrs::default())),
        ])
    }

    #[test]
    fn t_test_formula_splits_by_two_level_group() {
        // t.test(c(1,2,3,10,20,30) ~ c("a","a","a","b","b","b"))
        // R: Welch t-test, equivalent to t.test(c(1,2,3), c(10,20,30))
        let values = nums(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let groups = chs(&["a", "a", "a", "b", "b", "b"]);
        let r = bi_t_test(&[evarg(formula(values, groups))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let g1 = inst.fields.get("group1").and_then(|v| match v {
                    RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
                    _ => None,
                }).unwrap();
                let g2 = inst.fields.get("group2").and_then(|v| match v {
                    RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
                    _ => None,
                }).unwrap();
                assert_eq!(g1, "a");
                assert_eq!(g2, "b");
                let est = inst.fields.get("estimate").unwrap();
                let means = est.as_reals().unwrap();
                assert!((means[0].unwrap() - 2.0).abs() < 1e-12);
                assert!((means[1].unwrap() - 20.0).abs() < 1e-12);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_test_formula_with_id_pairs_within_subject() {
        // 4 subjects, each measured pre and post. Strong increase post.
        // value: 10, 12, 8, 11, 11, 14, 9, 13   (alternating subj/time)
        // time:  pre, post, pre, post, pre, post, pre, post
        // subj:  s1, s1,  s2, s2,  s3, s3,  s4, s4
        let values = nums(&[10.0, 12.0, 8.0, 11.0, 11.0, 14.0, 9.0, 13.0]);
        let times  = chs(&["pre","post","pre","post","pre","post","pre","post"]);
        let subj   = chs(&["s1","s1","s2","s2","s3","s3","s4","s4"]);
        let r = bi_t_test(&[
            evarg(formula(values, times)),
            EvalArg { name: Some(Arc::from("id")), value: subj },
            EvalArg { name: Some(Arc::from("paired")),
                      value: RVal::Logical(vec![Some(true)].into(), Attrs::default()) },
        ]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let method = inst.fields.get("method").and_then(|v| match v {
                    RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
                    _ => None,
                }).unwrap();
                assert_eq!(method, "Paired t-test");
                // df = n_subjects - 1 = 3
                let df = inst.fields.get("parameter")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((df - 3.0).abs() < 1e-12, "expected df=3, got {}", df);
                // Mean of (pre - post) differences. With pairs
                // (10,12) (8,11) (11,14) (9,13), diffs = -2, -3, -3, -4 → mean = -3.
                let est = inst.fields.get("estimate")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((est - (-3.0)).abs() < 1e-12, "expected mean diff=-3, got {}", est);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_test_paired_reports_pearson_r_and_uses_n_minus_1_df() {
        // Strongly-correlated paired data: y = 2x + small noise.
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = nums(&[2.1, 4.05, 5.95, 8.1, 9.9]);  // ~2x
        let r = bi_t_test(&[
            evarg(x), evarg(y),
            EvalArg { name: Some(Arc::from("paired")), value: RVal::Logical(vec![Some(true)].into(), Attrs::default()) },
        ]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                // Paired df = n - 1 = 4.
                let df = inst.fields.get("parameter")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((df - 4.0).abs() < 1e-12, "paired df should be n-1 = 4, got {}", df);
                // Pearson r should be near 1.0 (y ≈ 2x).
                let cor = inst.fields.get("cor")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!(cor > 0.999, "expected r ≈ 1, got {}", cor);
                // Method label flips to "Paired t-test".
                let method = inst.fields.get("method").and_then(|v| match v {
                    RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
                    _ => None,
                }).unwrap();
                assert_eq!(method, "Paired t-test");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_test_two_sample_uses_welch_df_not_pooled() {
        // x ~ tight cluster, y ~ wider — unequal variances.
        // R: t.test(c(1,2,3,4,5), c(10,20,30))$parameter ≈ 2.0602 (Welch df)
        // Pooled df would be n1+n2-2 = 6 — much larger and wrong.
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = nums(&[10.0, 20.0, 30.0]);
        let r = bi_t_test(&[evarg(x), evarg(y)]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let df = inst.fields.get("parameter")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((df - 2.0602).abs() < 1e-3,
                    "Welch df should match R's 2.0602, got {} (pooled would be 6)", df);
                // Sanity: must NOT be the pooled value.
                assert!(df < 5.0, "df {} looks like pooled n1+n2-2", df);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_test_one_sample_zero_mean_against_zero() {
        // x ~ centered, mu=0 → t ≈ 0, p ≈ 1.
        let r = bi_t_test(&[evarg(nums(&[-1.0, 0.0, 1.0]))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!(p > 0.5, "expected p > 0.5, got {}", p);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn cor_test_perfect_correlation_p_zero() {
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = nums(&[2.0, 4.0, 6.0, 8.0, 10.0]);
        let r = bi_cor_test(&[evarg(x), evarg(y)]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let est = inst.fields.get("estimate").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((est - 1.0).abs() < 1e-12, "estimate should be 1, got {}", est);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn chisq_test_uniform_observations_high_p() {
        // Equal observations → chi² = 0 → p ≈ 1.
        let r = bi_chisq_test(&[evarg(nums(&[10.0, 10.0, 10.0, 10.0]))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!(p > 0.99, "expected p ≈ 1, got {}", p);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_cdf_uses_beta_identity_no_normal_shortcut() {
        // The pre-R.10 implementation switched to a normal approximation
        // for df > 30, which gave ~5e-3 absolute error at moderate df.
        // The new code routes through `incomplete_beta` for all df.
        // R: pt(1.96, df=30) = 0.97032884. Lentz CF reaches ~1e-9.
        // (The pre-Lentz test used an imprecise 0.9703358 reference that
        // only survived because the trapezoidal rule ran at 2e-3.)
        let p = t_cdf(1.96, 30.0);
        assert!((p - 0.97032884).abs() < 1e-6, "got {}", p);
        // Reflection symmetry across t = 0 holds exactly.
        let p_neg = t_cdf(-1.96, 10.0);
        let p_pos = t_cdf(1.96, 10.0);
        assert!(((p_neg + p_pos) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn incomplete_beta_exact_via_lentz() {
        // pbeta(0.5, 2, 3) = 0.6875 exactly; Lentz CF is accurate to ~1e-12.
        let v = incomplete_beta(2.0, 3.0, 0.5);
        assert!((v - 0.6875).abs() < 1e-10, "got {}", v);
        // Symmetry I_x(a,b) = 1 - I_{1-x}(b,a), incl. the b < 1 path
        // (the case t_cdf exercises with b = 0.5).
        let lhs = incomplete_beta(0.5, 3.0, 0.3);
        let rhs = 1.0 - incomplete_beta(3.0, 0.5, 0.7);
        assert!((lhs - rhs).abs() < 1e-12, "symmetry: {} vs {}", lhs, rhs);
    }

    #[test]
    fn f_sf_matches_closed_form_and_handles_infinity() {
        // For df1 = 2 the F survival has the closed form
        // S(f) = (1 + f/3)^(-3) at df2 = 6.
        let f = 12.39623_f64;
        let expected = (1.0 + f / 3.0).powi(-3);
        let got = f_sf(f, 2.0, 6.0);
        assert!((got - expected).abs() < 1e-9, "f_sf {} vs closed form {}", got, expected);
        // f = +∞ (zero residual) → p = 0, not the old approximation's p = 1.
        assert_eq!(f_sf(f64::INFINITY, 2.0, 6.0), 0.0);
        // f = 0 → whole mass above → p = 1.
        assert_eq!(f_sf(0.0, 3.0, 10.0), 1.0);
    }

    #[test]
    fn fisher_test_classic_2x2_matches_r() {
        // R: fisher.test(matrix(c(3,1,1,3), nrow=2))$p.value ≈ 0.4857
        // Exact two-sided hypergeometric.
        use r2_types::Matrix;
        let m = Matrix::new(vec![3.0, 1.0, 1.0, 3.0], 2, 2);
        let r = bi_fisher_test(&[evarg(RVal::Matrix(m))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                // R reports 0.4857; our sum-of-equally-or-less-likely tail
                // gives the same family of outcomes. Allow ±0.02.
                assert!((p - 0.4857).abs() < 0.02, "got p = {}", p);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn fisher_test_independent_zero_off_diag() {
        // 2x2 with one zero cell: c(8,0,0,8). R: p = 0.0001554...
        use r2_types::Matrix;
        let m = Matrix::new(vec![8.0, 0.0, 0.0, 8.0], 2, 2);
        let r = bi_fisher_test(&[evarg(RVal::Matrix(m))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!(p < 0.001, "expected very small p, got {}", p);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn shapiro_test_returns_w_and_p() {
        let r = bi_shapiro_test(&[evarg(nums(&[1.0, 2.0, 3.0, 4.0, 5.0]))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                assert!(inst.fields.contains_key("statistic"));
                assert!(inst.fields.contains_key("p.value"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn fmt_pval_scales() {
        assert_eq!(fmt_pval(1e-20), "<2e-16");
        assert!(fmt_pval(0.0001).contains("e"));
        assert_eq!(fmt_pval(0.05), "0.05");
        assert_eq!(fmt_pval(1.0), "1");
    }

    #[test]
    fn signif_stars_thresholds() {
        assert_eq!(signif_stars(0.0001), "***");
        assert_eq!(signif_stars(0.005), "**");
        assert_eq!(signif_stars(0.02), "*");
        assert_eq!(signif_stars(0.08), ".");
        assert_eq!(signif_stars(0.5), " ");
    }
}
