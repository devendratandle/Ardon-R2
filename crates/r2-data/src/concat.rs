//! `c()` — concatenation. Phase R.2 step 2.
//!
//! Mode selection (matches r2-engine semantics bit-for-bit):
//!   - If any argument is `Character`, all args coerce to character as
//!     `as.character` writes them (NA stays NA; a factor gives its codes).
//!   - Otherwise the narrowest of logical < integer < double that holds
//!     every argument (`NA` is logical, so `c(TRUE, NA)` stays logical).
//!   - Names come from the arguments' names and their elements' names.
//!
//! Pure — no Engine reference; uses RVal methods (Phase R.1 step 2).

use r2_types::*;
use std::sync::Arc;

pub fn bi_c(args: &[EvalArg]) -> Result<RVal, R2Err> {
    // `recursive =` / `use.names =` are c()'s own options, not elements.
    let owned: Vec<EvalArg>;
    let args = if args.iter().any(|a| matches!(a.name.as_deref(), Some("recursive" | "use.names"))) {
        owned = args.iter().filter(|a| !matches!(a.name.as_deref(), Some("recursive" | "use.names"))).cloned().collect();
        &owned[..]
    } else { args };
    let has_str = args.iter().any(|a| matches!(&a.value, RVal::Character(..)));
    if has_str {
        // Character coercion path
        let mut s: Vec<Character> = Vec::new();
        for a in args {
            match &a.value {
                // A factor contributes its integer codes, as R's c() does.
                RVal::Factor(f) => s.extend(f.codes.iter().map(|c| c.map(|c| Arc::from((c + 1).to_string().as_str())))),
                RVal::Null => {}
                // Everything atomic as as.character writes it; NA stays NA.
                other => match factor_labels(other) {
                    Some((labels, _)) => s.extend(labels),
                    None => return Err(R2Err { msg: format!("c(): cannot combine {} with character", other.type_name()), kind: ErrKind::Type }),
                },
            }
        }
        return Ok(RVal::Character(s, names_of(args)));
    }
    // R's c() type hierarchy: logical < integer < double. Preserve the
    // narrowest type so `c(TRUE,FALSE)` stays logical (which()/all() need
    // that) and `c(1L,2L)` stays integer — only widen to double when a
    // double is actually present.
    let only_lgl = args.iter().all(|a| matches!(&a.value, RVal::Logical(..) | RVal::Null))
        && args.iter().any(|a| matches!(&a.value, RVal::Logical(..)));
    if only_lgl {
        let mut lg: Vec<Logical> = Vec::new();
        for a in args { if let RVal::Logical(v, _) = &a.value { lg.extend(v.iter().cloned()); } }
        return Ok(RVal::Logical(lg.into(), names_of(args)));
    }
    let only_int = args.iter().all(|a| matches!(&a.value, RVal::Integer(..) | RVal::Logical(..) | RVal::Null))
        && args.iter().any(|a| matches!(&a.value, RVal::Integer(..)));
    if only_int {
        let mut ints: Vec<Integer> = Vec::new();
        for a in args {
            match &a.value {
                RVal::Integer(v, _) => ints.extend(v.iter().cloned()),
                RVal::Logical(v, _) => ints.extend(v.iter().map(|x| x.map(|b| if b { 1 } else { 0 }))),
                _ => {}
            }
        }
        return Ok(RVal::Integer(ints.into(), names_of(args)));
    }

    // Numeric path — RVal::as_reals handles Numeric/Integer/Logical/Matrix
    let mut nums: Vec<Real> = Vec::new();
    for a in args {
        nums.extend(a.value.as_reals()?);
    }
    // Preserve a shared Date/POSIXct class so `c(d1, d2)` stays a Date
    // (R dispatches to c.Date/c.POSIXct). Only when EVERY argument carries
    // the same date class — mixed/absent classes drop it, as R does.
    let shared_class: Option<Arc<str>> = {
        let first = match &args.first().map(|a| &a.value) {
            Some(RVal::Numeric(_, at)) => at.class.as_deref(),
            _ => None,
        };
        match first {
            Some(c) if matches!(c, "Date" | "POSIXct" | "POSIXt")
                && args.iter().all(|a| matches!(&a.value, RVal::Numeric(_, at) if at.class.as_deref() == Some(c)))
                => Some(Arc::from(c)),
            _ => None,
        }
    };
    Ok(RVal::Numeric(nums.into(), Attrs { class: shared_class, ..names_of(args) }))
}

/// The names of `c(...)`'s result: an argument's name for a length-1
/// element (`c(a = 1)`), `name1`, `name2`, ... for a longer one, and the
/// element's own names otherwise ("" where there are none). No names at
/// all when nothing is named.
fn names_of(args: &[EvalArg]) -> Attrs {
    let own = |v: &RVal| match v {
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) | RVal::Character(_, at) => at.names.clone(),
        _ => None,
    };
    if args.iter().all(|a| a.name.is_none() && own(&a.value).is_none()) { return Attrs::default(); }
    let mut names: Vec<Arc<str>> = Vec::new();
    for a in args {
        let n = rval_length(&a.value);
        let inner = own(&a.value);
        for i in 0..n {
            let e = inner.as_ref().and_then(|ns| ns.get(i).cloned()).filter(|s| !s.is_empty());
            names.push(match (&a.name, e) {
                (Some(p), Some(e)) => Arc::from(format!("{}.{}", p, e).as_str()),
                (Some(p), None) if n == 1 => p.clone(),
                (Some(p), None) => Arc::from(format!("{}{}", p, i + 1).as_str()),
                (None, Some(e)) => e,
                (None, None) => Arc::from(""),
            });
        }
    }
    Attrs { names: Some(names), ..Default::default() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn c_concatenates_numeric() {
        let a = vec![
            evarg(RVal::Numeric(vec![Some(1.0), Some(2.0)].into(), Attrs::default())),
            evarg(RVal::Numeric(vec![Some(3.0), Some(4.0), Some(5.0)].into(), Attrs::default())),
        ];
        let r = bi_c(&a).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                assert_eq!(v.len(), 5);
                assert_eq!(v.as_vec(), &vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]);
            }
            _ => panic!("expected Numeric"),
        }
    }

    #[test]
    fn c_promotes_to_character_when_any_string() {
        let a = vec![
            evarg(RVal::Numeric(vec![Some(1.0), Some(2.0)].into(), Attrs::default())),
            evarg(RVal::Character(vec![Some(Arc::from("hi"))], Attrs::default())),
        ];
        let r = bi_c(&a).unwrap();
        match r {
            RVal::Character(v, _) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[2].as_ref().map(|s| s.as_ref()), Some("hi"));
            }
            _ => panic!("expected Character"),
        }
    }

    #[test]
    fn c_coerces_integer_to_numeric() {
        let a = vec![
            evarg(RVal::Integer(vec![Some(10), Some(20)].into(), Attrs::default())),
            evarg(RVal::Numeric(vec![Some(0.5)].into(), Attrs::default())),
        ];
        let r = bi_c(&a).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                assert_eq!(v.as_vec(), &vec![Some(10.0), Some(20.0), Some(0.5)]);
            }
            _ => panic!("expected Numeric"),
        }
    }

    #[test]
    fn c_empty_returns_empty_numeric() {
        let r = bi_c(&[]).unwrap();
        match r {
            RVal::Numeric(v, _) => assert!(v.is_empty()),
            _ => panic!("expected empty Numeric"),
        }
    }
}
