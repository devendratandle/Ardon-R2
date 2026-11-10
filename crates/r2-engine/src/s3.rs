//! S3 dispatch, as R does it: `UseMethod("g")` in a generic calls `g.<class>`
//! for the first class of the object that has one, else `g.default`;
//! `NextMethod()` in a method calls the next class's method, then the default,
//! then the builtin. A builtin called on a classed object (`print`, `summary`,
//! `format`, `length`, ...) and an operator on one (`+.money`, `Ops.money`)
//! dispatch the same way when a user has defined the method. Auto-print uses
//! `print.<class>`.

use std::sync::Arc;

use r2_types::*;

use crate::{err, Engine};

/// One dispatch in progress: what `NextMethod` and `.Generic` read.
pub(crate) struct S3Call {
    pub(crate) generic: Arc<str>,
    /// The classes after the one whose method is running.
    pub(crate) rest: Vec<Arc<str>>,
    pub(crate) args: Vec<EvalArg>,
}

/// The classes a value dispatches on: its class attribute, else R's implicit
/// class (`c("double", "numeric")` for a number, `"function"`, ...).
pub(crate) fn dispatch_classes(v: &RVal) -> Vec<Arc<str>> {
    let explicit = explicit_classes(v);
    if !explicit.is_empty() { return explicit; }
    let names: &[&str] = match v {
        RVal::Numeric(..) => &["double", "numeric"],
        RVal::Integer(..) => &["integer", "numeric"],
        RVal::Character(..) => &["character"],
        RVal::Logical(..) => &["logical"],
        RVal::List(..) => &["list"],
        RVal::Matrix(..) => &["matrix", "array", "double", "numeric"],
        RVal::Closure(..) | RVal::BuiltinFn(..) => &["function"],
        RVal::Null => &["NULL"],
        _ => &[],
    };
    names.iter().map(|s| Arc::from(*s)).collect()
}

/// The classes a value was given, which are the only ones a builtin or an
/// operator dispatches on.
fn explicit_classes(v: &RVal) -> Vec<Arc<str>> {
    match v {
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Character(_, at) | RVal::Logical(_, at) | RVal::List(_, at) => at.classes(),
        RVal::TypeInstance(i) => vec![i.type_name.clone()],
        RVal::DataFrame(_) => vec![Arc::from("data.frame")],
        RVal::Factor(_) => vec![Arc::from("factor")],
        _ => Vec::new(),
    }
}

fn no_method(generic: &str, classes: &[Arc<str>]) -> R2Err {
    let shown = match classes {
        [one] => one.to_string(),
        many => format!("c({})", many.iter().map(|c| format!("'{}'", c)).collect::<Vec<_>>().join(", ")),
    };
    R2Err { msg: format!("no applicable method for '{}' applied to an object of class \"{}\"", generic, shown), kind: ErrKind::Runtime }
}

impl Engine {
    /// A function named `name` (`print.money`) where the running code sees it.
    fn s3_method(&self, name: &str) -> Option<RVal> {
        if !self.s3_names.contains(name) { return None; }
        let found = self.frames.last().and_then(|f| f.lookup(name)).or_else(|| self.global_env.lookup(name));
        found.filter(|f| matches!(f, RVal::Closure(_)))
    }

    /// Call `method` as the method for `generic`, with `rest` left for NextMethod.
    fn run_method(&mut self, method: &RVal, generic: &str, rest: Vec<Arc<str>>, args: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
        self.s3_stack.push(S3Call { generic: Arc::from(generic), rest, args: args.to_vec() });
        let r = self.call_fn(method, args, env);
        self.s3_stack.pop();
        r
    }

    /// The first method among `generic.<class>` for `classes`, with the
    /// classes after it.
    fn first_method(&self, generic: &str, classes: &[Arc<str>]) -> Option<(RVal, Vec<Arc<str>>)> {
        classes.iter().enumerate().find_map(|(i, c)| {
            self.s3_method(&format!("{}.{}", generic, c)).map(|m| (m, classes[i + 1..].to_vec()))
        })
    }

    /// `UseMethod(generic, object)`: run the method and return its value from
    /// the generic, as R does (the rest of the generic's body never runs).
    pub(crate) fn use_method(&mut self, args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
        let generic = match args.first() {
            Some(a) => match self.eval_in(&a.value, env)? {
                RVal::Character(s, _) => s.first().cloned().flatten(),
                _ => None,
            },
            None => None,
        };
        let Some(generic) = generic else { return err!(Runtime, "'generic' argument must be a character string") };
        let Some(frame) = self.nse_stack.last() else { return err!(Runtime, "UseMethod called from outside a function") };
        let call_args = frame.args.clone();
        let object = match args.get(1) {
            Some(a) => self.eval_in(&a.value, env)?,
            None => call_args.first().map(|a| a.value.clone()).unwrap_or(RVal::Null),
        };
        let classes = dispatch_classes(&object);
        let (method, rest) = match self.first_method(&generic, &classes) {
            Some(found) => found,
            None => match self.s3_method(&format!("{}.default", generic)) {
                Some(m) => (m, Vec::new()),
                None if self.registry.resolve(&generic).is_some() => (RVal::BuiltinFn(generic.clone()), Vec::new()),
                None => return Err(no_method(&generic, &classes)),
            },
        };
        let v = self.run_method(&method, &generic, rest, &call_args, env)?;
        Err(R2Err { msg: String::new(), kind: ErrKind::CtrlReturn(Box::new(v)) })
    }

    /// `NextMethod()`: the method of the next class, the default, or the
    /// builtin, with the arguments the current method was called with.
    pub(crate) fn next_method(&mut self, env: &EnvRef) -> Result<RVal, R2Err> {
        let Some(call) = self.s3_stack.last() else { return err!(Runtime, "NextMethod called from outside a method dispatch") };
        let (generic, rest, args) = (call.generic.clone(), call.rest.clone(), call.args.clone());
        if let Some((method, after)) = self.first_method(&generic, &rest) {
            return self.run_method(&method, &generic, after, &args, env);
        }
        if let Some(m) = self.s3_method(&format!("{}.default", generic)) {
            return self.run_method(&m, &generic, Vec::new(), &args, env);
        }
        match self.registry.resolve(&generic) {
            // the builtin itself, without dispatching again
            Some((f, _)) => {
                let r = f(self, &args, env);
                self.after_builtin(&generic);
                r
            }
            None => err!(Runtime, "no more methods for '{}'", generic),
        }
    }

    /// A user's method for builtin `name` on the classed first argument.
    pub(crate) fn builtin_method(&mut self, name: &str, args: &[EvalArg], env: &EnvRef) -> Option<Result<RVal, R2Err>> {
        if self.s3_names.is_empty() { return None; }
        let classes = explicit_classes(&args.first()?.value);
        if classes.is_empty() { return None; }
        let (method, rest) = self.first_method(name, &classes)?;
        Some(self.run_method(&method, name, rest, args, env))
    }

    /// A user's method for an operator: `+.money`, else `Ops.money`, on the
    /// first operand's classes, then the second's.
    pub(crate) fn operator_method(&mut self, op: &str, lhs: &RVal, rhs: &RVal) -> Option<Result<RVal, R2Err>> {
        if self.s3_names.is_empty() { return None; }
        let found = [lhs, rhs].iter().find_map(|side| {
            let classes = explicit_classes(side);
            self.first_method(op, &classes)
                .or_else(|| self.first_method("Ops", &classes))
        })?;
        let args = [EvalArg { name: None, value: lhs.clone() }, EvalArg { name: None, value: rhs.clone() }];
        let env = self.global_env.clone();
        Some(self.run_method(&found.0, op, found.1, &args, &env))
    }

    /// Auto-print through `print.<class>` when the user defined one: true when
    /// it ran, so the caller prints nothing more.
    pub fn print_by_method(&mut self, val: &RVal) -> Result<bool, R2Err> {
        let env = self.global_env.clone();
        match self.builtin_method("print", &[EvalArg { name: None, value: val.clone() }], &env) {
            Some(r) => r.map(|_| true),
            None => Ok(false),
        }
    }

    /// `.Generic` inside a method: the generic or operator being dispatched.
    pub(crate) fn dispatch_generic(&self) -> Option<RVal> {
        self.s3_stack.last().map(|c| RVal::Character(vec![Some(c.generic.clone())], Attrs::default()))
    }

    /// Remember a function whose name could make it a method (`print.money`).
    pub(crate) fn note_s3_name(&mut self, name: &Arc<str>, val: &RVal) {
        if matches!(val, RVal::Closure(_)) && name.contains('.') { self.s3_names.insert(name.clone()); }
    }
}
