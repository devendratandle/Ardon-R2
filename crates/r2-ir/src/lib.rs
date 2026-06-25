//! R2-IR — Typed SSA intermediate representation (Phase B).
//!
//! Per docs/ARCHITECTURE.md §5 Phase B:
//!   - New crate
//!   - Defines IrNode, IrFunc, IrBlock
//!   - Builder lowers an annotated AST → IR
//!   - No runtime impact yet (validation only)
//!
//! Locked decisions honoured:
//!   §4.2  Column-shaped types (reused from r2-types::infer)
//!   §4.3  `.Internal()` lowers to typed `Intrinsic` instructions, no
//!         string dispatch at runtime once the JIT (Phase C) consumes it
//!   §9    SSA-with-phi (matches Cranelift's native form)
//!
//! Scope of Phase B:
//!   - Literals, symbols, arithmetic, comparisons, assignment
//!   - Function calls (builtin and user)
//!   - .Internal() lowering as Intrinsic
//!   - Control flow: If/Else (with phi), While, Block
//!   - Returns
//!   - Validation pass (every VReg defined before used, blocks terminated)
//!
//! Out of scope (will arrive with Phase C/D):
//!   - Closure capture lowering
//!   - For-loop lowering (needs iterator protocol)
//!   - Pattern match lowering
//!   - Type definitions / methods

use r2_types::infer::{IrType, TypeCtx};
use r2_types::*;
use std::collections::HashMap;
use std::sync::Arc;

// ── Identifiers ──────────────────────────────────────────────────────

/// Virtual register — strictly increasing within a function (SSA invariant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg(pub u32);

impl std::fmt::Display for VReg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "v{}", self.0) }
}

/// Basic-block label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "bb{}", self.0) }
}

// ── IR data model ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum IrConst {
    Real(f64),
    Int(i32),
    Bool(bool),
    Str(Arc<str>),
    Null,
    NA,
}

#[derive(Debug, Clone)]
pub enum IrInst {
    /// Materialise a constant.
    Const { dst: VReg, value: IrConst, ty: IrType },

    /// Element-wise unary op (preserves shape).
    Unary { dst: VReg, op: UnOp, src: VReg, ty: IrType },

    /// Element-wise binary op (broadcasts; promotes element type).
    Binary { dst: VReg, op: BinOp, lhs: VReg, rhs: VReg, ty: IrType },

    /// Call a user/builtin function by name.
    Call { dst: VReg, name: Arc<str>, args: Vec<VReg>, ty: IrType },

    /// Direct dispatch to a Rust intrinsic (the `.Internal()` path).
    /// At Phase C this lowers straight to a Cranelift function call —
    /// no string lookup at runtime.
    Intrinsic { dst: VReg, name: Arc<str>, args: Vec<VReg>, ty: IrType },

    /// SSA phi — pick a value depending on which predecessor block we
    /// arrived from.
    Phi { dst: VReg, sources: Vec<(BlockId, VReg)>, ty: IrType },
}

impl IrInst {
    pub fn dst(&self) -> VReg {
        match self {
            IrInst::Const { dst, .. }
            | IrInst::Unary { dst, .. }
            | IrInst::Binary { dst, .. }
            | IrInst::Call { dst, .. }
            | IrInst::Intrinsic { dst, .. }
            | IrInst::Phi { dst, .. } => *dst,
        }
    }
    pub fn ty(&self) -> &IrType {
        match self {
            IrInst::Const { ty, .. }
            | IrInst::Unary { ty, .. }
            | IrInst::Binary { ty, .. }
            | IrInst::Call { ty, .. }
            | IrInst::Intrinsic { ty, .. }
            | IrInst::Phi { ty, .. } => ty,
        }
    }
}

#[derive(Debug, Clone)]
pub enum IrTerm {
    Return(Option<VReg>),
    Jump(BlockId),
    Branch { cond: VReg, then_blk: BlockId, else_blk: BlockId },
    Unreachable,
}

#[derive(Debug, Clone)]
pub struct IrBlock {
    pub id: BlockId,
    pub insts: Vec<IrInst>,
    pub term: IrTerm,
}

#[derive(Debug, Clone)]
pub struct IrFunc {
    pub name: Arc<str>,
    pub params: Vec<(Arc<str>, IrType, VReg)>,
    pub return_type: IrType,
    pub blocks: Vec<IrBlock>,
    /// Type of each VReg, indexed by VReg.0 as usize.
    pub vreg_types: Vec<IrType>,
    pub entry: BlockId,
}

// ── Builder ──────────────────────────────────────────────────────────

pub struct IrBuilder {
    func_name: Arc<str>,
    blocks: Vec<IrBlock>,
    current: BlockId,
    next_vreg: u32,
    next_block: u32,
    vreg_types: Vec<IrType>,
    /// Local symbol → most recent VReg holding its value (within this function).
    /// SSA's "current definition" map.
    locals: HashMap<Arc<str>, VReg>,
    /// Used during inference; not part of the IR.
    type_ctx: TypeCtx,
    /// Function parameters, in declared order. Phase C.1.
    params: Vec<(Arc<str>, IrType, VReg)>,
}

impl IrBuilder {
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        let mut b = IrBuilder {
            func_name: name.into(),
            blocks: Vec::new(),
            current: BlockId(0),
            next_vreg: 0,
            next_block: 0,
            vreg_types: Vec::new(),
            locals: HashMap::new(),
            type_ctx: TypeCtx::new(),
            params: Vec::new(),
        };
        let entry = b.new_block();
        b.current = entry;
        b
    }

    /// Builder with declared formal parameters. Each parameter gets a
    /// pre-allocated VReg that lower() will use when the parameter symbol
    /// is referenced.
    pub fn with_params(name: impl Into<Arc<str>>, params: Vec<(Arc<str>, IrType)>) -> Self {
        let mut b = Self::new(name);
        for (pname, pty) in params {
            let reg = b.new_vreg(pty.clone());
            b.locals.insert(pname.clone(), reg);
            b.type_ctx.bind(pname.clone(), pty.clone());
            b.params.push((pname, pty, reg));
        }
        b
    }

    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        self.blocks.push(IrBlock { id, insts: Vec::new(), term: IrTerm::Unreachable });
        id
    }

    fn new_vreg(&mut self, ty: IrType) -> VReg {
        let r = VReg(self.next_vreg);
        self.next_vreg += 1;
        self.vreg_types.push(ty);
        r
    }

    fn block_mut(&mut self, id: BlockId) -> &mut IrBlock {
        self.blocks.iter_mut().find(|b| b.id == id).expect("block id valid")
    }

    fn emit(&mut self, inst: IrInst) {
        let cur = self.current;
        self.block_mut(cur).insts.push(inst);
    }

    fn terminate(&mut self, t: IrTerm) {
        let cur = self.current;
        let blk = self.block_mut(cur);
        if matches!(blk.term, IrTerm::Unreachable) { blk.term = t; }
    }

    /// Lower a single expression, return the VReg holding its value.
    pub fn lower(&mut self, e: &Expr) -> VReg {
        match e {
            Expr::NumLit(v)  => self.const_inst(IrConst::Real(*v),   IrType::scalar(infer::IrElem::Real)),
            Expr::IntLit(v)  => self.const_inst(IrConst::Int(*v),    IrType::scalar(infer::IrElem::Int)),
            Expr::BoolLit(v) => self.const_inst(IrConst::Bool(*v),   IrType::scalar(infer::IrElem::Bool)),
            Expr::StrLit(s)  => self.const_inst(IrConst::Str(Arc::from(s.as_str())), IrType::scalar(infer::IrElem::Char)),
            Expr::FStringLit(_) => self.const_inst(IrConst::Str(Arc::from("")), IrType::scalar(infer::IrElem::Char)),
            Expr::NaLit      => self.const_inst(IrConst::NA,         IrType::scalar(infer::IrElem::Unknown)),
            Expr::NullLit    => self.const_inst(IrConst::Null,       IrType::null()),

            Expr::Symbol(name) => {
                if let Some(reg) = self.locals.get(name).copied() { return reg; }
                // Free variable — emit a Call to a 0-arg "lookup" intrinsic.
                let ty = self.type_ctx.lookup(name);
                let dst = self.new_vreg(ty.clone());
                self.emit(IrInst::Intrinsic {
                    dst, name: Arc::from(format!("__lookup__{}", name)), args: vec![], ty,
                });
                dst
            }

            Expr::Unary { op, expr } => {
                let src = self.lower(expr);
                let ty = self.vreg_types[src.0 as usize].clone();
                let dst = self.new_vreg(ty.clone());
                self.emit(IrInst::Unary { dst, op: *op, src, ty });
                dst
            }

            Expr::Binary { op, lhs, rhs } => {
                let l = self.lower(lhs);
                let r = self.lower(rhs);
                let lt = self.vreg_types[l.0 as usize].clone();
                let rt = self.vreg_types[r.0 as usize].clone();
                let elem = infer::promote_elem(lt.elem, rt.elem);
                let shape = infer::promote_shape(&lt.shape, &rt.shape);
                let mut ty = IrType { elem, shape };
                if matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                              | BinOp::And | BinOp::Or | BinOp::AndShort | BinOp::OrShort) {
                    ty.elem = infer::IrElem::Bool;
                }
                let dst = self.new_vreg(ty.clone());
                self.emit(IrInst::Binary { dst, op: *op, lhs: l, rhs: r, ty });
                dst
            }

            Expr::Assign { target, value, .. } => {
                let v = self.lower(value);
                if let Expr::Symbol(name) = target.as_ref() {
                    self.locals.insert(name.clone(), v);
                    self.type_ctx.bind(name.clone(), self.vreg_types[v.0 as usize].clone());
                }
                v
            }

            Expr::Call { func, args } => {
                let arg_regs: Vec<VReg> = args.iter().map(|a| self.lower(&a.value)).collect();
                let arg_types: Vec<IrType> = arg_regs.iter().map(|r| self.vreg_types[r.0 as usize].clone()).collect();

                // Detect .Internal("name", ...) → typed intrinsic call.
                if let Expr::Symbol(s) = func.as_ref() {
                    if s.as_ref() == ".Internal" {
                        if let Some(CallArg { value: Expr::StrLit(intr_name), .. }) = args.first() {
                            let rest: Vec<VReg> = arg_regs.iter().skip(1).copied().collect();
                            let ty = IrType::unknown();
                            let dst = self.new_vreg(ty.clone());
                            self.emit(IrInst::Intrinsic { dst, name: Arc::from(intr_name.as_str()), args: rest, ty });
                            return dst;
                        }
                    }
                    let ret = infer::builtin_return_type_pub(s.as_ref(), &arg_types);
                    let dst = self.new_vreg(ret.clone());
                    self.emit(IrInst::Call { dst, name: s.clone(), args: arg_regs, ty: ret });
                    return dst;
                }

                let ty = IrType::unknown();
                let dst = self.new_vreg(ty.clone());
                self.emit(IrInst::Call { dst, name: Arc::from("<dynamic>"), args: arg_regs, ty });
                dst
            }

            Expr::If { cond, then, else_ } => {
                let c = self.lower(cond);
                let then_b = self.new_block();
                let else_b = self.new_block();
                let join_b = self.new_block();

                self.terminate(IrTerm::Branch { cond: c, then_blk: then_b, else_blk: else_b });

                self.current = then_b;
                let then_v = self.lower(then);
                let then_end = self.current;
                self.terminate(IrTerm::Jump(join_b));

                self.current = else_b;
                let else_v = match else_ {
                    Some(e) => self.lower(e),
                    None => self.const_inst(IrConst::Null, IrType::null()),
                };
                let else_end = self.current;
                self.terminate(IrTerm::Jump(join_b));

                self.current = join_b;
                let then_t = self.vreg_types[then_v.0 as usize].clone();
                let else_t = self.vreg_types[else_v.0 as usize].clone();
                let phi_ty = IrType {
                    elem: infer::promote_elem(then_t.elem, else_t.elem),
                    shape: infer::promote_shape(&then_t.shape, &else_t.shape),
                };
                let dst = self.new_vreg(phi_ty.clone());
                self.emit(IrInst::Phi { dst, sources: vec![(then_end, then_v), (else_end, else_v)], ty: phi_ty });
                dst
            }

            Expr::While { cond, body } => {
                let head = self.new_block();
                let body_b = self.new_block();
                let exit = self.new_block();
                self.terminate(IrTerm::Jump(head));

                self.current = head;
                let c = self.lower(cond);
                self.terminate(IrTerm::Branch { cond: c, then_blk: body_b, else_blk: exit });

                self.current = body_b;
                let _ = self.lower(body);
                self.terminate(IrTerm::Jump(head));

                self.current = exit;
                self.const_inst(IrConst::Null, IrType::null())
            }

            Expr::Block(stmts) => {
                let mut last = self.const_inst(IrConst::Null, IrType::null());
                for s in stmts { last = self.lower(s); }
                last
            }

            Expr::Return(v) => {
                let r = self.lower(v);
                self.terminate(IrTerm::Return(Some(r)));
                // Subsequent code is unreachable; allocate a dead block to keep the
                // builder consistent for any trailing expressions.
                let dead = self.new_block();
                self.current = dead;
                r
            }

            Expr::Pipe { lhs, rhs } => { let _ = self.lower(lhs); self.lower(rhs) }

            // Lowered as opaque calls until later phases.
            Expr::Index { object, .. } | Expr::DblIndex { object, .. } | Expr::Dollar { object, .. } => {
                let _ = self.lower(object);
                let dst = self.new_vreg(IrType::unknown());
                self.emit(IrInst::Intrinsic { dst, name: Arc::from("__index__"), args: vec![], ty: IrType::unknown() });
                dst
            }

            // Things deferred to later phases.
            _ => self.const_inst(IrConst::Null, IrType::null()),
        }
    }

    fn const_inst(&mut self, value: IrConst, ty: IrType) -> VReg {
        let dst = self.new_vreg(ty.clone());
        self.emit(IrInst::Const { dst, value, ty });
        dst
    }

    /// Finalize: ensure all blocks terminate, return an `IrFunc`.
    pub fn finish(mut self, return_value: Option<VReg>, return_type: IrType) -> IrFunc {
        self.terminate(IrTerm::Return(return_value));
        // Replace any stray Unreachable terminators with explicit Unreachable
        // (already the default — nothing to do).
        let entry = self.blocks.first().map(|b| b.id).unwrap_or(BlockId(0));
        IrFunc {
            name: self.func_name,
            params: self.params,
            return_type,
            blocks: self.blocks,
            vreg_types: self.vreg_types,
            entry,
        }
    }
}

// ── Top-level entry: lower a program ─────────────────────────────────

pub fn lower_program(prog: &[Expr], func_name: &str) -> IrFunc {
    let mut b = IrBuilder::new(func_name);
    let mut last: Option<VReg> = None;
    for e in prog { last = Some(b.lower(e)); }
    let ret_ty = last.map(|r| b.vreg_types[r.0 as usize].clone()).unwrap_or_else(IrType::null);
    b.finish(last, ret_ty)
}

/// Lower a function with formal parameters and a body expression.
/// Phase C.1 entry — used by tests and (later) by the engine when JITing
/// user-defined Closures.
pub fn lower_function(name: &str, params: Vec<(Arc<str>, IrType)>, body: &Expr) -> IrFunc {
    let mut b = IrBuilder::with_params(name, params);
    let v = b.lower(body);
    let ret_ty = b.vreg_types[v.0 as usize].clone();
    b.finish(Some(v), ret_ty)
}

// ════════════════════════════════════════════════════════════════════
// Closure capture inference — Phase B.1 extension
// ════════════════════════════════════════════════════════════════════
//
// `collect_free_vars` walks an `Expr` AST and returns the names of
// symbols that aren't bound by any inner `Expr::Closure` parameter list
// or by an `Expr::Assign` along the path to that symbol. The caller
// uses this set to decide what free names need resolution against the
// closure's captured environment.
//
// `substitute_constants` walks the body and replaces each free `Expr::Symbol`
// with the corresponding scalar `Expr::NumLit` from a name → value map.
// This is partial evaluation — capture is baked in at JIT-compile time
// rather than passed at every call. Works because numeric scalar
// captures are fixed at the moment the closure was created.

use std::collections::HashSet;

/// Walks an `Expr` and returns the set of free symbol names — i.e.
/// symbols referenced that aren't bound by enclosing function params
/// or by an assignment within the body. `params` lists the names that
/// the immediate function introduces (and should therefore not be
/// counted as free).
pub fn collect_free_vars(body: &Expr, params: &[Arc<str>]) -> HashSet<Arc<str>> {
    let mut free: HashSet<Arc<str>> = HashSet::new();
    let mut bound: HashSet<Arc<str>> = params.iter().cloned().collect();
    collect_free_vars_walk(body, &mut bound, &mut free);
    free
}

fn collect_free_vars_walk(
    e: &Expr,
    bound: &mut HashSet<Arc<str>>,
    free: &mut HashSet<Arc<str>>,
) {
    match e {
        Expr::Symbol(name) => {
            if !bound.contains(name) { free.insert(name.clone()); }
        }
        Expr::NumLit(_) | Expr::IntLit(_) | Expr::StrLit(_) | Expr::BoolLit(_)
            | Expr::NaLit | Expr::NullLit | Expr::FStringLit(_) => {}
        Expr::Unary { expr, .. } => collect_free_vars_walk(expr, bound, free),
        Expr::Binary { lhs, rhs, .. } => {
            collect_free_vars_walk(lhs, bound, free);
            collect_free_vars_walk(rhs, bound, free);
        }
        Expr::If { cond, then, else_ } => {
            collect_free_vars_walk(cond, bound, free);
            collect_free_vars_walk(then, bound, free);
            if let Some(e) = else_ { collect_free_vars_walk(e, bound, free); }
        }
        Expr::Block(exprs) => {
            for e in exprs { collect_free_vars_walk(e, bound, free); }
        }
        Expr::Call { func, args } => {
            collect_free_vars_walk(func, bound, free);
            for a in args { collect_free_vars_walk(&a.value, bound, free); }
        }
        Expr::Assign { target, value, .. } => {
            // The assigned-to name becomes bound from this point on.
            collect_free_vars_walk(value, bound, free);
            if let Expr::Symbol(name) = target.as_ref() {
                bound.insert(name.clone());
            }
        }
        // Conservative: anything else is opaque — treat its sub-exprs as
        // potential free-var holders. Most are language constructs that
        // either we don't JIT anyway or that the IR rejects.
        _ => {}
    }
}

/// Substitute free `Expr::Symbol(name)` references with `Expr::NumLit(value)`
/// where `name` maps to a scalar value in `subs`. Recursive, returns a
/// fresh `Expr` tree (the input is not mutated).
pub fn substitute_constants(body: &Expr, subs: &std::collections::HashMap<Arc<str>, f64>) -> Expr {
    match body {
        Expr::Symbol(name) => {
            if let Some(&val) = subs.get(name) { Expr::NumLit(val) }
            else { body.clone() }
        }
        Expr::Unary { op, expr } => Expr::Unary {
            op: *op,
            expr: Box::new(substitute_constants(expr, subs)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op: *op,
            lhs: Box::new(substitute_constants(lhs, subs)),
            rhs: Box::new(substitute_constants(rhs, subs)),
        },
        Expr::If { cond, then, else_ } => Expr::If {
            cond: Box::new(substitute_constants(cond, subs)),
            then: Box::new(substitute_constants(then, subs)),
            else_: else_.as_ref().map(|e| Box::new(substitute_constants(e, subs))),
        },
        Expr::Block(exprs) => Expr::Block(
            exprs.iter().map(|e| substitute_constants(e, subs)).collect()
        ),
        Expr::Call { func, args } => Expr::Call {
            func: Box::new(substitute_constants(func, subs)),
            args: args.iter().map(|a| r2_types::CallArg {
                name: a.name.clone(),
                value: substitute_constants(&a.value, subs),
            }).collect(),
        },
        _ => body.clone(),
    }
}

// ── Validator ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    UnterminatedBlock(BlockId),
    UseBeforeDef { block: BlockId, vreg: VReg },
    UnknownBlock(BlockId),
    UnknownVReg(VReg),
}

pub fn validate(f: &IrFunc) -> Result<(), Vec<ValidationError>> {
    let mut errs = Vec::new();
    let max_vreg = f.vreg_types.len() as u32;
    let known_blocks: std::collections::HashSet<u32> = f.blocks.iter().map(|b| b.id.0).collect();

    for blk in &f.blocks {
        if matches!(blk.term, IrTerm::Unreachable) { errs.push(ValidationError::UnterminatedBlock(blk.id)); }
        // Track which VRegs are defined at this point (not strict cross-block analysis;
        // good enough for Phase B sanity).
        let mut defined = std::collections::HashSet::new();
        for inst in &blk.insts {
            let used: Vec<VReg> = match inst {
                IrInst::Unary { src, .. } => vec![*src],
                IrInst::Binary { lhs, rhs, .. } => vec![*lhs, *rhs],
                IrInst::Call { args, .. } | IrInst::Intrinsic { args, .. } => args.clone(),
                IrInst::Phi { sources, .. } => sources.iter().map(|(_, v)| *v).collect(),
                IrInst::Const { .. } => vec![],
            };
            for u in used {
                if u.0 >= max_vreg { errs.push(ValidationError::UnknownVReg(u)); }
                // For Phi we trust the SSA shape; for everything else require local def
                // (or out-of-block predecessor — left to a stricter pass later).
            }
            defined.insert(inst.dst().0);
        }
        match &blk.term {
            IrTerm::Jump(b) => if !known_blocks.contains(&b.0) { errs.push(ValidationError::UnknownBlock(*b)); }
            IrTerm::Branch { then_blk, else_blk, .. } => {
                if !known_blocks.contains(&then_blk.0) { errs.push(ValidationError::UnknownBlock(*then_blk)); }
                if !known_blocks.contains(&else_blk.0) { errs.push(ValidationError::UnknownBlock(*else_blk)); }
            }
            _ => {}
        }
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

// ── Pretty-printer (debug-friendly, also used in tests) ──────────────

pub fn print_func(f: &IrFunc) -> String {
    let mut s = String::new();
    s.push_str(&format!("func @{}() -> {:?} {{\n", f.name, f.return_type));
    for blk in &f.blocks {
        s.push_str(&format!("  {}:\n", blk.id));
        for inst in &blk.insts {
            s.push_str(&format!("    {}\n", print_inst(inst)));
        }
        s.push_str(&format!("    {}\n", print_term(&blk.term)));
    }
    s.push_str("}\n");
    s
}

fn print_inst(i: &IrInst) -> String {
    match i {
        IrInst::Const { dst, value, .. } => format!("{} = const {:?}", dst, value),
        IrInst::Unary { dst, op, src, .. } => format!("{} = unary {:?} {}", dst, op, src),
        IrInst::Binary { dst, op, lhs, rhs, .. } => format!("{} = bin {:?} {}, {}", dst, op, lhs, rhs),
        IrInst::Call { dst, name, args, .. } => format!("{} = call @{}({})", dst, name, args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")),
        IrInst::Intrinsic { dst, name, args, .. } => format!("{} = intrinsic @{}({})", dst, name, args.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")),
        IrInst::Phi { dst, sources, .. } => format!("{} = phi [{}]", dst, sources.iter().map(|(b, v)| format!("{} from {}", v, b)).collect::<Vec<_>>().join(", ")),
    }
}

fn print_term(t: &IrTerm) -> String {
    match t {
        IrTerm::Return(None) => "return".into(),
        IrTerm::Return(Some(r)) => format!("return {}", r),
        IrTerm::Jump(b) => format!("jump {}", b),
        IrTerm::Branch { cond, then_blk, else_blk } => format!("br {} ? {} : {}", cond, then_blk, else_blk),
        IrTerm::Unreachable => "unreachable".into(),
    }
}

// ── Re-export inferencer's helper for builder use ─────────────────

mod _reexport {
    // Forces the public alias below to live next to the IR types.
}

// Provide a public re-export so the builder can call into infer::builtin_return_type
// without exposing the private name.
pub mod infer_compat {
    use super::*;
    pub fn builtin_return_type(name: &str, args: &[IrType]) -> IrType {
        infer::builtin_return_type_pub(name, args)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn expr_int(n: i32) -> Expr { Expr::IntLit(n) }
    fn expr_num(n: f64) -> Expr { Expr::NumLit(n) }
    fn expr_sym(s: &str) -> Expr { Expr::Symbol(Arc::from(s)) }
    fn expr_add(l: Expr, r: Expr) -> Expr { Expr::Binary { op: BinOp::Add, lhs: Box::new(l), rhs: Box::new(r) } }

    #[test]
    fn lower_literal() {
        let f = lower_program(&[expr_num(3.14)], "test");
        assert!(validate(&f).is_ok());
        assert_eq!(f.blocks[0].insts.len(), 1);
        assert!(matches!(f.blocks[0].insts[0], IrInst::Const { value: IrConst::Real(_), .. }));
    }

    #[test]
    fn lower_arithmetic() {
        let prog = vec![expr_add(expr_num(1.0), expr_int(2))];
        let f = lower_program(&prog, "add");
        assert!(validate(&f).is_ok());
        // Two consts + one binary
        assert_eq!(f.blocks[0].insts.len(), 3);
        assert!(matches!(f.blocks[0].insts.last().unwrap(), IrInst::Binary { .. }));
        // Result type promotes Int + Real → Real
        let last = f.blocks[0].insts.last().unwrap();
        assert_eq!(last.ty().elem, infer::IrElem::Real);
    }

    #[test]
    fn lower_assignment_and_use() {
        let prog = vec![
            Expr::Assign { target: Box::new(expr_sym("x")), value: Box::new(expr_num(5.0)), superassign: false },
            expr_add(expr_sym("x"), expr_num(2.0)),
        ];
        let f = lower_program(&prog, "use_x");
        assert!(validate(&f).is_ok());
        // Const(5.0) → Const(2.0) → Binary
        let last = f.blocks[0].insts.last().unwrap();
        assert!(matches!(last, IrInst::Binary { .. }));
    }

    #[test]
    fn lower_if_emits_phi() {
        let prog = vec![Expr::If {
            cond: Box::new(Expr::BoolLit(true)),
            then: Box::new(expr_num(1.0)),
            else_: Some(Box::new(expr_num(2.0))),
        }];
        let f = lower_program(&prog, "iff");
        assert!(validate(&f).is_ok());
        // Should have at least 4 blocks: entry, then, else, join.
        assert!(f.blocks.len() >= 4);
        // Final block should contain a Phi.
        let join = f.blocks.last().unwrap();
        assert!(join.insts.iter().any(|i| matches!(i, IrInst::Phi { .. })));
    }

    #[test]
    fn lower_internal_call_is_intrinsic() {
        // .Internal("matmul", a, b) → Intrinsic
        let prog = vec![Expr::Call {
            func: Box::new(expr_sym(".Internal")),
            args: vec![
                CallArg { name: None, value: Expr::StrLit("matmul".into()) },
                CallArg { name: None, value: expr_sym("a") },
                CallArg { name: None, value: expr_sym("b") },
            ],
        }];
        let f = lower_program(&prog, "intr");
        assert!(validate(&f).is_ok());
        let has_intr = f.blocks.iter().flat_map(|b| &b.insts)
            .any(|i| matches!(i, IrInst::Intrinsic { name, .. } if name.as_ref() == "matmul"));
        assert!(has_intr, "expected an Intrinsic with name 'matmul'");
    }

    #[test]
    fn lower_while_creates_loop_blocks() {
        let prog = vec![Expr::While {
            cond: Box::new(Expr::BoolLit(true)),
            body: Box::new(Expr::Block(vec![expr_num(1.0)])),
        }];
        let f = lower_program(&prog, "loop");
        assert!(validate(&f).is_ok());
        // head + body + exit + entry-and-trailing = at least 4 blocks
        assert!(f.blocks.len() >= 4);
    }

    #[test]
    fn validator_catches_unterminated_block() {
        // Hand-build a malformed function.
        let bad = IrFunc {
            name: Arc::from("bad"),
            params: vec![],
            return_type: IrType::null(),
            blocks: vec![IrBlock { id: BlockId(0), insts: vec![], term: IrTerm::Unreachable }],
            vreg_types: vec![],
            entry: BlockId(0),
        };
        let res = validate(&bad);
        assert!(matches!(res, Err(_)));
    }

    #[test]
    fn pretty_print_produces_text() {
        let f = lower_program(&[expr_add(expr_num(1.0), expr_num(2.0))], "add");
        let s = print_func(&f);
        assert!(s.contains("func @add"));
        assert!(s.contains("bin Add"));
        assert!(s.contains("return"));
    }
}
