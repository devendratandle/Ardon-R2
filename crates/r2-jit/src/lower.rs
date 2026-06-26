//! IR -> Cranelift body lowering (blocks, phis, instructions).

use cranelift::prelude::*;
use r2_ir::{BlockId, IrConst, IrFunc, IrInst, IrTerm, VReg};
use r2_types::BinOp;
use std::collections::HashMap;
use crate::*;


/// Per-IR-block summary of phi instructions: ordered list of (dst, sources).
pub(crate) struct PhiInfo {
    pub(crate) dst_regs: Vec<VReg>,
    /// For each phi position, the predecessor → source-VReg mapping.
    pub(crate) sources_per_phi: Vec<HashMap<u32, VReg>>,
}

pub(crate) fn lower_func_body(bcx: &mut FunctionBuilder, func: &IrFunc, math_refs: MathRefs<'_>) -> JitResult<()> {
    // 1. Pre-create one Cranelift block per IR block.
    let mut block_map: HashMap<u32, Block> = HashMap::new();
    for blk in &func.blocks {
        block_map.insert(blk.id.0, bcx.create_block());
    }

    // 2. Scan each IR block for leading Phi instructions; reserve block params for them.
    let mut phi_info: HashMap<u32, PhiInfo> = HashMap::new();
    for blk in &func.blocks {
        let cl = block_map[&blk.id.0];
        let mut info = PhiInfo { dst_regs: Vec::new(), sources_per_phi: Vec::new() };
        for inst in &blk.insts {
            if let IrInst::Phi { dst, sources, .. } = inst {
                bcx.append_block_param(cl, types::F64);
                info.dst_regs.push(*dst);
                let map: HashMap<u32, VReg> = sources.iter().map(|(b, v)| (b.0, *v)).collect();
                info.sources_per_phi.push(map);
            } else { break; }
        }
        phi_info.insert(blk.id.0, info);
    }

    // 3. Entry block also receives the function's parameters (after any phi params).
    let entry_cl = *block_map.get(&func.entry.0).ok_or(JitError::UndefinedBlock(func.entry))?;
    bcx.append_block_params_for_function_params(entry_cl);

    // 4. env: VReg index → Cranelift Value
    let mut env: HashMap<u32, Value> = HashMap::new();

    // 5. Switch into entry, bind formal parameters.
    bcx.switch_to_block(entry_cl);
    let entry_params: Vec<Value> = bcx.block_params(entry_cl).to_vec();
    let entry_phi_count = phi_info[&func.entry.0].dst_regs.len();
    for (i, dst) in phi_info[&func.entry.0].dst_regs.iter().enumerate() {
        env.insert(dst.0, entry_params[i]);
    }
    for ((_, _, vreg), v) in func.params.iter().zip(entry_params.iter().skip(entry_phi_count)) {
        env.insert(vreg.0, *v);
    }

    // 6. Lower each block.
    for blk in &func.blocks {
        let cl = block_map[&blk.id.0];
        if blk.id != func.entry {
            bcx.switch_to_block(cl);
            // Bind phi destinations to block parameters.
            let cl_params = bcx.block_params(cl).to_vec();
            for (i, dst) in phi_info[&blk.id.0].dst_regs.iter().enumerate() {
                env.insert(dst.0, cl_params[i]);
            }
        }
        // Lower instructions, skipping leading phis (already bound).
        let phi_count = phi_info[&blk.id.0].dst_regs.len();
        for inst in blk.insts.iter().skip(phi_count) {
            let v = lower_inst(bcx, inst, &env, math_refs)?;
            env.insert(inst.dst().0, v);
        }

        // Terminator.
        match &blk.term {
            IrTerm::Return(Some(reg)) => {
                let v = *env.get(&reg.0).ok_or(JitError::UndefinedVReg(*reg))?;
                bcx.ins().return_(&[v]);
            }
            IrTerm::Return(None) => {
                let zero = bcx.ins().f64const(0.0);
                bcx.ins().return_(&[zero]);
            }
            IrTerm::Jump(target) => {
                let target_cl = *block_map.get(&target.0).ok_or(JitError::UndefinedBlock(*target))?;
                let args = phi_args(&blk.id, target, &phi_info, &env)?;
                bcx.ins().jump(target_cl, &args);
            }
            IrTerm::Branch { cond, then_blk, else_blk } => {
                let c = *env.get(&cond.0).ok_or(JitError::UndefinedVReg(*cond))?;
                let zero = bcx.ins().f64const(0.0);
                let cond_b = bcx.ins().fcmp(FloatCC::NotEqual, c, zero);
                let then_cl = *block_map.get(&then_blk.0).ok_or(JitError::UndefinedBlock(*then_blk))?;
                let else_cl = *block_map.get(&else_blk.0).ok_or(JitError::UndefinedBlock(*else_blk))?;
                let then_args = phi_args(&blk.id, then_blk, &phi_info, &env)?;
                let else_args = phi_args(&blk.id, else_blk, &phi_info, &env)?;
                bcx.ins().brif(cond_b, then_cl, &then_args, else_cl, &else_args);
            }
            IrTerm::Unreachable => {
                bcx.ins().trap(TrapCode::UnreachableCodeReached);
            }
        }
    }

    // All blocks emitted; seal everything.
    bcx.seal_all_blocks();
    Ok(())
}

/// Build the argument list for a Jump/Branch into `to`, picking each phi's
/// source from the `from` predecessor.
pub(crate) fn phi_args(
    from: &BlockId,
    to: &BlockId,
    phi_info: &HashMap<u32, PhiInfo>,
    env: &HashMap<u32, Value>,
) -> JitResult<Vec<Value>> {
    let info = match phi_info.get(&to.0) { Some(i) if !i.dst_regs.is_empty() => i, _ => return Ok(Vec::new()) };
    let mut args = Vec::with_capacity(info.dst_regs.len());
    for src_map in &info.sources_per_phi {
        let src = src_map.get(&from.0).ok_or_else(|| {
            JitError::CraneliftError(format!("phi in {} has no source from {}", to, from))
        })?;
        let v = *env.get(&src.0).ok_or(JitError::UndefinedVReg(*src))?;
        args.push(v);
    }
    Ok(args)
}

/// Lowering context for `Call` instructions. `math_refs` maps each
/// R-level name (e.g. `"sqrt"`) to the Cranelift `FuncRef` already
/// declared in the current function builder. `None` means math externs
/// aren't registered for this compilation (the legacy paths that
/// pre-date Call support pass `None`; such paths reject `IrInst::Call`).
pub(crate) type MathRefs<'a> = Option<&'a HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef>>;

pub(crate) fn lower_inst(
    bcx: &mut FunctionBuilder,
    inst: &IrInst,
    env: &HashMap<u32, Value>,
    math_refs: MathRefs<'_>,
) -> JitResult<Value> {
    match inst {
        IrInst::Const { value, .. } => match value {
            IrConst::Real(x) => Ok(bcx.ins().f64const(*x)),
            IrConst::Int(x) => {
                let v = bcx.ins().iconst(types::I64, *x as i64);
                Ok(bcx.ins().fcvt_from_sint(types::F64, v))
            }
            IrConst::Bool(b) => Ok(bcx.ins().f64const(if *b { 1.0 } else { 0.0 })),
            IrConst::Null | IrConst::NA => Ok(bcx.ins().f64const(0.0)),
            IrConst::Str(_) => Err(JitError::Unsupported("string const".into())),
        },
        IrInst::Binary { op, lhs, rhs, .. } => {
            let l = *env.get(&lhs.0).ok_or(JitError::UndefinedVReg(*lhs))?;
            let r = *env.get(&rhs.0).ok_or(JitError::UndefinedVReg(*rhs))?;
            // BinOp::Pow lowers to a Cranelift call to `r2_math_pow` if
            // math externs are available — we don't have a native f64
            // `**` instruction. Falls back to error otherwise.
            if matches!(op, BinOp::Pow) {
                let refs = math_refs.ok_or_else(|| JitError::Unsupported(
                    "binop Pow needs math externs (compile path didn't register them)".into()
                ))?;
                let fref = refs.get("^").ok_or_else(|| JitError::CraneliftError(
                    "math extern `^` (pow) not declared".into()
                ))?;
                let call = bcx.ins().call(*fref, &[l, r]);
                return Ok(bcx.inst_results(call)[0]);
            }
            Ok(match op {
                BinOp::Add => bcx.ins().fadd(l, r),
                BinOp::Sub => bcx.ins().fsub(l, r),
                BinOp::Mul => bcx.ins().fmul(l, r),
                BinOp::Div => bcx.ins().fdiv(l, r),
                BinOp::Lt => cmp_to_f64(bcx, l, r, FloatCC::LessThan),
                BinOp::Gt => cmp_to_f64(bcx, l, r, FloatCC::GreaterThan),
                BinOp::Le => cmp_to_f64(bcx, l, r, FloatCC::LessThanOrEqual),
                BinOp::Ge => cmp_to_f64(bcx, l, r, FloatCC::GreaterThanOrEqual),
                BinOp::Eq => cmp_to_f64(bcx, l, r, FloatCC::Equal),
                BinOp::Ne => cmp_to_f64(bcx, l, r, FloatCC::NotEqual),
                other => return Err(JitError::Unsupported(format!("binop {:?}", other))),
            })
        }
        IrInst::Unary { op, src, .. } => {
            let v = *env.get(&src.0).ok_or(JitError::UndefinedVReg(*src))?;
            Ok(match op {
                r2_types::UnOp::Neg => bcx.ins().fneg(v),
                r2_types::UnOp::Pos => v,
                r2_types::UnOp::Not => {
                    let zero = bcx.ins().f64const(0.0);
                    cmp_to_f64(bcx, v, zero, FloatCC::Equal)
                }
            })
        }
        // Lower a builtin call. Two-tier strategy:
        //   1. **Native Cranelift instruction** for math ops that have
        //      direct hardware support (`sqrt`, `abs`, `floor`, `ceil`,
        //      `trunc`, `min`, `max`). One x86 instruction per element,
        //      no call overhead, no register spill across a call.
        //   2. **Rust call via stable ABI** for the transcendentals
        //      (`sin`, `cos`, `exp`, `log`, etc.) that don't have direct
        //      CPU support. These are Rust functions (not C library
        //      functions) — the `extern "C"` keyword on the wrappers
        //      sets the *calling convention* so Cranelift can predictably
        //      `call` them; the function bodies themselves are pure Rust
        //      delegating to `f64::sin()` etc. No OS-level FFI is involved.
        //
        // The native path matters at scale: SIMD `sqrtpd` retires one
        // double per cycle; a Rust call adds ~5 ns dispatch overhead +
        // libm cost. For `sqrt(x*x + 1)` over 1e6 elements that's the
        // difference between 6 ms and 15 ms.
        IrInst::Call { name, args, .. } => {
            let arg_vals: Vec<Value> = args.iter()
                .map(|reg| env.get(&reg.0).copied().ok_or(JitError::UndefinedVReg(*reg)))
                .collect::<JitResult<Vec<_>>>()?;
            // Try the native-instruction fast path first.
            match (name.as_ref(), arg_vals.len()) {
                ("sqrt",  1) => return Ok(bcx.ins().sqrt(arg_vals[0])),
                ("abs",   1) => return Ok(bcx.ins().fabs(arg_vals[0])),
                ("floor", 1) => return Ok(bcx.ins().floor(arg_vals[0])),
                ("ceil",  1) => return Ok(bcx.ins().ceil(arg_vals[0])),
                ("trunc", 1) => return Ok(bcx.ins().trunc(arg_vals[0])),
                // Cranelift `nearest` rounds half-to-even ("banker's
                // rounding") — same as R's `round()`. Match.
                ("round", 1) => return Ok(bcx.ins().nearest(arg_vals[0])),
                ("min",   2) => return Ok(bcx.ins().fmin(arg_vals[0], arg_vals[1])),
                ("max",   2) => return Ok(bcx.ins().fmax(arg_vals[0], arg_vals[1])),
                _ => {} // fall through to extern call
            }
            // Rust-call fallback for sin/cos/log/exp/etc. — emits a
            // Cranelift `call` to a Rust function (extern "C" wrapper
            // for stable ABI). Not foreign-function-interface in the
            // OS sense; just a JIT → Rust handoff.
            let refs = math_refs.ok_or_else(|| JitError::Unsupported(
                "Call requires math-extern registration (compile path didn't enable it)".into()
            ))?;
            let me = find_math_extern(name.as_ref()).ok_or_else(|| JitError::Unsupported(
                format!("Call to unsupported builtin `{}`", name)
            ))?;
            if args.len() != me.arity {
                return Err(JitError::Unsupported(format!(
                    "Call `{}` arity {} != expected {}", name, args.len(), me.arity
                )));
            }
            let fref = refs.get(me.r_name).ok_or_else(|| JitError::CraneliftError(
                format!("math extern `{}` not declared in this module", name)
            ))?;
            let call = bcx.ins().call(*fref, &arg_vals);
            Ok(bcx.inst_results(call)[0])
        }
        IrInst::Phi { .. } => Err(JitError::CraneliftError(
            "phi must appear at the start of a block (handled by lower_func_body)".into(),
        )),
        other => Err(JitError::Unsupported(format!("instruction {:?}", other))),
    }
}

/// Lower a comparison to a Cranelift f64 1.0 / 0.0 value (Bool ABI placeholder).
pub(crate) fn cmp_to_f64(bcx: &mut FunctionBuilder, l: Value, r: Value, cc: FloatCC) -> Value {
    let cmp = bcx.ins().fcmp(cc, l, r);                     // i8 (b1)
    let one = bcx.ins().f64const(1.0);
    let zero = bcx.ins().f64const(0.0);
    bcx.ins().select(cmp, one, zero)
}
