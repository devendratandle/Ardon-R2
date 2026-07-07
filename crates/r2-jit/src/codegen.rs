//! Vectorized codegen: SIMD-clean detection + map/reduce builders.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use r2_ir::{IrConst, IrFunc, IrInst, IrTerm, VReg};
use r2_types::BinOp;
use std::collections::HashMap;
use crate::*;

/// Returns `true` when an IR body is suitable for f64x2 SIMD vectorization:
/// single block, only `Const`/`Unary`/`Binary` arithmetic + `Call` to the
/// natively-vectorizable math instructions. Anything outside this subset
/// (branches, phis, extern math calls like sin/cos/exp/log, comparisons,
/// Pow) bails to the scalar path.
pub(crate) fn body_is_simd_clean(body_ir: &IrFunc) -> bool {
    if body_ir.blocks.len() != 1 { return false; }
    let blk = &body_ir.blocks[0];
    if !matches!(blk.term, IrTerm::Return(Some(_))) { return false; }
    for inst in &blk.insts {
        match inst {
            IrInst::Const { value, .. } => match value {
                IrConst::Real(_) | IrConst::Int(_) | IrConst::Bool(_) => {}
                _ => return false,
            },
            IrInst::Unary { op, .. } => match op {
                r2_types::UnOp::Neg | r2_types::UnOp::Pos => {}
                _ => return false,
            },
            IrInst::Binary { op, .. } => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {}
                _ => return false,
            },
            IrInst::Call { name, args, .. } => {
                let ok_unary = args.len() == 1 && matches!(name.as_ref(),
                    "sqrt" | "abs" | "floor" | "ceil" | "trunc" | "round");
                let ok_binary = args.len() == 2 && matches!(name.as_ref(),
                    "min" | "max");
                if !ok_unary && !ok_binary { return false; }
            }
            _ => return false, // Phi, Intrinsic, etc.
        }
    }
    true
}

/// Lower a single IR instruction to a SIMD `F64X2` Cranelift Value.
/// Mirrors `lower_inst` but emits vector instructions throughout.
pub(crate) fn lower_inst_simd(
    bcx: &mut FunctionBuilder,
    inst: &IrInst,
    env: &HashMap<u32, Value>,
) -> JitResult<Value> {
    match inst {
        IrInst::Const { value, .. } => {
            let scalar = match value {
                IrConst::Real(x) => bcx.ins().f64const(*x),
                IrConst::Int(x) => {
                    let v = bcx.ins().iconst(types::I64, *x as i64);
                    bcx.ins().fcvt_from_sint(types::F64, v)
                }
                IrConst::Bool(b) => bcx.ins().f64const(if *b { 1.0 } else { 0.0 }),
                _ => return Err(JitError::Unsupported("simd: unsupported const".into())),
            };
            // Splat scalar to F64X2 (broadcasts the value across both lanes).
            Ok(bcx.ins().splat(types::F64X2, scalar))
        }
        IrInst::Binary { op, lhs, rhs, .. } => {
            let l = *env.get(&lhs.0).ok_or(JitError::UndefinedVReg(*lhs))?;
            let r = *env.get(&rhs.0).ok_or(JitError::UndefinedVReg(*rhs))?;
            Ok(match op {
                BinOp::Add => bcx.ins().fadd(l, r),
                BinOp::Sub => bcx.ins().fsub(l, r),
                BinOp::Mul => bcx.ins().fmul(l, r),
                BinOp::Div => bcx.ins().fdiv(l, r),
                _ => return Err(JitError::Unsupported(format!("simd: binop {:?}", op))),
            })
        }
        IrInst::Unary { op, src, .. } => {
            let v = *env.get(&src.0).ok_or(JitError::UndefinedVReg(*src))?;
            Ok(match op {
                r2_types::UnOp::Neg => bcx.ins().fneg(v),
                r2_types::UnOp::Pos => v,
                _ => return Err(JitError::Unsupported("simd: unsupported unop".into())),
            })
        }
        IrInst::Call { name, args, .. } => {
            let arg_vals: Vec<Value> = args.iter()
                .map(|reg| env.get(&reg.0).copied().ok_or(JitError::UndefinedVReg(*reg)))
                .collect::<JitResult<Vec<_>>>()?;
            match (name.as_ref(), arg_vals.len()) {
                ("sqrt",  1) => Ok(bcx.ins().sqrt(arg_vals[0])),
                ("abs",   1) => Ok(bcx.ins().fabs(arg_vals[0])),
                ("floor", 1) => Ok(bcx.ins().floor(arg_vals[0])),
                ("ceil",  1) => Ok(bcx.ins().ceil(arg_vals[0])),
                ("trunc", 1) => Ok(bcx.ins().trunc(arg_vals[0])),
                ("round", 1) => Ok(bcx.ins().nearest(arg_vals[0])),
                ("min",   2) => Ok(bcx.ins().fmin(arg_vals[0], arg_vals[1])),
                ("max",   2) => Ok(bcx.ins().fmax(arg_vals[0], arg_vals[1])),
                _ => Err(JitError::Unsupported(format!("simd: Call to `{}`", name))),
            }
        }
        _ => Err(JitError::Unsupported("simd: unsupported instruction".into())),
    }
}

/// Reduction op for `compile_vector_map_reduce`. Sum/Prod have
/// well-defined associative identities suitable for fusion.
/// (Mean is `Sum / len`, computed by the engine after the JIT call.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedReduceOp {
    /// Σ identity = 0, combine: acc + v
    Sum,
    /// Π identity = 1, combine: acc * v
    Prod,
}

/// Codegen for Phase C.9 — fused map-reduce.
///
/// Loop structure:
///   entry → header(i, acc) → body(loads x[i], computes f(x[i]), combines into acc) → header(i+1, new_acc)
///                        ↓ when i >= len, return acc.
///
/// `acc` is carried as a block parameter (Cranelift Phi via block param).
pub(crate) fn compile_map_reduce_inner(
    body_ir: &IrFunc,
    reduce_op: FusedReduceOp,
) -> JitResult<CompiledFn> {
    if body_ir.blocks.is_empty() {
        return Err(JitError::Unsupported("empty IR body".into()));
    }

    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
    register_math_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);
    let math_ids = declare_math_imports(&mut module)?;

    // Signature: (in_ptr: i64, len: i64) -> f64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::F64));

    let func_id = module
        .declare_function("__jit_map_reduce", Linkage::Export, &sig)
        .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let math_refs: HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef> =
            math_ids.iter()
                .map(|(k, id)| (*k, module.declare_func_in_func(*id, &mut bcx.func)))
                .collect();
        let math_refs_opt: MathRefs<'_> = Some(&math_refs);

        // Outer scaffold blocks.
        let entry  = bcx.create_block();
        let header = bcx.create_block();   // (i: i64, acc: f64) block params
        let load_b = bcx.create_block();   // (i: i64, acc: f64) block params — runs body
        let exit   = bcx.create_block();   // (acc: f64) block param

        bcx.append_block_param(header, types::I64);
        bcx.append_block_param(header, types::F64);
        bcx.append_block_param(load_b, types::I64);
        bcx.append_block_param(load_b, types::F64);
        bcx.append_block_param(exit,   types::F64);

        // Identity element for the reduction.
        let identity = match reduce_op {
            FusedReduceOp::Sum  => 0.0,
            FusedReduceOp::Prod => 1.0,
        };

        // Pre-create one Cranelift block per IR block of the inner body.
        // Each inner block receives (i, acc) block params first, then
        // phi params for the IR's own Phis, then the loaded element
        // as the IR's formal parameter on the entry block.
        let mut block_map: HashMap<u32, Block> = HashMap::new();
        let mut phi_info: HashMap<u32, PhiInfo> = HashMap::new();
        for blk in &body_ir.blocks {
            let cl = bcx.create_block();
            block_map.insert(blk.id.0, cl);
            bcx.append_block_param(cl, types::I64); // i carried through
            bcx.append_block_param(cl, types::F64); // acc carried through
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
        let ir_entry_cl = *block_map.get(&body_ir.entry.0)
            .ok_or(JitError::UndefinedBlock(body_ir.entry))?;
        // Inner body's formal param (the loaded x[i]) — one F64 block param on IR entry.
        bcx.append_block_param(ir_entry_cl, types::F64);

        // ── Entry: collect args, jump to header(0, identity) ────────
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        let entry_params: Vec<Value> = bcx.block_params(entry).to_vec();
        let in_ptr = entry_params[0];
        let len    = entry_params[1];
        let zero_i = bcx.ins().iconst(types::I64, 0);
        let id_v = bcx.ins().f64const(identity);
        bcx.ins().jump(header, &[zero_i, id_v]);

        // ── Header: while i < len ───────────────────────────────────
        bcx.switch_to_block(header);
        let i_h   = bcx.block_params(header)[0];
        let acc_h = bcx.block_params(header)[1];
        let lt = bcx.ins().icmp(IntCC::SignedLessThan, i_h, len);
        bcx.ins().brif(lt, load_b, &[i_h, acc_h], exit, &[acc_h]);

        // ── load_b: read in_ptr[i], jump into IR entry ──────────────
        bcx.switch_to_block(load_b);
        let i_l   = bcx.block_params(load_b)[0];
        let acc_l = bcx.block_params(load_b)[1];
        let eight = bcx.ins().iconst(types::I64, 8);
        let off = bcx.ins().imul(i_l, eight);
        let addr = bcx.ins().iadd(in_ptr, off);
        let mflags = MemFlags::trusted();
        let elem = bcx.ins().load(types::F64, mflags, addr, 0);
        bcx.ins().jump(ir_entry_cl, &[i_l, acc_l, elem]);

        // ── Lower each IR block, threading (i, acc) through ─────────
        // env: VReg -> Cranelift Value. Shared across IR blocks (defs
        // from a dominator block visible in dominated blocks).
        let mut env: HashMap<u32, Value> = HashMap::new();

        for blk in &body_ir.blocks {
            let cl = block_map[&blk.id.0];
            bcx.switch_to_block(cl);
            let cl_params: Vec<Value> = bcx.block_params(cl).to_vec();
            let i_here   = cl_params[0];
            let acc_here = cl_params[1];
            let phi_count = phi_info[&blk.id.0].dst_regs.len();
            for (k, dst) in phi_info[&blk.id.0].dst_regs.iter().enumerate() {
                env.insert(dst.0, cl_params[2 + k]);
            }
            if blk.id == body_ir.entry {
                // Bind IR formal param to the loaded element.
                let elem = cl_params[2 + phi_count];
                env.insert(body_ir.params[0].2.0, elem);
            }
            // Lower instructions (skip leading Phis).
            for inst in blk.insts.iter().skip(phi_count) {
                let v = lower_inst(&mut bcx, inst, &env, math_refs_opt)?;
                env.insert(inst.dst().0, v);
            }
            // Terminator. On Return, combine the result into acc and
            // continue to header(i+1, new_acc). On Jump/Branch, thread
            // (i, acc) through to the target IR block.
            match &blk.term {
                IrTerm::Return(Some(reg)) => {
                    let v = *env.get(&reg.0).ok_or(JitError::UndefinedVReg(*reg))?;
                    let new_acc = match reduce_op {
                        FusedReduceOp::Sum  => bcx.ins().fadd(acc_here, v),
                        FusedReduceOp::Prod => bcx.ins().fmul(acc_here, v),
                    };
                    let one = bcx.ins().iconst(types::I64, 1);
                    let next_i = bcx.ins().iadd(i_here, one);
                    bcx.ins().jump(header, &[next_i, new_acc]);
                }
                IrTerm::Return(None) => {
                    // Skip this iteration's contribution; advance i, keep acc.
                    let one = bcx.ins().iconst(types::I64, 1);
                    let next_i = bcx.ins().iadd(i_here, one);
                    bcx.ins().jump(header, &[next_i, acc_here]);
                }
                IrTerm::Jump(target) => {
                    let target_cl = *block_map.get(&target.0)
                        .ok_or(JitError::UndefinedBlock(*target))?;
                    let mut args = vec![i_here, acc_here];
                    args.extend(phi_args(&blk.id, target, &phi_info, &env)?);
                    bcx.ins().jump(target_cl, &args);
                }
                IrTerm::Branch { cond, then_blk, else_blk } => {
                    let c = *env.get(&cond.0).ok_or(JitError::UndefinedVReg(*cond))?;
                    let zero = bcx.ins().f64const(0.0);
                    let cond_b = bcx.ins().fcmp(FloatCC::NotEqual, c, zero);
                    let then_cl = *block_map.get(&then_blk.0).ok_or(JitError::UndefinedBlock(*then_blk))?;
                    let else_cl = *block_map.get(&else_blk.0).ok_or(JitError::UndefinedBlock(*else_blk))?;
                    let mut then_args = vec![i_here, acc_here];
                    then_args.extend(phi_args(&blk.id, then_blk, &phi_info, &env)?);
                    let mut else_args = vec![i_here, acc_here];
                    else_args.extend(phi_args(&blk.id, else_blk, &phi_info, &env)?);
                    bcx.ins().brif(cond_b, then_cl, &then_args, else_cl, &else_args);
                }
                IrTerm::Unreachable => { bcx.ins().trap(TrapCode::UnreachableCodeReached); }
            }
        }

        // ── Exit: return acc ────────────────────────────────────────
        bcx.switch_to_block(exit);
        let final_acc = bcx.block_params(exit)[0];
        bcx.ins().return_(&[final_acc]);

        bcx.seal_all_blocks();
        bcx.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| JitError::CraneliftError(format!("define: {:?}", e)))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::CraneliftError(format!("finalize: {:?}", e)))?;

    let ptr = module.get_finalized_function(func_id);
    Ok(CompiledFn { ptr, arity: 1, kind: r2_types::JitKind::Vector1ToScalar, _module: module })
}

/// Phase J.2 — fused BINARY map-reduce: `reduce(f(a[i], b[i]))` over two
/// same-length vectors → scalar (e.g. `sum(x*w)` dot product). Signature
/// `(a_ptr, b_ptr, len) -> f64`. Mirrors `compile_map_reduce_inner` exactly
/// but loads from two pointers and binds two inner params per iteration.
pub(crate) fn compile_binary_map_reduce_inner(
    body_ir: &IrFunc,
    reduce_op: FusedReduceOp,
) -> JitResult<CompiledFn> {
    if body_ir.blocks.is_empty() {
        return Err(JitError::Unsupported("empty IR body".into()));
    }
    if body_ir.params.len() != 2 {
        return Err(JitError::Unsupported("binary map-reduce expects 2 inner params".into()));
    }
    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
    register_math_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);
    let math_ids = declare_math_imports(&mut module)?;

    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64)); // a_ptr
    sig.params.push(AbiParam::new(types::I64)); // b_ptr
    sig.params.push(AbiParam::new(types::I64)); // len
    sig.returns.push(AbiParam::new(types::F64));

    let func_id = module
        .declare_function("__jit_binary_map_reduce", Linkage::Export, &sig)
        .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let math_refs: HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef> =
            math_ids.iter().map(|(k, id)| (*k, module.declare_func_in_func(*id, &mut bcx.func))).collect();
        let math_refs_opt: MathRefs<'_> = Some(&math_refs);

        let entry  = bcx.create_block();
        let header = bcx.create_block();
        let load_b = bcx.create_block();
        let exit   = bcx.create_block();
        bcx.append_block_param(header, types::I64);
        bcx.append_block_param(header, types::F64);
        bcx.append_block_param(load_b, types::I64);
        bcx.append_block_param(load_b, types::F64);
        bcx.append_block_param(exit,   types::F64);

        let identity = match reduce_op { FusedReduceOp::Sum => 0.0, FusedReduceOp::Prod => 1.0 };

        let mut block_map: HashMap<u32, Block> = HashMap::new();
        let mut phi_info: HashMap<u32, PhiInfo> = HashMap::new();
        for blk in &body_ir.blocks {
            let cl = bcx.create_block();
            block_map.insert(blk.id.0, cl);
            bcx.append_block_param(cl, types::I64);
            bcx.append_block_param(cl, types::F64);
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
        let ir_entry_cl = *block_map.get(&body_ir.entry.0).ok_or(JitError::UndefinedBlock(body_ir.entry))?;
        bcx.append_block_param(ir_entry_cl, types::F64); // a[i]
        bcx.append_block_param(ir_entry_cl, types::F64); // b[i]

        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        let entry_params: Vec<Value> = bcx.block_params(entry).to_vec();
        let a_ptr = entry_params[0];
        let b_ptr = entry_params[1];
        let len   = entry_params[2];
        let zero_i = bcx.ins().iconst(types::I64, 0);
        let id_v = bcx.ins().f64const(identity);
        bcx.ins().jump(header, &[zero_i, id_v]);

        bcx.switch_to_block(header);
        let i_h = bcx.block_params(header)[0];
        let acc_h = bcx.block_params(header)[1];
        let lt = bcx.ins().icmp(IntCC::SignedLessThan, i_h, len);
        bcx.ins().brif(lt, load_b, &[i_h, acc_h], exit, &[acc_h]);

        bcx.switch_to_block(load_b);
        let i_l = bcx.block_params(load_b)[0];
        let acc_l = bcx.block_params(load_b)[1];
        let eight = bcx.ins().iconst(types::I64, 8);
        let off = bcx.ins().imul(i_l, eight);
        let mflags = MemFlags::trusted();
        let a_addr = bcx.ins().iadd(a_ptr, off);
        let a_elem = bcx.ins().load(types::F64, mflags, a_addr, 0);
        let b_addr = bcx.ins().iadd(b_ptr, off);
        let b_elem = bcx.ins().load(types::F64, mflags, b_addr, 0);
        bcx.ins().jump(ir_entry_cl, &[i_l, acc_l, a_elem, b_elem]);

        let mut env: HashMap<u32, Value> = HashMap::new();
        for blk in &body_ir.blocks {
            let cl = block_map[&blk.id.0];
            bcx.switch_to_block(cl);
            let cl_params: Vec<Value> = bcx.block_params(cl).to_vec();
            let i_here = cl_params[0];
            let acc_here = cl_params[1];
            let phi_count = phi_info[&blk.id.0].dst_regs.len();
            for (k, dst) in phi_info[&blk.id.0].dst_regs.iter().enumerate() {
                env.insert(dst.0, cl_params[2 + k]);
            }
            if blk.id == body_ir.entry {
                let a_e = cl_params[2 + phi_count];
                let b_e = cl_params[2 + phi_count + 1];
                env.insert(body_ir.params[0].2.0, a_e);
                env.insert(body_ir.params[1].2.0, b_e);
            }
            for inst in blk.insts.iter().skip(phi_count) {
                let v = lower_inst(&mut bcx, inst, &env, math_refs_opt)?;
                env.insert(inst.dst().0, v);
            }
            match &blk.term {
                IrTerm::Return(Some(reg)) => {
                    let v = *env.get(&reg.0).ok_or(JitError::UndefinedVReg(*reg))?;
                    let new_acc = match reduce_op {
                        FusedReduceOp::Sum => bcx.ins().fadd(acc_here, v),
                        FusedReduceOp::Prod => bcx.ins().fmul(acc_here, v),
                    };
                    let one = bcx.ins().iconst(types::I64, 1);
                    let next_i = bcx.ins().iadd(i_here, one);
                    bcx.ins().jump(header, &[next_i, new_acc]);
                }
                IrTerm::Return(None) => {
                    let one = bcx.ins().iconst(types::I64, 1);
                    let next_i = bcx.ins().iadd(i_here, one);
                    bcx.ins().jump(header, &[next_i, acc_here]);
                }
                IrTerm::Jump(target) => {
                    let target_cl = *block_map.get(&target.0).ok_or(JitError::UndefinedBlock(*target))?;
                    let mut args = vec![i_here, acc_here];
                    args.extend(phi_args(&blk.id, target, &phi_info, &env)?);
                    bcx.ins().jump(target_cl, &args);
                }
                IrTerm::Branch { cond, then_blk, else_blk } => {
                    let c = *env.get(&cond.0).ok_or(JitError::UndefinedVReg(*cond))?;
                    let zero = bcx.ins().f64const(0.0);
                    let cond_b = bcx.ins().fcmp(FloatCC::NotEqual, c, zero);
                    let then_cl = *block_map.get(&then_blk.0).ok_or(JitError::UndefinedBlock(*then_blk))?;
                    let else_cl = *block_map.get(&else_blk.0).ok_or(JitError::UndefinedBlock(*else_blk))?;
                    let mut then_args = vec![i_here, acc_here];
                    then_args.extend(phi_args(&blk.id, then_blk, &phi_info, &env)?);
                    let mut else_args = vec![i_here, acc_here];
                    else_args.extend(phi_args(&blk.id, else_blk, &phi_info, &env)?);
                    bcx.ins().brif(cond_b, then_cl, &then_args, else_cl, &else_args);
                }
                IrTerm::Unreachable => { bcx.ins().trap(TrapCode::UnreachableCodeReached); }
            }
        }

        bcx.switch_to_block(exit);
        let final_acc = bcx.block_params(exit)[0];
        bcx.ins().return_(&[final_acc]);
        bcx.seal_all_blocks();
        bcx.finalize();
    }

    module.define_function(func_id, &mut ctx).map_err(|e| JitError::CraneliftError(format!("define: {:?}", e)))?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| JitError::CraneliftError(format!("finalize: {:?}", e)))?;
    let ptr = module.get_finalized_function(func_id);
    Ok(CompiledFn { ptr, arity: 2, kind: r2_types::JitKind::Vector2ToScalar, _module: module })
}

/// Shared codegen for SIMD f64x2 N-input vector maps. Emits a SIMD loop
/// with stride 2 over the bulk + a scalar remainder loop for the tail.
pub(crate) fn compile_vector_n_simd_map(
    body_ir: &IrFunc,
    n_in: usize,
    fn_name: &str,
    kind: r2_types::JitKind,
) -> JitResult<CompiledFn> {
    if !body_is_simd_clean(body_ir) {
        return Err(JitError::Unsupported("body is not SIMD-clean".into()));
    }
    let blk = &body_ir.blocks[0];
    let ret_reg = match &blk.term {
        IrTerm::Return(Some(r)) => *r,
        _ => return Err(JitError::Unsupported("simd: body must end with Return".into())),
    };

    let jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
    let mut module = JITModule::new(jit_builder);

    // Signature: (in_ptr_1..N, out_ptr, len) all as i64.
    let mut sig = module.make_signature();
    for _ in 0..n_in { sig.params.push(AbiParam::new(types::I64)); }
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));

    let func_id = module
        .declare_function(fn_name, Linkage::Export, &sig)
        .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let entry      = bcx.create_block();
        let simd_hdr   = bcx.create_block(); // i is block param
        let simd_body  = bcx.create_block();
        let rem_hdr    = bcx.create_block(); // i is block param
        let rem_body   = bcx.create_block();
        let exit       = bcx.create_block();

        bcx.append_block_param(simd_hdr, types::I64);
        bcx.append_block_param(rem_hdr,  types::I64);

        // ── Entry: pull args, compute simd_end = len & ~1, jump to simd_hdr(0).
        // Constants are re-created per-block to satisfy Cranelift's strict
        // SSA dominance verifier (using `entry`'s constants in `simd_body`
        // fails because `entry` doesn't directly dominate `simd_body`).
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        let entry_params: Vec<Value> = bcx.block_params(entry).to_vec();
        let in_ptrs: Vec<Value> = entry_params[..n_in].to_vec();
        let out_ptr = entry_params[n_in];
        let len     = entry_params[n_in + 1];
        let one_e = bcx.ins().iconst(types::I64, 1);
        let not_one = bcx.ins().bnot(one_e);
        let simd_end = bcx.ins().band(len, not_one);
        let zero_e = bcx.ins().iconst(types::I64, 0);
        bcx.ins().jump(simd_hdr, &[zero_e]);

        // ── SIMD header: while i < simd_end, process 2 elements per iter.
        bcx.switch_to_block(simd_hdr);
        let i_sh = bcx.block_params(simd_hdr)[0];
        let cond = bcx.ins().icmp(IntCC::SignedLessThan, i_sh, simd_end);
        bcx.ins().brif(cond, simd_body, &[], rem_hdr, &[i_sh]);

        // ── SIMD body: load f64x2, run body, store f64x2.
        bcx.switch_to_block(simd_body);
        let i_sb = i_sh;
        let eight_b = bcx.ins().iconst(types::I64, 8);
        let off_bytes = bcx.ins().imul(i_sb, eight_b);
        let mflags = MemFlags::trusted();
        let mut env: HashMap<u32, Value> = HashMap::new();
        for (k, p) in in_ptrs.iter().enumerate() {
            let addr = bcx.ins().iadd(*p, off_bytes);
            let v = bcx.ins().load(types::F64X2, mflags, addr, 0);
            env.insert(body_ir.params[k].2.0, v);
        }
        for inst in &blk.insts {
            let v = lower_inst_simd(&mut bcx, inst, &env)?;
            env.insert(inst.dst().0, v);
        }
        let result = *env.get(&ret_reg.0).ok_or(JitError::UndefinedVReg(ret_reg))?;
        let out_addr = bcx.ins().iadd(out_ptr, off_bytes);
        bcx.ins().store(mflags, result, out_addr, 0);
        let two_b = bcx.ins().iconst(types::I64, 2);
        let next = bcx.ins().iadd(i_sb, two_b);
        bcx.ins().jump(simd_hdr, &[next]);

        // ── Remainder header: while i < len, process 1 element per iter.
        bcx.switch_to_block(rem_hdr);
        let i_rh = bcx.block_params(rem_hdr)[0];
        let cond = bcx.ins().icmp(IntCC::SignedLessThan, i_rh, len);
        bcx.ins().brif(cond, rem_body, &[], exit, &[]);

        // ── Remainder body: same lowering as SIMD body but scalar.
        bcx.switch_to_block(rem_body);
        let i_rb = i_rh;
        let eight_r = bcx.ins().iconst(types::I64, 8);
        let off_bytes = bcx.ins().imul(i_rb, eight_r);
        let mut env_s: HashMap<u32, Value> = HashMap::new();
        for (k, p) in in_ptrs.iter().enumerate() {
            let addr = bcx.ins().iadd(*p, off_bytes);
            let v = bcx.ins().load(types::F64, mflags, addr, 0);
            env_s.insert(body_ir.params[k].2.0, v);
        }
        for inst in &blk.insts {
            let v = lower_inst(&mut bcx, inst, &env_s, None)?;
            env_s.insert(inst.dst().0, v);
        }
        let result_s = *env_s.get(&ret_reg.0).ok_or(JitError::UndefinedVReg(ret_reg))?;
        let out_addr = bcx.ins().iadd(out_ptr, off_bytes);
        bcx.ins().store(mflags, result_s, out_addr, 0);
        let one_b = bcx.ins().iconst(types::I64, 1);
        let next = bcx.ins().iadd(i_rb, one_b);
        bcx.ins().jump(rem_hdr, &[next]);

        // ── Exit.
        bcx.switch_to_block(exit);
        bcx.ins().return_(&[]);

        bcx.seal_all_blocks();
        bcx.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| JitError::CraneliftError(format!("define: {:?}", e)))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::CraneliftError(format!("finalize: {:?}", e)))?;

    let ptr = module.get_finalized_function(func_id);
    Ok(CompiledFn { ptr, arity: n_in, kind, _module: module })
}

/// Shared codegen for 1-arg and 3-arg branchy element-wise vector maps.
/// Lowers an arbitrary-multi-block IR body inside a per-element row loop.
/// `n_in` is the number of input vector pointers; output is one f64 vector.
pub(crate) fn compile_vector_n_map_generic(
    body_ir: &IrFunc,
    n_in: usize,
    fn_name: &str,
    kind: r2_types::JitKind,
) -> JitResult<CompiledFn> {
    if body_ir.blocks.is_empty() {
        return Err(JitError::Unsupported("empty IR body".into()));
    }

    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
    register_math_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);
    let math_ids = declare_math_imports(&mut module)?;

    // Signature: (in_ptr_1, ..., in_ptr_N, out_ptr, len) all as i64.
    let mut sig = module.make_signature();
    for _ in 0..n_in { sig.params.push(AbiParam::new(types::I64)); }
    sig.params.push(AbiParam::new(types::I64)); // out_ptr
    sig.params.push(AbiParam::new(types::I64)); // len

    let func_id = module
        .declare_function(fn_name, Linkage::Export, &sig)
        .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);

        // Per-function FuncRefs for the math externs.
        let math_refs: HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef> =
            math_ids.iter()
                .map(|(k, id)| (*k, module.declare_func_in_func(*id, &mut bcx.func)))
                .collect();
        let math_refs_opt: MathRefs<'_> = Some(&math_refs);

        // Outer scaffold blocks.
        let entry  = bcx.create_block();
        let header = bcx.create_block();   // i is block param
        let load_b = bcx.create_block();   // i is block param
        let tail   = bcx.create_block();   // (i, result) are block params
        let exit   = bcx.create_block();

        bcx.append_block_param(header, types::I64);
        bcx.append_block_param(load_b, types::I64);
        bcx.append_block_param(tail,   types::I64);
        bcx.append_block_param(tail,   types::F64);

        // Pre-create one Cranelift block per IR block, with `i: i64` as first
        // block param, then F64 block params for each leading Phi, then —
        // for the IR entry block only — F64 block params for the loaded
        // input elements (one per IR formal param).
        let mut block_map: HashMap<u32, Block> = HashMap::new();
        let mut phi_info: HashMap<u32, PhiInfo> = HashMap::new();
        for blk in &body_ir.blocks {
            let cl = bcx.create_block();
            block_map.insert(blk.id.0, cl);
            // First param of every IR block: row index `i`.
            bcx.append_block_param(cl, types::I64);
            // Then F64 params for leading Phis.
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
        // The IR entry block gets N extra F64 block params (the loaded elements).
        let ir_entry_cl = *block_map.get(&body_ir.entry.0)
            .ok_or(JitError::UndefinedBlock(body_ir.entry))?;
        for _ in 0..n_in { bcx.append_block_param(ir_entry_cl, types::F64); }

        // ── Entry: collect function args, jump to header with i=0 ────────
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        let entry_params: Vec<Value> = bcx.block_params(entry).to_vec();
        let in_ptrs: Vec<Value> = entry_params[..n_in].to_vec();
        let out_ptr = entry_params[n_in];
        let len     = entry_params[n_in + 1];
        let zero_i = bcx.ins().iconst(types::I64, 0);
        bcx.ins().jump(header, &[zero_i]);

        // ── Header: while i < len ───────────────────────────────────────
        bcx.switch_to_block(header);
        let i_h = bcx.block_params(header)[0];
        let lt = bcx.ins().icmp(IntCC::SignedLessThan, i_h, len);
        bcx.ins().brif(lt, load_b, &[i_h], exit, &[]);

        // ── load_b: read in_ptrs[j][i] for j in 0..N, jump to IR entry ──
        bcx.switch_to_block(load_b);
        let i_l = bcx.block_params(load_b)[0];
        let eight = bcx.ins().iconst(types::I64, 8);
        let off = bcx.ins().imul(i_l, eight);
        let mflags = MemFlags::trusted();
        let mut loaded: Vec<Value> = Vec::with_capacity(n_in);
        for p in &in_ptrs {
            let addr = bcx.ins().iadd(*p, off);
            loaded.push(bcx.ins().load(types::F64, mflags, addr, 0));
        }
        // Jump to IR entry: args = [i, ...phi_args (none for entry), ...loaded]
        let mut entry_args: Vec<Value> = vec![i_l];
        // entry has no incoming phi sources (it's the IR entry), so phi_args empty.
        entry_args.extend(loaded.iter().copied());
        bcx.ins().jump(ir_entry_cl, &entry_args);

        // env: VReg -> Cranelift Value. Shared across IR blocks so defs from
        // a dominator block (e.g. entry) are visible in dominated blocks
        // (e.g. then/else branches). Cranelift's SSA verifier enforces real
        // dominance; we only need env for name-to-Value lookup.
        let mut env: HashMap<u32, Value> = HashMap::new();

        // ── Lower each IR block ─────────────────────────────────────────
        for blk in &body_ir.blocks {
            let cl = block_map[&blk.id.0];
            bcx.switch_to_block(cl);

            // Block param layout: [i: i64, ...phi_dsts: f64, [entry-only: ...loaded: f64]]
            let cl_params: Vec<Value> = bcx.block_params(cl).to_vec();
            let i_here = cl_params[0];
            let phi_count = phi_info[&blk.id.0].dst_regs.len();
            for (k, dst) in phi_info[&blk.id.0].dst_regs.iter().enumerate() {
                env.insert(dst.0, cl_params[1 + k]);
            }
            if blk.id == body_ir.entry {
                // Bind IR formal params to the loaded elements.
                for (k, (_, _, vreg)) in body_ir.params.iter().enumerate() {
                    env.insert(vreg.0, cl_params[1 + phi_count + k]);
                }
            }

            // Lower instructions (skip leading Phis — already bound).
            for inst in blk.insts.iter().skip(phi_count) {
                let v = lower_inst(&mut bcx, inst, &env, math_refs_opt)?;
                env.insert(inst.dst().0, v);
            }

            // Lower terminator. Threads `i_here` as the first arg of every
            // outgoing edge into other IR blocks; Return jumps to `tail`.
            match &blk.term {
                IrTerm::Return(Some(reg)) => {
                    let result = *env.get(&reg.0).ok_or(JitError::UndefinedVReg(*reg))?;
                    bcx.ins().jump(tail, &[i_here, result]);
                }
                IrTerm::Return(None) => {
                    let nan = bcx.ins().f64const(f64::NAN);
                    bcx.ins().jump(tail, &[i_here, nan]);
                }
                IrTerm::Jump(target) => {
                    let target_cl = *block_map.get(&target.0)
                        .ok_or(JitError::UndefinedBlock(*target))?;
                    let mut args = vec![i_here];
                    args.extend(phi_args(&blk.id, target, &phi_info, &env)?);
                    bcx.ins().jump(target_cl, &args);
                }
                IrTerm::Branch { cond, then_blk, else_blk } => {
                    let c = *env.get(&cond.0).ok_or(JitError::UndefinedVReg(*cond))?;
                    let zero = bcx.ins().f64const(0.0);
                    let cond_b = bcx.ins().fcmp(FloatCC::NotEqual, c, zero);
                    let then_cl = *block_map.get(&then_blk.0)
                        .ok_or(JitError::UndefinedBlock(*then_blk))?;
                    let else_cl = *block_map.get(&else_blk.0)
                        .ok_or(JitError::UndefinedBlock(*else_blk))?;
                    let mut then_args = vec![i_here];
                    then_args.extend(phi_args(&blk.id, then_blk, &phi_info, &env)?);
                    let mut else_args = vec![i_here];
                    else_args.extend(phi_args(&blk.id, else_blk, &phi_info, &env)?);
                    bcx.ins().brif(cond_b, then_cl, &then_args, else_cl, &else_args);
                }
                IrTerm::Unreachable => {
                    bcx.ins().trap(TrapCode::UnreachableCodeReached);
                }
            }
        }

        // ── Tail: store result, increment i, jump back to header ────────
        bcx.switch_to_block(tail);
        let i_t = bcx.block_params(tail)[0];
        let result = bcx.block_params(tail)[1];
        let off_t = bcx.ins().imul(i_t, eight);
        let out_addr = bcx.ins().iadd(out_ptr, off_t);
        bcx.ins().store(mflags, result, out_addr, 0);
        let one = bcx.ins().iconst(types::I64, 1);
        let next = bcx.ins().iadd(i_t, one);
        bcx.ins().jump(header, &[next]);

        // ── Exit: return ────────────────────────────────────────────────
        bcx.switch_to_block(exit);
        bcx.ins().return_(&[]);

        bcx.seal_all_blocks();
        bcx.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| JitError::CraneliftError(format!("define: {:?}", e)))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::CraneliftError(format!("finalize: {:?}", e)))?;

    let arity = n_in;
    let ptr = module.get_finalized_function(func_id);
    Ok(CompiledFn { ptr, arity, kind, _module: module })
}

// ── Phase J.4 brick 2 — multi-reduction scalar kernels ───────────────
//
// Compiles a scalar-returning function whose body *combines several whole-
// vector reductions* — the shape the single-reduction paths can't express:
//   function(x, y) sum(x*y) / sum(x*x)                 (regression coefficient)
//   function(x)    { m <- mean(x); sum((x-m)*(x-m)) }  (centred sum / variance)
//   function(x, y) { mx<-mean(x); my<-mean(y); sum((x-mx)*(y-my)) } (covariance)
//
// Each `sum`/`prod`/`mean` emits its own fused loop (no intermediate vector is
// ever materialised); scalar locals thread between the loops as plain SSA
// values. Reuses the `Vector1/2ToScalar` ABI (`(ptr[,ptr],len)->f64`) and the
// existing engine dispatch — no new ABI surface.

struct Kctx<'a> {
    x_ptr: Value,
    y_ptr: Option<Value>,
    len: Value,   // i64 element count
    len_f: Value, // f64 element count (for mean / length)
    names: &'a [std::sync::Arc<str>],
    math: &'a HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef>,
}

/// Emit a pure element-wise expression (inside a reduction loop). `env` maps the
/// vector param names → the loaded element value, and any scalar locals → their
/// (loop-invariant) values. No reductions here — those live at the scalar level.
fn emit_elem(bcx: &mut FunctionBuilder, k: &Kctx, e: &r2_types::Expr, env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Value> {
    use r2_types::Expr::*;
    match e {
        NumLit(x) => Ok(bcx.ins().f64const(*x)),
        IntLit(x) => Ok(bcx.ins().f64const(*x as f64)),
        BoolLit(b) => Ok(bcx.ins().f64const(if *b { 1.0 } else { 0.0 })),
        Symbol(s) => env.get(s).copied()
            .ok_or_else(|| JitError::Unsupported(format!("element expr references non-vector `{}`", s))),
        Unary { op, expr } => {
            let v = emit_elem(bcx, k, expr, env)?;
            Ok(match op {
                r2_types::UnOp::Neg => bcx.ins().fneg(v),
                r2_types::UnOp::Pos => v,
                r2_types::UnOp::Not => { let z = bcx.ins().f64const(0.0); cmp_to_f64(bcx, v, z, FloatCC::Equal) }
            })
        }
        Binary { op, lhs, rhs } => {
            let a = emit_elem(bcx, k, lhs, env)?;
            let b = emit_elem(bcx, k, rhs, env)?;
            emit_binop(bcx, k, *op, a, b)
        }
        Call { func, args } => {
            let fname = match func.as_ref() { Symbol(s) => s.clone(), _ => return Err(JitError::Unsupported("call target".into())) };
            let mut av = Vec::with_capacity(args.len());
            for a in args { av.push(emit_elem(bcx, k, &a.value, env)?); }
            emit_math_call(bcx, k, fname.as_ref(), &av)
        }
        If { cond, then, else_ } => {
            let e2 = else_.as_ref().ok_or_else(|| JitError::Unsupported("if without else in element expr".into()))?;
            let c = emit_elem(bcx, k, cond, env)?;
            let t = emit_elem(bcx, k, then, env)?;
            let f = emit_elem(bcx, k, e2, env)?;
            let z = bcx.ins().f64const(0.0);
            let nz = bcx.ins().fcmp(FloatCC::NotEqual, c, z);
            Ok(bcx.ins().select(nz, t, f))
        }
        _ => Err(JitError::Unsupported("unsupported node in element expr".into())),
    }
}

fn emit_binop(bcx: &mut FunctionBuilder, k: &Kctx, op: BinOp, a: Value, b: Value) -> JitResult<Value> {
    Ok(match op {
        BinOp::Add => bcx.ins().fadd(a, b),
        BinOp::Sub => bcx.ins().fsub(a, b),
        BinOp::Mul => bcx.ins().fmul(a, b),
        BinOp::Div => bcx.ins().fdiv(a, b),
        BinOp::Lt => cmp_to_f64(bcx, a, b, FloatCC::LessThan),
        BinOp::Gt => cmp_to_f64(bcx, a, b, FloatCC::GreaterThan),
        BinOp::Le => cmp_to_f64(bcx, a, b, FloatCC::LessThanOrEqual),
        BinOp::Ge => cmp_to_f64(bcx, a, b, FloatCC::GreaterThanOrEqual),
        BinOp::Eq => cmp_to_f64(bcx, a, b, FloatCC::Equal),
        BinOp::Ne => cmp_to_f64(bcx, a, b, FloatCC::NotEqual),
        BinOp::Pow => {
            let f = k.math.get("^").ok_or_else(|| JitError::CraneliftError("pow extern missing".into()))?;
            let call = bcx.ins().call(*f, &[a, b]);
            bcx.inst_results(call)[0]
        }
        other => return Err(JitError::Unsupported(format!("binop {:?}", other))),
    })
}

fn emit_math_call(bcx: &mut FunctionBuilder, k: &Kctx, name: &str, av: &[Value]) -> JitResult<Value> {
    match (name, av.len()) {
        ("sqrt", 1) => return Ok(bcx.ins().sqrt(av[0])),
        ("abs", 1) => return Ok(bcx.ins().fabs(av[0])),
        ("floor", 1) => return Ok(bcx.ins().floor(av[0])),
        ("ceil", 1) => return Ok(bcx.ins().ceil(av[0])),
        ("trunc", 1) => return Ok(bcx.ins().trunc(av[0])),
        ("round", 1) => return Ok(bcx.ins().nearest(av[0])),
        ("min", 2) => return Ok(bcx.ins().fmin(av[0], av[1])),
        ("max", 2) => return Ok(bcx.ins().fmax(av[0], av[1])),
        _ => {}
    }
    let me = find_math_extern(name).ok_or_else(|| JitError::Unsupported(format!("call `{}` unsupported", name)))?;
    if me.arity != av.len() { return Err(JitError::Unsupported(format!("`{}` arity", name))); }
    let f = k.math.get(me.r_name).ok_or_else(|| JitError::CraneliftError(format!("extern `{}` missing", name)))?;
    let call = bcx.ins().call(*f, av);
    Ok(bcx.inst_results(call)[0])
}

/// Emit a fused reduction loop: `acc = identity; for i in 0..len { acc = acc
/// <op> elem(i) }`. `scalar_env` provides loop-invariant locals visible in the
/// element expression. Returns the accumulator value.
fn emit_reduction(bcx: &mut FunctionBuilder, k: &Kctx, elem: &r2_types::Expr, op: FusedReduceOp, scalar_env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Value> {
    let header = bcx.create_block();
    let body = bcx.create_block();
    let exit = bcx.create_block();
    bcx.append_block_param(header, types::I64);
    bcx.append_block_param(header, types::F64);
    bcx.append_block_param(body, types::I64);
    bcx.append_block_param(body, types::F64);
    bcx.append_block_param(exit, types::F64);

    let identity = match op { FusedReduceOp::Sum => 0.0, FusedReduceOp::Prod => 1.0 };
    let zero_i = bcx.ins().iconst(types::I64, 0);
    let id_v = bcx.ins().f64const(identity);
    bcx.ins().jump(header, &[zero_i, id_v]);

    bcx.switch_to_block(header);
    let i_h = bcx.block_params(header)[0];
    let acc_h = bcx.block_params(header)[1];
    let lt = bcx.ins().icmp(IntCC::SignedLessThan, i_h, k.len);
    bcx.ins().brif(lt, body, &[i_h, acc_h], exit, &[acc_h]);

    bcx.switch_to_block(body);
    let i_b = bcx.block_params(body)[0];
    let acc_b = bcx.block_params(body)[1];
    let eight = bcx.ins().iconst(types::I64, 8);
    let off = bcx.ins().imul(i_b, eight);
    let mf = MemFlags::trusted();
    let mut env = scalar_env.clone();
    let xa = bcx.ins().iadd(k.x_ptr, off);
    let xv = bcx.ins().load(types::F64, mf, xa, 0);
    env.insert(k.names[0].clone(), xv);
    if let Some(yp) = k.y_ptr {
        let ya = bcx.ins().iadd(yp, off);
        let yv = bcx.ins().load(types::F64, mf, ya, 0);
        if k.names.len() > 1 { env.insert(k.names[1].clone(), yv); }
    }
    let e = emit_elem(bcx, k, elem, &env)?;
    let new_acc = match op { FusedReduceOp::Sum => bcx.ins().fadd(acc_b, e), FusedReduceOp::Prod => bcx.ins().fmul(acc_b, e) };
    let one = bcx.ins().iconst(types::I64, 1);
    let i_next = bcx.ins().iadd(i_b, one);
    bcx.ins().jump(header, &[i_next, new_acc]);

    bcx.switch_to_block(exit);
    Ok(bcx.block_params(exit)[0])
}

/// Phase J.4 brick 4 — emit a *single* loop that computes several reductions at
/// once (one accumulator each), loading each vector element just once. The
/// reductions' element expressions typically share sub-terms (e.g. `x[i]-mx`);
/// Cranelift's value numbering folds those to one computation. Matches what a
/// hand-written native `cor`/`cov` does: one pass for `sxy`,`sxx`,`syy`.
fn emit_reduction_wave(bcx: &mut FunctionBuilder, k: &Kctx, reds: &[(&r2_types::Expr, FusedReduceOp)], scalar_env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Vec<Value>> {
    let n = reds.len();
    let header = bcx.create_block();
    let body = bcx.create_block();
    let exit = bcx.create_block();
    bcx.append_block_param(header, types::I64);
    for _ in 0..n { bcx.append_block_param(header, types::F64); }
    bcx.append_block_param(body, types::I64);
    for _ in 0..n { bcx.append_block_param(body, types::F64); }
    for _ in 0..n { bcx.append_block_param(exit, types::F64); }

    let mut init_args = Vec::with_capacity(n + 1);
    init_args.push(bcx.ins().iconst(types::I64, 0));
    for (_, op) in reds { init_args.push(bcx.ins().f64const(match op { FusedReduceOp::Sum => 0.0, FusedReduceOp::Prod => 1.0 })); }
    bcx.ins().jump(header, &init_args);

    bcx.switch_to_block(header);
    let hp: Vec<Value> = bcx.block_params(header).to_vec();
    let i_h = hp[0];
    let lt = bcx.ins().icmp(IntCC::SignedLessThan, i_h, k.len);
    let then_args: Vec<Value> = hp.clone();          // (i, accs...)
    let else_args: Vec<Value> = hp[1..].to_vec();    // (accs...)
    bcx.ins().brif(lt, body, &then_args, exit, &else_args);

    bcx.switch_to_block(body);
    let bp: Vec<Value> = bcx.block_params(body).to_vec();
    let i_b = bp[0];
    let eight = bcx.ins().iconst(types::I64, 8);
    let off = bcx.ins().imul(i_b, eight);
    let mf = MemFlags::trusted();
    let mut env = scalar_env.clone();
    let xa = bcx.ins().iadd(k.x_ptr, off);
    let xv = bcx.ins().load(types::F64, mf, xa, 0);
    env.insert(k.names[0].clone(), xv);
    if let Some(yp) = k.y_ptr {
        let ya = bcx.ins().iadd(yp, off);
        let yv = bcx.ins().load(types::F64, mf, ya, 0);
        if k.names.len() > 1 { env.insert(k.names[1].clone(), yv); }
    }
    let mut next = Vec::with_capacity(n + 1);
    let one = bcx.ins().iconst(types::I64, 1);
    next.push(bcx.ins().iadd(i_b, one));
    for (j, (elem, op)) in reds.iter().enumerate() {
        let e = emit_elem(bcx, k, elem, &env)?;
        let acc = bp[1 + j];
        next.push(match op { FusedReduceOp::Sum => bcx.ins().fadd(acc, e), FusedReduceOp::Prod => bcx.ins().fmul(acc, e) });
    }
    bcx.ins().jump(header, &next);

    bcx.switch_to_block(exit);
    Ok(bcx.block_params(exit).to_vec())
}

/// Emit a scalar expression whose leaves are reductions, scalar locals,
/// literals, and scalar math — combined by arithmetic.
fn emit_scalar(bcx: &mut FunctionBuilder, k: &Kctx, e: &r2_types::Expr, scalar_env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Value> {
    use r2_types::Expr::*;
    let is_vec = |s: &str| k.names.iter().any(|n| n.as_ref() == s);
    match e {
        NumLit(x) => Ok(bcx.ins().f64const(*x)),
        IntLit(x) => Ok(bcx.ins().f64const(*x as f64)),
        BoolLit(b) => Ok(bcx.ins().f64const(if *b { 1.0 } else { 0.0 })),
        Symbol(s) => {
            if is_vec(s) { return Err(JitError::Unsupported(format!("bare vector `{}` used as scalar", s))); }
            scalar_env.get(s).copied().ok_or_else(|| JitError::Unsupported(format!("unknown scalar `{}`", s)))
        }
        Unary { op, expr } => {
            let v = emit_scalar(bcx, k, expr, scalar_env)?;
            Ok(match op { r2_types::UnOp::Neg => bcx.ins().fneg(v), r2_types::UnOp::Pos => v,
                r2_types::UnOp::Not => { let z = bcx.ins().f64const(0.0); cmp_to_f64(bcx, v, z, FloatCC::Equal) } })
        }
        Binary { op, lhs, rhs } => {
            let a = emit_scalar(bcx, k, lhs, scalar_env)?;
            let b = emit_scalar(bcx, k, rhs, scalar_env)?;
            emit_binop(bcx, k, *op, a, b)
        }
        Call { func, args } => {
            let fname = match func.as_ref() { Symbol(s) => s.clone(), _ => return Err(JitError::Unsupported("call target".into())) };
            match (fname.as_ref(), args.len()) {
                ("sum", 1) => emit_reduction(bcx, k, &args[0].value, FusedReduceOp::Sum, scalar_env),
                ("prod", 1) => emit_reduction(bcx, k, &args[0].value, FusedReduceOp::Prod, scalar_env),
                ("mean", 1) => {
                    let s = emit_reduction(bcx, k, &args[0].value, FusedReduceOp::Sum, scalar_env)?;
                    Ok(bcx.ins().fdiv(s, k.len_f))
                }
                ("length", 1) => match &args[0].value {
                    Symbol(v) if is_vec(v) => Ok(k.len_f),
                    _ => Err(JitError::Unsupported("length of non-vector".into())),
                },
                _ => {
                    let mut av = Vec::with_capacity(args.len());
                    for a in args { av.push(emit_scalar(bcx, k, &a.value, scalar_env)?); }
                    emit_math_call(bcx, k, fname.as_ref(), &av)
                }
            }
        }
        _ => Err(JitError::Unsupported("unsupported node in scalar body".into())),
    }
}

/// Does `e` reference any symbol in `names`? Used to decide whether a reduction
/// can join the current fused wave (it cannot if its element expr depends on a
/// scalar local being produced by another reduction in the same wave).
fn expr_refs_any(e: &r2_types::Expr, names: &std::collections::HashSet<std::sync::Arc<str>>) -> bool {
    use r2_types::Expr::*;
    match e {
        Symbol(s) => names.contains(s),
        Unary { expr, .. } => expr_refs_any(expr, names),
        Binary { lhs, rhs, .. } => expr_refs_any(lhs, names) || expr_refs_any(rhs, names),
        Call { func, args } => expr_refs_any(func, names) || args.iter().any(|a| expr_refs_any(&a.value, names)),
        If { cond, then, else_ } => expr_refs_any(cond, names) || expr_refs_any(then, names)
            || else_.as_ref().map_or(false, |x| expr_refs_any(x, names)),
        _ => false,
    }
}

/// Emit the pending reduction batch as one fused loop and bind each result
/// (dividing by `len` for `mean`) into `scalar_env`; then clear the batch.
fn flush_wave(
    bcx: &mut FunctionBuilder, k: &Kctx,
    batch: &mut Vec<(std::sync::Arc<str>, &r2_types::Expr, FusedReduceOp, bool)>,
    batch_names: &mut std::collections::HashSet<std::sync::Arc<str>>,
    scalar_env: &mut HashMap<std::sync::Arc<str>, Value>,
) -> JitResult<()> {
    if batch.is_empty() { return Ok(()); }
    let reds: Vec<(&r2_types::Expr, FusedReduceOp)> = batch.iter().map(|(_, e, op, _)| (*e, *op)).collect();
    let vals = emit_reduction_wave(bcx, k, &reds, scalar_env)?;
    for ((name, _, _, is_mean), v) in batch.iter().zip(vals.into_iter()) {
        let bound = if *is_mean { bcx.ins().fdiv(v, k.len_f) } else { v };
        scalar_env.insert(name.clone(), bound);
    }
    batch.clear();
    batch_names.clear();
    Ok(())
}

/// Compile a multi-reduction scalar kernel over 1–2 vector params. `body` is a
/// scalar expression, or a `Block` of `local <- <scalar>` assignments followed
/// by a final scalar expression.
pub(crate) fn compile_reduction_kernel(body: &r2_types::Expr, param_names: &[std::sync::Arc<str>]) -> JitResult<CompiledFn> {
    let n_vec = param_names.len();
    if !(1..=2).contains(&n_vec) { return Err(JitError::Unsupported("reduction kernel needs 1-2 params".into())); }

    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
    register_math_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);
    let math_ids = declare_math_imports(&mut module)?;

    let mut sig = module.make_signature();
    for _ in 0..n_vec { sig.params.push(AbiParam::new(types::I64)); }
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::F64));

    let func_id = module.declare_function("__jit_reduction_kernel", Linkage::Export, &sig)
        .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let math: HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef> =
            math_ids.iter().map(|(kk, id)| (*kk, module.declare_func_in_func(*id, &mut bcx.func))).collect();

        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        let ps: Vec<Value> = bcx.block_params(entry).to_vec();
        let x_ptr = ps[0];
        let y_ptr = if n_vec == 2 { Some(ps[1]) } else { None };
        let len = ps[n_vec];
        let len_f = bcx.ins().fcvt_from_sint(types::F64, len);
        let k = Kctx { x_ptr, y_ptr, len, len_f, names: param_names, math: &math };

        let mut scalar_env: HashMap<std::sync::Arc<str>, Value> = HashMap::new();
        let result = match body {
            r2_types::Expr::Block(stmts) => {
                if stmts.is_empty() { return Err(JitError::Unsupported("empty body".into())); }
                let (last, init) = stmts.split_last().unwrap();
                // Batch independent reductions into single fused passes (brick 4).
                // `batch` holds (name, element-expr, op, is_mean); flushed when the
                // next reduction depends on a batch member or a non-reduction stmt
                // / the final expr is reached.
                let mut batch: Vec<(std::sync::Arc<str>, &r2_types::Expr, FusedReduceOp, bool)> = Vec::new();
                let mut batch_names: std::collections::HashSet<std::sync::Arc<str>> = std::collections::HashSet::new();

                for st in init {
                    let (nm, value) = match st {
                        r2_types::Expr::Assign { target, value, .. } => match target.as_ref() {
                            r2_types::Expr::Symbol(n) => (n.clone(), value.as_ref()),
                            _ => return Err(JitError::Unsupported("non-symbol assign in kernel".into())),
                        },
                        _ => return Err(JitError::Unsupported("non-assign statement before final expr".into())),
                    };
                    // Is this assignment a single-vector reduction?
                    let red: Option<(&r2_types::Expr, FusedReduceOp, bool)> = match value {
                        r2_types::Expr::Call { func, args } if args.len() == 1 => match func.as_ref() {
                            r2_types::Expr::Symbol(s) if s.as_ref() == "sum" => Some((&args[0].value, FusedReduceOp::Sum, false)),
                            r2_types::Expr::Symbol(s) if s.as_ref() == "prod" => Some((&args[0].value, FusedReduceOp::Prod, false)),
                            r2_types::Expr::Symbol(s) if s.as_ref() == "mean" => Some((&args[0].value, FusedReduceOp::Sum, true)),
                            _ => None,
                        },
                        _ => None,
                    };
                    match red {
                        Some((elem, op, is_mean)) => {
                            if expr_refs_any(elem, &batch_names) {
                                flush_wave(&mut bcx, &k, &mut batch, &mut batch_names, &mut scalar_env)?;
                            }
                            batch_names.insert(nm.clone());
                            batch.push((nm, elem, op, is_mean));
                        }
                        None => {
                            flush_wave(&mut bcx, &k, &mut batch, &mut batch_names, &mut scalar_env)?;
                            let v = emit_scalar(&mut bcx, &k, value, &scalar_env)?;
                            scalar_env.insert(nm, v);
                        }
                    }
                }
                flush_wave(&mut bcx, &k, &mut batch, &mut batch_names, &mut scalar_env)?;
                emit_scalar(&mut bcx, &k, last, &scalar_env)?
            }
            other => emit_scalar(&mut bcx, &k, other, &scalar_env)?,
        };
        bcx.ins().return_(&[result]);
        bcx.seal_all_blocks();
        bcx.finalize();
    }
    module.define_function(func_id, &mut ctx).map_err(|e| JitError::CraneliftError(format!("define: {:?}", e)))?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| JitError::CraneliftError(format!("finalize: {:?}", e)))?;
    let ptr = module.get_finalized_function(func_id);
    let kind = if n_vec == 1 { r2_types::JitKind::Vector1ToScalar } else { r2_types::JitKind::Vector2ToScalar };
    Ok(CompiledFn { ptr, arity: n_vec, kind, _module: module })
}
