//! IR-based fused map-reduce builders: scalar and F64X2 (4x-unrolled) sum/prod reductions over 1-2 input vectors. Consumes typed IR via lower_inst / lower_inst_simd.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use r2_ir::{IrFunc, IrInst, IrTerm, VReg};
use std::collections::HashMap;
use crate::*;

/// SIMD (F64X2) fused map-reduce over `n_in` (1 or 2) input vectors:
/// `reduce(f(x[i][, y[i]]))` → scalar. A 2-elements-per-iteration main loop
/// accumulates into an F64X2 vector, horizontally reduces to a scalar, then a
/// scalar tail handles the odd element. Requires a SIMD-clean IR body (single
/// block, arithmetic + native-instruction math only); returns `Err` otherwise
/// so the caller uses the scalar loop. Signature `(in_ptr..N, len) -> f64`.
pub(crate) fn compile_simd_map_reduce_n(
    body_ir: &IrFunc,
    n_in: usize,
    reduce_op: FusedReduceOp,
    fn_name: &str,
) -> JitResult<CompiledFn> {
    if !body_is_simd_clean(body_ir) {
        return Err(JitError::Unsupported("body is not SIMD-clean".into()));
    }
    let blk = &body_ir.blocks[0];
    let ret_reg = match &blk.term {
        IrTerm::Return(Some(r)) => *r,
        _ => return Err(JitError::Unsupported("simd map-reduce: body must Return".into())),
    };

    let jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
    let mut module = JITModule::new(jit_builder);

    let mut sig = module.make_signature();
    for _ in 0..n_in { sig.params.push(AbiParam::new(types::I64)); }
    sig.params.push(AbiParam::new(types::I64)); // len
    sig.returns.push(AbiParam::new(types::F64));

    let func_id = module.declare_function(fn_name, Linkage::Export, &sig)
        .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let identity = match reduce_op { FusedReduceOp::Sum => 0.0, FusedReduceOp::Prod => 1.0 };
        // 4× unroll: 4 independent F64X2 accumulators break the reduction's
        // dependency chain (a single accumulator is fadd-latency-bound), so the
        // loop saturates the FP units — 8 elements per iteration.
        const U: usize = 4;

        let entry = bcx.create_block();
        let shdr = bcx.create_block();      // (i, acc0..acc3 : f64x2)
        let sbody = bcx.create_block();
        let sexit = bcx.create_block();     // (i, acc0..acc3) — combine + horizontal reduce
        let rhdr = bcx.create_block();      // (i, acc:f64)
        let rbody = bcx.create_block();
        let exit = bcx.create_block();      // (acc:f64)
        bcx.append_block_param(shdr, types::I64);
        for _ in 0..U { bcx.append_block_param(shdr, types::F64X2); }
        bcx.append_block_param(sexit, types::I64);
        for _ in 0..U { bcx.append_block_param(sexit, types::F64X2); }
        bcx.append_block_param(rhdr, types::I64);
        bcx.append_block_param(rhdr, types::F64);
        bcx.append_block_param(exit, types::F64);

        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        let ep: Vec<Value> = bcx.block_params(entry).to_vec();
        let in_ptrs: Vec<Value> = ep[..n_in].to_vec();
        let len = ep[n_in];
        let simd_end = bcx.ins().band_imm(len, -(2 * U as i64)); // multiple of 8
        let zero = bcx.ins().iconst(types::I64, 0);
        let id_s = bcx.ins().f64const(identity);
        let id_v = bcx.ins().splat(types::F64X2, id_s);
        let mut init = vec![zero]; for _ in 0..U { init.push(id_v); }
        bcx.ins().jump(shdr, &init);

        // SIMD header.
        bcx.switch_to_block(shdr);
        let hp: Vec<Value> = bcx.block_params(shdr).to_vec();
        let cond = bcx.ins().icmp(IntCC::SignedLessThan, hp[0], simd_end);
        bcx.ins().brif(cond, sbody, &[], sexit, &hp);

        // SIMD body: U independent F64X2 accumulators, 2*U elements per iter.
        bcx.switch_to_block(sbody);
        let sp: Vec<Value> = bcx.block_params(shdr).to_vec();
        let i_sb = sp[0];
        let eight = bcx.ins().iconst(types::I64, 8);
        let mf = MemFlags::trusted();
        let mut next = vec![]; // filled below (i_next first)
        let mut new_accs = Vec::with_capacity(U);
        for u in 0..U {
            let two_u = bcx.ins().iconst(types::I64, 2 * u as i64);
            let idx = bcx.ins().iadd(i_sb, two_u);
            let off = bcx.ins().imul(idx, eight);
            let mut env: HashMap<u32, Value> = HashMap::new();
            for (j, p) in in_ptrs.iter().enumerate() {
                let addr = bcx.ins().iadd(*p, off);
                env.insert(body_ir.params[j].2.0, bcx.ins().load(types::F64X2, mf, addr, 0));
            }
            for inst in &blk.insts { let v = lower_inst_simd(&mut bcx, inst, &env)?; env.insert(inst.dst().0, v); }
            let e = *env.get(&ret_reg.0).ok_or(JitError::UndefinedVReg(ret_reg))?;
            let acc = sp[1 + u];
            new_accs.push(match reduce_op { FusedReduceOp::Sum => bcx.ins().fadd(acc, e), FusedReduceOp::Prod => bcx.ins().fmul(acc, e) });
        }
        let stride = bcx.ins().iconst(types::I64, 2 * U as i64);
        let i_next = bcx.ins().iadd(i_sb, stride);
        next.push(i_next); next.extend(new_accs);
        bcx.ins().jump(shdr, &next);

        // SIMD exit: combine the U accumulators, then horizontal reduce to scalar.
        bcx.switch_to_block(sexit);
        let xp: Vec<Value> = bcx.block_params(sexit).to_vec();
        let mut comb = xp[1];
        for u in 1..U { comb = match reduce_op { FusedReduceOp::Sum => bcx.ins().fadd(comb, xp[1 + u]), FusedReduceOp::Prod => bcx.ins().fmul(comb, xp[1 + u]) }; }
        let l0 = bcx.ins().extractlane(comb, 0);
        let l1 = bcx.ins().extractlane(comb, 1);
        let hacc = match reduce_op { FusedReduceOp::Sum => bcx.ins().fadd(l0, l1), FusedReduceOp::Prod => bcx.ins().fmul(l0, l1) };
        bcx.ins().jump(rhdr, &[xp[0], hacc]);

        // Scalar remainder.
        bcx.switch_to_block(rhdr);
        let rp: Vec<Value> = bcx.block_params(rhdr).to_vec();
        let rcond = bcx.ins().icmp(IntCC::SignedLessThan, rp[0], len);
        bcx.ins().brif(rcond, rbody, &[], exit, &[rp[1]]);
        bcx.switch_to_block(rbody);
        let i_rb = bcx.block_params(rhdr)[0];
        let acc_rb = bcx.block_params(rhdr)[1];
        let eight_r = bcx.ins().iconst(types::I64, 8);
        let off_r = bcx.ins().imul(i_rb, eight_r);
        let mut env_s: HashMap<u32, Value> = HashMap::new();
        for (j, p) in in_ptrs.iter().enumerate() {
            let addr = bcx.ins().iadd(*p, off_r);
            env_s.insert(body_ir.params[j].2.0, bcx.ins().load(types::F64, mf, addr, 0));
        }
        for inst in &blk.insts { let v = lower_inst(&mut bcx, inst, &env_s, None)?; env_s.insert(inst.dst().0, v); }
        let e_s = *env_s.get(&ret_reg.0).ok_or(JitError::UndefinedVReg(ret_reg))?;
        let racc = match reduce_op { FusedReduceOp::Sum => bcx.ins().fadd(acc_rb, e_s), FusedReduceOp::Prod => bcx.ins().fmul(acc_rb, e_s) };
        let one = bcx.ins().iconst(types::I64, 1);
        let i_rn = bcx.ins().iadd(i_rb, one);
        bcx.ins().jump(rhdr, &[i_rn, racc]);

        bcx.switch_to_block(exit);
        let fin = bcx.block_params(exit)[0];
        bcx.ins().return_(&[fin]);
        bcx.seal_all_blocks();
        bcx.finalize();
    }
    module.define_function(func_id, &mut ctx).map_err(|e| JitError::CraneliftError(format!("define: {:?}", e)))?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| JitError::CraneliftError(format!("finalize: {:?}", e)))?;
    let ptr = module.get_finalized_function(func_id);
    let kind = if n_in == 1 { r2_types::JitKind::Vector1ToScalar } else { r2_types::JitKind::Vector2ToScalar };
    Ok(CompiledFn { ptr, arity: n_in, kind, _module: module })
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
    // SIMD fast-path for a SIMD-clean element body; scalar loop otherwise.
    if let Ok(c) = compile_simd_map_reduce_n(body_ir, 1, reduce_op, "__jit_simd_map_reduce") {
        return Ok(c);
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
    // SIMD fast-path (dot products `sum(x*w)` etc.); scalar loop otherwise.
    if let Ok(c) = compile_simd_map_reduce_n(body_ir, 2, reduce_op, "__jit_simd_binary_map_reduce") {
        return Ok(c);
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
