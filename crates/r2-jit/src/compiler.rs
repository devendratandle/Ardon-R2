//! `JitCompiler` — the public compile_* entry points.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use r2_ir::IrFunc;
use r2_types::infer::IrElem;
use std::collections::HashMap;
use crate::*;

// ── Compiler ─────────────────────────────────────────────────────────

pub struct JitCompiler;

impl JitCompiler {
    /// Compile an `IrFunc` whose params and return type are scalar Real/Int/Bool.
    pub fn compile(func: &IrFunc) -> JitResult<CompiledFn> {
        for (name, ty, _) in &func.params {
            if !is_scalar_numeric(&ty.elem) {
                return Err(JitError::Unsupported(format!(
                    "param '{}' has unsupported type {:?}", name, ty.elem
                )));
            }
        }
        if !is_scalar_numeric(&func.return_type.elem) && func.return_type.elem != IrElem::Unknown {
            return Err(JitError::Unsupported(format!("return type {:?}", func.return_type.elem)));
        }

        let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
        register_math_symbols(&mut jit_builder);
        let mut module = JITModule::new(jit_builder);
        let math_ids = declare_math_imports(&mut module)?;

        let mut sig = module.make_signature();
        for _ in &func.params { sig.params.push(AbiParam::new(types::F64)); }
        sig.returns.push(AbiParam::new(types::F64));

        let func_id = module
            .declare_function(func.name.as_ref(), Linkage::Export, &sig)
            .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;

        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        let mut fbctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
            // Materialize per-function FuncRefs from the math FuncIds.
            let math_refs: HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef> =
                math_ids.iter()
                    .map(|(k, id)| (*k, module.declare_func_in_func(*id, &mut bcx.func)))
                    .collect();
            lower_func_body(&mut bcx, func, Some(&math_refs))?;
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
        Ok(CompiledFn { ptr, arity: func.params.len(), kind: r2_types::JitKind::Scalar, _module: module })
    }

    /// Phase C.3: compile a vector reduction `(v) -> scalar`.
    /// `reduction` is one of "sum", "mean", "length", "prod".
    /// The compiled native fn has signature `(*const f64, i64) -> f64`,
    /// internally calling the corresponding R2 extern.
    pub fn compile_vector_reduction(reduction: &str) -> JitResult<CompiledFn> {
        let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;

        // Register the Rust externs under stable names so Cranelift can resolve them.
        jit_builder.symbol("__r2_sum",    r2_extern_sum    as *const u8);
        jit_builder.symbol("__r2_mean",   r2_extern_mean   as *const u8);
        jit_builder.symbol("__r2_length", r2_extern_length as *const u8);
        jit_builder.symbol("__r2_prod",   r2_extern_prod   as *const u8);

        let mut module = JITModule::new(jit_builder);

        // External function signature: (*const f64, i64) -> f64.
        let mut ext_sig = module.make_signature();
        ext_sig.params.push(AbiParam::new(types::I64));
        ext_sig.params.push(AbiParam::new(types::I64));
        ext_sig.returns.push(AbiParam::new(types::F64));

        let extern_name = match reduction {
            "sum"    => "__r2_sum",
            "mean"   => "__r2_mean",
            "length" => "__r2_length",
            "prod"   => "__r2_prod",
            other    => return Err(JitError::Unsupported(format!("reduction {:?}", other))),
        };
        let extern_id = module
            .declare_function(extern_name, Linkage::Import, &ext_sig)
            .map_err(|e| JitError::CraneliftError(format!("declare extern: {:?}", e)))?;

        // Our compiled wrapper has the same signature: (ptr, len) -> f64.
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::F64));

        let func_id = module
            .declare_function(&format!("__jit_vec_{}", reduction), Linkage::Export, &sig)
            .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;

        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        let mut fbctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
            let entry = bcx.create_block();
            bcx.append_block_params_for_function_params(entry);
            bcx.switch_to_block(entry);
            bcx.seal_block(entry);

            let extern_ref = module.declare_func_in_func(extern_id, &mut bcx.func);
            let params = bcx.block_params(entry).to_vec();
            let call = bcx.ins().call(extern_ref, &params);
            let result = bcx.inst_results(call)[0];
            bcx.ins().return_(&[result]);
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

    /// Phase C.4: compile an element-wise `(v) -> v_op_scalar` vector map.
    /// Generates a real native loop; no extern call.
    /// Body: `for i in 0..len: out[i] = in[i] OP scalar`.
    pub fn compile_vector_map_scalar_op(op: r2_types::BinOp, scalar: f64) -> JitResult<CompiledFn> {
        let supported = matches!(op, r2_types::BinOp::Add | r2_types::BinOp::Sub
                                    | r2_types::BinOp::Mul | r2_types::BinOp::Div);
        if !supported { return Err(JitError::Unsupported(format!("vector op {:?}", op))); }

        let jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
        let mut module = JITModule::new(jit_builder);

        // (in_ptr: i64, out_ptr: i64, len: i64) -> ()
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        // No return.

        let func_id = module
            .declare_function("__jit_vec_map", Linkage::Export, &sig)
            .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;

        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        let mut fbctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);

            let entry = bcx.create_block();
            let header = bcx.create_block();
            let body = bcx.create_block();
            let exit = bcx.create_block();
            // Loop counter `i` is a block parameter on `header`.
            bcx.append_block_param(header, types::I64);

            // Entry: pull args, jump to header with i=0.
            bcx.append_block_params_for_function_params(entry);
            bcx.switch_to_block(entry);
            let in_ptr  = bcx.block_params(entry)[0];
            let out_ptr = bcx.block_params(entry)[1];
            let len     = bcx.block_params(entry)[2];
            let zero_i = bcx.ins().iconst(types::I64, 0);
            bcx.ins().jump(header, &[zero_i]);

            // Header: cmp i<len; if so → body, else → exit.
            bcx.switch_to_block(header);
            let i = bcx.block_params(header)[0];
            let cond = bcx.ins().icmp(IntCC::SignedLessThan, i, len);
            bcx.ins().brif(cond, body, &[], exit, &[]);

            // Body: load in_ptr[i], op with scalar, store out_ptr[i], i+1, back to header.
            bcx.switch_to_block(body);
            let eight = bcx.ins().iconst(types::I64, 8);
            let off = bcx.ins().imul(i, eight);
            let in_addr  = bcx.ins().iadd(in_ptr, off);
            let out_addr = bcx.ins().iadd(out_ptr, off);
            let mflags = MemFlags::trusted();
            let v = bcx.ins().load(types::F64, mflags, in_addr, 0);
            let s = bcx.ins().f64const(scalar);
            let r = match op {
                r2_types::BinOp::Add => bcx.ins().fadd(v, s),
                r2_types::BinOp::Sub => bcx.ins().fsub(v, s),
                r2_types::BinOp::Mul => bcx.ins().fmul(v, s),
                r2_types::BinOp::Div => bcx.ins().fdiv(v, s),
                _ => unreachable!(),
            };
            bcx.ins().store(mflags, r, out_addr, 0);
            let one = bcx.ins().iconst(types::I64, 1);
            let next = bcx.ins().iadd(i, one);
            bcx.ins().jump(header, &[next]);

            // Exit: return ().
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
        Ok(CompiledFn { ptr, arity: 1, kind: r2_types::JitKind::VectorMap, _module: module })
    }

    /// Phase C.4-full: compile element-wise vector⊗vector op.
    /// Body: `for i in 0..len: out[i] = a[i] OP b[i]`.
    pub fn compile_vector_binary_op(op: r2_types::BinOp) -> JitResult<CompiledFn> {
        let supported = matches!(op, r2_types::BinOp::Add | r2_types::BinOp::Sub
                                    | r2_types::BinOp::Mul | r2_types::BinOp::Div);
        if !supported { return Err(JitError::Unsupported(format!("vec-vec op {:?}", op))); }

        let jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
        let mut module = JITModule::new(jit_builder);

        // (a_ptr, b_ptr, out_ptr, len) -> ()
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));

        let func_id = module
            .declare_function("__jit_vec_binary", Linkage::Export, &sig)
            .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;

        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        let mut fbctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);

            let entry = bcx.create_block();
            let header = bcx.create_block();
            let body = bcx.create_block();
            let exit = bcx.create_block();
            bcx.append_block_param(header, types::I64);

            bcx.append_block_params_for_function_params(entry);
            bcx.switch_to_block(entry);
            let a_ptr  = bcx.block_params(entry)[0];
            let b_ptr  = bcx.block_params(entry)[1];
            let out_ptr = bcx.block_params(entry)[2];
            let len    = bcx.block_params(entry)[3];
            let zero_i = bcx.ins().iconst(types::I64, 0);
            bcx.ins().jump(header, &[zero_i]);

            bcx.switch_to_block(header);
            let i = bcx.block_params(header)[0];
            let cond = bcx.ins().icmp(IntCC::SignedLessThan, i, len);
            bcx.ins().brif(cond, body, &[], exit, &[]);

            bcx.switch_to_block(body);
            let eight = bcx.ins().iconst(types::I64, 8);
            let off = bcx.ins().imul(i, eight);
            let a_addr   = bcx.ins().iadd(a_ptr,   off);
            let b_addr   = bcx.ins().iadd(b_ptr,   off);
            let out_addr = bcx.ins().iadd(out_ptr, off);
            let mflags = MemFlags::trusted();
            let av = bcx.ins().load(types::F64, mflags, a_addr, 0);
            let bv = bcx.ins().load(types::F64, mflags, b_addr, 0);
            let r = match op {
                r2_types::BinOp::Add => bcx.ins().fadd(av, bv),
                r2_types::BinOp::Sub => bcx.ins().fsub(av, bv),
                r2_types::BinOp::Mul => bcx.ins().fmul(av, bv),
                r2_types::BinOp::Div => bcx.ins().fdiv(av, bv),
                _ => unreachable!(),
            };
            bcx.ins().store(mflags, r, out_addr, 0);
            let one = bcx.ins().iconst(types::I64, 1);
            let next = bcx.ins().iadd(i, one);
            bcx.ins().jump(header, &[next]);

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
        Ok(CompiledFn { ptr, arity: 2, kind: r2_types::JitKind::VectorBinaryMap, _module: module })
    }

    /// Phase C.4-full part 2: compile a 1-arg closure whose body is a pure
    /// scalar arithmetic expression (no calls, no control flow), generating
    /// a fused element-wise loop. The param VReg is loaded from `in[i]`
    /// once per iteration; everything else lowers like the scalar JIT.
    ///
    /// Accepts e.g. `function(v) (v + 1) * 2`, `function(v) v*v - 1`, etc.
    pub fn compile_vector_map_generic(body_ir: &IrFunc) -> JitResult<CompiledFn> {
        if body_ir.params.len() != 1 {
            return Err(JitError::Unsupported("generic vector map expects 1 param".into()));
        }
        compile_vector_n_map_generic(body_ir, 1, "__jit_vec_map_generic", r2_types::JitKind::VectorMap)
    }

    /// Phase C.7 — element-wise **2-arg** general vector map.
    /// `function(a, b) BODY` over two same-length vectors where BODY can be
    /// arbitrary multi-block IR (including math-extern Calls, comparisons,
    /// branches, phis). The simpler `compile_vector_binary_op` handles only
    /// `function(a, b) a OP b` for a single fused binop — this is its
    /// generalisation. ABI matches `VectorBinaryMap`:
    /// `(*const f64, *const f64, *mut f64, i64) -> ()`.
    ///
    /// Closes the `sqrt(x*x + y*y)`-shape perf gap: pre-C.7 this fell
    /// back to the interpreter+columnar path; post-C.7 it compiles to a
    /// single fused native loop with one `fsqrt` per iteration.
    pub fn compile_vector_binary_map_generic(body_ir: &IrFunc) -> JitResult<CompiledFn> {
        if body_ir.params.len() != 2 {
            return Err(JitError::Unsupported("vector binary map expects 2 params".into()));
        }
        compile_vector_n_map_generic(body_ir, 2, "__jit_vec_map_binary", r2_types::JitKind::VectorBinaryMap)
    }

    /// Phase C.5 — element-wise ternary map. `function(c, a, b) BODY` over three
    /// same-length vectors. Body may be multi-block (branchy) — this is the
    /// main motivation. ABI: `(*const f64, *const f64, *const f64, *mut f64, i64) -> ()`.
    pub fn compile_vector_ternary_map_generic(body_ir: &IrFunc) -> JitResult<CompiledFn> {
        if body_ir.params.len() != 3 {
            return Err(JitError::Unsupported("vector ternary map expects 3 params".into()));
        }
        compile_vector_n_map_generic(body_ir, 3, "__jit_vec_map_ternary", r2_types::JitKind::VectorTernaryMap)
    }

    /// Phase C.9 — **Fused map-reduce: vector in, scalar out, no
    /// intermediate vector allocated.**
    ///
    /// Compiles closures like `function(x) sum(f(x))` or
    /// `function(x) sum(x*x + 1)` into a single Cranelift loop that:
    ///
    ///   1. Loads `x[i]` from the input pointer.
    ///   2. Evaluates the inner expression `f(x[i])` to produce one f64.
    ///   3. Accumulates that f64 into a running reduce-state.
    ///   4. After the loop, returns the final reduced scalar.
    ///
    /// **Why this matters**: without fusion, `sum(f(x))` on 1e7 elements
    /// allocates an 8 MB intermediate vector for `f(x)`, then sums it.
    /// Two passes over memory: 16 MB intermediate traffic. With fusion,
    /// only the 8 MB input is read and a single f64 is returned.
    ///
    /// **Supported reduction ops**: sum, prod. Mean is `sum / len`
    /// computed in the caller. Min/max would need different identity
    /// + combine; future extension.
    ///
    /// `body_ir` is the IR of the **inner map** function (the `f` in
    /// `sum(f(x))`), single-param `function(x) ...`. Caller decides
    /// which reduction to fuse via the `reduce_op` argument.
    pub fn compile_vector_map_reduce(
        body_ir: &IrFunc,
        reduce_op: FusedReduceOp,
    ) -> JitResult<CompiledFn> {
        if body_ir.params.len() != 1 {
            return Err(JitError::Unsupported("map-reduce expects 1 inner param".into()));
        }
        compile_map_reduce_inner(body_ir, reduce_op)
    }

    /// Phase J.2 — fused binary map-reduce `reduce(f(a[i], b[i]))` (e.g.
    /// `sum(x*w)`). Inner body has two scalar params (the two elements).
    pub fn compile_vector_binary_map_reduce(
        body_ir: &IrFunc,
        reduce_op: FusedReduceOp,
    ) -> JitResult<CompiledFn> {
        compile_binary_map_reduce_inner(body_ir, reduce_op)
    }

    /// Phase C.8 — **SIMD f64x2 vectorized 1-arg vector map.**
    ///
    /// When the IR body is "SIMD-clean" (single block, only arithmetic +
    /// native-instr math + constants, no branches, no extern calls),
    /// emit a Cranelift loop that processes **two f64s per iteration**
    /// via SSE2 `F64X2` SIMD instructions (`fadd.f64x2`, `fmul.f64x2`,
    /// `sqrt.f64x2` etc.). A scalar remainder handles the tail when
    /// `n` is odd.
    ///
    /// **Why it matters:** SSE2's `sqrtpd` is 1 instruction for 2
    /// doubles in one register; the scalar version executes one
    /// `sqrtsd` per element with full load/store/branch loop overhead
    /// per element. For `sqrt(x*x + 1)` over 1e6 elements, this closes
    /// the per-element gap from ~14 ns to ~6-8 ns — comparable to R's
    /// libm-vectorized path.
    ///
    /// **Targets:** SSE2 is mandatory on x86_64; Cranelift's `F64X2`
    /// lowers to native SSE2 on x86_64 and to NEON `vsqrtq_f64` on
    /// aarch64. So the SIMD path is enabled unconditionally on those
    /// targets and disabled on others.
    pub fn compile_vector_simd_map_f64x2(body_ir: &IrFunc) -> JitResult<CompiledFn> {
        if body_ir.params.len() != 1 {
            return Err(JitError::Unsupported("simd vector map expects 1 param".into()));
        }
        compile_vector_n_simd_map(body_ir, 1, "__jit_vec_simd_map", r2_types::JitKind::VectorMap)
    }
}
