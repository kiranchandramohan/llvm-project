// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=off %s -o %t.cir
// RUN: cir-opt %t.cir -cir-raise-matmul='assume-no-alias allow-contract' -o %t.linalg
// RUN: FileCheck %s --check-prefix=LINALG < %t.linalg
// RUN: cir-opt %t.linalg -cir-matmul-to-sme -o %t.sme
// RUN: FileCheck %s --check-prefix=SME --implicit-check-not=linalg.matmul < %t.sme
// RUN: cir-opt %t.sme -cir-sme-to-llvm | mlir-translate -mlir-to-llvmir -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM < %t.ll
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sme -O3 -S -x ir %t.ll -o - | FileCheck %s --check-prefix=ASM
// RUN: not cir-opt %t.cir -cir-raise-matmul=require-kernel -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONTRACT
// RUN: not cir-opt %t.cir -cir-raise-matmul='require-kernel assume-no-alias' -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONTRACT
// RUN: cir-opt %t.cir -cir-raise-matmul=allow-contract -o %t.guarded
// RUN: FileCheck %s --check-prefix=GUARD < %t.guarded
// RUN: cir-opt %t.guarded -cir-matmul-to-sme -cir-sme-to-llvm | mlir-translate -mlir-to-llvmir -o %t.guarded.ll
// RUN: FileCheck %s --check-prefix=GUARD-LLVM < %t.guarded.ll
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sme -O3 -S -x ir %t.guarded.ll -o - | FileCheck %s --check-prefix=GUARD-ASM
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=on %s -o %t.on
// RUN: cir-opt %t.on -cir-raise-matmul -cir-matmul-to-sme | FileCheck %s --check-prefix=CONTRACT-SME
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast %s -o %t.fast
// RUN: cir-opt %t.fast -cir-raise-matmul -cir-matmul-to-sme | FileCheck %s --check-prefix=CONTRACT-SME
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DMUL_OFF %s -o %t.partial
// RUN: not cir-opt %t.partial -cir-raise-matmul=require-kernel -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONTRACT
// RUN: cir-opt %t.partial -cir-raise-matmul=allow-contract -o /dev/null
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DADD_OFF %s -o %t.partial
// RUN: not cir-opt %t.partial -cir-raise-matmul=require-kernel -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONTRACT
// RUN: cir-opt %t.partial -cir-raise-matmul=allow-contract -o /dev/null
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DEPILOGUE_OFF %s -o %t.epilogue
// RUN: cir-opt %t.epilogue -cir-raise-matmul=assume-no-alias | FileCheck %s --check-prefix=UNFUSED-EPILOGUE
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DALPHA_OFF %s -o %t.epilogue
// RUN: cir-opt %t.epilogue -cir-raise-matmul=assume-no-alias | FileCheck %s --check-prefix=BETA-FUSED
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict %s -o %t.strict
// RUN: not cir-opt %t.strict -cir-raise-matmul='require-kernel allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=off -DSHARED_BOUND %s -o %t.bad
// RUN: cir-opt %t.bad -cir-raise-matmul='require-kernel assume-no-alias allow-contract' | FileCheck %s --check-prefix=LINALG
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=off -DWRONG_INDEX %s -o %t.bad
// RUN: not cir-opt %t.bad -cir-raise-matmul='require-kernel assume-no-alias allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=off -DSUBTRACT %s -o %t.bad
// RUN: not cir-opt %t.bad -cir-raise-matmul='require-kernel assume-no-alias allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=off -DEXTRA_STORE %s -o %t.bad
// RUN: not cir-opt %t.bad -cir-raise-matmul='require-kernel assume-no-alias allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=off -DVOLATILE %s -o %t.bad
// RUN: not cir-opt %t.bad -cir-raise-matmul='require-kernel assume-no-alias allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=off -fwrapv %s -o %t.bad
// RUN: not cir-opt %t.bad -cir-raise-matmul='require-kernel assume-no-alias allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O0 -ffp-contract=off -Dmatmul=renamed %s -o %t.renamed
// RUN: cir-opt %t.renamed -cir-raise-matmul='assume-no-alias allow-contract' | FileCheck %s --check-prefix=RENAMED

// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DSNAPSHOT %s -o %t.snapshot
// RUN: not cir-opt %t.snapshot -cir-raise-matmul='require-kernel allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DFROZEN_ALPHA %s -o %t.snapshot
// RUN: not cir-opt %t.snapshot -cir-raise-matmul='require-kernel allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: cir-opt %t.snapshot -cir-raise-matmul -cir-matmul-to-sme -cir-sme-to-llvm | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=SKIP
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sme -fclangir -fclangir-matmul-to-sme -fclangir-matmul-options=require-kernel -O3 -ffp-contract=fast -DINLINE -fvisibility=hidden -emit-llvm %s -o - | FileCheck %s --check-prefix=INLINE
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sme -fclangir -fclangir-matmul-to-sme -fclangir-matmul-options=require-kernel -O3 -ffp-contract=fast -DINLINE -S %s -o - | FileCheck %s --check-prefix=INLINE-ASM
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DINLINE -fvisibility=hidden %s -o %t.mixed
// RUN: cir-opt %t.mixed -cir-raise-matmul=require-kernel -cir-matmul-to-sme -cir-sme-to-llvm | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=INLINE-RAW
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -fvisibility=hidden %s -o %t.hidden
// RUN: cir-opt %t.hidden -cir-raise-matmul=require-kernel -cir-matmul-to-sme -cir-sme-to-llvm | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=HIDDEN
// RUN: cir-opt %t.hidden -cir-raise-matmul='require-kernel assume-no-alias' -cir-matmul-to-sme -cir-sme-to-llvm | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=HIDDEN

// SKIP-NOT: cir_matmul
// SKIP: define {{(dso_local )?}}void @_Z6matmul
// SKIP-NOT: cir_matmul
// INLINE: @global = hidden
// INLINE: @llvm.global_ctors
// INLINE: define internal {{.*}}@_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme
// INLINE: @llvm.aarch64.sme.mopa
// INLINE: define {{(dso_local )?}}hidden void @_Z6caller
// INLINE: call {{.*}}@_Z6matmulbbiiifPfiS_ifS_ib(
// INLINE: define linkonce_odr hidden void @_Z6matmulbbiiifPfiS_ifS_ib(
// INLINE-SAME: comdat
// INLINE: call {{.*}}@_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme
// INLINE-RAW: @unrelated = hidden global
// INLINE-RAW: define {{(dso_local )?}}hidden void @_Z6caller
// INLINE-RAW: define linkonce_odr hidden void @_Z6matmulbbiiifPfiS_ifS_ib(
// INLINE-RAW-SAME: comdat
// INLINE-ASM: fmopa
// HIDDEN: define internal void @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_fallback(
// HIDDEN: define {{(dso_local )?}}hidden void @_Z6matmulbbiiifPfiS_ifS_ib(

// LINALG: affine_map<(d0, d1, d2) -> (d2, d0)>
// LINALG-LABEL: func.func private @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme(
// LINALG: arith.maxsi
// LINALG: scf.if
// LINALG: builtin.unrealized_conversion_cast
// LINALG: arith.select %arg0
// LINALG: arith.select %arg1
// LINALG: linalg.fill
// LINALG: linalg.matmul indexing_maps =
// LINALG: %[[BETA:.*]] = arith.cmpf oeq, %arg10
// LINALG: scf.if %[[BETA]] {
// LINALG-NOT: llvm.load
// LINALG: llvm.store
// LINALG: } else {
// LINALG: llvm.load
// LINALG: arith.mulf
// LINALG: llvm.intr.fma
// LINALG: llvm.store
// LINALG-NOT: memref.alloc
// LINALG-NOT: memref.dealloc
// SME-LABEL: func.func private @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme(
// SME: vector.vscale
// SME: scf.for {{.*}} iter_args
// SME: arm_sme.outerproduct
// SME: arm_sme.tile_store
// LLVM-LABEL: define internal void @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme(
// LLVM: call void @llvm.aarch64.sme.mopa
// LLVM-DAG: "aarch64_pstate_sm_body"
// LLVM-DAG: "aarch64_new_za"
// ASM-LABEL: _Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme:
// ASM: smstart
// ASM: fmopa za0.s
// ASM: smstop
// CONTRACT: MatMul reduction requires contraction permission on its multiply/add instructions or the allow-contract option
// CONTRACT-SME: func.func private @{{.*}}.cir_matmul_sme(
// CONTRACT-SME: arm_sme.outerproduct
// CONTRACT-SME: llvm.intr.fma
// BETA-FUSED: llvm.intr.fma(%arg10,
// UNFUSED-EPILOGUE: linalg.matmul
// UNFUSED-EPILOGUE-NOT: llvm.intr.fma
// UNFUSED-EPILOGUE: arith.addf
// UNFUSED-EPILOGUE-NOT: llvm.intr.fma
// GUARD-LABEL: llvm.func internal @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul(
// GUARD: arith.extsi %arg15 : i32 to i128
// GUARD: scf.if
// GUARD: llvm.ptrtoint %arg19
// GUARD: llvm.ptrtoint %arg21
// GUARD: llvm.ptrtoint %arg24
// GUARD: arith.cmpi sle, {{.*}} : i128
// GUARD: llvm.call @malloc
// GUARD: llvm.icmp "ne"
// GUARD: scf.if
// GUARD: func.call @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme(
// GUARD: llvm.call @free
// GUARD: } else {
// GUARD: llvm.call @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_fallback(
// GUARD: } else {
// GUARD: llvm.call @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_fallback(
// GUARD-LABEL: llvm.func internal @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_fallback(
// GUARD-NOT: contract
// GUARD: llvm.fmul
// GUARD-NOT: contract
// GUARD: llvm.fadd
// GUARD-NOT: contract
// GUARD-LLVM-LABEL: define internal void @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme(
// GUARD-LLVM-SAME: ) #[[SMEATTR:[0-9]+]] {
// GUARD-LLVM: call void @llvm.aarch64.sme.mopa
// GUARD-LLVM: call float @llvm.fma.f32
// GUARD-LLVM-LABEL: define internal void @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul(
// GUARD-LLVM: icmp sle i128
// GUARD-LLVM: call ptr @malloc
// GUARD-LLVM: icmp ne ptr
// GUARD-LLVM: call void @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme(
// GUARD-LLVM: call void @free
// GUARD-LLVM: call void @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_fallback(
// GUARD-LLVM-LABEL: define internal void @_Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_fallback(
// GUARD-LLVM: fmul float
// GUARD-LLVM: fadd float
// GUARD-LLVM: attributes #[[SMEATTR]] = { {{.*}}"aarch64_new_za" "aarch64_pstate_sm_body"
// GUARD-ASM-LABEL: _Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme:
// GUARD-ASM: smstart
// GUARD-ASM: fmopa za0.s
// GUARD-ASM: fmadd
// GUARD-ASM: smstop
// GUARD-ASM-LABEL: _Z6matmulbbiiifPfiS_ifS_ib:
// GUARD-ASM-NOT: smstart
// GUARD-ASM: bl _Z6matmulbbiiifPfiS_ifS_ib.cir_matmul_sme
// GUARD-ASM-NOT: smstart
// REJECT: unsupported MatMul kernel
// RENAMED: func.func private @_Z7renamedbbiiifPfiS_ifS_ib.cir_matmul_sme(
// RENAMED: linalg.matmul

#ifdef INLINE
inline
#endif
void matmul(bool transA, bool transB, int rows_a, int rows_b, int width,
           float alpha,
#ifdef VOLATILE
           volatile
#endif
           float *a,
           int lda, float *b, int ldb, float beta, float *c, int ldc,
           bool cache = false) {
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < rows_b; ++j) {
      float sum = 0.0;
#ifdef SNAPSHOT
      const float initial = sum;
#endif
#ifdef FROZEN_ALPHA
      const float scaled = alpha * sum;
#endif
#ifdef SHARED_BOUND
      for (int k = 0; k < rows_a; ++k) {
#else
      for (int k = 0; k < width; ++k) {
#endif
#ifdef WRONG_INDEX
        const float a_val = transA ? a[k * lda + j] : a[i * lda + k];
#else
        const float a_val = transA ? a[k * lda + i] : a[i * lda + k];
#endif
        const float b_val = transB ? b[j * ldb + k] : b[k * ldb + j];
#ifdef SNAPSHOT
        sum = initial + a_val * b_val;
#elif defined(SUBTRACT)
        sum -= a_val * b_val;
#elif defined(MUL_OFF)
        float product;
        {
#pragma clang fp contract(off)
          product = a_val * b_val;
        }
        sum += product;
#elif defined(ADD_OFF)
      const float product = a_val * b_val;
      {
#pragma clang fp contract(off)
        sum += product;
      }
#else
      sum += a_val * b_val;
#endif
#ifdef EXTRA_STORE
        a[0] = 1.0;
#endif
      }
      {
#ifdef EPILOGUE_OFF
#pragma clang fp contract(off)
#endif
        if (beta == 0.0) {
#ifdef FROZEN_ALPHA
          c[i * ldc + j] = scaled;
#else
          c[i * ldc + j] = alpha * sum;
#endif
        } else {
#ifdef ALPHA_OFF
          float scaled;
          {
#pragma clang fp contract(off)
            scaled = alpha * sum;
          }
          c[i * ldc + j] = scaled + beta * c[i * ldc + j];
#else
#ifdef FROZEN_ALPHA
          c[i * ldc + j] = scaled + beta * c[i * ldc + j];
#else
          c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
#endif
#endif
        }
      }
    }
  }
}

#ifdef INLINE
// Exercise C++ metadata, constructors and global initialization alongside the
// inline kernel. These functions must retain their normal lowering.
struct Unrelated {
  int value;
  Unrelated() : value(7) {}
  ~Unrelated() {}
};
Unrelated unrelated;
extern int initialize();
int global = initialize();
void caller(int m, int n, int k, float *a, float *b, float *c) {
  matmul(false, false, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
}
#endif
