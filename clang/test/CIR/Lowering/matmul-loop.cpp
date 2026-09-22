// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast %s -o %t.cir
// RUN: cir-opt %t.cir -cir-raise-matmul=require-kernel -o %t.linalg
// RUN: FileCheck %s --check-prefix=RAISE < %t.linalg
// RUN: cir-opt %t.linalg -cir-matmul-to-sme -cir-sme-to-llvm | mlir-translate -mlir-to-llvmir -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM < %t.ll
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sme -O3 -S -x ir %t.ll -o - | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sme -fclangir -fclangir-matmul-to-sme -fclangir-matmul-options=require-kernel -O3 -ffp-contract=fast -S %s -o - | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O0 -ffp-contract=on %s -o %t.on
// RUN: cir-opt %t.on -cir-raise-matmul=require-kernel -cir-matmul-to-sme | FileCheck %s --check-prefix=SME
// RUN: cir-opt %t.on -cir-raise-matmul=require-kernel | FileCheck %s --check-prefix=BETA
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DINVALID %s -o %t.bad
// RUN: not cir-opt %t.bad -cir-raise-matmul='require-kernel allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DESCAPED %s -o %t.bad
// RUN: not cir-opt %t.bad -cir-raise-matmul='require-kernel allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -emit-cir -O1 -ffp-contract=fast -DEXTERNAL_IV %s -o %t.bad
// RUN: not cir-opt %t.bad -cir-raise-matmul='require-kernel allow-contract' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT

#ifdef REFERENCE
#define reordered reordered_reference
#define plain plain_reference
#define fixed fixed_reference
#define unconditional unconditional_reference
#define multiple multiple_reference
#define scaled_output scaled_output_reference
#endif

extern "C" {
#if defined(INVALID) || defined(ESCAPED) || defined(EXTERNAL_IV)
extern void escape(int *);
void invalid(float *c, const float *a, const float *b, int m, int n, int k) {
#ifdef ESCAPED
  escape(&k);
#endif
#ifdef EXTERNAL_IV
  int i;
  for (i = 0; i < m; ++i) {
#else
  for (int i = 0; i < m; ++i) {
#endif
    for (int j = 0; j < n; ++j) {
#ifdef INVALID
      double sum = 0.0;
#else
      float sum = 0.0f;
#endif
      for (int r = 0; r < k; ++r)
        sum += a[i * k + r] * b[r * n + j];
      c[i * n + j] = sum;
    }
  }
#ifdef EXTERNAL_IV
  c[0] = i;
#endif
}
#else
// Argument order/count, return type and surrounding side effects are unrelated
// to the matrix loop. The unused argument is deliberately not a boolean.
int reordered(float *c, int m, const float *b, int n, const float *a, int k,
              int ldc, int ldb, int lda, float beta, bool tb, float alpha,
              bool ta, long unused, int *observed) {
  *observed += 7;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int r = 0; r < k; ++r) {
        float left = ta ? a[r * lda + i] : a[i * lda + r];
        float right = tb ? b[j * ldb + r] : b[r * ldb + j];
        sum += left * right;
      }
      if (beta == 0.0f)
        c[i * ldc + j] = alpha * sum;
      else
        c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
    }
  }
  *observed += 11;
  return *observed;
}

// No transpose, scale, leading-dimension or unused parameters. Local bounds
// are computed before the loop; the output epilogue has no extra arithmetic.
int plain(float *c, const float *a, const float *b, int m, int n, int k,
          int tag) {
  int rows = m + 1;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int r = 0; r < k; ++r)
        sum += a[i * k + r] * b[r * n + j];
      c[i * n + j] = sum;
    }
  }
  return tag + 3;
}

// Constants and fixed transposed layouts are inferred from the accesses.
void fixed(float *c, const float *a, const float *b) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      float sum = 0.0f;
      for (int r = 0; r < 5; ++r)
        sum += a[r * 3 + i] * b[j * 5 + r];
      c[i * 4 + j] = 2.0f * sum;
    }
  }
}

// An unconditional output read must remain unconditional, even for beta == 0.
void unconditional(float *c, const float *a, const float *b, int m, int n,
                   int k, float alpha, float beta) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int r = 0; r < k; ++r)
        sum += a[i * k + r] * b[r * n + j];
      c[i * n + j] = alpha * sum + beta * c[i * n + j];
    }
  }
}

// The contractable product is on the right when alpha is omitted.
void scaled_output(float *c, const float *a, const float *b, int m, int n,
                   int k, float beta) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int r = 0; r < k; ++r)
        sum += a[i * k + r] * b[r * n + j];
      c[i * n + j] = sum + beta * c[i * n + j];
    }
  }
}

void multiple(float *c, const float *a, const float *b, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int r = 0; r < k; ++r)
        sum += a[i * k + r] * b[r * n + j];
      c[i * n + j] = sum;
    }
  }
  c[0] += 1.0f;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int r = 0; r < k; ++r)
        sum += a[i * k + r] * b[r * n + j];
      c[i * n + j] = sum + c[i * n + j];
    }
  }
}
#endif
}

// RAISE-LABEL: func.func private @reordered.cir_matmul_sme
// RAISE: linalg.matmul
// RAISE-LABEL: llvm.func internal @reordered.cir_matmul(
// RAISE: llvm.ptrtoint
// RAISE: llvm.call @malloc
// RAISE: llvm.call @reordered.cir_matmul_fallback
// RAISE-LABEL: llvm.func @reordered(
// RAISE: llvm.add
// RAISE: llvm.call @reordered.cir_matmul(
// RAISE: llvm.add
// RAISE: llvm.return {{.*}} : i32
// RAISE-LABEL: func.func private @plain.cir_matmul_sme
// RAISE: linalg.matmul
// RAISE-NOT: arith.mulf
// RAISE-NOT: llvm.load
// RAISE: llvm.store
// RAISE-LABEL: func.func private @fixed.cir_matmul_sme
// RAISE: linalg.matmul
// RAISE: arith.mulf
// RAISE-NOT: llvm.load
// RAISE: llvm.store
// RAISE-LABEL: func.func private @unconditional.cir_matmul_sme
// RAISE: linalg.matmul
// RAISE-NOT: arith.cmpf
// RAISE: llvm.load
// RAISE: llvm.intr.fma
// RAISE-LABEL: func.func private @multiple.cir_matmul_sme
// RAISE: linalg.matmul
// RAISE-LABEL: func.func private @multiple.cir_matmul__sme
// RAISE: linalg.matmul
// RAISE-NOT: arith.mulf
// RAISE: llvm.load
// RAISE: arith.addf
// RAISE-NOT: llvm.intr.fma
// RAISE-LABEL: llvm.func @multiple(
// RAISE: llvm.call @multiple.cir_matmul(
// RAISE: llvm.fadd
// RAISE: llvm.call @multiple.cir_matmul_(
// LLVM-COUNT-7: call void @llvm.aarch64.sme.mopa
// SME-COUNT-7: arm_sme.outerproduct
// ASM-COUNT-7: fmopa
// REJECT: unsupported MatMul kernel: no eligible FP32 loop nest found

// BETA-LABEL: func.func private @scaled_output.cir_matmul_sme(
// BETA: linalg.matmul
// BETA: llvm.intr.fma(%arg10,
