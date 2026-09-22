// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -ffp-contract=fast -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -ffp-contract=fast -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple aarch64-linux-gnu -ffp-contract=fast -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -ffp-contract=off -emit-cir %s -o - | FileCheck %s --check-prefix=OFF --implicit-check-not=cir.contract
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fclangir -ffp-contract=fast -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-cir %s -o - | FileCheck %s --check-prefix=STRICT --implicit-check-not=cir.contract

// Retain permission across statement boundaries, including a reused product.
float contract(float a, float b, float c, float *p) {
  float product = a * b;
  *p = product;
  return product + c;
}
// CIR-LABEL: @contract(
// CIR: cir.fmul {{.*}} : !cir.float {cir.contract}
// CIR: cir.fadd {{.*}} : !cir.float {cir.contract}
// LLVM-LABEL: @contract(
// LLVM: fmul contract float
// LLVM: fadd contract float
// OFF-LABEL: @contract(
// OFF: cir.fmul
// OFF: cir.fadd
// STRICT-LABEL: @contract(
// STRICT: cir.fmul {{.*}} : !cir.float {fenv =
// STRICT: cir.fadd {{.*}} : !cir.float {fenv =

float mul_off(float a, float b, float c) {
  float product;
  {
#pragma clang fp contract(off)
    product = a * b;
  }
  return product + c;
}
// CIR-LABEL: @mul_off(
// CIR: cir.fmul {{.*}} : !cir.float loc(
// CIR: cir.fadd {{.*}} : !cir.float {cir.contract}
// LLVM-LABEL: @mul_off(
// LLVM: fmul float
// LLVM: fadd contract float

float add_off(float a, float b, float c) {
  float product = a * b;
  {
#pragma clang fp contract(off)
    return product + c;
  }
}
// CIR-LABEL: @add_off(
// CIR: cir.fmul {{.*}} : !cir.float {cir.contract}
// CIR: cir.fadd {{.*}} : !cir.float loc(
// LLVM-LABEL: @add_off(
// LLVM: fmul contract float
// LLVM: fadd float

typedef float float4 __attribute__((ext_vector_type(4)));
float4 vector_contract(float4 a, float4 b, float4 c) {
  return a * b + c;
}
// CIR-LABEL: @vector_contract(
// CIR: cir.fmul {{.*}} : !cir.vector<4 x !cir.float> {cir.contract}
// CIR: cir.fadd {{.*}} : !cir.vector<4 x !cir.float> {cir.contract}
// LLVM-LABEL: @vector_contract(
// LLVM: fmul contract <4 x float>
// LLVM: fadd contract <4 x float>
