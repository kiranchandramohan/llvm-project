//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Similar to MLIR/LLVM's "opt" tools but also deals with analysis and custom
// arguments. TODO: this is basically a copy from MlirOptMain.cpp, but capable
// of module emission as specified by the user.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/InitAllDialects.h"
#include "clang/CIR/Passes.h"

#ifdef CLANG_INCLUDE_TESTS
namespace cir::test {
void registerTestCIRAliasAnalysisPass();
} // namespace cir::test
#endif

struct CIRToLLVMPipelineOptions
    : public mlir::PassPipelineOptions<CIRToLLVMPipelineOptions> {
  Option<bool> enableOpenMP{
      *this, "enable-openmp",
      llvm::cl::desc("Add OpenMP-specific CIR-to-LLVM lowering passes"),
      llvm::cl::init(false)};
};

int main(int argc, char **argv) {
  // TODO: register needed MLIR passes for CIR?
  mlir::DialectRegistry registry;
  cir::registerAllDialects(registry);

#ifdef CLANG_INCLUDE_TESTS
  cir::test::registerTestCIRAliasAnalysisPass();
#endif
  registry.insert<mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
                  mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::arm_sme::ArmSMEDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::ub::UBDialect, mlir::vector::VectorDialect>();

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRCanonicalizePass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRSimplifyPass();
  });

  mlir::PassPipelineRegistration<CIRToLLVMPipelineOptions> pipeline(
      "cir-to-llvm", "",
      [](mlir::OpPassManager &pm, const CIRToLLVMPipelineOptions &options) {
        cir::direct::populateCIRToLLVMPasses(pm, options.enableOpenMP);
      });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRFlattenCFGPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIREHABILoweringPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createHoistAllocasPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createGotoSolverPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCXXABILoweringPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createIdiomRecognizerPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCallConvLoweringPass();
  });

  mlir::registerPass([] { return cir::createRaiseMatMulPass(); });
  mlir::registerPass([] { return cir::createVectorizeMatMulPass(); });
  mlir::PassPipelineRegistration<>(
      "cir-matmul-to-sme",
      "Lower raised MatMul through scalable vectors to ArmSME",
      [](mlir::OpPassManager &pm) { cir::populateMatMulToSMEPipeline(pm); });
  mlir::PassPipelineRegistration<>(
      "cir-sme-to-llvm", "Lower MatMul ArmSME to LLVM with streaming/ZA ABI",
      [](mlir::OpPassManager &pm) { cir::populateSMEToLLVMPipeline(pm); });

  mlir::omp::registerOpenMPPasses();
  mlir::registerTransformsPasses();

  return mlir::asMainReturnCode(MlirOptMain(
      argc, argv, "Clang IR analysis and optimization tool\n", registry));
}
