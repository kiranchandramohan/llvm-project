//===- CIRToSME.cpp - Experimental MatMul SME pipeline ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
namespace {
// The generated masked outer product hides the accumulator use inside a
// region, which the generic transfer-hoisting pass does not follow. Recognize
// this one producer/consumer chain and make its accumulator loop-carried.
static void hoistAccumulator(func::FuncOp f) {
  SmallVector<scf::ForOp> loops;
  f.walk([&](scf::ForOp loop) {
    moveLoopInvariantCode(cast<LoopLikeOpInterface>(loop.getOperation()));
    loops.push_back(loop);
  });
  IRRewriter rewriter(f.getContext());
  for (scf::ForOp loop : loops) {
    for (auto read : loop.getBody()->getOps<vector::TransferReadOp>()) {
      if (read.getVectorType().getRank() != 2 || !read->hasOneUse() ||
          (!read.getBase().getDefiningOp() ||
           !read.getBase().getDefiningOp()->hasAttr("cir.matmul_scratch")) ||
          !llvm::all_of(read->getOperands(), [&](Value v) {
            return loop.isDefinedOutsideOfLoop(v);
          }))
        continue;
      auto product =
          dyn_cast<vector::OuterProductOp>(*read->getUsers().begin());
      auto mask = product ? dyn_cast<vector::MaskOp>(product->getParentOp())
                          : vector::MaskOp();
      if (!mask || !mask->hasOneUse() || product.getAcc() != read.getResult())
        continue;
      auto write = dyn_cast<vector::TransferWriteOp>(*mask->getUsers().begin());
      if (!write || write->getParentOp() != loop ||
          write.getBase() != read.getBase() ||
          write.getIndices() != read.getIndices() ||
          write.getPermutationMap() != read.getPermutationMap() ||
          write.getMask() != read.getMask())
        continue;
      // Refuse any other use of the accumulator buffer inside the loop,
      // including aliases. Packing gives A, B, and sums disjoint workspace
      // slices.
      if (llvm::any_of(read.getBase().getUsers(), [&](Operation *user) {
            return loop->isAncestor(user) && user != read && user != write;
          }))
        continue;
      // Sums has been initialized even for K == 0. This masked read and write
      // are valid for every iteration of the enclosing nonempty M/N loops.
      read->moveBefore(loop);
      write->moveAfter(loop);
      auto replacement = loop.replaceWithAdditionalYields(
          rewriter, ValueRange{read.getResult()},
          /*replaceInitOperandUsesInLoop=*/true,
          [&](OpBuilder &, Location, ArrayRef<BlockArgument>) {
            return SmallVector<Value>{write.getVector()};
          });
      assert(succeeded(replacement) &&
             "scf.for must support additional yields");
      write.getValueToStoreMutable().assign(
          replacement->getOperation()->getResults().back());
      break;
    }
  }
}

struct CIRVectorizeMatMul
    : PassWrapper<CIRVectorizeMatMul, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIRVectorizeMatMul)
  StringRef getArgument() const final { return "cir-vectorize-matmul"; }
  StringRef getDescription() const final {
    return "Tile MatMul to scalable f32 SME tiles and form vector outer "
           "products";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<vector::VectorDialect, scf::SCFDialect, arith::ArithDialect,
             memref::MemRefDialect>();
  }
  void runOnOperation() override {
    auto f = getOperation();
    if (!f->hasAttr("cir.matmul"))
      return;
    SmallVector<linalg::MatmulOp> matmuls;
    f.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (auto matmul : matmuls) {
      rewriter.setInsertionPoint(matmul);
      Location loc = matmul.getLoc();
      Value four = arith::ConstantIndexOp::create(rewriter, loc, 4);
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value scale = vector::VectorScaleOp::create(rewriter, loc);
      Value tile = arith::MulIOp::create(rewriter, loc, four, scale);
      linalg::LinalgTilingOptions options;
      options.setTileSizes(SmallVector<Value, 4>{tile, tile, one});
      auto tiled = linalg::tileLinalgOp(rewriter, matmul, options);
      if (failed(tiled))
        return signalPassFailure();
      rewriter.eraseOp(matmul);
      if (failed(linalg::vectorize(rewriter, tiled->op, {4, 4, 1},
                                   {true, true, false})))
        return signalPassFailure();
      rewriter.eraseOp(tiled->op);
    }
    RewritePatternSet patterns(&getContext());
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateSinkVectorOpsPatterns(patterns);
    if (failed(applyPatternsGreedily(f, std::move(patterns))))
      return signalPassFailure();
    RewritePatternSet outerProducts(&getContext());
    vector::populateVectorContractLoweringPatterns(
        outerProducts, vector::VectorContractLowering::OuterProduct,
        /*benefit=*/1, /*disableOuterProductLowering=*/true);
    vector::populateVectorMaskOpLoweringPatterns(outerProducts);
    vector::populateVectorTransferDropUnitDimsPatterns(outerProducts);
    if (failed(applyPatternsGreedily(f, std::move(outerProducts))))
      return signalPassFailure();
    RewritePatternSet foldViews(&getContext());
    memref::populateFoldMemRefAliasOpPatterns(foldViews);
    if (failed(applyPatternsGreedily(f, std::move(foldViews))))
      return signalPassFailure();
    // The output scratch memref is initialized even when K is zero. Moving
    // its accumulator read/write outside K is safe and keeps the tile in ZA.
    // This pass only operates on functions produced by the MatMul raiser.
    hoistAccumulator(f);
    f->removeAttr("cir.matmul");
  }
};
} // namespace

std::unique_ptr<Pass> cir::createVectorizeMatMulPass() {
  return std::make_unique<CIRVectorizeMatMul>();
}

static void populateMatMulToSMEStage(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(cir::createVectorizeMatMulPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addPass(arm_sme::createVectorLegalizationPass());
  pm.addPass(createArithToArmSMEConversionPass());
  pm.addPass(createConvertVectorToArmSMEPass());
}

static void populateSMEToLLVMStage(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertArmSMEToSCFPass());
  pm.addPass(createConvertVectorToSCFPass(
      VectorTransferToSCFOptions().enableFullUnroll()));
  pm.addNestedPass<func::FuncOp>(arm_sme::createEnableArmStreamingPass(
      arm_sme::ArmStreamingMode::StreamingLocally, arm_sme::ArmZaMode::NewZA,
      /*ifRequiredByOps=*/true));
  pm.addPass(createSCFToControlFlowPass());
  pm.addNestedPass<func::FuncOp>(createConvertArmSMEToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createUBToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

namespace {
// Dialect conversion folds operations as well as applying explicit patterns.
// Do not expose unrelated CIR-lowered LLVM bodies to that folding: in
// particular, LLVM lifetime markers must retain their alloca provenance.
// Only generated helpers and dispatchers participate in these MLIR pipelines.
struct IsolatedMatMulPipeline
    : PassWrapper<IsolatedMatMulPipeline, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IsolatedMatMulPipeline)
  bool lowerToLLVM;
  explicit IsolatedMatMulPipeline(bool lowerToLLVM) : lowerToLLVM(lowerToLLVM) {}
  IsolatedMatMulPipeline(const IsolatedMatMulPipeline &other)
      : PassWrapper(other), lowerToLLVM(other.lowerToLLVM) {}

  void populate(OpPassManager &pm) const {
    if (lowerToLLVM)
      populateSMEToLLVMStage(pm);
    else
      populateMatMulToSMEStage(pm);
  }

  void getDependentDialects(DialectRegistry &r) const override {
    OpPassManager pm(ModuleOp::getOperationName());
    populate(pm);
    pm.getDependentDialects(r);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> generated;
    for (Operation &op : module.getBody()->getOperations())
      if (op.hasAttr("cir.matmul_helper") || op.hasAttr("cir.matmul_dispatch"))
        generated.push_back(&op);
    if (generated.empty())
      return;

    OpBuilder b(module.getBodyRegion());
    auto stage = ModuleOp::create(b, module.getLoc());
    stage->setAttrs(module->getAttrs());
    stage->removeAttr("sym_name");
    // Inherit the surrounding target layout instead of duplicating its
    // pointer specifications in a nested data-layout scope.
    stage->removeAttr("dlti.dl_spec");
    // Declarations satisfy symbol verification in the temporary symbol table.
    // Keep their bodies exclusively in the parent module.
    llvm::SmallPtrSet<Operation *, 8> declarations;
    for (Operation *op : generated)
      op->moveBefore(stage.getBody(), stage.getBody()->end());
    for (Operation *op : generated)
      op->walk([&](LLVM::CallOp call) {
        auto callee = call.getCallee();
        if (!callee || stage.lookupSymbol(*callee))
          return;
        auto original = module.lookupSymbol<LLVM::LLVMFuncOp>(*callee);
        if (!original)
          return;
        auto decl = cast<LLVM::LLVMFuncOp>(original->cloneWithoutRegions());
        decl.setLinkage(LLVM::Linkage::External);
        decl.removeComdatAttr();
        stage.getBody()->push_back(decl);
        declarations.insert(decl);
      });
    for (auto comdat : module.getOps<LLVM::ComdatOp>()) {
      Operation *clone = comdat->clone();
      stage.getBody()->push_back(clone);
      declarations.insert(clone);
    }
    OpPassManager pm(ModuleOp::getOperationName());
    populate(pm);
    LogicalResult result = runPipeline(pm, stage);
    // Move back even on failure, so the diagnostic module retains all bodies.
    for (Operation &op : llvm::make_early_inc_range(*stage.getBody())) {
      if (declarations.contains(&op))
        continue;
      if (lowerToLLVM) {
        op.removeAttr("cir.matmul_helper");
        op.removeAttr("cir.matmul_dispatch");
        op.removeAttr("cir.matmul");
      }
      op.moveBefore(stage);
    }
    stage.erase();
    if (failed(result))
      signalPassFailure();
  }
};
} // namespace

void cir::populateMatMulToSMEPipeline(OpPassManager &pm) {
  pm.addPass(std::make_unique<IsolatedMatMulPipeline>(false));
}

void cir::populateSMEToLLVMPipeline(OpPassManager &pm) {
  pm.addPass(std::make_unique<IsolatedMatMulPipeline>(true));
}
