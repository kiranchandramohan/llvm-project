//===- CIRToLinalg.cpp - Experimental MatMul raising -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Recognize FP32 MatMul loop nests before flattening CIR.
// This is an opt-in experiment, not a general C loop-to-linalg conversion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/TargetParser/Triple.h"

using namespace mlir;

namespace {
enum class EpilogueFusion { None, Alpha, Beta };

struct MatMulEpilogue {
  bool scaleAlpha = true;
  bool readOutput = true;
  bool scaleBeta = true;
  bool guardBeta = true;
  EpilogueFusion fusion = EpilogueFusion::None;
};

// The normalized interface is independent of the enclosing function's ABI.
// Transpose flags, M/N/K, alpha, A/lda, B/ldb, beta, C/ldc.
static SmallVector<Type> getMatMulInputTypes(MLIRContext *context) {
  Type boolean = cir::BoolType::get(context);
  Type integer = cir::IntType::get(context, 32, true);
  Type single = cir::SingleType::get(context);
  Type pointer = cir::PointerType::get(single);
  return {boolean, boolean, integer, integer, integer, single, pointer,
          integer, pointer, integer, single,  pointer, integer};
}

// Describe the complete loop nest, including all stores and control flow.
// Local names and source locations are irrelevant. Only single-assignment local
// temporaries are forwarded. Unexpected operations cause a closed failure.
class MatMulMatcher {
  cir::ScopeOp scope;
  DenseMap<Value, unsigned> leafNumbers;
  DenseMap<Attribute, unsigned> constantNumbers;
  SmallVector<Value> leaves;
  SmallVector<Value> inputs;
  SmallVector<TypedAttr> constants;
  MatMulEpilogue epilogue;
  DenseMap<Value, Value> singleStores;
  DenseMap<Value, std::string> variables;
  DominanceInfo dominance;
  bool valid = true;
  bool reductionContract = false;

  // Remember when mutable locals were read. Forwarding a single-assignment
  // temporary must not turn a saved value into a new load after a store or a
  // loop iteration (e.g. const float initial = sum before the reduction).
  DenseMap<Value, unsigned> epochs;
  DenseMap<Value, unsigned> loadEpochs;

  bool current(Value value, unsigned depth = 0) {
    if (depth > 64)
      return false;
    Operation *def = value.getDefiningOp();
    if (!def)
      return true;
    if (auto load = dyn_cast<cir::LoadOp>(def)) {
      if (variables.contains(load.getAddr()))
        return loadEpochs.contains(value) &&
               loadEpochs.lookup(value) == epochs.lookup(load.getAddr());
      if (singleStores.contains(load.getAddr()))
        return current(singleStores.lookup(load.getAddr()), depth + 1);
    }
    for (Value operand : def->getOperands())
      if (!current(operand, depth + 1))
        return false;
    if (auto ternary = dyn_cast<cir::TernaryOp>(def))
      for (Region &branch : ternary->getRegions()) {
        if (!llvm::hasSingleElement(branch))
          return false;
        for (Value operand : branch.front().getTerminator()->getOperands())
          if (!current(operand, depth + 1))
            return false;
      }
    return true;
  }

  bool checkLoadTimes(Region &region) {
    if (region.empty())
      return true;
    if (!llvm::hasSingleElement(region))
      return false;
    for (Operation &op : region.front()) {
      if (auto load = dyn_cast<cir::LoadOp>(op)) {
        if (variables.contains(load.getAddr()))
          loadEpochs[load.getResult()] = epochs.lookup(load.getAddr());
      } else if (auto store = dyn_cast<cir::StoreOp>(op)) {
        if (!singleStores.contains(store.getAddr()) &&
            (!current(store.getValue()) || !current(store.getAddr())))
          return false;
        if (variables.contains(store.getAddr()))
          ++epochs[store.getAddr()];
      } else if (auto loop = dyn_cast<cir::ForOp>(op)) {
        auto invalidate = [&] {
          loop.walk([&](cir::StoreOp store) {
            if (variables.contains(store.getAddr()))
              ++epochs[store.getAddr()];
          });
        };
        // Crossing a loop boundary may cross arbitrarily many stores.
        invalidate();
        for (Region *part : {&loop.getCond(), &loop.getBody(), &loop.getStep(),
                             &loop.getCleanup()})
          if (!checkLoadTimes(*part))
            return false;
        invalidate();
        continue;
      } else if (isa<cir::ConditionOp, cir::IfOp, cir::TernaryOp>(op)) {
        if (!current(op.getOperand(0)))
          return false;
      }
      for (Region &nested : op.getRegions())
        if (!checkLoadTimes(nested))
          return false;
    }
    return true;
  }

  // Only speculate reads of nonescaping locals that are unchanged throughout
  // this nest. Other live-ins must already be SSA values defined above it.
  // In particular, do not hoist matrix/record/global loads out of zero-trip
  // loops, or read a local whose address could alias an output matrix.
  bool isInvariant(Value value) {
    Operation *def = value.getDefiningOp();
    if (!def || !scope->isAncestor(def))
      return dominance.dominates(value, scope);
    if (isa<cir::ConstantOp>(def))
      return true;
    auto load = dyn_cast<cir::LoadOp>(def);
    auto alloca =
        load ? load.getAddr().getDefiningOp<cir::AllocaOp>() : nullptr;
    if (!alloca || scope->isAncestor(alloca) || load.getIsVolatile() ||
        load.getMemOrder() ||
        !dominance.dominates(alloca.getOperation(), scope))
      return false;
    bool initialized = false;
    for (Operation *user : alloca.getResult().getUsers()) {
      if (auto store = dyn_cast<cir::StoreOp>(user)) {
        initialized |= dominance.dominates(store, scope);
        if (store.getAddr() != alloca.getResult() || scope->isAncestor(store))
          return false;
      } else if (!isa<cir::LoadOp, cir::LifetimeStartOp, cir::LifetimeEndOp>(
                     user))
        return false;
    }
    return initialized;
  }

  std::string leaf(Value value) {
    unsigned number;
    if (auto constant = value.getDefiningOp<cir::ConstantOp>()) {
      auto [it, inserted] =
          constantNumbers.try_emplace(constant.getValue(), leaves.size());
      number = it->second;
      if (inserted)
        leaves.push_back(value);
    } else {
      Value key = value;
      if (auto load = value.getDefiningOp<cir::LoadOp>())
        if (scope->isAncestor(load))
          key = load.getAddr();
      auto [it, inserted] = leafNumbers.try_emplace(key, leaves.size());
      number = it->second;
      if (inserted)
        leaves.push_back(value);
    }
    return "v" + std::to_string(number);
  }

  Value resolveTemporary(Value value) {
    for (unsigned i = 0; i < 64; ++i) {
      auto load = value.getDefiningOp<cir::LoadOp>();
      if (!load || !singleStores.count(load.getAddr()))
        break;
      value = singleStores.lookup(load.getAddr());
    }
    return value;
  }

  bool hasContract(Value value) {
    return value.getDefiningOp() &&
           value.getDefiningOp()->getAttrOfType<UnitAttr>("cir.contract");
  }

  // A per-instruction permission is required on both members of a pair.
  // cir.fmuladd already represents exactly that permission.
  bool canContract(Value value, bool lhsProduct = false) {
    value = resolveTemporary(value);
    if (auto fma = value.getDefiningOp<cir::FMulAddOp>()) {
      // expression() puts a bare accumulator before the product. Preserve
      // which pair was fused, including sum + beta*C without an alpha scale.
      bool productFirst = expression(fma->getOperand(2)) != "sum";
      return lhsProduct == productFirst;
    }
    auto add = value.getDefiningOp<cir::FAddOp>();
    if (!add || !hasContract(value))
      return false;
    Value product = resolveTemporary(lhsProduct ? add.getLhs() : add.getRhs());
    return product.getDefiningOp<cir::FMulOp>() && hasContract(product);
  }

  std::string expression(Value value, unsigned depth = 0) {
    if (depth > 64 || !value) {
      valid = false;
      return "?";
    }
    if (!value.getDefiningOp() || !scope->isAncestor(value.getDefiningOp())) {
      if (isInvariant(value))
        return leaf(value);
      valid = false;
      return "?";
    }
    Operation *op = value.getDefiningOp();
    auto e = [&](Value v) { return expression(v, depth + 1); };
    if (auto load = dyn_cast<cir::LoadOp>(op)) {
      Value addr = load.getAddr();
      if (auto it = variables.find(addr); it != variables.end())
        return it->second;
      if (auto it = singleStores.find(addr); it != singleStores.end())
        return e(it->second);
      if (isInvariant(value))
        return leaf(value);
      return "load(" + e(addr) + ")";
    }
    if (auto c = dyn_cast<cir::ConstantOp>(op)) {
      if (auto i = dyn_cast<cir::IntAttr>(c.getValue()))
        if (i.getValue().isZero())
          return "0";
      if (auto f = dyn_cast<cir::FPAttr>(c.getValue()))
        if (f.getValue().isPosZero())
          return "0";
    }
    if (isInvariant(value))
      return leaf(value);
    if (auto cast = dyn_cast<cir::CastOp>(op)) {
      Type src = cast.getSrc().getType(), dst = cast.getType();
      // Only the sign extension of address arithmetic, lossless promotion of
      // beta for comparison, and the literal double zero initializer.
      if ((isa<cir::IntType>(src) && isa<cir::IntType>(dst) &&
           cast.getKind() == cir::CastKind::integral &&
           llvm::cast<cir::IntType>(src).isSigned() &&
           llvm::cast<cir::IntType>(src).getWidth() == 32 &&
           llvm::cast<cir::IntType>(dst).isSigned() &&
           llvm::cast<cir::IntType>(dst).getWidth() == 64) ||
          (isa<cir::SingleType>(src) && isa<cir::DoubleType>(dst)) ||
          (isa<cir::DoubleType>(src) && isa<cir::SingleType>(dst) &&
           e(cast.getSrc()) == "0"))
        return e(cast.getSrc());
    }
    if (auto ternary = dyn_cast<cir::TernaryOp>(op)) {
      auto branch = [&](Region &r) -> std::string {
        if (!llvm::hasSingleElement(r)) {
          valid = false;
          return "?";
        }
        auto yield = dyn_cast<cir::YieldOp>(r.front().getTerminator());
        if (!yield || yield->getNumOperands() != 1 || !region(r).empty()) {
          valid = false;
          return "?";
        }
        return e(yield->getOperand(0));
      };
      return "select(" + e(op->getOperand(0)) + "," + branch(op->getRegion(0)) +
             "," + branch(op->getRegion(1)) + ")";
    }
    if (isa<cir::FMulAddOp>(op)) {
      std::string product =
          "fmul(" + e(op->getOperand(0)) + "," + e(op->getOperand(1)) + ")";
      std::string addend = e(op->getOperand(2));
      // The reduction uses sum + A*B; CIR's contractable op puts the
      // multiplication first. The epilogue already has that order.
      return addend == "sum" ? "fadd(sum," + product + ")"
                             : "fadd(" + product + "," + addend + ")";
    }
    if (auto cmp = dyn_cast<cir::CmpOp>(op))
      return std::string(cir::stringifyCmpOpKind(cmp.getKind())) + "(" +
             e(op->getOperand(0)) + "," + e(op->getOperand(1)) + ")";
    if (isa<cir::AddOp, cir::MulOp, cir::FAddOp, cir::FMulOp, cir::PtrStrideOp,
            cir::IncOp>(op)) {
      std::string result = op->getName().stripDialect().str() + "(";
      for (auto [i, operand] : llvm::enumerate(op->getOperands()))
        result += (i ? "," : "") + e(operand);
      return result + ")";
    }
    valid = false;
    return "?";
  }

  std::string region(Region &r) {
    if (r.empty())
      return "";
    if (!llvm::hasSingleElement(r)) {
      valid = false;
      return "?";
    }
    std::string result;
    for (Operation &op : r.front()) {
      if (auto store = dyn_cast<cir::StoreOp>(op)) {
        if (singleStores.contains(store.getAddr()))
          continue;
        std::string addr = variables.count(store.getAddr())
                               ? variables.lookup(store.getAddr())
                               : expression(store.getAddr());
        std::string rhs = expression(store.getValue());
        if (addr == "sum" && rhs != "0")
          reductionContract = canContract(store.getValue());
        if (StringRef(addr).starts_with("ptr_stride(") &&
            StringRef(rhs).starts_with("fadd(")) {
          if (canContract(store.getValue(), /*lhsProduct=*/true))
            epilogue.fusion = EpilogueFusion::Alpha;
          else if (canContract(store.getValue()))
            epilogue.fusion = EpilogueFusion::Beta;
        }
        result += addr + "=" + rhs + ";";
      } else if (isa<cir::ScopeOp, cir::CleanupScopeOp>(op)) {
        for (Region &nested : op.getRegions())
          result += region(nested);
      } else if (auto loop = dyn_cast<cir::ForOp>(op)) {
        result += "for(" + region(loop.getCond()) + "){" +
                  region(loop.getBody()) + "}{" + region(loop.getStep()) + "}";
        result += region(loop.getCleanup());
      } else if (isa<cir::IfOp>(op)) {
        result += "if(" + expression(op.getOperand(0)) + "){" +
                  region(op.getRegion(0)) + "}{" + region(op.getRegion(1)) +
                  "}";
      } else if (isa<cir::ConditionOp>(op)) {
        result += expression(op.getOperand(0));
      } else if (isa<cir::ReturnOp>(op)) {
        if (op.getNumOperands())
          valid = false;
        result += "return;";
      } else if (isa<cir::AllocaOp, cir::YieldOp, cir::LifetimeStartOp,
                     cir::LifetimeEndOp>(op)) {
        continue;
      } else if (isa<cir::LoadOp, cir::ConstantOp, cir::CastOp, cir::CmpOp,
                     cir::AddOp, cir::MulOp, cir::FAddOp, cir::FMulOp,
                     cir::PtrStrideOp, cir::IncOp, cir::TernaryOp,
                     cir::FMulAddOp>(op)) {
        // Validate even unused expressions, including ternary bodies.
        (void)expression(op.getResult(0));
      } else {
        valid = false;
      }
    }
    return result;
  }

  // Match only invariant leaves as parameters. Repeated placeholders must
  // designate the same value, regardless of argument position or local name.
  bool matchPattern(StringRef pattern, StringRef actual) {
    DenseMap<unsigned, StringRef> bindings;
    while (!pattern.empty()) {
      if (!pattern.consume_front("$")) {
        if (actual.empty() || pattern.front() != actual.front())
          return false;
        pattern = pattern.drop_front();
        actual = actual.drop_front();
        continue;
      }
      unsigned role;
      if (pattern.consumeInteger(10, role))
        return false;
      StringRef start = actual;
      if (actual.consume_front("v")) {
        unsigned number;
        if (actual.consumeInteger(10, number))
          return false;
      } else if (!actual.consume_front("0"))
        return false;
      StringRef token = start.take_front(start.size() - actual.size());
      auto [it, inserted] = bindings.try_emplace(role, token);
      if (!inserted && it->second != token)
        return false;
    }
    if (!actual.empty())
      return false;
    SmallVector<Type> types = getMatMulInputTypes(scope.getContext());
    inputs.assign(types.size(), Value());
    constants.assign(types.size(), TypedAttr());
    for (auto [role, token] : bindings) {
      if (token == "0") {
        if (isa<cir::SingleType>(types[role]))
          constants[role] = cir::FPAttr::getZero(types[role]);
        else if (auto integer = dyn_cast<cir::IntType>(types[role]))
          constants[role] = cir::IntAttr::get(integer, 0);
        else
          return false;
        continue;
      }
      unsigned number;
      if (token.drop_front().getAsInteger(10, number) ||
          number >= leaves.size())
        return false;
      Value value = leaves[number];
      if (value.getType() != types[role] || !isInvariant(value))
        return false;
      inputs[role] = value;
    }
    return true;
  }

public:
  explicit MatMulMatcher(cir::ScopeOp scope) : scope(scope) {}

  bool permitsReductionFusion() const { return reductionContract; }
  MatMulEpilogue getEpilogue() const { return epilogue; }

  // Outline exactly the checked scope. Capture its original live-ins for the
  // scalar fallback and append normalized scalar inputs for the dispatcher.
  // The surrounding function (including its ABI and other effects) is retained.
  cir::FuncOp outline(StringRef name) {
    cir::CIRBaseBuilderTy b(*scope.getContext());
    b.setInsertionPoint(scope);
    Location loc = scope.getLoc();
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(scope->getRegions(), captures);
    SmallVector<Value> arguments(captures.begin(), captures.end());
    for (auto [input, constant] : llvm::zip(inputs, constants)) {
      if (constant)
        arguments.push_back(cir::ConstantOp::create(b, loc, constant));
      else if (auto *def = input.getDefiningOp(); def && scope->isAncestor(def))
        arguments.push_back(b.clone(*def)->getResult(0));
      else
        arguments.push_back(input);
    }
    auto parent = scope->getParentOfType<cir::FuncOp>();
    b.setInsertionPoint(parent);
    auto type = cir::FuncType::get(llvm::to_vector(TypeRange(arguments)),
                                   cir::VoidType::get(b.getContext()));
    auto outlined = cir::FuncOp::create(b, loc, name, type);
    outlined.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
    SymbolTable::setSymbolVisibility(outlined,
                                     SymbolTable::Visibility::Private);
    // Preserve target attributes needed by normal CIR lowering.
    for (NamedAttribute attr : parent->getAttrs())
      if (attr.getName().strref().starts_with("cir."))
        outlined->setAttr(attr.getName(), attr.getValue());
    Block *entry = outlined.addEntryBlock();
    b.setInsertionPointToStart(entry);
    IRMapping mapping;
    for (auto [capture, argument] : llvm::zip(captures, entry->getArguments()))
      mapping.map(capture, argument);
    b.clone(*scope, mapping);
    cir::ReturnOp::create(b, loc);
    b.setInsertionPoint(scope);
    b.createCallOp(loc, outlined, arguments);
    scope.erase();
    return outlined;
  }

  bool matches() {
    if (scope->getNumResults() ||
        !llvm::hasSingleElement(scope.getScopeRegion()))
      return false;
    for (Operation *parent = scope->getParentOp(); parent;
         parent = parent->getParentOp())
      if (parent->hasAttr("fenv"))
        return false;
    unsigned numLoops = 0;
    scope.walk([&](cir::ForOp) {
      return ++numLoops > 3 ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (numLoops != 3)
      return false;
    scope.walk([&](cir::AllocaOp alloca) {
      SmallVector<cir::StoreOp> stores;
      for (Operation *user : alloca.getResult().getUsers()) {
        if (auto store = dyn_cast<cir::StoreOp>(user)) {
          if (store.getAddr() != alloca.getResult())
            valid = false;
          stores.push_back(store);
        } else if (!isa<cir::LoadOp, cir::LifetimeStartOp, cir::LifetimeEndOp>(
                       user)) {
          valid = false;
        }
      }
      if (stores.size() == 1) {
        for (Operation *user : alloca.getResult().getUsers()) {
          if (!isa<cir::LoadOp>(user))
            continue;
          Operation *definition = stores[0];
          // A plain lexical scope executes unconditionally. Follow it outward
          // for locals assigned inside a scoped FP pragma. Never look through
          // conditionals or loops, which might not execute the store.
          while (!dominance.dominates(definition, user) &&
                 isa<cir::ScopeOp>(definition->getParentOp()))
            definition = definition->getParentOp();
          if (!dominance.dominates(definition, user))
            valid = false;
        }
        singleStores[alloca.getResult()] = stores[0].getValue();
      }
    });
    unsigned loopNumber = 0;
    scope.walk<WalkOrder::PreOrder>([&](cir::ForOp loop) {
      if (!llvm::hasSingleElement(loop.getCond())) {
        valid = false;
        return;
      }
      auto condition =
          dyn_cast<cir::ConditionOp>(loop.getCond().front().getTerminator());
      auto cmp = condition
                     ? condition->getOperand(0).getDefiningOp<cir::CmpOp>()
                     : nullptr;
      auto load =
          cmp ? cmp->getOperand(0).getDefiningOp<cir::LoadOp>() : nullptr;
      if (!load || loopNumber >= 3 || variables.contains(load.getAddr())) {
        valid = false;
        return;
      }
      auto alloca = load.getAddr().getDefiningOp<cir::AllocaOp>();
      if (!alloca || !scope->isAncestor(alloca)) {
        valid = false;
        return;
      }
      auto type = dyn_cast<cir::IntType>(load.getType());
      if (!type || !type.isSigned() || type.getWidth() != 32) {
        valid = false;
        return;
      }
      variables[load.getAddr()] = std::string(1, "ijk"[loopNumber++]);
    });
    unsigned accumulators = 0;
    scope.walk([&](cir::AllocaOp alloca) {
      Value addr = alloca.getResult();
      if (!singleStores.contains(addr) && !variables.contains(addr)) {
        if (!isa<cir::SingleType>(alloca.getAllocaType()))
          valid = false;
        variables[addr] = "sum";
        ++accumulators;
      }
    });
    scope.walk([&](Operation *op) {
      if (op->hasAttr("fenv"))
        valid = false;
      // Widening address arithmetic is only valid when signed overflow in
      // the original computation was undefined. In particular, reject
      // -fwrapv and saturating arithmetic.
      if (auto add = dyn_cast<cir::AddOp>(op))
        valid &= add.getNoSignedWrap() && !add.getSaturated();
      if (auto mul = dyn_cast<cir::MulOp>(op))
        valid &= mul.getNoSignedWrap();
      if (auto inc = dyn_cast<cir::IncOp>(op))
        valid &= inc.getNoSignedWrap();
      if (auto load = dyn_cast<cir::LoadOp>(op))
        valid &= !load.getIsVolatile() && !load.getMemOrder();
      if (auto store = dyn_cast<cir::StoreOp>(op))
        valid &= !store.getIsVolatile() && !store.getMemOrder();
    });
    if (!valid || loopNumber != 3 || accumulators != 1 ||
        !checkLoadTimes(scope.getScopeRegion()))
      return false;
    std::string actual = region(scope.getScopeRegion());
    if (!valid)
      return false;
    // Both layouts may be fixed or selected by an invariant transpose flag.
    const std::string aNormal = "load(ptr_stride($6,add(mul(i,$7),k)))";
    const std::string aTranspose = "load(ptr_stride($6,add(mul(k,$7),i)))";
    const std::string bNormal = "load(ptr_stride($8,add(mul(k,$9),j)))";
    const std::string bTranspose = "load(ptr_stride($8,add(mul(j,$9),k)))";
    SmallVector<std::string> aPatterns{
        "select($0," + aTranspose + "," + aNormal + ")", aNormal, aTranspose};
    SmallVector<std::string> bPatterns{
        "select($1," + bTranspose + "," + bNormal + ")", bNormal, bTranspose};
    const std::string c = "ptr_stride($11,add(mul(i,$12),j))";
    enum class OutputKind { Guarded, Scaled, None, Unscaled };
    for (auto [ai, a] : llvm::enumerate(aPatterns))
      for (auto [bi, b] : llvm::enumerate(bPatterns))
        for (bool scaleAlpha : {true, false})
          for (OutputKind output : {OutputKind::Guarded, OutputKind::Scaled,
                                    OutputKind::None, OutputKind::Unscaled}) {
            std::string scaled = scaleAlpha ? "fmul($5,sum)" : "sum";
            std::string old = output == OutputKind::Unscaled
                                  ? "load(" + c + ")"
                                  : "fmul($10,load(" + c + "))";
            std::string store = c + "=fadd(" + scaled + "," + old + ");";
            if (output == OutputKind::Guarded)
              store = "if(eq($10,0)){" + c + "=" + scaled + ";}{" + store + "}";
            else if (output == OutputKind::None)
              store = c + "=" + scaled + ";";
            std::string pattern =
                "i=0;for(lt(i,$2)){j=0;for(lt(j,$3)){sum=0;k=0;for(lt(k,$4)){"
                "sum=fadd(sum,fmul(" +
                a + "," + b + "));}{k=inc(k);}" + store +
                "}{j=inc(j);}}{i=inc(i);}";
            if (!matchPattern(pattern, actual))
              continue;
            MLIRContext *context = scope.getContext();
            if (ai)
              constants[0] = cir::BoolAttr::get(context, ai == 2);
            if (bi)
              constants[1] = cir::BoolAttr::get(context, bi == 2);
            if (!scaleAlpha)
              constants[5] = cir::FPAttr::get(cir::SingleType::get(context),
                                              APFloat(1.0f));
            if (output == OutputKind::None || output == OutputKind::Unscaled)
              constants[10] = cir::FPAttr::get(
                  cir::SingleType::get(context),
                  APFloat(output == OutputKind::None ? 0.0f : 1.0f));
            epilogue.scaleAlpha = scaleAlpha;
            epilogue.readOutput = output != OutputKind::None;
            epilogue.scaleBeta = output != OutputKind::Unscaled;
            epilogue.guardBeta = output == OutputKind::Guarded;
            return true;
          }
    return false;
  }
};

// Keep the C ABI (raw pointers and scalar arguments) at the boundary. Packing
// normalizes both transpose flags and arbitrary leading dimensions to
// contiguous memrefs. The initial prototype deliberately favors a simple,
// inspectable lowering over packing/tiling profitability; all scratch storage
// is released.
static func::FuncOp raiseMatMul(LLVM::LLVMFuncOp original, StringRef name,
                                MatMulEpilogue epilogue) {
  OpBuilder b(original);
  Location loc = original.getLoc();
  Type i1 = b.getI1Type(), i32 = b.getI32Type(), f32 = b.getF32Type();
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto type = b.getFunctionType({i1, i1, i32, i32, i32, f32, ptr, i32, ptr, i32,
                                 f32, ptr, i32, ptr, ptr, ptr},
                                {});
  auto f = func::FuncOp::create(b, loc, name, type);
  f->setAttr("cir.matmul", b.getUnitAttr());
  f->setAttr("cir.matmul_helper", b.getUnitAttr());
  Block *entry = f.addEntryBlock();
  b.setInsertionPointToStart(entry);
  auto arg = [&](unsigned i) -> Value { return entry->getArgument(i); };
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value fpzero =
      arith::ConstantFloatOp::create(b, loc, b.getF32Type(), APFloat(0.0f));
  auto index = [&](unsigned i) -> Value {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), arg(i));
  };
  Value m = arith::MaxSIOp::create(b, loc, index(2), zero);
  Value n = arith::MaxSIOp::create(b, loc, index(3), zero);
  Value k = arith::MaxSIOp::create(b, loc, index(4), zero);
  Value lda = index(7), ldb = index(9), ldc = index(12);
  Value nonempty = arith::AndIOp::create(
      b, loc, arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, m, zero),
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, n, zero));
  auto guard = scf::IfOp::create(b, loc, nonempty, false);
  b.setInsertionPointToStart(&guard.getThenRegion().front());
  auto memrefType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
  LLVMTypeConverter converter(b.getContext());
  auto view = [&](Value pointer, Value rows, Value cols) -> Value {
    auto descriptor =
        MemRefDescriptor::poison(b, loc, converter.convertType(memrefType));
    descriptor.setAllocatedPtr(b, loc, pointer);
    descriptor.setAlignedPtr(b, loc, pointer);
    descriptor.setConstantOffset(b, loc, 0);
    Value r = arith::IndexCastOp::create(b, loc, b.getI64Type(), rows);
    Value c = arith::IndexCastOp::create(b, loc, b.getI64Type(), cols);
    descriptor.setSize(b, loc, 0, r);
    descriptor.setSize(b, loc, 1, c);
    descriptor.setStride(b, loc, 0, c);
    descriptor.setConstantStride(b, loc, 1, 1);
    auto cast = UnrealizedConversionCastOp::create(
        b, loc, TypeRange{memrefType}, ValueRange{descriptor});
    cast->setAttr("cir.matmul_scratch", b.getUnitAttr());
    return cast.getResult(0);
  };
  Value a = view(arg(13), k, m);
  Value bb = view(arg(14), k, n);
  Value sums = view(arg(15), m, n);
  auto offset = [&](Value row, Value stride, Value col) -> Value {
    return arith::AddIOp::create(
        b, loc, arith::MulIOp::create(b, loc, row, stride), col);
  };
  auto address = [&](Value p, Value off) -> Value {
    Value i64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), off);
    return LLVM::GEPOp::create(b, loc, ptr, f32, p, ValueRange{i64});
  };
  auto loop2 = [&](Value x, Value y, auto body) {
    auto outer = scf::ForOp::create(b, loc, zero, x, one);
    b.setInsertionPointToStart(outer.getBody());
    auto inner = scf::ForOp::create(b, loc, zero, y, one);
    b.setInsertionPointToStart(inner.getBody());
    body(outer.getInductionVar(), inner.getInductionVar());
    b.setInsertionPointAfter(outer);
  };
  loop2(m, k, [&](Value i, Value kk) {
    Value off = arith::SelectOp::create(b, loc, arg(0), offset(kk, lda, i),
                                        offset(i, lda, kk));
    Value v = LLVM::LoadOp::create(b, loc, f32, address(arg(6), off));
    memref::StoreOp::create(b, loc, v, a, ValueRange{kk, i});
  });
  loop2(k, n, [&](Value kk, Value j) {
    Value off = arith::SelectOp::create(b, loc, arg(1), offset(j, ldb, kk),
                                        offset(kk, ldb, j));
    Value v = LLVM::LoadOp::create(b, loc, f32, address(arg(8), off));
    memref::StoreOp::create(b, loc, v, bb, ValueRange{kk, j});
  });
  linalg::FillOp::create(b, loc, ValueRange{fpzero}, ValueRange{sums});
  auto matmul =
      linalg::MatmulOp::create(b, loc, ValueRange{a, bb}, ValueRange{sums});
  // Store the packed left operand as K x M: each outer product then reads
  // contiguous vectors from both operands, regardless of source transposition.
  AffineExpr i, j, kk;
  bindDims(b.getContext(), i, j, kk);
  matmul.setIndexingMapsAttr(
      b.getAffineMapArrayAttr({AffineMap::get(3, 0, {kk, i}, b.getContext()),
                               AffineMap::get(3, 0, {kk, j}, b.getContext()),
                               AffineMap::get(3, 0, {i, j}, b.getContext())}));
  Value betaZero;
  if (epilogue.guardBeta)
    betaZero = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, arg(10),
                                     fpzero);
  loop2(m, n, [&](Value i, Value j) {
    Value sum = memref::LoadOp::create(b, loc, sums, ValueRange{i, j});
    Value scaled = sum;
    if (epilogue.scaleAlpha)
      scaled = arith::MulFOp::create(b, loc, arg(5), sum);
    Value dest = address(arg(11), offset(i, ldc, j));
    scf::IfOp condition;
    if (epilogue.guardBeta) {
      condition = scf::IfOp::create(b, loc, betaZero, true);
      b.setInsertionPointToStart(&condition.getThenRegion().front());
      LLVM::StoreOp::create(b, loc, scaled, dest);
      b.setInsertionPointToStart(&condition.getElseRegion().front());
    }
    Value result = scaled;
    if (epilogue.readOutput) {
      Value old = LLVM::LoadOp::create(b, loc, f32, dest);
      Value product = old;
      if (epilogue.scaleBeta)
        product = arith::MulFOp::create(b, loc, arg(10), old);
      if (epilogue.fusion == EpilogueFusion::Alpha && epilogue.scaleAlpha)
        result =
            LLVM::FMAOp::create(b, loc, f32, ValueRange{arg(5), sum, product});
      else if (epilogue.fusion == EpilogueFusion::Beta && epilogue.scaleBeta)
        result =
            LLVM::FMAOp::create(b, loc, f32, ValueRange{arg(10), old, scaled});
      else
        result = arith::AddFOp::create(b, loc, scaled, product);
    }
    LLVM::StoreOp::create(b, loc, result, dest);
    if (condition)
      b.setInsertionPointAfter(condition);
  });
  b.setInsertionPointAfter(guard);
  func::ReturnOp::create(b, loc);
  return f;
}

// Compute conservative byte intervals with wide arithmetic. The source uses
// signed 32-bit dimensions and strides: all extent computations fit in i128,
// even after multiplying by sizeof(float) and adding a 64-bit address.
// Negative/zero strides and transposed storage use the same interval formula.
static void buildDispatch(LLVM::LLVMFuncOp wrapper, func::FuncOp helper,
                          LLVM::LLVMFuncOp fallback, LLVM::LLVMFuncOp mallocFn,
                          LLVM::LLVMFuncOp freeFn, bool assumeNoAlias) {
  OpBuilder b(wrapper);
  Location loc = wrapper.getLoc();
  wrapper->setAttr("cir.matmul_dispatch", b.getUnitAttr());
  Block *entry = wrapper.addEntryBlock(b);
  b.setInsertionPointToStart(entry);
  auto normalized = entry->getArguments().take_back(13);
  auto arg = [&](unsigned i) -> Value { return normalized[i]; };
  Type wide = b.getIntegerType(128);
  auto constant = [&](uint64_t v) -> Value {
    return arith::ConstantOp::create(b, loc,
                                     b.getIntegerAttr(wide, APInt(128, v)));
  };
  Value zero = constant(0), one = constant(1), four = constant(4);
  // Tagged pointers, address wraparound, and ranges outside the untagged
  // AArch64 address space conservatively select the original computation.
  Value addressLimit = constant(uint64_t(1) << 56);
  auto integer = [&](unsigned i) -> Value {
    return arith::ExtSIOp::create(b, loc, wide, arg(i));
  };
  auto nonnegative = [&](unsigned i) -> Value {
    return arith::MaxSIOp::create(b, loc, integer(i), zero);
  };
  auto both = [&](Value x, Value y) -> Value {
    return arith::AndIOp::create(b, loc, x, y);
  };
  auto either = [&](Value x, Value y) -> Value {
    return arith::OrIOp::create(b, loc, x, y);
  };
  auto compare = [&](arith::CmpIPredicate pred, Value x, Value y) -> Value {
    return arith::CmpIOp::create(b, loc, pred, x, y);
  };
  Value m = nonnegative(2), n = nonnegative(3), k = nonnegative(4);
  Value nonempty = both(compare(arith::CmpIPredicate::sgt, m, zero),
                        compare(arith::CmpIPredicate::sgt, n, zero));
  auto active = scf::IfOp::create(b, loc, nonempty, false);
  b.setInsertionPointToStart(&active.getThenRegion().front());
  struct Interval {
    Value low, end, valid;
  };
  auto interval = [&](Value pointer, Value rows, Value cols, Value stride) {
    Value address = LLVM::PtrToIntOp::create(b, loc, b.getI64Type(), pointer);
    Value base = arith::ExtUIOp::create(b, loc, wide, address);
    Value lastRow = arith::MulIOp::create(
        b, loc, arith::SubIOp::create(b, loc, rows, one), stride);
    Value lowOffset = arith::MinSIOp::create(b, loc, lastRow, zero);
    Value endOffset = arith::AddIOp::create(
        b, loc, arith::MaxSIOp::create(b, loc, lastRow, zero), cols);
    Value low = arith::AddIOp::create(
        b, loc, base, arith::MulIOp::create(b, loc, lowOffset, four));
    Value end = arith::AddIOp::create(
        b, loc, base, arith::MulIOp::create(b, loc, endOffset, four));
    Value valid = both(compare(arith::CmpIPredicate::sge, low, zero),
                       compare(arith::CmpIPredicate::sle, end, addressLimit));
    return Interval{low, end, valid};
  };
  auto select = [&](Value flag, Value t, Value f) -> Value {
    return arith::SelectOp::create(b, loc, flag, t, f);
  };
  Value safe = arith::ConstantIntOp::create(b, loc, 1, 1);
  if (!assumeNoAlias) {
    Interval a = interval(arg(6), select(arg(0), k, m), select(arg(0), m, k),
                          integer(7));
    Interval bb = interval(arg(8), select(arg(1), n, k), select(arg(1), k, n),
                           integer(9));
    Interval c = interval(arg(11), m, n, integer(12));
    auto disjoint = [&](Interval x, Interval y) {
      return either(compare(arith::CmpIPredicate::sle, x.end, y.low),
                    compare(arith::CmpIPredicate::sle, y.end, x.low));
    };
    Value valid = both(both(a.valid, bb.valid), c.valid);
    safe = both(valid, both(disjoint(a, c), disjoint(bb, c)));
    // With K <= 0 neither input is read, irrespective of their pointer values.
    safe = either(compare(arith::CmpIPredicate::eq, k, zero), safe);
  }
  Value km = arith::MulIOp::create(b, loc, k, m);
  Value kn = arith::MulIOp::create(b, loc, k, n);
  Value mn = arith::MulIOp::create(b, loc, m, n);
  Value inputs = arith::AddIOp::create(b, loc, km, kn);
  Value elements = arith::AddIOp::create(b, loc, inputs, mn);
  Value bytes = arith::MulIOp::create(b, loc, elements, four);
  // Bound the single allocation and every subsequent signed index/GEP.
  safe = both(safe,
              compare(arith::CmpIPredicate::sle, bytes, constant(INT64_MAX)));
  auto dispatch = scf::IfOp::create(b, loc, safe, true);
  b.setInsertionPointToStart(&dispatch.getThenRegion().front());
  auto narrow = [&](Value value) -> Value {
    return arith::TruncIOp::create(b, loc, b.getI64Type(), value);
  };
  Value workspace =
      LLVM::CallOp::create(b, loc, mallocFn, ValueRange{narrow(bytes)})
          .getResult();
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  Value null = LLVM::ZeroOp::create(b, loc, ptr);
  Value allocated =
      LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::ne, workspace, null);
  auto allocation = scf::IfOp::create(b, loc, allocated, true);
  b.setInsertionPointToStart(&allocation.getThenRegion().front());
  SmallVector<Value> arguments(normalized);
  arguments.push_back(workspace);
  for (Value offset : {km, inputs})
    arguments.push_back(LLVM::GEPOp::create(
        b, loc, ptr, b.getF32Type(), workspace, ValueRange{narrow(offset)}));
  func::CallOp::create(b, loc, helper, arguments);
  LLVM::CallOp::create(b, loc, freeFn, ValueRange{workspace});
  b.setInsertionPointToStart(&allocation.getElseRegion().front());
  LLVM::CallOp::create(b, loc, fallback, entry->getArguments());
  b.setInsertionPointToStart(&dispatch.getElseRegion().front());
  LLVM::CallOp::create(b, loc, fallback, entry->getArguments());
  b.setInsertionPointAfter(active);
  LLVM::ReturnOp::create(b, loc, ValueRange{});
}

struct CIRRaiseMatMul : PassWrapper<CIRRaiseMatMul, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIRRaiseMatMul)
  bool enableOpenMP;
  explicit CIRRaiseMatMul(bool enableOpenMP = false)
      : enableOpenMP(enableOpenMP) {}
  CIRRaiseMatMul(const CIRRaiseMatMul &other)
      : PassWrapper(other), enableOpenMP(other.enableOpenMP) {}
  Option<bool> requireKernel{
      *this, "require-kernel",
      llvm::cl::desc("Fail if no eligible MatMul kernel is found"),
      llvm::cl::init(false)};
  Option<bool> assumeNoAlias{*this, "assume-no-alias",
                             llvm::cl::desc("Assume C does not overlap A or B"),
                             llvm::cl::init(false)};
  Option<bool> allowContract{
      *this, "allow-contract",
      llvm::cl::desc("Permit fused multiply-add in MatMul arithmetic"),
      llvm::cl::init(false)};
  StringRef getArgument() const final { return "cir-raise-matmul"; }
  StringRef getDescription() const final {
    return "Raise a checked CIR MatMul kernel to linalg and memref "
           "(experimental)";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
             linalg::LinalgDialect, memref::MemRefDialect, scf::SCFDialect>();
    OpPassManager pipeline(ModuleOp::getOperationName());
    cir::direct::populateCIRToLLVMPasses(pipeline, enableOpenMP);
    pipeline.getDependentDialects(r);
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto triple = module->getAttrOfType<StringAttr>("cir.triple");
    llvm::Triple target(triple ? triple.getValue() : StringRef());
    if (!triple || target.getArch() != llvm::Triple::aarch64 ||
        target.getEnvironment() == llvm::Triple::GNUILP32) {
      module.emitError(
          "MatMul raising currently requires an AArch64 LP64 CIR module");
      return signalPassFailure();
    }
    struct Kernel {
      std::string name;
      MatMulEpilogue epilogue;
    };
    SmallVector<Kernel> kernels;
    llvm::SmallSetVector<StringAttr, 4> comdatFunctions;
    bool missingContract = false;
    SmallVector<cir::ScopeOp> candidates;
    module.walk<WalkOrder::PreOrder>(
        [&](cir::ScopeOp scope) { candidates.push_back(scope); });
    // Outlining invalidates nested candidates; collect matches first and skip
    // descendants of scopes that have already been selected.
    SmallVector<cir::ScopeOp> selected;
    SmallVector<std::unique_ptr<MatMulMatcher>> matches;
    for (auto scope : candidates) {
      if (!scope->getParentOfType<cir::FuncOp>())
        continue;
      if (llvm::any_of(selected, [&](cir::ScopeOp parent) {
            return parent->isAncestor(scope);
          }))
        continue;
      auto matcher = std::make_unique<MatMulMatcher>(scope);
      if (!matcher->matches())
        continue;
      if (!allowContract && !matcher->permitsReductionFusion()) {
        missingContract = true;
        continue;
      }
      selected.push_back(scope);
      matches.push_back(std::move(matcher));
    }
    for (auto [scope, matcher] : llvm::zip(selected, matches)) {
      auto parent = scope->getParentOfType<cir::FuncOp>();
      if (parent.getComdat())
        comdatFunctions.insert(parent.getSymNameAttr());
      std::string name = (parent.getSymName() + ".cir_matmul").str();
      while (module.lookupSymbol(name))
        name += "_";
      MatMulEpilogue epilogue = matcher->getEpilogue();
      if (allowContract)
        epilogue.fusion =
            epilogue.scaleAlpha ? EpilogueFusion::Alpha : EpilogueFusion::Beta;
      matcher->outline(name);
      kernels.push_back({name, epilogue});
    }
    if (kernels.empty() && requireKernel) {
      module.emitError(
          missingContract
              ? "MatMul reduction requires contraction permission on its "
                "multiply/add instructions or the allow-contract option"
              : "unsupported MatMul kernel: no eligible FP32 loop nest found");
      return signalPassFailure();
    }

    // Match before flattening, then use normal CIR lowering for the entire
    // module, including unrelated functions, callers, globals and constructors.
    // The outlined scalar fallback retains the original memory/FP operations.
    OpPassManager pipeline(ModuleOp::getOperationName());
    cir::direct::populateCIRToLLVMPasses(pipeline, enableOpenMP);
    if (failed(runPipeline(pipeline, module)))
      return signalPassFailure();

    // Preserve CIR's string function annotations for standalone mlir-translate
    // as well as the frontend (which normally has CIR's translation extension).
    for (auto f : module.getOps<LLVM::LLVMFuncOp>()) {
      // CIR's C++ function classification is consumed by ABI lowering. Its
      // record types must not leak into standalone LLVM-dialect input.
      f->removeAttr("func_info");
      SmallVector<Attribute> passthrough;
      if (auto attrs = f.getPassthrough())
        llvm::append_range(passthrough, *attrs);
      SmallVector<StringAttr> remove;
      OpBuilder b(f);
      for (NamedAttribute attr : f->getAttrs()) {
        if (!attr.getName().strref().starts_with("cir."))
          continue;
        if (auto str = dyn_cast<StringAttr>(attr.getValue()))
          passthrough.push_back(b.getArrayAttr(
              {b.getStringAttr(attr.getName().strref().drop_front(4)), str}));
        remove.push_back(attr.getName());
      }
      for (StringAttr attr : remove)
        f->removeAttr(attr);
      if (!passthrough.empty())
        f.setPassthroughAttr(b.getArrayAttr(passthrough));
    }
    auto uniqueName = [&](StringRef base) {
      std::string name = base.str();
      while (module.lookupSymbol(name))
        name += "_";
      return name;
    };
    // CIR's normal lowering does not yet transfer COMDAT. Retain it on the
    // original enclosing functions, whose signatures/linkage were not changed.
    for (StringAttr name : comdatFunctions) {
      auto original = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
      if (!original || original.getComdat())
        continue;
      OpBuilder b(module.getBodyRegion());
      auto comdat = LLVM::ComdatOp::create(b, original.getLoc(),
                                           uniqueName("__cir_matmul_comdat"));
      b.setInsertionPointToStart(&comdat.getBody().back());
      auto selector =
          LLVM::ComdatSelectorOp::create(b, original.getLoc(), name.getValue(),
                                         LLVM::comdat::Comdat::Any, nullptr);
      original.setComdatAttr(SymbolRefAttr::get(
          b.getContext(), comdat.getSymName(),
          FlatSymbolRefAttr::get(selector.getSymNameAttr())));
    }
    for (const Kernel &kernel : kernels) {
      auto fallback = module.lookupSymbol<LLVM::LLVMFuncOp>(kernel.name);
      assert(fallback && "CIR lowering must retain the kernel definition");
      OpBuilder b(fallback);
      auto mallocFn = LLVM::lookupOrCreateMallocFn(b, module, b.getI64Type());
      auto freeFn = LLVM::lookupOrCreateFreeFn(b, module);
      if (failed(mallocFn) || failed(freeFn))
        return signalPassFailure();
      auto wrapper = cast<LLVM::LLVMFuncOp>(fallback->cloneWithoutRegions());
      fallback.setSymName(uniqueName(kernel.name + "_fallback"));
      b.insert(wrapper);
      fallback.setLinkage(LLVM::Linkage::Internal);
      fallback.setVisibility_(LLVM::Visibility::Default);
      fallback.removeComdatAttr();
      fallback.removeAlwaysInlineAttr();
      fallback.removeInlineHintAttr();
      fallback.setNoInline(true);
      auto helper = raiseMatMul(wrapper, uniqueName(kernel.name + "_sme"),
                                kernel.epilogue);
      helper.setPrivate();
      helper->setAttr(
          "llvm.linkage",
          LLVM::LinkageAttr::get(&getContext(), LLVM::Linkage::Internal));
      SmallVector<Attribute> helperAttrs{b.getStringAttr("noinline")};
      if (auto attrs = wrapper.getPassthrough())
        llvm::append_range(helperAttrs, *attrs);
      helper->setAttr("llvm.passthrough", b.getArrayAttr(helperAttrs));
      buildDispatch(wrapper, helper, fallback, *mallocFn, *freeFn,
                    assumeNoAlias);
    }
    SmallVector<StringAttr> remove;
    for (NamedAttribute attr : module->getAttrs())
      if (attr.getName().strref().starts_with("cir."))
        remove.push_back(attr.getName());
    for (StringAttr attr : remove)
      module->removeAttr(attr);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cir::createRaiseMatMulPass(bool enableOpenMP) {
  return std::make_unique<CIRRaiseMatMul>(enableOpenMP);
}
