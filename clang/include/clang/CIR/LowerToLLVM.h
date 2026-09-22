//====- LowerToLLVM.h- Lowering from CIR to LLVM --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for converting CIR modules to LLVM IR.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_LOWERTOLLVM_H
#define CLANG_CIR_LOWERTOLLVM_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace llvm {
class LLVMContext;
class Module;
namespace vfs {
class FileSystem;
} // namespace vfs
} // namespace llvm

namespace mlir {
class ModuleOp;
class OpPassManager;
} // namespace mlir

namespace cir {

namespace direct {
std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp mlirModule,
                             llvm::LLVMContext &llvmCtx, bool enableOpenMP,
                             llvm::StringRef mlirSaveTempsOutFile = {},
                             llvm::vfs::FileSystem *fs = nullptr,
                             llvm::function_ref<void(mlir::OpPassManager &)>
                                 populatePipeline = {});
} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERTOLLVM_H
