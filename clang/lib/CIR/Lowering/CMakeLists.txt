set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LLVM_OPTIONAL_SOURCES CIRToLinalg.cpp CIRToSME.cpp)

add_clang_library(clangCIRLoweringCommon
  CIRPasses.cpp
  LoweringHelpers.cpp

  LINK_LIBS
  clangCIR
  ${dialect_libs}
  MLIRCIR
  MLIRCIRTransforms
  MLIRTransforms
  MLIRSupport
  )

add_subdirectory(DirectToLLVM)

# Opt-in MatMul recognition, exposed through cir-opt and the ClangIR frontend.
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_clang_library(clangCIRLoweringToLinalg
  CIRToLinalg.cpp
  CIRToSME.cpp

  LINK_LIBS
  clangCIRLoweringDirectToLLVM
  MLIRCIR
  ${dialect_libs}
  ${conversion_libs}
  MLIRLinalgTransforms
  MLIRVectorTransforms
  MLIRArmSMETransforms
  MLIRTransforms
  )
