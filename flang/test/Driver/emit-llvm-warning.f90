! This intentionally exercises LLVM IR generation to test the
! flang-integration-test-check GitHub workflow.
! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

subroutine emit_llvm_warning_test
end subroutine

! CHECK-LABEL: define void @emit_llvm_warning_test_()
