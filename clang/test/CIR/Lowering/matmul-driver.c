// REQUIRES: aarch64-registered-target
// RUN: %clang -### --target=aarch64-linux-gnu -fclangir -fclangir-matmul-to-sme -fclangir-matmul-options='allow-contract assume-no-alias' -c %s 2>&1 | FileCheck %s --check-prefix=FORWARD
// RUN: not %clang -### -fclangir-matmul-to-sme -c %s 2>&1 | FileCheck %s --check-prefix=NEEDS-CIR
// RUN: not %clang -### -fclangir -fclangir-matmul-options=allow-contract -c %s 2>&1 | FileCheck %s --check-prefix=NEEDS-PASS
// RUN: not %clang --target=aarch64-linux-gnu -fclangir -fclangir-matmul-to-sme -S %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEEDS-SME
// RUN: not %clang --target=aarch64-linux-gnu -march=armv9-a+sme -fclangir -fclangir-matmul-to-sme -fclangir-matmul-options=invalid -S %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-OPTION
// FORWARD: "-fclangir" "-fclangir-matmul-to-sme" "-fclangir-matmul-options=allow-contract assume-no-alias"
// NEEDS-CIR: argument '-fclangir-matmul-to-sme' only allowed with '-fclangir'
// NEEDS-PASS: argument '-fclangir-matmul-options=' only allowed with '-fclangir-matmul-to-sme'
// NEEDS-SME: -fclangir-matmul-to-sme requires an AArch64 LP64 target with SME
// BAD-OPTION: invalid
int unrelated(int a) { return a + 1; }
