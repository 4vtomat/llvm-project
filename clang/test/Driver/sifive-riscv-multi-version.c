// RUN: %clang -target riscv32-unknown-elf -march=rv32ia2p1 -### %s -fsyntax-only 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+a"

// RUN: not %clang -target riscv32-unknown-elf -march=rv32ia2p2 -### %s -fsyntax-only 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-UNSUPPORTED-VERSION %s
// CHECK-UNSUPPORTED-VERSION: error: invalid arch name 'rv32ia2p2', unsupported version number 2.2 for
// CHECK-UNSUPPORTED-VERSION: extension 'a' (this compiler supports 2.1)

// RUN: %clang -target riscv32-unknown-elf -march=rv32g_zve64x_zvkg1p0 -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-EXP-VERSION %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32g_zve64x_zvkg0p1 -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-EXP-VERSION-1 %s
//
// CHECK-EXP-VERSION: "-target-feature" "+zvkg"
// CHECK-EXP-VERSION-1: "-target-feature" "+zvkg0p1"

// RUN: not %clang -target riscv32-unknown-elf -march=rv32gzba1p1 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-EXP-UNSUPPORTED-VERSION %s
//
// CHECK-EXP-UNSUPPORTED-VERSION: error: invalid arch name 'rv32gzba1p1', unsupported version number 1.1
// CHECK-EXP-UNSUPPORTED-VERSION: for extension 'zba' (this compiler supports 1.0)
