// RUN: %clang -target riscv32-unknown-elf -march=rv32ia2p1 -### %s -fsyntax-only 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+a"

// RUN: %clang -target riscv32-unknown-elf -march=rv32ia2p2 -### %s -fsyntax-only 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-UNSUPPORTED-VERSION %s
// CHECK-UNSUPPORTED-VERSION: error: invalid arch name 'rv32ia2p2', unsupported version number 2.2 for
// CHECK-UNSUPPORTED-VERSION: extension 'a' (this compiler supports 2.1)

// RUN: %clang -target riscv32-unknown-elf -march=rv32gzba1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-EXP-VERSION %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32gzba0p93 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-EXP-VERSION-1 %s
//
// CHECK-EXP-VERSION: "-target-feature" "+zba"
// CHECK-EXP-VERSION-1: "-target-feature" "+zba0p93"

// RUN: %clang -target riscv32-unknown-elf -march=rv32gzba1p1 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-EXP-UNSUPPORTED-VERSION %s
//
// CHECK-EXP-UNSUPPORTED-VERSION: error: invalid arch name 'rv32gzba1p1', unsupported version number 1.1
// CHECK-EXP-UNSUPPORTED-VERSION: for extension 'zba' (this compiler supports 1.0, 0.93)
