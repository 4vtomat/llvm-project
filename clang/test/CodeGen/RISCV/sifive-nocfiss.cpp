// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 %s -emit-llvm -fsanitize=shadow-call-stack -o - | FileCheck %s

// CHECK-NOT: shadowcallstack
[[riscv::no_cfi_ss]] void bar2(void) {}
