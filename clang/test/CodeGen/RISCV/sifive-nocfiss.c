// RUN: %clang --target=riscv64 -march=rv64gc -fsanitize=shadow-call-stack %s -S -emit-llvm -o - | FileCheck %s

// CHECK-NOT: shadowcallstack
__attribute__((no_cfi_ss)) void bar(void) {}
[[riscv::no_cfi_ss]] void bar2(void) {}
