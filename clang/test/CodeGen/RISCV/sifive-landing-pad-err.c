// RUN: ! %clang --target=riscv64 -march=rv64gc %s -S -emit-llvm -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: expected landing pad value fits in unsigned 20-bits, but the input label value is 1048576
__attribute__((landing_pad_value(1048576)))
void bar (void){}

// CHECK: error: expected landing pad value fits in unsigned 20-bits, but the input label value is 4294967295
__attribute__((landing_pad_value(-1)))
void bar2 (void){}
