// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64-unknown-linux-gnu -emit-llvm \
// RUN: -target-feature +v -target-abi lp64d -debug-info-kind=constructor \
// RUN: -O1 -fcxx-exceptions %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: Invalid size request on a scalable vector.

typedef __rvv_uint32m4_t a;
void b() { a c; }
