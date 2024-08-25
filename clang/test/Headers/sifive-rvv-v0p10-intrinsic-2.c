// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding %s -verify
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding -flax-vector-conversions=none %s -verify
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding -x c++ %s -verify
// expected-no-diagnostics

#define __rvv_0p10_compatible_intrinsics
#include <riscv_vector.h>

vint8mf8_t test_vadd_vv_i8mf8(vint8mf8_t op1, vint8mf8_t op2, size_t vl) {
  return vadd_vv_i8mf8(op1, op2, vl);
}

vint8mf8_t test_vadd(vint8mf8_t op1, vint8mf8_t op2, size_t vl) {
  return vadd(op1, op2, vl);
}
