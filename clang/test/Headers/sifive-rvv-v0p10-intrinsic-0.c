// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding %s -verify
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding -flax-vector-conversions=none %s -verify
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding -x c++ %s -verify
// expected-no-diagnostics


#include <rvv_v0p10_compatible/riscv_vector.h>

vint8mf8_t test_vadd_vv_i8mf8(vint8mf8_t op1, vint8mf8_t op2, size_t vl) {
  return vadd_vv_i8mf8(op1, op2, vl);
}
