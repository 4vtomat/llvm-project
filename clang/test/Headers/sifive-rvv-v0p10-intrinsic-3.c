// RUN: %clang_cc1 -U __rvv_0p10_compatible_intrinsics -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding %s -verify
// RUN: %clang_cc1 -U __rvv_0p10_compatible_intrinsics -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding -flax-vector-conversions=none %s -verify

#include <riscv_vector.h>

vint8mf8_t test_vadd_vv_i8mf8(vint8mf8_t op1, vint8mf8_t op2, size_t vl) {
  return vadd_vv_i8mf8(op1, op2, vl); /* expected-error {{call to undeclared function 'vadd_vv_i8mf8'; ISO C99 and later do not support implicit function declarations}} expected-error {{returning 'int' from a function with incompatible result type 'vint8mf8_t' (aka '__rvv_int8mf8_t')}} */
}

vint8mf8_t test_vadd(vint8mf8_t op1, vint8mf8_t op2, size_t vl) {
  return vadd(op1, op2, vl); /* expected-error {{call to undeclared function 'vadd'; ISO C99 and later do not support implicit function declarations}} expected-error {{returning 'int' from a function with incompatible result type 'vint8mf8_t' (aka '__rvv_int8mf8_t')}} */
}
