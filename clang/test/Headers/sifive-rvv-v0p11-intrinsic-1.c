// RUN: %clang_cc1 -U __rvv_0p11_compatible_intrinsics -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding %s -verify
// RUN: %clang_cc1 -U __rvv_0p11_compatible_intrinsics -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding -flax-vector-conversions=none %s -verify

#define __rvv_0p10_compatible_intrinsics
#include <riscv_vector.h>

// expected-warning@rvv_v0p10_compatible/riscv_vector.h:4 {{The RVV intrinsic version 0.10 compatible header is deprecated and will be removed in the next release. Please refer to the latest version of the interfaces.}}

void test_vlseg2e32_v_i32m1(vint32m1_t *v0, vint32m1_t *v1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_v_i32m1(v0, v1, base, vl); /* expected-error {{call to undeclared function '__riscv_vlseg2e32_v_i32m1'; ISO C99 and later do not support implicit function declarations}} expected-error {{void function 'test_vlseg2e32_v_i32m1' should not return a value}} */
}

vint8mf8_t test_vaadd_vv_i8mf8(vint8mf8_t op1, vint8mf8_t op2, size_t vl) {
  return __riscv_vaadd_vv_i8mf8(op1, op2, vl); /* expected-error {{too few arguments to function call, expected 4, have 3}} */
}
