// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding -I %resource_dir/rvv_v0p10_compatible %s -verify
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding -flax-vector-conversions=none -I %resource_dir/rvv_v0p10_compatible %s -verify
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only -ffreestanding -x c++ -I %resource_dir/rvv_v0p10_compatible %s -verify

#define __rvv_0p11_compatible_intrinsics
#include <riscv_vector.h>

// expected-warning@rvv_v0p10_compatible/riscv_vector.h:4 {{The RVV intrinsic version 0.10 compatible header is deprecated and will be removed in the next release. Please refer to the latest version of the interfaces.}}
// expected-warning@rvv_v0p11_compatible/riscv_vector.h:4 {{The RVV intrinsic version 0.11 compatible header is deprecated and will be removed in the next release. Please refer to the latest version of the interfaces.}}

vint8mf8_t test_vadd_vv_i8mf8(vint8mf8_t op1, vint8mf8_t op2, size_t vl) {
  return vadd_vv_i8mf8(op1, op2, vl);
}
