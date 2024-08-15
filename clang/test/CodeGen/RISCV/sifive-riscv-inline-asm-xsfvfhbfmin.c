// RUN: %clang_cc1 -triple riscv32 -target-feature +f \
// RUN:     -target-feature +v \
// RUN:     -target-feature +xsfvfhbfmin -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +f \
// RUN:     -target-feature +v \
// RUN:     -target-feature +xsfvfhbfmin -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// There is no __bf16 for RISC-V in the current stage. We use int16 to store
// the values of __bf16.
#include <riscv_vector.h>

vfloat32m2_t from_bf16(vint16m1_t in) {
// CHECK-LABEL: define{{.*}} @from_bf16
// CHECK: %0 = tail call <vscale x 4 x float> asm sideeffect "sf.vfwcvt.f.bf16.v $0, $1", "=^vr,^vr,~{vl},~{vtype}"(<vscale x 4 x i16> %in)
  vfloat32m2_t ret;
  asm volatile ("sf.vfwcvt.f.bf16.v %0, %1" : "=vr"(ret) : "vr"(in));
  return ret;
}

vint16m1_t to_bf16(vfloat32m2_t in) {
// CHECK-LABEL: define{{.*}} @to_bf16
// CHECK: %0 = tail call <vscale x 4 x i16> asm sideeffect "sf.vfncvt.bf16.f.w $0, $1", "=^vr,^vr,~{vl},~{vtype}"(<vscale x 4 x float> %in)
  vint16m1_t ret;
  asm volatile ("sf.vfncvt.bf16.f.w %0, %1" : "=vr"(ret) : "vr"(in));
  return ret;
}
