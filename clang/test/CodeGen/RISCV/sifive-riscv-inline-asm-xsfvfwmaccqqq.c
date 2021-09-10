// RUN: %clang_cc1 -triple riscv32 -target-feature +v \
// RUN:     -target-feature +f -target-feature +xsfvfwmaccqqq \
// RUN:     -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v\
// RUN:     -target-feature +f -target-feature +xsfvfwmaccqqq \
// RUN:     -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

#include <riscv_vector.h>

vfloat32m2_t sf_vfwmacc_4x4x4 (vint16m1_t a, vint16m1_t b) {
// CHECK-LABEL: define{{.*}} @sf_vfwmacc_4x4x4
// CHECK: %0 = tail call <vscale x 4 x float> asm sideeffect "sf.vfwmacc.4x4x4 $0, $1, $2", "=&^vr,^vr,^vr"(<vscale x 4 x i16> %a, <vscale x 4 x i16> %b)
vfloat32m2_t ret;
  asm volatile ("sf.vfwmacc.4x4x4 %0, %1, %2" : "=&vr"(ret) : "vr"(a), "vr"(b));
  return ret;
}
