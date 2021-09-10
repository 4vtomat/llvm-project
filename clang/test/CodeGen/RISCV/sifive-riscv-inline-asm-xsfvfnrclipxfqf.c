// RUN: %clang_cc1 -triple riscv32 -target-feature +v \
// RUN:     -target-feature +f -target-feature +xsfvfnrclipxfqf \
// RUN:     -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:     -target-feature +f -target-feature +xsfvfnrclipxfqf \
// RUN:     -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

#include <riscv_vector.h>

vuint8m1_t sf_vfnrclip_xu_f_qf (vfloat32m4_t a, float b) {
// CHECK-LABEL: define{{.*}} @sf_vfnrclip_xu_f_qf
// CHECK: %0 = tail call <vscale x 8 x i8> asm sideeffect "sf.vfnrclip.xu.f.qf $0, $1, $2", "=^vr,^vr,f"(<vscale x 8 x float> %a, float %b)
vuint8m1_t ret;
  asm volatile ("sf.vfnrclip.xu.f.qf %0, %1, %2" : "=vr"(ret) : "vr"(a), "f"(b));
  return ret;
}

vint8m1_t sf_vfnrclip_x_f_qf (vfloat32m4_t a, float b) {
// CHECK-LABEL: define{{.*}} @sf_vfnrclip_x_f_qf
// CHECK: %0 = tail call <vscale x 8 x i8> asm sideeffect "sf.vfnrclip.x.f.qf $0, $1, $2", "=^vr,^vr,f"(<vscale x 8 x float> %a, float %b)
vint8m1_t ret;
  asm volatile ("sf.vfnrclip.x.f.qf %0, %1, %2" : "=vr"(ret) : "vr"(a), "f"(b));
  return ret;
}
