# RUN: not llvm-mc -triple=riscv64 --mattr=+v,+xsfvfwmaccqqq,+xsfvqmaccdod,+xsfvqmaccqoq %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

sf.vfwmacc.4x4x4 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vfwmacc.4x4x4 v2, v2, v4

sf.vqmaccu.4x8x4 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vqmaccu.4x8x4 v2, v2, v4

sf.vqmacc.4x8x4 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vqmacc.4x8x4 v2, v2, v4

sf.vqmaccus.4x8x4 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vqmaccus.4x8x4 v2, v2, v4

sf.vqmaccsu.4x8x4 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vqmaccsu.4x8x4 v2, v2, v4

sf.vqmaccu.2x8x2 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vqmaccu.2x8x2 v2, v2, v4

sf.vqmacc.2x8x2 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vqmacc.2x8x2 v2, v2, v4

sf.vqmaccus.2x8x2 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vqmaccus.2x8x2 v2, v2, v4

sf.vqmaccsu.2x8x2 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vqmaccsu.2x8x2 v2, v2, v4
