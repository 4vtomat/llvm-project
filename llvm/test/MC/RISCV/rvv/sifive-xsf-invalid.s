# RUN: not llvm-mc -triple=riscv64 --mattr=+v,+xsfvfwmaccqqq %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

sf.vfwmacc.4x4x4 v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: sf.vfwmacc.4x4x4 v2, v2, v4
