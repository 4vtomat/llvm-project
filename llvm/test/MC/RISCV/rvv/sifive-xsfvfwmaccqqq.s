# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xsfvfwmaccqqq,+f %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfvfwmaccqqq,+f %s \
# RUN:        | llvm-objdump -d --mattr=+xsfvfwmaccqqq,+f - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfvfwmaccqqq,+f %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sf.vfwmacc.4x4x4 v4, v8, v12
# CHECK-INST: sf.vfwmacc.4x4x4 v4, v8, v12
# CHECK-ENCODING: [0x5b,0x12,0xc4,0xf2]
# CHECK-ERROR: instruction requires the following: 'Xsfvfwmaccqqq' (SiFive custom BF16 matrix arithmetic vector instructions)
# CHECK-UNKNOWN: 5b 12 c4 f2 <unknown>
