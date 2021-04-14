# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+xsfvfhbfmin %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvfhbfmin %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvfhbfmin - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvfhbfmin %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sf.vfwcvt.f.bf16.v v8, v4, v0.t
# CHECK-INST: sf.vfwcvt.f.bf16.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x46,0x48]
# CHECK-ERROR: instruction requires the following: 'Xsfvfhbfmin'
# CHECK-UNKNOWN: 57 94 46 48 <unknown>

sf.vfwcvt.f.bf16.v v8, v4
# CHECK-INST: sf.vfwcvt.f.bf16.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x46,0x4a]
# CHECK-ERROR: instruction requires the following: 'Xsfvfhbfmin'
# CHECK-UNKNOWN: 57 94 46 4a <unknown>

sf.vfncvt.bf16.f.w v8, v4, v0.t
# CHECK-INST: sf.vfncvt.bf16.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4e,0x48]
# CHECK-ERROR: instruction requires the following: 'Xsfvfhbfmin'
# CHECK-UNKNOWN: 57 94 4e 48 <unknown>

sf.vfncvt.bf16.f.w v8, v4
# CHECK-INST: sf.vfncvt.bf16.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4e,0x4a]
# CHECK-ERROR: instruction requires the following: 'Xsfvfhbfmin'
# CHECK-UNKNOWN: 57 94 4e 4a <unknown>
