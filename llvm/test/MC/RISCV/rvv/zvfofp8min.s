# RUN: llvm-mc -triple=riscv32 -show-encoding -mattr=+experimental-zvfofp8min %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+experimental-zvfofp8min %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfofp8min - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+experimental-zvfofp8min %s \
# RUN:    | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfofp8min %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfofp8min %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfofp8min - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfofp8min %s \
# RUN:    | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# CHECK-INST: vfwcvtbf16.f.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x46,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvfbfmin' or 'Zvfofp8min'{{$}}
# CHECK-UNKNOWN: 48469457 <unknown>
vfwcvtbf16.f.f.v v8, v4, v0.t

# CHECK-INST: vfwcvtbf16.f.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x46,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvfbfmin' or 'Zvfofp8min'{{$}}
# CHECK-UNKNOWN: 4a469457 <unknown>
vfwcvtbf16.f.f.v v8, v4

# CHECK-INST: vfncvtbf16.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4e,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvfbfmin' or 'Zvfofp8min'{{$}}
# CHECK-UNKNOWN: 484e9457 <unknown>
vfncvtbf16.f.f.w v8, v4, v0.t

# CHECK-INST: vfncvtbf16.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4e,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvfbfmin' or 'Zvfofp8min'{{$}}
# CHECK-UNKNOWN: 4a4e9457 <unknown>
vfncvtbf16.f.f.w v8, v4

# CHECK-INST: vfncvtbf16.sat.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4f,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvfofp8min' (OFP8 conversion extension Zvfofp8min){{$}}
# CHECK-UNKNOWN: 484f9457 <unknown>
vfncvtbf16.sat.f.f.w v8, v4, v0.t

# CHECK-INST: vfncvtbf16.sat.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4f,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvfofp8min' (OFP8 conversion extension Zvfofp8min){{$}}
# CHECK-UNKNOWN: 4a4f9457 <unknown>
vfncvtbf16.sat.f.f.w v8, v4

# CHECK-INST: vfncvt.f.f.q v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4c,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvfofp8min' (OFP8 conversion extension Zvfofp8min){{$}}
# CHECK-UNKNOWN: 484c9457 <unknown>
vfncvt.f.f.q v8, v4, v0.t

# CHECK-INST: vfncvt.f.f.q v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4c,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvfofp8min' (OFP8 conversion extension Zvfofp8min){{$}}
# CHECK-UNKNOWN: 4a4c9457 <unknown>
vfncvt.f.f.q v8, v4

# CHECK-INST: vfncvt.sat.f.f.q v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4d,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvfofp8min' (OFP8 conversion extension Zvfofp8min){{$}}
# CHECK-UNKNOWN: 484d9457 <unknown>
vfncvt.sat.f.f.q v8, v4, v0.t

# CHECK-INST: vfncvt.sat.f.f.q v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4d,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvfofp8min' (OFP8 conversion extension Zvfofp8min){{$}}
# CHECK-UNKNOWN: 4a4d9457 <unknown>
vfncvt.sat.f.f.q v8, v4
