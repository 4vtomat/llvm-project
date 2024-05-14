# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+xsfvqdotq %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvqdotq %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvqdotq - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvqdotq %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sf.vqdot.vv v8, v4, v20, v0.t
# CHECK-INST: sf.vqdot.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'Xsfvqdotq' (SiFive Vector Quad-Widening 4D Dot Product Instructions){{$}}
# CHECK-UNKNOWN: b04a2457 <unknown>

sf.vqdotu.vv v8, v4, v20, v0.t
# CHECK-INST: sf.vqdotu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'Xsfvqdotq' (SiFive Vector Quad-Widening 4D Dot Product Instructions){{$}}
# CHECK-UNKNOWN: a04a2457 <unknown>

sf.vqdotsu.vv v8, v4, v20, v0.t
# CHECK-INST: sf.vqdotsu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'Xsfvqdotq' (SiFive Vector Quad-Widening 4D Dot Product Instructions){{$}}
# CHECK-UNKNOWN: a84a2457 <unknown>

sf.vqdot.vx v8, v4, s4, v0.t
# CHECK-INST: sf.vqdot.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'Xsfvqdotq' (SiFive Vector Quad-Widening 4D Dot Product Instructions){{$}}
# CHECK-UNKNOWN: b04a6457 <unknown>

sf.vqdotu.vx v8, v4, s4, v0.t
# CHECK-INST: sf.vqdotu.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'Xsfvqdotq' (SiFive Vector Quad-Widening 4D Dot Product Instructions){{$}}
# CHECK-UNKNOWN: a04a6457 <unknown>

sf.vqdotsu.vx v8, v4, s4, v0.t
# CHECK-INST: sf.vqdotsu.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'Xsfvqdotq' (SiFive Vector Quad-Widening 4D Dot Product Instructions){{$}}
# CHECK-UNKNOWN: a84a6457 <unknown>

sf.vqdotus.vx v8, v4, s4, v0.t
# CHECK-INST: sf.vqdotus.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb8]
# CHECK-ERROR: instruction requires the following: 'Xsfvqdotq' (SiFive Vector Quad-Widening 4D Dot Product Instructions){{$}}
# CHECK-UNKNOWN: b84a6457 <unknown>
