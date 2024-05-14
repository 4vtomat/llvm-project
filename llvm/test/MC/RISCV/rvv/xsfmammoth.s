# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a %s \
# RUN:        | llvm-objdump -d  --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a %s \
# RUN:        | llvm-objdump -d  --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# CHECK-INST: sf.vsettnm a0, a1, a2, e8, w1
# CHECK-ENCODING: [0x57,0xf5,0xc5,0x90]
# CHECK-ERROR: instruction requires the following: XSfmm32ea/XSfmmbase{{$}}
# CHECK-UNKNOWN: 90c5f557 <unknown>
sf.vsettnm a0, a1, a2, e8, w1

# CHECK-INST: sf.vsettn a0, a1
# CHECK-ENCODING: [0x57,0xf5,0x05,0x84]
# CHECK-ERROR: instruction requires the following: XSfmm32ea/XSfmmbase{{$}}
# CHECK-UNKNOWN: 8405f557 <unknown>
sf.vsettn a0, a1

# CHECK-INST: sf.vsettm a0, a1
# CHECK-ENCODING: [0x57,0xf5,0x15,0x84]
# CHECK-ERROR: instruction requires the following: XSfmm32ea/XSfmmbase{{$}}
# CHECK-UNKNOWN: 8415f557 <unknown>
sf.vsettm a0, a1

# CHECK-INST: sf.vsettk a0, a1
# CHECK-ENCODING: [0x57,0xf5,0x25,0x84]
# CHECK-ERROR: instruction requires the following: XSfmm32ea/XSfmmbase{{$}}
# CHECK-UNKNOWN: 8425f557 <unknown>
sf.vsettk a0, a1

# CHECK-INST: sf.vlte8  a0, (a1)
# CHECK-ENCODING: [0x07,0xf0,0xa5,0x12]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' All non arithmetic instructions for all TEWs and sf.vtzero{{$}}
# CHECK-UNKNOWN: 12a5f007 <unknown>
sf.vlte8  a0, (a1)

# CHECK-INST: sf.vlte16 a0, (a1)
# CHECK-ENCODING: [0x07,0xf0,0xa5,0x32]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' All non arithmetic instructions for all TEWs and sf.vtzero{{$}}
# CHECK-UNKNOWN: 32a5f007 <unknown>
sf.vlte16 a0, (a1)

# CHECK-INST: sf.vlte32 a0, (a1)
# CHECK-ENCODING: [0x07,0xf0,0xa5,0x52]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' All non arithmetic instructions for all TEWs and sf.vtzero{{$}}
# CHECK-UNKNOWN: 52a5f007 <unknown>
sf.vlte32 a0, (a1)

# CHECK-INST: sf.vlte64 a0, (a1)
# CHECK-ENCODING: [0x07,0xf0,0xa5,0x72]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' All non arithmetic instructions for all TEWs and sf.vtzero{{$}}
# CHECK-UNKNOWN: 72a5f007 <unknown>
sf.vlte64 a0, (a1)

# CHECK-INST: sf.vste8  a0, (a1)
# CHECK-ENCODING: [0x27,0xf0,0xa5,0x12]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' All non arithmetic instructions for all TEWs and sf.vtzero{{$}}
# CHECK-UNKNOWN: 12a5f027 <unknown>
sf.vste8  a0, (a1)

# CHECK-INST: sf.vste16 a0, (a1)
# CHECK-ENCODING: [0x27,0xf0,0xa5,0x32]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' All non arithmetic instructions for all TEWs and sf.vtzero{{$}}
# CHECK-UNKNOWN: 32a5f027 <unknown>
sf.vste16 a0, (a1)

# CHECK-INST: sf.vste32 a0, (a1)
# CHECK-ENCODING: [0x27,0xf0,0xa5,0x52]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' All non arithmetic instructions for all TEWs and sf.vtzero{{$}}
# CHECK-UNKNOWN: 52a5f027 <unknown>
sf.vste32 a0, (a1)

# CHECK-INST: sf.vste64 a0, (a1)
# CHECK-ENCODING: [0x27,0xf0,0xa5,0x72]
# CHECK-ERROR: instruction requires the following: 'XSfmmbase' All non arithmetic instructions for all TEWs and sf.vtzero{{$}}
# CHECK-UNKNOWN: 72a5f027 <unknown>
sf.vste64 a0, (a1)

# CHECK-INST: sf.vtmv.v.t v8, a0
# CHECK-ENCODING: [0x57,0x64,0xf5,0x43]
# CHECK-ERROR: instruction requires the following: XSfmm32ea/XSfmmbase{{$}}
# CHECK-UNKNOWN: 43f56457 <unknown>
sf.vtmv.v.t v8, a0

# CHECK-INST: sf.vtmv.t.v a0, v8
# CHECK-ENCODING: [0x57,0x60,0x85,0x5e]
# CHECK-ERROR: instruction requires the following: XSfmm32ea/XSfmmbase{{$}}
# CHECK-UNKNOWN: 5e856057 <unknown>
sf.vtmv.t.v a0, v8

# CHECK-INST: sf.mm.f.f mt2, v8, v9
# CHECK-ENCODING: [0x77,0x92,0x84,0xf2]
# CHECK-ERROR: instruction requires the following: XSfmm32ea/XSfmm32a/XSfmm64a{{$}}
# CHECK-UNKNOWN: f2849277 <unknown>
sf.mm.f.f mt2, v8, v9

# CHECK-INST: sf.mm.bf.bf mt4, v8, v9
# CHECK-ENCODING: [0xf7,0x94,0x84,0xf2]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a' (TEW=32-bit accumulation) operands - int: 8b; float: fp16, bf16, fp32{{$}}
# CHECK-UNKNOWN: f28494f7 <unknown>
sf.mm.bf.bf mt4, v8, v9

# CHECK-INST: sf.mm.f8p3.f8p3 mt0, v8, v9
# CHECK-ENCODING: [0xf7,0x90,0x84,0xf6]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation) operands - float: fp8{{$}}
# CHECK-UNKNOWN: f68490f7 <unknown>
sf.mm.f8p3.f8p3 mt0, v8, v9

# CHECK-INST: sf.mm.f8p3.f8p4 mt4, v8, v9
# CHECK-ENCODING: [0x77,0x95,0x84,0xf6]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation) operands - float: fp8{{$}}
# CHECK-UNKNOWN: f6849577 <unknown>
sf.mm.f8p3.f8p4 mt4, v8, v9

# CHECK-INST: sf.mm.f8p3.f8p5 mt8, v8, v9
# CHECK-ENCODING: [0xf7,0x99,0x84,0xf6]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation) operands - float: fp8{{$}}
# CHECK-UNKNOWN: f68499f7 <unknown>
sf.mm.f8p3.f8p5 mt8, v8, v9

# CHECK-INST: sf.mm.f8p4.f8p3 mt12, v8, v9
# CHECK-ENCODING: [0xf7,0x9c,0x84,0xfa]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation) operands - float: fp8{{$}}
# CHECK-UNKNOWN: fa849cf7 <unknown>
sf.mm.f8p4.f8p3 mt12, v8, v9

# CHECK-INST: sf.mm.f8p4.f8p4 mt0, v8, v9
# CHECK-ENCODING: [0x77,0x91,0x84,0xfa]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation) operands - float: fp8{{$}}
# CHECK-UNKNOWN: fa849177 <unknown>
sf.mm.f8p4.f8p4 mt0, v8, v9

# CHECK-INST: sf.mm.f8p4.f8p5 mt0, v8, v9
# CHECK-ENCODING: [0xf7,0x91,0x84,0xfa]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation) operands - float: fp8{{$}}
# CHECK-UNKNOWN: fa8491f7 <unknown>
sf.mm.f8p4.f8p5 mt0, v8, v9

# CHECK-INST: sf.mm.f8p5.f8p3 mt0, v8, v9
# CHECK-ENCODING: [0xf7,0x90,0x84,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation) operands - float: fp8{{$}}
# CHECK-UNKNOWN: fe8490f7 <unknown>
sf.mm.f8p5.f8p3 mt0, v8, v9

# CHECK-INST: sf.mm.f8p5.f8p4 mt0, v8, v9
# CHECK-ENCODING: [0x77,0x91,0x84,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation) operands - float: fp8{{$}}
# CHECK-UNKNOWN: fe849177 <unknown>
sf.mm.f8p5.f8p4 mt0, v8, v9

# CHECK-INST: sf.mm.f8p5.f8p5 mt0, v8, v9
# CHECK-ENCODING: [0xf7,0x91,0x84,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a8f' (TEW=32-bit accumulation) operands - float: fp8{{$}}
# CHECK-UNKNOWN: fe8491f7 <unknown>
sf.mm.f8p5.f8p5 mt0, v8, v9

# CHECK-INST: sf.mm.u.u mt0, v8, v9
# CHECK-ENCODING: [0x77,0x80,0x84,0xf2]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a' (TEW=32-bit accumulation) operands - int: 8b; float: fp16, bf16, fp32{{$}}
# CHECK-UNKNOWN: f2848077 <unknown>
sf.mm.u.u mt0, v8, v9

# CHECK-INST: sf.mm.s.u mt4, v8, v9
# CHECK-ENCODING: [0x77,0x84,0x84,0xf6]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a' (TEW=32-bit accumulation) operands - int: 8b; float: fp16, bf16, fp32{{$}}
# CHECK-UNKNOWN: f6848477 <unknown>
sf.mm.s.u mt4, v8, v9

# CHECK-INST: sf.mm.u.s mt8, v8, v9
# CHECK-ENCODING: [0xf7,0x88,0x84,0xf2]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a' (TEW=32-bit accumulation) operands - int: 8b; float: fp16, bf16, fp32{{$}}
# CHECK-UNKNOWN: f28488f7 <unknown>
sf.mm.u.s mt8, v8, v9

# CHECK-INST: sf.mm.s.s mt12, v8, v9
# CHECK-ENCODING: [0xf7,0x8c,0x84,0xf6]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a' (TEW=32-bit accumulation) operands - int: 8b; float: fp16, bf16, fp32{{$}}
# CHECK-UNKNOWN: f6848cf7 <unknown>
sf.mm.s.s mt12, v8, v9

# CHECK-INST: sf.p2mm.u.u mt0, v8, v9
# CHECK-ENCODING: [0x77,0x80,0x84,0xfa]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a4i' (TEW=32-bit accumulation) operands - int: 4b (packed){{$}}
# CHECK-UNKNOWN: fa848077 <unknown>
sf.p2mm.u.u mt0, v8, v9

# CHECK-INST: sf.p2mm.s.u mt4, v8, v9
# CHECK-ENCODING: [0x77,0x84,0x84,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a4i' (TEW=32-bit accumulation) operands - int: 4b (packed){{$}}
# CHECK-UNKNOWN: fe848477 <unknown>
sf.p2mm.s.u mt4, v8, v9

# CHECK-INST: sf.p2mm.u.s mt8, v8, v9
# CHECK-ENCODING: [0xf7,0x88,0x84,0xfa]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a4i' (TEW=32-bit accumulation) operands - int: 4b (packed){{$}}
# CHECK-UNKNOWN: fa8488f7 <unknown>
sf.p2mm.u.s mt8, v8, v9

# CHECK-INST: sf.p2mm.s.s mt12, v8, v9
# CHECK-ENCODING: [0xf7,0x8c,0x84,0xfe]
# CHECK-ERROR: instruction requires the following: 'XSfmm32a4i' (TEW=32-bit accumulation) operands - int: 4b (packed){{$}}
# CHECK-UNKNOWN: fe848cf7 <unknown>
sf.p2mm.s.s mt12, v8, v9

# CHECK-INST: sf.vtzero.t mt15
# CHECK-ENCODING: [0x57,0x6f,0xe0,0x43]
# CHECK-ERROR: instruction requires the following: XSfmm32ea/XSfmmbase{{$}}
# CHECK-UNKNOWN: 43e06f57 <unknown>
sf.vtzero.t mt15

# CHECK-INST: vsetvl a2, a0, a1
# CHECK-ENCODING: [0x57,0x76,0xb5,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 80b57657 <unknown>
vsetvl a2, a0, a1
