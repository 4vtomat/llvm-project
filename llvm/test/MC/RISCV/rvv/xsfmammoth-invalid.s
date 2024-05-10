# RUN: not llvm-mc -triple=riscv32 --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 --mattr=+xsfmm32ea,+xsfmmbase, \
# RUN:     --mattr=+xsfmm32a,+xsfmm32a8f,+xsfmm32a4i,+xsfmm64a %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# CHECK-ERROR: operand must be e[8|16|32|64],w[1|2|4]
# CHECK-ERROR-LABEL: sf.vsettnt a0, a1, a2, e128, w1{{$}}
sf.vsettnt a0, a1, a2, e128, w1

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f.f mt1, v8, v9{{$}}
sf.mm.f.f mt1, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.bf.bf mt3, v8, v9{{$}}
sf.mm.bf.bf mt3, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f8p3.f8p3 mt2, v8, v9{{$}}
sf.mm.f8p3.f8p3 mt2, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f8p3.f8p4 mt6, v8, v9{{$}}
sf.mm.f8p3.f8p4 mt6, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f8p3.f8p5 mt10, v8, v9{{$}}
sf.mm.f8p3.f8p5 mt10, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f8p4.f8p3 mt14, v8, v9{{$}}
sf.mm.f8p4.f8p3 mt14, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f8p4.f8p4 mt2, v8, v9{{$}}
sf.mm.f8p4.f8p4 mt2, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f8p4.f8p5 mt6, v8, v9{{$}}
sf.mm.f8p4.f8p5 mt6, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f8p5.f8p3 mt10, v8, v9{{$}}
sf.mm.f8p5.f8p3 mt10, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f8p5.f8p4 mt14, v8, v9{{$}}
sf.mm.f8p5.f8p4 mt14, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.f8p5.f8p5 mt2, v8, v9{{$}}
sf.mm.f8p5.f8p5 mt2, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.u.u mt1, v8, v9{{$}}
sf.mm.u.u mt1, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.s.u mt2, v8, v9{{$}}
sf.mm.s.u mt2, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.u.s mt3, v8, v9{{$}}
sf.mm.u.s mt3, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.mm.s.s mt1, v8, v9{{$}}
sf.mm.s.s mt1, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.p2mm.u.u mt1, v8, v9{{$}}
sf.p2mm.u.u mt1, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.p2mm.s.u mt1, v8, v9{{$}}
sf.p2mm.s.u mt1, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.p2mm.u.s mt1, v8, v9{{$}}
sf.p2mm.u.s mt1, v8, v9

# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: sf.p2mm.s.s mt1, v8, v9{{$}}
sf.p2mm.s.s mt1, v8, v9
