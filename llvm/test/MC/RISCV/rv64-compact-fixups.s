# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d,+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d,-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=NORELAX-RELOC %s
# RUN: llvm-mc -triple riscv64 -mattr=+d,+relax < %s -show-encoding > %t 2> %t.err
# RUN: FileCheck -check-prefix=RELAX-FIXUP %s < %t
# RUN: FileCheck -check-prefix=CHECK-STDERR %s < %t.err

# GPREL LLA

# CHECK-STDERR: warning: compact code model operand specifiers are deprecated

lui a0, %gprel_hi(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_HI20 foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_hi(foo), relocation type: 194
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

add a0, gp, a0, %gprel(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_ADD foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_ADD foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel(foo), relocation type: 197
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

addi a0, a0, %gprel_lo(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

# GOT GPREL LA

lui a0, %got_gprel_hi(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_HI20 foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_hi(foo), relocation type: 200
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

add a0, gp, a0, %got_gprel(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_ADD foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_ADD foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel(foo), relocation type: 202
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

addi a0, a0, %got_gprel_lo(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

# GPREL LA TLS GD

lui a0, %tls_gd_gprel_hi(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_HI20 foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_gd_gprel_hi(foo), relocation type: 208

add a0, gp, a0, %tls_gd_gprel(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_ADD foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_ADD foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_gd_gprel(foo), relocation type: 210

addi a0, a0, %tls_gd_gprel_lo(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_LO12_I foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_gd_gprel_lo(foo), relocation type: 209

# GPREL LA TLS IE

lui a0, %tls_ie_gprel_hi(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_HI20 foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_ie_gprel_hi(foo), relocation type: 205

add a0, gp, a0, %tls_ie_gprel(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_ADD foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_ADD foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_ie_gprel(foo), relocation type: 207

lw a0, %tls_ie_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_ie_gprel_lo(foo), relocation type: 206

# GPREL LO12_I

lb a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lh a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lw a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

ld a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lbu a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lhu a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lwu a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

flw fa0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

fld fa0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 195
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

# GPREL LO12_S

sb a0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 196
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

sh a0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 196
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

sw a0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 196
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

sd a0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 196
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

fsw fa0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 196
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

fsd fa0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), relocation type: 196
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

# GOT GPREL LO12_I

lb a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lh a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lw a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

ld a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lbu a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lhu a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

lwu a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

flw fa0, %got_gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51

fld fa0, %got_gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), relocation type: 201
# RELAX-FIXUP: fixup B - offset: 0, value: 0, relocation type: 51
