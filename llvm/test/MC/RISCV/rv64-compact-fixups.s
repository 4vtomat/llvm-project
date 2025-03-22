# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d,+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+d,-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=NORELAX-RELOC %s
# RUN: llvm-mc -triple riscv64 -mattr=+d,+relax < %s -show-encoding 2>&1 \
# RUN:     | FileCheck -check-prefix=RELAX-FIXUP %s

# GPREL LLA

# RELAX-FIXUP: warning: compact code model operand modifiers are deprecated

lui a0, %gprel_hi(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_HI20 foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_hi(foo), kind: fixup_riscv_gprel_hi20
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

add a0, gp, a0, %gprel(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_ADD foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_ADD foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel(foo), kind: fixup_riscv_gprel_add
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

addi a0, a0, %gprel_lo(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

# GOT GPREL LA

lui a0, %got_gprel_hi(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_HI20 foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_hi(foo), kind: fixup_riscv_got_gprel_hi20
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

add a0, gp, a0, %got_gprel(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_ADD foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_ADD foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel(foo), kind: fixup_riscv_got_gprel_add
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

addi a0, a0, %got_gprel_lo(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

# GPREL LA TLS GD

lui a0, %tls_gd_gprel_hi(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_HI20 foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_gd_gprel_hi(foo), kind: fixup_riscv_tls_gd_gprel_hi20

add a0, gp, a0, %tls_gd_gprel(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_ADD foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_ADD foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_gd_gprel(foo), kind: fixup_riscv_tls_gd_gprel_add

addi a0, a0, %tls_gd_gprel_lo(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GD_GPREL_LO12_I foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_gd_gprel_lo(foo), kind: fixup_riscv_tls_gd_gprel_lo12_i

# GPREL LA TLS IE

lui a0, %tls_ie_gprel_hi(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_HI20 foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_HI20 foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_ie_gprel_hi(foo), kind: fixup_riscv_tls_got_gprel_hi20

add a0, gp, a0, %tls_ie_gprel(foo)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_ADD foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_ADD foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_ie_gprel(foo), kind: fixup_riscv_tls_got_gprel_add

lw a0, %tls_ie_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_TLS_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-FIXUP: fixup A - offset: 0, value: %tls_ie_gprel_lo(foo), kind: fixup_riscv_tls_got_gprel_lo12_i

# GPREL LO12_I

lb a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lh a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lw a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

ld a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lbu a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lhu a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lwu a0, %gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

flw fa0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

fld fa0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

# GPREL LO12_S

sb a0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

sh a0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

sw a0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

sd a0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

fsw fa0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

fsd fa0, %gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GPREL_LO12_S foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %gprel_lo(foo), kind: fixup_riscv_gprel_lo12_s
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

# GOT GPREL LO12_I

lb a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lh a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lw a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

ld a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lbu a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lhu a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

lwu a0, %got_gprel_lo(foo)(a0)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

flw fa0, %got_gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax

fld fa0, %got_gprel_lo(foo)(a1)
# NORELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# NORELAX-RELOC-NOT: R_RISCV_RELAX
# RELAX-RELOC: R_RISCV_SIFIVE_GOT_GPREL_LO12_I foo 0x0
# RELAX-RELOC: R_RISCV_RELAX - 0x0
# RELAX-FIXUP: fixup A - offset: 0, value: %got_gprel_lo(foo), kind: fixup_riscv_got_gprel_lo12_i
# RELAX-FIXUP: fixup B - offset: 0, value: 0, kind: fixup_riscv_relax
