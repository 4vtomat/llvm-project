# RUN: llvm-mc %s -triple=riscv64 -mattr=+d,+zfhmin 2>&1 \
# RUN:     | FileCheck %s --check-prefixes=CHECK

# Set pseudo gp to gp if it isn't set.

# CHECK: warning: compact code model pseudoinstructions are deprecated

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, gp, a0, %gprel(foo)
# CHECK: addi a0, a0, %gprel_lo(foo)
lla.gprel a0, foo

# CHECK: lui a0, %got_gprel_hi(foo)
# CHECK: add a0, gp, a0, %got_gprel(foo)
# CHECK: ld a0, %got_gprel_lo(foo)(a0)
la.got.gprel a0, foo

# CHECK: lui a0, %tls_gd_gprel_hi(foo)
# CHECK: add a0, gp, a0, %tls_gd_gprel(foo)
# CHECK: addi a0, a0, %tls_gd_gprel_lo(foo)
la.tls.gd.gprel a0, foo

# CHECK: lui a0, %tls_ie_gprel_hi(foo)
# CHECK: add a0, gp, a0, %tls_ie_gprel(foo)
# CHECK: ld a0, %tls_ie_gprel_lo(foo)(a0)
la.tls.ie.gprel a0, foo

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, gp, a0, %gprel(foo)
# CHECK: lb a0, %gprel_lo(foo)(a0)
lb.gprel a0, foo

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, gp, a0, %gprel(foo)
# CHECK: lh a0, %gprel_lo(foo)(a0)
lh.gprel a0, foo

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, gp, a0, %gprel(foo)
# CHECK: lw a0, %gprel_lo(foo)(a0)
lw.gprel a0, foo

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, gp, a0, %gprel(foo)
# CHECK: ld a0, %gprel_lo(foo)(a0)
ld.gprel a0, foo

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, gp, a0, %gprel(foo)
# CHECK: lbu a0, %gprel_lo(foo)(a0)
lbu.gprel a0, foo

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, gp, a0, %gprel(foo)
# CHECK: lhu a0, %gprel_lo(foo)(a0)
lhu.gprel a0, foo

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, gp, a0, %gprel(foo)
# CHECK: lwu a0, %gprel_lo(foo)(a0)
lwu.gprel a0, foo

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, gp, a1, %gprel(foo)
# CHECK: flh fa0, %gprel_lo(foo)(a1)
flh.gprel fa0, foo, a1

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, gp, a1, %gprel(foo)
# CHECK: flw fa0, %gprel_lo(foo)(a1)
flw.gprel fa0, foo, a1

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, gp, a1, %gprel(foo)
# CHECK: fld fa0, %gprel_lo(foo)(a1)
fld.gprel fa0, foo, a1

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: a1, gp, a1, %gprel(foo)
# CHECK: sb a0, %gprel_lo(foo)(a1)
sb.gprel a0, foo, a1

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: a1, gp, a1, %gprel(foo)
# CHECK: sh a0, %gprel_lo(foo)(a1)
sh.gprel a0, foo, a1

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: a1, gp, a1, %gprel(foo)
# CHECK: sw a0, %gprel_lo(foo)(a1)
sw.gprel a0, foo, a1

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: a1, gp, a1, %gprel(foo)
# CHECK: sd a0, %gprel_lo(foo)(a1)
sd.gprel a0, foo, a1

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: a1, gp, a1, %gprel(foo)
# CHECK: fsh fa0, %gprel_lo(foo)(a1)
fsh.gprel fa0, foo, a1

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: a1, gp, a1, %gprel(foo)
# CHECK: fsw fa0, %gprel_lo(foo)(a1)
fsw.gprel fa0, foo, a1

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: a1, gp, a1, %gprel(foo)
# CHECK: fsd fa0, %gprel_lo(foo)(a1)
fsd.gprel fa0, foo, a1

# Set pseudo gp to a2.

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, a2, a0, %gprel(foo)
# CHECK: addi a0, a0, %gprel_lo(foo)
lla.gprel a0, foo, a2

# CHECK: lui a0, %got_gprel_hi(foo)
# CHECK: add a0, a2, a0, %got_gprel(foo)
# CHECK: ld a0, %got_gprel_lo(foo)(a0)
la.got.gprel a0, foo, a2

# CHECK: lui a0, %tls_gd_gprel_hi(foo)
# CHECK: add a0, a2, a0, %tls_gd_gprel(foo)
# CHECK: addi a0, a0, %tls_gd_gprel_lo(foo)
la.tls.gd.gprel a0, foo, a2

# CHECK: lui a0, %tls_ie_gprel_hi(foo)
# CHECK: add a0, a2, a0, %tls_ie_gprel(foo)
# CHECK: ld a0, %tls_ie_gprel_lo(foo)(a0)
la.tls.ie.gprel a0, foo, a2

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, a2, a0, %gprel(foo)
# CHECK: lb a0, %gprel_lo(foo)(a0)
lb.gprel a0, foo, a2

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, a2, a0, %gprel(foo)
# CHECK: lh a0, %gprel_lo(foo)(a0)
lh.gprel a0, foo, a2

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, a2, a0, %gprel(foo)
# CHECK: lw a0, %gprel_lo(foo)(a0)
lw.gprel a0, foo, a2

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, a2, a0, %gprel(foo)
# CHECK: ld a0, %gprel_lo(foo)(a0)
ld.gprel a0, foo, a2

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, a2, a0, %gprel(foo)
# CHECK: lbu a0, %gprel_lo(foo)(a0)
lbu.gprel a0, foo, a2

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, a2, a0, %gprel(foo)
# CHECK: lhu a0, %gprel_lo(foo)(a0)
lhu.gprel a0, foo, a2

# CHECK: lui a0, %gprel_hi(foo)
# CHECK: add a0, a2, a0, %gprel(foo)
# CHECK: lwu a0, %gprel_lo(foo)(a0)
lwu.gprel a0, foo, a2

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, a2, a1, %gprel(foo)
# CHECK: flh fa0, %gprel_lo(foo)(a1)
flh.gprel fa0, foo, a1, a2

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, a2, a1, %gprel(foo)
# CHECK: flw fa0, %gprel_lo(foo)(a1)
flw.gprel fa0, foo, a1, a2

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, a2, a1, %gprel(foo)
# CHECK: sb a0, %gprel_lo(foo)(a1)
sb.gprel a0, foo, a1, a2

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, a2, a1, %gprel(foo)
# CHECK: sh a0, %gprel_lo(foo)(a1)
sh.gprel a0, foo, a1, a2

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, a2, a1, %gprel(foo)
# CHECK: sw a0, %gprel_lo(foo)(a1)
sw.gprel a0, foo, a1, a2

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, a2, a1, %gprel(foo)
# CHECK: sd a0, %gprel_lo(foo)(a1)
sd.gprel a0, foo, a1, a2

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, a2, a1, %gprel(foo)
# CHECK: fsh fa0, %gprel_lo(foo)(a1)
fsh.gprel fa0, foo, a1, a2

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, a2, a1, %gprel(foo)
# CHECK: fsw fa0, %gprel_lo(foo)(a1)
fsw.gprel fa0, foo, a1, a2

# CHECK: lui a1, %gprel_hi(foo)
# CHECK: add a1, a2, a1, %gprel(foo)
# CHECK: fsd fa0, %gprel_lo(foo)(a1)
fsd.gprel fa0, foo, a1, a2
