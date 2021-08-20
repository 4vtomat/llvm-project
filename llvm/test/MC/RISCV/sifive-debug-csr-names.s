# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
#
# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

###########################
# Debug Trigger Registers
###########################

# tselect
# name
# CHECK-INST: csrrs t1, tselect, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x7a]
# CHECK-INST-ALIAS: csrr t1, tselect
# uimm12
# CHECK-INST: csrrs t2, tselect, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7a]
# CHECK-INST-ALIAS: csrr t2, tselect
# name
csrrs t1, tselect, zero
# uimm12
csrrs t2, 0x7A0, zero

# tdata1
# name
# CHECK-INST: csrrs t1, tdata1, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x7a]
# CHECK-INST-ALIAS: csrr t1, tdata1
# uimm12
# CHECK-INST: csrrs t2, tdata1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7a]
# CHECK-INST-ALIAS: csrr t2, tdata1
# name
csrrs t1, tdata1, zero
# uimm12
csrrs t2, 0x7A1, zero

# tdata2
# name
# CHECK-INST: csrrs t1, tdata2, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x7a]
# CHECK-INST-ALIAS: csrr t1, tdata2
# uimm12
# CHECK-INST: csrrs t2, tdata2, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7a]
# CHECK-INST-ALIAS: csrr t2, tdata2
csrrs t1, tdata2, zero
# uimm12
csrrs t2, 0x7A2, zero

# tdata3
# name
# CHECK-INST: csrrs t1, tdata3, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x7a]
# CHECK-INST-ALIAS: csrr t1, tdata3
# uimm12
# CHECK-INST: csrrs t2, tdata3, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7a]
# CHECK-INST-ALIAS: csrr t2, tdata3
# name
csrrs t1, tdata3, zero
# uimm12
csrrs t2, 0x7A3, zero

# tinfo
# name
# CHECK-INST: csrrs t1, tinfo, zero
# CHECK-ENC: encoding: [0x73,0x23,0x40,0x7a]
# CHECK-INST-ALIAS: csrr t1, tinfo
# uimm12
# CHECK-INST: csrrs t2, tinfo, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x40,0x7a]
# CHECK-INST-ALIAS: csrr t2, tinfo
# name
csrrs t1, tinfo, zero
# uimm12
csrrs t2, 0x7A4, zero

# tcontrol
# name
# CHECK-INST: csrrs t1, tcontrol, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x7a]
# CHECK-INST-ALIAS: csrr t1, tcontrol
# uimm12
# CHECK-INST: csrrs t2, tcontrol, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x7a]
# CHECK-INST-ALIAS: csrr t2, tcontrol
# name
csrrs t1, tcontrol, zero
# uimm12
csrrs t2, 0x7A5, zero

# hcontext
# name
# CHECK-INST: csrrs t1, hcontext, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x6a]
# CHECK-INST-ALIAS: csrr t1, hcontext
# uimm12
# CHECK-INST: csrrs t2, hcontext, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x6a]
# CHECK-INST-ALIAS: csrr t2, hcontext
# name
csrrs t1, hcontext, zero
# uimm12
csrrs t2, 0x6A8, zero

# scontext
# name
# CHECK-INST: csrrs t1, scontext, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x5a]
# CHECK-INST-ALIAS: csrr t1, scontext
# uimm12
# CHECK-INST: csrrs t2, scontext, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x5a]
# CHECK-INST-ALIAS: csrr t2, scontext
# name
csrrs t1, scontext, zero
# uimm12
csrrs t2, 0x5A8, zero

# mcontext
# name
# CHECK-INST: csrrs t1, mcontext, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x7a]
# CHECK-INST-ALIAS: csrr t1, mcontext
# uimm12
# CHECK-INST: csrrs t2, mcontext, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x7a]
# CHECK-INST-ALIAS: csrr t2, mcontext
# name
csrrs t1, mcontext, zero
# uimm12
csrrs t2, 0x7A8, zero

# mscontext
# name
# CHECK-INST: csrrs t1, mscontext, zero
# CHECK-ENC: encoding: [0x73,0x23,0xa0,0x7a]
# CHECK-INST-ALIAS: csrr t1, mscontext
# uimm12
# CHECK-INST: csrrs t2, mscontext, zero
# CHECK-ENC: encoding: [0xf3,0x23,0xa0,0x7a]
# CHECK-INST-ALIAS: csrr t2, mscontext
# name
csrrs t1, mscontext, zero
# uimm12
csrrs t2, 0x7AA, zero

#######################
# Debug Core Registers
#######################

# dcsr
# name
# CHECK-INST: csrrs t1, dcsr, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0x7b]
# CHECK-INST-ALIAS: csrr t1, dcsr
# uimm12
# CHECK-INST: csrrs t2, dcsr, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0x7b]
# CHECK-INST-ALIAS: csrr t2, dcsr
# name
csrrs t1, dcsr, zero
# uimm12
csrrs t2, 0x7B0, zero

# dpc
# name
# CHECK-INST: csrrs t1, dpc, zero
# CHECK-ENC: encoding: [0x73,0x23,0x10,0x7b]
# CHECK-INST-ALIAS: csrr t1, dpc
# uimm12
# CHECK-INST: csrrs t2, dpc, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x10,0x7b]
# CHECK-INST-ALIAS: csrr t2, dpc
# name
csrrs t1, dpc, zero
# uimm12
csrrs t2, 0x7B1, zero

# dscratch0
# name
# CHECK-INST: csrrs t1, dscratch0, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t1, dscratch0
# uimm12
# CHECK-INST: csrrs t2, dscratch0, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t2, dscratch0
# name
csrrs t1, dscratch0, zero
# uimm12
csrrs t2, 0x7B2, zero

# dscratch
# name
# CHECK-INST: csrrs t1, dscratch0, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t1, dscratch0
# uimm12
# CHECK-INST: csrrs t2, dscratch0, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x7b]
# CHECK-INST-ALIAS: csrr t2, dscratch0
# name
csrrs t1, dscratch, zero
# uimm12
csrrs t2, 0x7B2, zero

# dscratch1
# name
# CHECK-INST: csrrs t1, dscratch1, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x7b]
# CHECK-INST-ALIAS: csrr t1, dscratch1
# uimm12
# CHECK-INST: csrrs t2, dscratch1, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x7b]
# CHECK-INST-ALIAS: csrr t2, dscratch1
# name
csrrs t1, dscratch1, zero
# uimm12
csrrs t2, 0x7B3, zero

