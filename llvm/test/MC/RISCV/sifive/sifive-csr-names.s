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

##################################
# Interrupt Clic CSRs
##################################

# mtvt
# name
# CHECK-INST: csrrs t1, mtvt, zero
# CHECK-ENC: encoding: [0x73,0x23,0x70,0x30]
# CHECK-INST-ALIAS: csrr t1, mtvt
# uimm12
# CHECK-INST: csrrs t2, mtvt, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x70,0x30]
# CHECK-INST-ALIAS: csrr t2, mtvt
csrrs t1, mtvt, zero
# uimm12
csrrs t2, 0x307, zero

# mnxti
# name
# CHECK-INST: csrrs t1, mnxti, zero
# CHECK-ENC: encoding: [0x73,0x23,0x50,0x34]
# CHECK-INST-ALIAS: csrr t1, mnxti
# uimm12
# CHECK-INST: csrrs t2, mnxti, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x34]
# CHECK-INST-ALIAS: csrr t2, mnxti
csrrs t1, mnxti, zero
# uimm12
csrrs t2, 0x345, zero

# mintstatus
# name
# CHECK-INST: csrrs t1, mintstatus, zero
# CHECK-ENC: encoding: [0x73,0x23,0x60,0x34]
# CHECK-INST-ALIAS: csrr t1, mintstatus
# uimm12
# CHECK-INST: csrrs t2, mintstatus, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x60,0x34]
# CHECK-INST-ALIAS: csrr t2, mintstatus
csrrs t1, mintstatus, zero
# uimm12
csrrs t2, 0x346, zero

# mscratchcsw
# name
# CHECK-INST: csrrs t1, mscratchcsw, zero
# CHECK-ENC: encoding: [0x73,0x23,0x80,0x34]
# CHECK-INST-ALIAS: csrr t1, mscratchcsw
# uimm12
# CHECK-INST: csrrs t2, mscratchcsw, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x80,0x34]
# CHECK-INST-ALIAS: csrr t2, mscratchcsw
csrrs t1, mscratchcsw, zero
# uimm12
csrrs t2, 0x348, zero
