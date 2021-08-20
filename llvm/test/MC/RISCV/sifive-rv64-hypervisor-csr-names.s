# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

# These machine mode CSR register names are RV32 only, but RV64
# can encode and disassemble these registers if given their value.

#####################################################
# Hypervisor Counter/Timer Virtualization Registers
#####################################################

# htimedeltah
# uimm12
# CHECK-INST: csrrs t2, 1557, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x50,0x61]
# CHECK-INST-ALIAS: csrr t2, 1557
csrrs t2, 0x615, zero
