# RUN: llvm-mc %s -triple=riscv32 --mattr=+zicsr -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 --mattr=+zicsr < %s \
# RUN:     | llvm-objdump -d --mattr=+zicsr - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
#
# RUN: llvm-mc %s -triple=riscv64 --mattr=+zicsr -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 --mattr=+zicsr < %s \
# RUN:     | llvm-objdump -d --mattr=+zicsr - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

##################################
# Sscofpmf CSRs
##################################

# scountovf
# name
# CHECK-INST: csrrs t1, scountovf, zero
# CHECK-ENC: encoding: [0x73,0x23,0x00,0xda]
# CHECK-INST-ALIAS: csrr t1, scountovf
# uimm12
# CHECK-INST: csrrs t2, scountovf, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x00,0xda]
# CHECK-INST-ALIAS: csrr t2, scountovf
# name
csrrs t1, scountovf, zero
# uimm12
csrrs t2, 0xda0, zero
