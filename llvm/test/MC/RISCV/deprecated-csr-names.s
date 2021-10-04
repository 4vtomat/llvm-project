# RUN: llvm-mc -triple riscv32 --mattr=+zicsr -riscv-no-aliases -show-encoding %s \
# RUN:     | FileCheck -check-prefixes CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype obj -triple riscv32 --mattr=+zicsr %s \
# RUN:     | llvm-objdump -d --mattr=+zicsr - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

# RUN: llvm-mc -triple riscv64 --mattr=+zicsr -riscv-no-aliases -show-encoding %s \
# RUN:     | FileCheck -check-prefixes CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype obj -triple riscv64 --mattr=+zicsr %s \
# RUN:     | llvm-objdump -d --mattr=+zicsr - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

# RUN: llvm-mc -triple riscv32 --mattr=+zicsr %s 2>&1 | FileCheck -check-prefix CHECK-WARN %s

# sbadaddr
# name
# CHECK-INST: csrrw zero, stval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x14]
# CHECK-INST-ALIAS: csrw stval, zero
# uimm12
# CHECK-INST: csrrw zero, stval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x14]
# CHECK-INST-ALIAS: csrw stval, zero
# name
csrw sbadaddr, zero
# uimm12
csrrw zero, 0x143, zero

# CHECK-WARN: warning: 'sbadaddr' is a deprecated alias for 'stval'

# mbadaddr
# name
# CHECK-INST: csrrw zero, mtval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x34]
# CHECK-INST-ALIAS: csrw mtval, zero
# uimm12
# CHECK-INST: csrrw zero, mtval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x34]
# CHECK-INST-ALIAS: csrw mtval, zero
# name
csrw mbadaddr, zero
# uimm12
csrrw zero, 0x343, zero

# CHECK-WARN: warning: 'mbadaddr' is a deprecated alias for 'mtval'

# ubadaddr
# name
# CHECK-INST: csrrw zero, utval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x04]
# CHECK-INST-ALIAS: csrw utval, zero
# uimm12
# CHECK-INST: csrrw zero, utval, zero
# CHECK-ENC: encoding: [0x73,0x10,0x30,0x04]
# CHECK-INST-ALIAS: csrw utval, zero
# name
csrw ubadaddr, zero
# uimm12
csrrw zero, 0x043, zero

# CHECK-WARN: warning: 'ubadaddr' is a deprecated alias for 'utval'
# CHECK-WARN: warning: 'utval' is deprecated

# sptbr
# name
# CHECK-INST: csrrw zero, satp, zero
# CHECK-ENC: encoding: [0x73,0x10,0x00,0x18]
# CHECK-INST-ALIAS: csrw satp, zero
# uimm12
# CHECK-INST: csrrw zero, satp, zero
# CHECK-ENC: encoding: [0x73,0x10,0x00,0x18]
# CHECK-INST-ALIAS: csrw satp, zero
# name
csrw sptbr, zero
# uimm12
csrrw zero, 0x180, zero

# CHECK-WARN: warning: 'sptbr' is a deprecated alias for 'satp'

# ustatus
# name
# CHECK-INST: csrrs t1, ustatus, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0x00]
# CHECK-INST-ALIAS: csrr t1, ustatus
# uimm12
# CHECK-INST: csrrs t2, ustatus, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0x00]
# CHECK-INST-ALIAS: csrr t2, ustatus
csrrs t1, ustatus, zero
# uimm12
csrrs t2, 0x000, zero

# CHECK-WARN: warning: 'ustatus' is deprecated

# uie
# name
# CHECK-INST: csrrs t1, uie, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x00]
# CHECK-INST-ALIAS: csrr t1, uie
# uimm12
# CHECK-INST: csrrs t2, uie, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x00]
# CHECK-INST-ALIAS: csrr t2, uie
# name
csrrs t1, uie, zero
# uimm12
csrrs t2, 0x004, zero

# CHECK-WARN: warning: 'uie' is deprecated

# utvec
# name
# CHECK-INST: csrrs t1, utvec, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0x00]
# CHECK-INST-ALIAS: csrr t1, utvec
# uimm12
# CHECK-INST: csrrs t2, utvec, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0x00]
# CHECK-INST-ALIAS: csrr t2, utvec
# name
csrrs t1, utvec, zero
# uimm12
csrrs t2, 0x005, zero

# CHECK-WARN: warning: 'utvec' is deprecated

# uscratch
# name
# CHECK-INST: csrrs t1, uscratch, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0x04]
# CHECK-INST-ALIAS: csrr t1, uscratch
# uimm12
# CHECK-INST: csrrs t2, uscratch, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0x04]
# CHECK-INST-ALIAS: csrr t2, uscratch
# name
csrrs t1, uscratch, zero
# uimm12
csrrs t2, 0x040, zero

# CHECK-WARN: warning: 'uscratch' is deprecated

# uepc
# name
# CHECK-INST: csrrs t1, uepc, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0x04]
# CHECK-INST-ALIAS: csrr t1, uepc
# uimm12
# CHECK-INST: csrrs t2, uepc, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0x04]
# CHECK-INST-ALIAS: csrr t2, uepc
# name
csrrs t1, uepc, zero
# uimm12
csrrs t2, 0x041, zero

# CHECK-WARN: warning: 'uepc' is deprecated

# ucause
# name
# CHECK-INST: csrrs t1, ucause, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0x04]
# CHECK-INST-ALIAS: csrr t1, ucause
# uimm12
# CHECK-INST: csrrs t2, ucause, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0x04]
# CHECK-INST-ALIAS: csrr t2, ucause
# name
csrrs t1, ucause, zero
# uimm12
csrrs t2, 0x042, zero

# CHECK-WARN: warning: 'ucause' is deprecated

# utval
# name
# CHECK-INST: csrrs t1, utval, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0x04]
# CHECK-INST-ALIAS: csrr t1, utval
# uimm12
# CHECK-INST: csrrs t2, utval, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0x04]

csrrs t1, utval, zero
# uimm12
csrrs t2, 0x043, zero

# CHECK-WARN: warning: 'utval' is deprecated

# uip
# name
# CHECK-INST: csrrs t1, uip, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x04]
# CHECK-INST-ALIAS: csrr t1, uip
# uimm12
# CHECK-INST: csrrs t2, uip, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x04]
# CHECK-INST-ALIAS: csrr t2, uip
#name
csrrs t1, uip, zero
# uimm12
csrrs t2, 0x044, zero

# CHECK-WARN: warning: 'uip' is deprecated

# sedeleg
# name
# CHECK-INST: csrrs t1, sedeleg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x20,0x10]
# CHECK-INST-ALIAS: csrr t1, sedeleg
# uimm12
# CHECK-INST: csrrs t2, sedeleg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x20,0x10]
# CHECK-INST-ALIAS: csrr t2, sedeleg
# name
csrrs t1, sedeleg, zero
# uimm12
csrrs t2, 0x102, zero

# CHECK-WARN: warning: 'sedeleg' is deprecated

# sideleg
# name
# CHECK-INST: csrrs t1, sideleg, zero
# CHECK-ENC: encoding: [0x73,0x23,0x30,0x10]
# CHECK-INST-ALIAS: csrr t1, sideleg
# uimm12
# CHECK-INST: csrrs t2, sideleg, zero
# CHECK-ENC: encoding: [0xf3,0x23,0x30,0x10]
# CHECK-INST-ALIAS: csrr t2, sideleg
# name
csrrs t1, sideleg, zero
# uimm12
csrrs t2, 0x103, zero

# CHECK-WARN: warning: 'sideleg' is deprecated
