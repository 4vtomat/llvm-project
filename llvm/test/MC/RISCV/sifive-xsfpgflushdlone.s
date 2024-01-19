# RUN: llvm-mc %s -triple=riscv32 --mattr=+xsfpgflushdlone \
# RUN:    -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 --mattr=+xsfpgflushdlone \
# RUN:    -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 --mattr=+xsfpgflushdlone < %s \
# RUN:     | llvm-objdump --mattr=+xsfpgflushdlone -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 --mattr=+xsfpgflushdlone < %s \
# RUN:     | llvm-objdump --mattr=+xsfpgflushdlone -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: sf.pgflush.d.l1
# CHECK: encoding: [0x73,0x00,0x30,0xfc]
sf.pgflush.d.l1
