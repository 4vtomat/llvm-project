# RUN: llvm-mc %s -triple=riscv32 -mattr=+xsfsci -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple riscv64 -mattr=+xsfsci -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xsfsci < %s \
# RUN:     | llvm-objdump --mattr=+xsfsci --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xsfsci < %s \
# RUN:     | llvm-objdump --mattr=+xsfsci --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: sf.sci 0, 0, a0, a1, a2
# CHECK-ASM: encoding: [0x7b,0x85,0xc5,0x00]
sf.sci 0, 0, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci 6, 127, a0, a1, a2
# CHECK-ASM: encoding: [0x7b,0xe5,0xc5,0xfe]
sf.sci 6, 127, a0, a1, a2
