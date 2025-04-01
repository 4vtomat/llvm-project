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

# CHECK-ASM-AND-OBJ: sf.sci.0.r 0, 0, a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x85,0xc5,0x00]
sf.sci.0.r 0, 0, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci.0.r 1, 63, a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x95,0xc5,0x7e]
sf.sci.0.r 1, 63, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci.1.r 2, 64, a0, a1, a2
# CHECK-ASM: encoding: [0x2b,0xa5,0xc5,0x80]
sf.sci.1.r 2, 64, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci.1.r 3, 45, a0, a1, a2
# CHECK-ASM: encoding: [0x2b,0xb5,0xc5,0x5a]
sf.sci.1.r 3, 45, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci.2.r 4, 10, a0, a1, a2
# CHECK-ASM: encoding: [0x5b,0xc5,0xc5,0x14]
sf.sci.2.r 4, 10, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci.2.r 5, 87, a0, a1, a2
# CHECK-ASM: encoding: [0x5b,0xd5,0xc5,0xae]
sf.sci.2.r 5, 87, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci.3.r 0, 0, a0, a1, a2
# CHECK-ASM: encoding: [0x7b,0x85,0xc5,0x00]
sf.sci.3.r 0, 0, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci.3.r 6, 127, a0, a1, a2
# CHECK-ASM: encoding: [0x7b,0xe5,0xc5,0xfe]
sf.sci.3.r 6, 127, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci.3.r 0, 0, a0, a1, a2
# CHECK-ASM: encoding: [0x7b,0x85,0xc5,0x00]
sf.sci 0, 0, a0, a1, a2

# CHECK-ASM-AND-OBJ: sf.sci.3.r 6, 127, a0, a1, a2
# CHECK-ASM: encoding: [0x7b,0xe5,0xc5,0xfe]
sf.sci 6, 127, a0, a1, a2
