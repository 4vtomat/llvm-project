# RUN: llvm-mc %s -triple=riscv32 -mattr=+xsfsci -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple riscv64 -mattr=+xsfsci -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xsfsci < %s \
# RUN:     | llvm-objdump --mattr=+xsfsci --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xsfsci < %s \
# RUN:     | llvm-objdump --mattr=+xsfsci --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s

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

# CHECK-OBJ: sf.sci.0.r 0, 38, a2, a3, s2
# CHECK-ASM: sf.sci.0.i 0, a2, a3, 1234
# CHECK-ASM: encoding: [0x0b,0x86,0x26,0x4d]
sf.sci.0.i 0, a2, a3, 1234

# CHECK-OBJ: sf.sci.1.r 1, 64, a3, a4, zero
# CHECK-ASM: sf.sci.1.i 1, a3, a4, 2048
# CHECK-ASM: encoding: [0xab,0x16,0x07,0x80]
sf.sci.1.i 1, a3, a4, 2048

# CHECK-OBJ: sf.sci.2.r 2, 63, a4, a5, t6
# CHECK-ASM: sf.sci.2.i 2, a4, a5, 2047
# CHECK-ASM: encoding: [0x5b,0xa7,0xf7,0x7f]
sf.sci.2.i 2, a4, a5, 2047

# CHECK-OBJ: sf.sci.3.r 3, 127, a5, a6, t6
# CHECK-ASM: sf.sci.3.i 3, a5, a6, 4095
# CHECK-ASM: encoding: [0xfb,0x37,0xf8,0xff]
sf.sci.3.i 3, a5, a6, 4095

# CHECK-OBJ: sf.sci.0.r 7, 127, a2, t6, t6
# CHECK-ASM: sf.sci.0.u a2, 1048575
# CHECK-ASM: encoding: [0x0b,0xf6,0xff,0xff]
sf.sci.0.u a2, 1048575

# CHECK-OBJ: sf.sci.1.r 0, 0, a3, zero, zero
# CHECK-ASM: sf.sci.1.u a3, 0
# CHECK-ASM: encoding: [0xab,0x06,0x00,0x00]
sf.sci.1.u a3, 0

# CHECK-OBJ: sf.sci.2.r 0, 15, a4, s0, sp
# CHECK-ASM: sf.sci.2.u a4, 123456
# CHECK-ASM: encoding: [0x5b,0x07,0x24,0x1e]
sf.sci.2.u a4, 123456

# CHECK-OBJ: sf.sci.3.r 7, 0, a5, t6, a5
# CHECK-ASM: sf.sci.3.u a5, 4095
# CHECK-ASM: encoding: [0xfb,0xf7,0xff,0x00]
sf.sci.3.u a5, 4095

# CHECK-OBJ: sf.sci.0.r 7, 127, t6, t6, t6
# CHECK-ASM: sf.sci.0.x 33554431
# CHECK-ASM: encoding: [0x8b,0xff,0xff,0xff]
sf.sci.0.x 33554431

# CHECK-OBJ: sf.sci.1.r 0, 0, zero, zero, zero
# CHECK-ASM: sf.sci.1.x 0
# CHECK-ASM: encoding: [0x2b,0x00,0x00,0x00]
sf.sci.1.x 0

# CHECK-OBJ: sf.sci.2.r 2, 0, zero, sp, a5
# CHECK-ASM: sf.sci.2.x 123456
# CHECK-ASM: encoding: [0x5b,0x20,0xf1,0x00]
sf.sci.2.x 123456

# CHECK-OBJ: sf.sci.3.r 7, 0, t6, a5, zero
# CHECK-ASM: sf.sci.3.x 4095
# CHECK-ASM: encoding: [0xfb,0xff,0x07,0x00]
sf.sci.3.x 4095
