# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zjid -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zjid -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zjid < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zjid --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zjid < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zjid --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: clean.id (t0)
# CHECK-ASM: encoding: [0x0f,0xa0,0x92,0x00]
# CHECK-NO-EXT: instruction requires the following: 'Zjid' (Instruction/Data Cache Synchronization){{$}}
clean.id (t0)

# CHECK-ASM-AND-OBJ: inval.i (t0)
# CHECK-ASM: encoding: [0x0f,0xa0,0x82,0x00]
# CHECK-NO-EXT: instruction requires the following: 'Zjid' (Instruction/Data Cache Synchronization){{$}}
inval.i (t0)

# CHECK-ASM-AND-OBJ: import.i
# CHECK-ASM: encoding: [0x0f,0x10,0x10,0x00]
# CHECK-NO-EXT: instruction requires the following: 'Zjid' (Instruction/Data Cache Synchronization){{$}}
import.i

# CHECK-ASM-AND-OBJ: fence.cfi
# CHECK-ASM: encoding: [0x0f,0x00,0x20,0x22]
# CHECK-NO-EXT: instruction requires the following: 'Zjid' (Instruction/Data Cache Synchronization){{$}}
fence.cfi

# CHECK-ASM-AND-OBJ: fence.iis
# CHECK-ASM: encoding: [0x0f,0x00,0x20,0x32]
# CHECK-NO-EXT: instruction requires the following: 'Zjid' (Instruction/Data Cache Synchronization){{$}}
fence.iis
