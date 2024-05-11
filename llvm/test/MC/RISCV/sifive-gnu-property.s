# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfilp < %s | llvm-readelf -x .note.gnu.property - | FileCheck --check-prefix=ZICFILP %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfiss < %s | llvm-readelf -x .note.gnu.property - | FileCheck --check-prefix=ZICFISS %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfilp,+experimental-zicfiss < %s | llvm-readelf -x .note.gnu.property - | FileCheck --check-prefix=ZICFISSLP %s

# ZICFILP:      0x00000000 04000000 10000000 05000000 474e5500
# ZICFILP-NEXT: 0x00000010 000000c0 04000000 01000000 00000000

# ZICFISS:      0x00000000 04000000 10000000 05000000 474e5500
# ZICFISS-NEXT: 0x00000010 000000c0 04000000 02000000 00000000

# ZICFISSLP:      0x00000000 04000000 10000000 05000000 474e5500
# ZICFISSLP-NEXT: 0x00000010 000000c0 04000000 03000000 00000000
