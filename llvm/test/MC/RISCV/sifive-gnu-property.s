# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zicfilp < %s | llvm-readelf -x .note.gnu.property - | FileCheck --check-prefix=ZICFILP-RV32 %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zicfilp < %s | llvm-readobj -n - | FileCheck --check-prefix=ZICFILP-NOTE %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zicfiss < %s | llvm-readelf -x .note.gnu.property - | FileCheck --check-prefix=ZICFISS-RV32 %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zicfiss < %s | llvm-readobj -n - | FileCheck --check-prefix=ZICFISS-NOTE %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zicfilp,+experimental-zicfiss < %s | llvm-readelf -x .note.gnu.property - | FileCheck --check-prefix=ZICFISSLP-RV32 %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zicfilp,+experimental-zicfiss < %s | llvm-readobj -n - | FileCheck --check-prefix=ZICFISSLP-NOTE %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfilp < %s | llvm-readelf -x .note.gnu.property - | FileCheck --check-prefix=ZICFILP-RV64 %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfilp < %s | llvm-readobj -n - | FileCheck --check-prefix=ZICFILP-NOTE %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfiss < %s | llvm-readelf -x .note.gnu.property - | FileCheck --check-prefix=ZICFISS-RV64 %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfiss < %s | llvm-readobj -n - | FileCheck --check-prefix=ZICFISS-NOTE %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfilp,+experimental-zicfiss < %s | llvm-readelf -x .note.gnu.property - | FileCheck --check-prefix=ZICFISSLP-RV64 %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfilp,+experimental-zicfiss < %s | llvm-readobj -n - | FileCheck --check-prefix=ZICFISSLP-NOTE %s

# ZICFILP-RV32:      0x00000000 04000000 0c000000 05000000 474e5500
# ZICFILP-RV32-NEXT: 0x00000010 000000c0 04000000 01000000
# ZICFILP-RV64:      0x00000000 04000000 10000000 05000000 474e5500
# ZICFILP-RV64-NEXT: 0x00000010 000000c0 04000000 01000000 00000000
# ZICFILP-NOTE:      riscv feature: ZICFILP

# ZICFISS-RV32:      0x00000000 04000000 0c000000 05000000 474e5500
# ZICFISS-RV32-NEXT: 0x00000010 000000c0 04000000 02000000
# ZICFISS-RV64:      0x00000000 04000000 10000000 05000000 474e5500
# ZICFISS-RV64-NEXT: 0x00000010 000000c0 04000000 02000000 00000000
# ZICFISS-NOTE:      riscv feature: ZICFISS

# ZICFISSLP-RV32:      0x00000000 04000000 0c000000 05000000 474e5500
# ZICFISSLP-RV32-NEXT: 0x00000010 000000c0 04000000 03000000
# ZICFISSLP-RV64:      0x00000000 04000000 10000000 05000000 474e5500
# ZICFISSLP-RV64-NEXT: 0x00000010 000000c0 04000000 03000000 00000000
# ZICFISSLP-NOTE:      riscv feature: ZICFILP, ZICFISS
