// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfilp -fcf-protection=branch -c %s -o - \
// RUN:     | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFILP-RV32 %s 
// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfilp -fcf-protection=branch -c %s -o - \
// RUN:     | llvm-readobj -n - | FileCheck --check-prefix=ZICFILP-NOTE %s 
// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfiss -fcf-protection=return \
// RUN:     -c %s -o - | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFISS-RV32 %s
// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfiss -fcf-protection=return \
// RUN:     -c %s -o - | llvm-readobj -n - | FileCheck --check-prefix=ZICFISS-NOTE %s
// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfilp_zicfiss -fcf-protection=full \
// RUN:     -c %s -o - | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFILPSS-RV32 %s
// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfilp_zicfiss -fcf-protection=full \
// RUN:     -c %s -o - | llvm-readobj -n - | FileCheck --check-prefix=ZICFILPSS-NOTE %s
//
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfilp -fcf-protection=branch -c %s -o - \
// RUN:     | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFILP-RV64 %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfilp -fcf-protection=branch -c %s -o - \
// RUN:     | llvm-readobj -n - | FileCheck --check-prefix=ZICFILP-NOTE %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfiss -fcf-protection=return \
// RUN:     -c %s -o - | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFISS-RV64 %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfiss -fcf-protection=return \
// RUN:     -c %s -o - | llvm-readobj -n - | FileCheck --check-prefix=ZICFISS-NOTE %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfilp_zicfiss -fcf-protection=full \
// RUN:     -c %s -o - | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFILPSS-RV64 %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfilp_zicfiss -fcf-protection=full \
// RUN:     -c %s -o - | llvm-readobj -n - | FileCheck --check-prefix=ZICFILPSS-NOTE %s
//
// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfilp_zicfiss \
// RUN:     -fcf-protection=branch -c %s -o - | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFILP-RV32 %s
// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfilp_zicfiss \
// RUN:     -fcf-protection=branch -c %s -o - | llvm-readobj -n - | FileCheck --check-prefix=ZICFILP-NOTE %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfilp_zicfiss \
// RUN:     -fcf-protection=branch -c %s -o - | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFILP-RV64 %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfilp_zicfiss \
// RUN:     -fcf-protection=branch -c %s -o - | llvm-readobj -n - | FileCheck --check-prefix=ZICFILP-NOTE %s
//
// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfilp_zicfiss \
// RUN:     -fcf-protection=return -c %s -o - | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFISS-RV32 %s
// RUN: %clang --target=riscv32 -menable-experimental-extensions -march=rv32g_zicfilp_zicfiss \
// RUN:     -fcf-protection=return -c %s -o - | llvm-readobj -n - | FileCheck --check-prefix=ZICFISS-NOTE %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfilp_zicfiss \
// RUN:     -fcf-protection=return -c %s -o - | llvm-readelf -x .note.gnu.property - \
// RUN:     | FileCheck --check-prefix=ZICFISS-RV64 %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions -march=rv64g_zicfilp_zicfiss \
// RUN:     -fcf-protection=return -c %s -o - | llvm-readobj -n - | FileCheck --check-prefix=ZICFISS-NOTE %s

void foo() {}

// ZICFILP-RV32:      0x00000000 04000000 0c000000 05000000 474e5500
// ZICFILP-RV32-NEXT: 0x00000010 000000c0 04000000 01000000
// ZICFILP-RV64:      0x00000000 04000000 10000000 05000000 474e5500
// ZICFILP-RV64-NEXT: 0x00000010 000000c0 04000000 01000000 00000000
// ZICFILP-NOTE:      riscv feature: ZICFILP

// ZICFISS-RV32:      0x00000000 04000000 0c000000 05000000 474e5500
// ZICFISS-RV32-NEXT: 0x00000010 000000c0 04000000 02000000
// ZICFISS-RV64:      0x00000000 04000000 10000000 05000000 474e5500
// ZICFISS-RV64-NEXT: 0x00000010 000000c0 04000000 02000000 00000000
// ZICFISS-NOTE:      riscv feature: ZICFISS

// ZICFILPSS-RV32:      0x00000000 04000000 0c000000 05000000 474e5500
// ZICFILPSS-RV32-NEXT: 0x00000010 000000c0 04000000 03000000
// ZICFILPSS-RV64:      0x00000000 04000000 10000000 05000000 474e5500
// ZICFILPSS-RV64-NEXT: 0x00000010 000000c0 04000000 03000000 00000000
// ZICFILPSS-NOTE:      riscv feature: ZICFILP, ZICFISS
