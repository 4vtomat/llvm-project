// Check handling Sifive specific features options.

// default 
// RUN: %clang -target riscv64-unknown-elf -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-mllvm" "-riscv-use-vla-vectorizer=false"
// CHECK-DEFAULT-NOT: "-mllvm" "-scalable-vectorization=off"

// -fuse-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -fuse-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-USE-VLA %s
// CHECK-USE-VLA-NOT: "-mllvm" "-riscv-use-vla-vectorizer=false"
// CHECK-USE-VLA-NOT: "-mllvm" "-scalable-vectorization=off"

// -fno-use-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -fno-use-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-USE-VLA %s
// CHECK-NO-USE-VLA: "-mllvm" "-riscv-use-vla-vectorizer=false"
// CHECK-NO-USE-VLA: "-mllvm" "-scalable-vectorization=off"

// -flto
// RUN: %clang -target riscv64-unknown-elf -flto -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LTO-DEFAULT %s
// CHECK-LTO-DEFAULT-NOT: "-plugin-opt=-riscv-use-vla-vectorizer=false"
// CHECK-LTO-DEFAULT-NOT: "-plugin-opt=-scalable-vectorization=off"

// -flto -fuse-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -flto -fuse-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LTO-USE-VLA %s
// CHECK-LTO-USE-VLA-NOT: "-mllvm" "-riscv-use-vla-vectorizer=false"
// CHECK-LTO-USE-VLA-NOT: "-plugin-opt=-riscv-use-vla-vectorizer=false"
// CHECK-LTO-USE-VLA-NOT: "-mllvm" "-scalable-vectorization=off"
// CHECK-LTO-USE-VLA-NOT: "-plugin-opt=-scalable-vectorization=off"

// -flto -fno-use-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -flto -fno-use-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LTO-NO-USE-VLA %s
// CHECK-LTO-NO-USE-VLA: "-mllvm" "-riscv-use-vla-vectorizer=false"
// CHECK-LTO-NO-USE-VLA: "-mllvm" "-scalable-vectorization=off"
// CHECK-LTO-NO-USE-VLA: "-plugin-opt=-riscv-use-vla-vectorizer=false"
// CHECK-LTO-NO-USE-VLA: "-plugin-opt=-scalable-vectorization=off"

// Pass --relax-zcmt to linker if -Os/-Oz presents.
// RUN: %clang -### -target riscv64-unknown-elf --gcc-toolchain="" %s -Os 2>&1 | FileCheck %s --check-prefix=RELAX-ZCMT
// RUN: %clang -### -target riscv64-unknown-elf --gcc-toolchain="" %s -Oz 2>&1 | FileCheck %s --check-prefix=RELAX-ZCMT
// If other optimization comes last, don't pass --relax-zcmt.
// RUN: %clang -### -target riscv64-unknown-elf --gcc-toolchain="" %s -Os -O3 2>&1 | FileCheck %s --check-prefix=NO-RELAX-ZCMT
// RUN: %clang -### -target riscv64-unknown-elf --gcc-toolchain="" %s -Oz -O3 2>&1 | FileCheck %s --check-prefix=NO-RELAX-ZCMT
// RUN: %clang -### -target riscv64-unknown-elf --gcc-toolchain="" %s 2>&1 | FileCheck %s --check-prefix=NO-RELAX-ZCMT
// lld doesn't support yet --relax-zcmt yet.
// RUN: %clang -### -target riscv64-unknown-elf --gcc-toolchain="" -fuse-ld=lld -B%S/Inputs/lld %s 2>&1 | FileCheck %s --check-prefix=NO-RELAX-ZCMT
// RELAX-ZCMT: "--relax-zcmt"
// NO-RELAX-ZCMT-NOT: "--relax-zcmt"

int main() { return 0; }
