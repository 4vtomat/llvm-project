// Check handling Sifive specific features options.

// default 
// RUN: %clang -target riscv64-unknown-elf -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-mllvm" "-riscv-use-vla-vectorizer=false"

// -fuse-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -fuse-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-USE-VLA %s
// CHECK-USE-VLA-NOT: "-mllvm" "-riscv-use-vla-vectorizer=false"

// -fno-use-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -fno-use-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-USE-VLA %s
// CHECK-NO-USE-VLA: "-mllvm" "-riscv-use-vla-vectorizer=false"

// -flto
// RUN: %clang -target riscv64-unknown-elf -flto -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LTO-DEFAULT %s
// CHECK-LTO-DEFAULT-NOT: "-plugin-opt=-riscv-use-vla-vectorizer=false"

// -flto -fuse-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -flto -fuse-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LTO-USE-VLA %s
// CHECK-LTO-USE-VLA-NOT: "-mllvm" "-riscv-use-vla-vectorizer=false" "-plugin-opt=-riscv-use-vla-vectorizer=false"

// -flto -fno-use-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -flto -fno-use-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LTO-NO-USE-VLA %s
// CHECK-LTO-NO-USE-VLA: "-mllvm" "-riscv-use-vla-vectorizer=false"
// CHECK-LTO-NO-USE-VLA: "-plugin-opt=-riscv-use-vla-vectorizer=false"

int main() { return 0; }
