// Check handling Sifive specific features options.
//
// default -use-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT-USE-VLA %s
// CHECK-DEFAULT-USE-VLA: "-mllvm" "-use-vla-vectorizer"

// -fuse-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -fuse-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-USE-VLA %s
// CHECK-USE-VLA: "-mllvm" "-use-vla-vectorizer"

// -fno-use-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -fno-use-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-USE-VLA %s
// CHECK-NO-USE-VLA-NOT: "-mllvm" "-use-vla-vectorizer"

// -flto
// RUN: %clang -target riscv64-unknown-elf -flto -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LTO-DEFAULT-USE-VLA %s
// CHECK-LTO-DEFAULT-USE-VLA: "-plugin-opt=-use-vla-vectorizer"

// -flto -fuse-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -flto -fuse-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LTO-USE-VLA %s
// CHECK-LTO-USE-VLA: "-plugin-opt=-use-vla-vectorizer"

// -flto -fno-use-vla-vectorizer
// RUN: %clang -target riscv64-unknown-elf -flto -fno-use-vla-vectorizer -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LTO-NO-USE-VLA %s
// CHECK-LTO-NO-USE-VLA-NOT: "-mllvm" "-use-vla-vectorizer" "-plugin-opt=-use-vla-vectorizer"

int main() { return 0; }
