// RUN: %clang_cc1 -triple riscv64-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -triple riscv64-linux-gnu %s -emit-llvm -o - -fno-strict-pointer-overflow | FileCheck %s --check-prefix=PTR
// RUN: %clang_cc1 -triple riscv64-linux-gnu %s -emit-llvm -o - -fwrapv | FileCheck %s --check-prefixes=PTR,SCALAR

void foo(char *base, unsigned offset, int x, int y) {
  // DEFAULT: add nsw i32
  // SCALAR: add i32
  // PTR-NOT: add i32
  volatile int z = x + y;

  // DEFAULT: getelementptr inbounds nuw i8, ptr
  // PTR: getelementptr i8, ptr
  volatile char *ptr = base + offset;
}
