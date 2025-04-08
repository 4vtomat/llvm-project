// RUN: %clang_cc1 -triple riscv64-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -triple riscv64-linux-gnu %s -emit-llvm -o - -fwrapv | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -triple riscv64-linux-gnu %s -emit-llvm -o - -ftrapv | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -triple riscv64-linux-gnu %s -emit-llvm -o - -fwrapv-pointer | FileCheck %s --check-prefix=FWRAPV-POINTER
// RUN: %clang_cc1 -triple riscv64-linux-gnu %s -emit-llvm -o - -fwrapv-pointer-arithmetic | FileCheck %s --check-prefix=FWRAPV-POINTER-ARITH

void test(void) {
  extern int* P;

  // DEFAULT: getelementptr inbounds nuw i32, ptr
  // FWRAPV-POINTER: getelementptr i32, ptr
  // FWRAPV-POINTER-ARITH: getelementptr inbounds nuw i32, ptr
  ++P;

  // DEFAULT: getelementptr inbounds i32, ptr
  // FWRAPV-POINTER: getelementptr i32, ptr
  // FWRAPV-POINTER-ARITH: getelementptr inbounds i32, ptr
  (void)P[8];

  // DEFAULT: getelementptr inbounds i32, ptr
  // FWRAPV-POINTER: getelementptr i32, ptr
  // FWRAPV-POINTER-ARITH: getelementptr i32, ptr
  P += 4;
}
