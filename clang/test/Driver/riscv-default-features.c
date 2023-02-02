// RUN: %clang --target=riscv32-unknown-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix=RV32
// RUN: %clang --target=riscv64-unknown-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix=RV64

<<<<<<< HEAD
// RV32: "target-features"="+32bit,+a,+c,+m,+relax,-save-restore"
// RV64: "target-features"="+64bit,+a,+c,+d,+f,+m,+relax,+zicsr,-save-restore"
=======
// RV32: "target-features"="+32bit,+a,+c,+m,+relax,
// RV32-SAME: -save-restore
// RV64: "target-features"="+64bit,+a,+c,+m,+relax,
// RV64-SAME: -save-restore
>>>>>>> revert-rvv-intrinsic-v0.11-patches

// Dummy function
int foo(void){
  return  3;
}
