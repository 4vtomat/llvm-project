// RUN: %clang --target=riscv32-unknown-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix=RV32
// RUN: %clang --target=riscv64-unknown-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix=RV64

// RV32: "target-features"="+32bit,+a,+c,+m,+relax,
<<<<<<< HEAD
// RV32-SAME: -save-restore
// SIFIVE_CUSTOMIZATION
// RV64: "target-features"="+64bit,+a,+c,+d,+f,+m,+relax,+zicsr
// end of SIFIVE_CUSTOMIZATION
// RV64-SAME: -save-restore
=======
// RV64: "target-features"="+64bit,+a,+c,+m,+relax,
>>>>>>> b51f8f13edf3f7ab6407d2b7b46285ea675730b6

// Dummy function
int foo(void){
  return  3;
}
