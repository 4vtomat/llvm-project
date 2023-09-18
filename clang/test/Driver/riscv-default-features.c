// RUN: %clang --target=riscv32-unknown-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix=RV32
// RUN: %clang --target=riscv64-unknown-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix=RV64

// RV32: "target-features"="+32bit,+a,+c,+m,+relax,
// RV32-SAME: -save-restore
// SIFIVE_CUSTOMIZATION
// RV64: "target-features"="+64bit,+a,+c,+d,+f,+m,+relax,+zicsr
// end of SIFIVE_CUSTOMIZATION
// RV64-SAME: -save-restore

// Dummy function
int foo(void){
  return  3;
}
