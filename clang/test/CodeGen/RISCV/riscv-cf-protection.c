// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
<<<<<<< HEAD
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

#if SIFIVE_CUSTOMIZATION
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=fixed-one -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,FIXED-ONE-MACRO %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=fixed-one -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FIXED-ONE-FLAG %s
#endif // SIFIVE_CUSTOMIZATION

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,FUNC-SIG-MACRO %s

=======
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
<<<<<<< HEAD
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
=======
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,UNLABELED-SCHEME-UNUSED %s

<<<<<<< HEAD
#if SIFIVE_CUSTOMIZATION
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=fixed-one -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,FIXED-ONE-SCHEME-UNUSED %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=fixed-one -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FIXED-ONE-SCHEME-UNUSED %s
#endif // SIFIVE_CUSTOMIZATION

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,FUNC-SIG-SCHEME-UNUSED %s

=======
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FUNC-SIG-SCHEME-UNUSED %s

<<<<<<< HEAD
#if SIFIVE_CUSTOMIZATION
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM -emit-llvm %s -o - | \
// RUN: FileCheck --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=fixed-one -E -dM -emit-llvm %s -o - | \
// RUN: FileCheck --check-prefixes=LPAD-MACRO,FIXED-ONE-MACRO %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=fixed-one -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FIXED-ONE-FLAG %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,FUNC-SIG-MACRO %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s
#endif // SIFIVE_CUSTOMIZATION

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,UNLABELED-SCHEME-UNUSED %s
=======
// RUN: %clang --target=riscv32 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

// RUN: %clang --target=riscv32 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=unlabeled -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,UNLABELED-SCHEME-UNUSED %s

<<<<<<< HEAD
#if SIFIVE_CUSTOMIZATION
// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=fixed-one -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,FIXED-ONE-SCHEME-UNUSED %s

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=fixed-one -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FIXED-ONE-SCHEME-UNUSED %s
#endif // SIFIVE_CUSTOMIZATION

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,FUNC-SIG-SCHEME-UNUSED %s

=======
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=func-sig -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FUNC-SIG-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
<<<<<<< HEAD
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

#if SIFIVE_CUSTOMIZATION
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=fixed-one -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,FIXED-ONE-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=fixed-one -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FIXED-ONE-FLAG %s
#endif // SIFIVE_CUSTOMIZATION

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,FUNC-SIG-MACRO %s

=======
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
<<<<<<< HEAD
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
=======
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,UNLABELED-SCHEME-UNUSED %s

<<<<<<< HEAD
#if SIFIVE_CUSTOMIZATION
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=fixed-one -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,FIXED-ONE-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=fixed-one -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FIXED-ONE-SCHEME-UNUSED %s
#endif // SIFIVE_CUSTOMIZATION

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,FUNC-SIG-SCHEME-UNUSED %s

=======
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FUNC-SIG-SCHEME-UNUSED %s

<<<<<<< HEAD
#if SIFIVE_CUSTOMIZATION
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=fixed-one -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,FIXED-ONE-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=fixed-one -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FIXED-ONE-FLAG %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,FUNC-SIG-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s
#endif // SIFIVE_CUSTOMIZATION

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,UNLABELED-SCHEME-UNUSED %s
=======
// RUN: %clang --target=riscv64 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

// RUN: %clang --target=riscv64 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=unlabeled -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,UNLABELED-SCHEME-UNUSED %s

<<<<<<< HEAD
#if SIFIVE_CUSTOMIZATION
// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=fixed-one -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,FIXED-ONE-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=fixed-one -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FIXED-ONE-SCHEME-UNUSED %s
#endif // SIFIVE_CUSTOMIZATION

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN: -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-MACRO,FUNC-SIG-SCHEME-UNUSED %s

=======
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=func-sig -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FUNC-SIG-SCHEME-UNUSED %s

<<<<<<< HEAD
#if SIFIVE_CUSTOMIZATION
// Default -mcf-branch-label-scheme is fixed-one
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch -S -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BRANCH-PROT-FLAG,FIXED-ONE-FLAG %s

// Default -mcf-branch-label-scheme is fixed-one
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch -S -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BRANCH-PROT-FLAG,FIXED-ONE-FLAG %s
#endif // SIFIVE_CUSTOMIZATION

// UNLABELED-SCHEME-UNUSED: warning: argument unused during compilation:
// UNLABELED-SCHEME-UNUSED-SAME: '-mcf-branch-label-scheme=unlabeled'
#if SIFIVE_CUSTOMIZATION
// FIXED-ONE-SCHEME-UNUSED: warning: argument unused during compilation:
// FIXED-ONE-SCHEME-UNUSED-SAME: '-mcf-branch-label-scheme=fixed-one'
#endif // SIFIVE_CUSTOMIZATION
// FUNC-SIG-SCHEME-UNUSED: warning: argument unused during compilation:
// FUNC-SIG-SCHEME-UNUSED-SAME: '-mcf-branch-label-scheme=func-sig'

// LPAD-MACRO: __riscv_landing_pad 1{{$}}
// UNLABELED-MACRO: __riscv_landing_pad_unlabeled 1{{$}}
#if SIFIVE_CUSTOMIZATION
// FIXED-ONE-MACRO: __riscv_landing_pad_fixed_one 1{{$}}
#endif // SIFIVE_CUSTOMIZATION
// FUNC-SIG-MACRO: __riscv_landing_pad_func_sig 1{{$}}
// NO-MACRO-NOT: __riscv_landing_pad
// NO-MACRO-NOT: __riscv_landing_pad_unlabeled
#if SIFIVE_CUSTOMIZATION
// NO-MACRO-NOT: __riscv_landing_pad_fixed_one
#endif // SIFIVE_CUSTOMIZATION
// NO-MACRO-NOT: __riscv_landing_pad_func_sig

// BRANCH-PROT-FLAG-DAG: [[P_FLAG:![0-9]+]] = !{i32 8, !"cf-protection-branch", i32 1}
// UNLABELED-FLAG-DAG: [[S_FLAG:![0-9]+]] = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
#if SIFIVE_CUSTOMIZATION
// FIXED-ONE-FLAG-DAG: [[S_FLAG:![0-9]+]] = !{i32 1, !"cf-branch-label-scheme", !"fixed-one"}
#endif // SIFIVE_CUSTOMIZATION
=======
// Default -mcf-branch-label-scheme is func-sig
// RUN: %clang --target=riscv32 -fcf-protection=branch -S -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// Default -mcf-branch-label-scheme is func-sig
// RUN: %clang --target=riscv64 -fcf-protection=branch -S -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// UNLABELED-SCHEME-UNUSED: warning: argument unused during compilation:
// UNLABELED-SCHEME-UNUSED-SAME: '-mcf-branch-label-scheme=unlabeled'
// FUNC-SIG-SCHEME-UNUSED: warning: argument unused during compilation:
// FUNC-SIG-SCHEME-UNUSED-SAME: '-mcf-branch-label-scheme=func-sig'

// BRANCH-PROT-FLAG-DAG: [[P_FLAG:![0-9]+]] = !{i32 8, !"cf-protection-branch", i32 1}
// UNLABELED-FLAG-DAG: [[S_FLAG:![0-9]+]] = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// FUNC-SIG-FLAG-DAG: [[S_FLAG:![0-9]+]] = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
// BRANCH-PROT-FLAG-DAG: !llvm.module.flags = !{{[{].*}}[[P_FLAG]]{{.*, }}[[S_FLAG]]{{(,.+)?[}]}}
// NO-FLAG-NOT: !{i32 8, !"cf-protection-branch", i32 1}
// NO-FLAG-NOT: !{i32 8, !"cf-branch-label-scheme", !"unlabeled"}
<<<<<<< HEAD
#if SIFIVE_CUSTOMIZATION
// NO-FLAG-NOT: !{i32 8, !"cf-branch-label-scheme", !"fixed-one"}
#endif // SIFIVE_CUSTOMIZATION
=======
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
// NO-FLAG-NOT: !{i32 8, !"cf-branch-label-scheme", !"func-sig"}

int main() { return 0; }
