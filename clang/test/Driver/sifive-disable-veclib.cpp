// SCT-4009 SiFive_NF library is compiled for rv64gcv
//
// RUN: %clang++ --target=riscv64 -march=rv64gcv -fveclib=SiFive_NF -v -S -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK %s
// RUN: %clang++ --target=riscv64 -march=rv64gc_zve64f_zve64d_zvl128b -fveclib=SiFive_NF -v -S -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_NONE %s
// RUN: %clang++ --target=riscv64 -march=rv64gc_zve64f_zve64d -fveclib=SiFive_NF -v -S -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_NONE %s
// RUN: %clang++ --target=riscv64 -march=rv64gc_zve64f -fveclib=SiFive_NF -v -S -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_NONE %s
// RUN: %clang++ --target=riscv64 -march=rv64gc -fveclib=SiFive_NF -v -S -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_NONE %s
// RUN: not %clang++ --target=riscv32 -march=rv32i -fveclib=SiFive_NF -v -S -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_ERROR %s

// CHECK: -fveclib=SiFive_NF
// CHECK-NOT: -fveclib=none
//
// CHECK_NONE: -fveclib=SiFive_NF
// CHECK_NONE: -fveclib=none
//
// CHECK_ERROR: error: unsupported option 'SiFive_NF' for target 'riscv32'
namespace std{
  extern "C" float erf(float);
}
void mylibmcall(float *r, const float *x, int n) {
     for (int i=0; i!=n;++i) {
              r[i] = std::erf(x[i]);
     }
}
