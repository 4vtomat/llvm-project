// RUN: %clang_cc1 -triple riscv64 -verify %s

// This file tests diagnostic messages when checking if rvv-related pragma is
// semantically legal. The test cases checks if at least the minimum vector
// extension 'zve32x' is enabled.

#define DUMMY_LOOP for (int i=0; i<n; ++i) c[i] = a[i] + b[i]

void foo (int *a, int *b, int *c, int n) {
#pragma clang rvv lmul_sew(m1, e32) /* expected-error {{Require at least zve32x to use the #pragma clang rvv lmul_sew}} */
  DUMMY_LOOP;
}
