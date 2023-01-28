// RUN: %clang_cc1 -triple riscv64 -target-feature +zve32x -verify %s

// This file tests diagnostic messages when checking if rvv-related pragma is
// semantically legal. The test cases maps to a VF of 1, which is not legal when
// minimum vector length is 32.

#define DUMMY_LOOP for (int i=0; i<n; ++i) c[i] = a[i] + b[i]

void foo (int *a, int *b, int *c, int n) {
#pragma clang rvv lmul_sew(mf8, e8) /* expected-error {{(LMUL, SEW) pair (mf8, e8) requires at max vector element width of more than 64}} */
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(mf4, e16) /* expected-error {{(LMUL, SEW) pair (mf4, e16) requires at max vector element width of more than 64}} */
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(mf2, e32) /* expected-error {{(LMUL, SEW) pair (mf2, e32) requires at max vector element width of more than 64}} */
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(m1, e64) /* expected-error {{(LMUL, SEW) pair (m1, e64) requires at max vector element width of more than 64}} */
  DUMMY_LOOP;
}
