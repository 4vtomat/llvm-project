// RUN: %clang_cc1 -triple riscv64 -target-feature +zve64x -verify %s

// This file tests diagnostic messages when checking if rvv-related pragma is
// semantically legal

#define DUMMY_LOOP for (int i=0; i<n; ++i) c[i] = a[i] + b[i]

void foo (int *a, int *b, int *c, int n) {
#pragma clang rvv lmul_sew(m1, e32) /* expected-error {{incompatible directives 'vectorize_width(4)' and 'lmul_sew(m1, e32)'}} */
#pragma clang loop vectorize_width(4)
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(m1, e32) /* expected-error {{incompatible directives 'interleave_count(4)' and 'lmul_sew(m1, e32)'}} */
#pragma clang loop interleave_count(4)
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(m1, e32) /* expected-error {{incompatible directives 'vectorize(disable)' and 'lmul_sew(m1, e32)}} */
#pragma clang loop vectorize(disable)
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(m1, e32)
#pragma clang rvv lmul_sew(m2, e32) /* expected-error {{duplicate directives 'lmul_sew(m1, e32)' and 'lmul_sew(m2, e32)'}} */
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(mf8, e16) /* expected-error {{(LMUL, SEW) pair (mf8, e16) does not map to a valid VF}}*/
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(mf8, e32) /* expected-error {{(LMUL, SEW) pair (mf8, e32) does not map to a valid VF}}*/
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(mf4, e32) /* expected-error {{(LMUL, SEW) pair (mf4, e32) does not map to a valid VF}}*/
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(mf8, e64) /* expected-error {{(LMUL, SEW) pair (mf8, e64) does not map to a valid VF}}*/
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(mf4, e64) /* expected-error {{(LMUL, SEW) pair (mf4, e64) does not map to a valid VF}}*/
  DUMMY_LOOP;

#pragma clang rvv lmul_sew(mf2, e64) /* expected-error {{(LMUL, SEW) pair (mf2, e64) does not map to a valid VF}}*/
  DUMMY_LOOP;

#pragma clang loop vectorize_width(4) /* expected-warning {{use #pragma clang rvv lmul_sew instead of vectorize_width for RISC-V vectors}} */
  DUMMY_LOOP;

#pragma clang loop vectorize_width(4, fixed) /* expected-warning {{use #pragma clang rvv lmul_sew instead of vectorize_width for RISC-V vectors}} */
  DUMMY_LOOP;

#pragma clang loop vectorize_width(4, scalable) /* expected-warning {{use #pragma clang rvv lmul_sew instead of vectorize_width for RISC-V vectors}} */
  DUMMY_LOOP;
}
