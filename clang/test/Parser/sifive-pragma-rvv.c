// RUN: %clang_cc1 -triple riscv64 -verify %s

// This file tests diagnostic messages when parsing rvv-related pragma

#define DUMMY_LOOP for (int i=0; i<n; ++i) c[i] = a[i] + b[i]

void foo (int *a, int *b, int *c, int n) {
#pragma clang rvv lmul_sew() /* expected-error {{missing argument to '#pragma clang rvv lmul_sew'; expected a legal LMUL (one of 'mf8', 'mf4', 'mf2, 'm1', 'm2', 'm4', 'm8') and a legal SEW (e8, e16, e32, e64)}} */
#pragma clang rvv lmul_sew(m1) /* expected-error {{missing argument to '#pragma clang rvv lmul_sew'; expected a legal LMUL (one of 'mf8', 'mf4', 'mf2, 'm1', 'm2', 'm4', 'm8') and a legal SEW (e8, e16, e32, e64)}} */
#pragma clang rvv lmul_sew(m1,) /* expected-error {{missing argument to '#pragma clang rvv lmul_sew'; expected a legal LMUL (one of 'mf8', 'mf4', 'mf2, 'm1', 'm2', 'm4', 'm8') and a legal SEW (e8, e16, e32, e64)}} */
#pragma clang rvv lmul_sew(m1 no_comma e32) /* expected-error {{expected ','}} */
#pragma clang rvv wrong_option /* expected-warning {{unexpected argument 'wrong_option' to '#pragma clang rvv'; expected 'lmul_sew'}} */
#pragma clang rvv lmul_sew /* expected-error {{expected '('}} */
#pragma clang rvv lmul_sew(m1, wrong_keyword) /* expected-error {{unexpected keyword for '#pragma clang rvv lmul_sew', expects a legal LMUL identifier(one of 'mf8', 'mf4', 'mf2, 'm1', 'm2', 'm4', 'm8') and a valid SEW (one of e8, e16, e32, e64)}} */
#pragma clang rvv lmul_sew(wrong_keyword, e32) /* expected-error {{unexpected keyword for '#pragma clang rvv lmul_sew', expects a legal LMUL identifier(one of 'mf8', 'mf4', 'mf2, 'm1', 'm2', 'm4', 'm8') and a valid SEW (one of e8, e16, e32, e64)}} */
#pragma clang rvv lmul_sew(1, e32) /* expected-error {{unexpected keyword for '#pragma clang rvv lmul_sew', expects a legal LMUL identifier(one of 'mf8', 'mf4', 'mf2, 'm1', 'm2', 'm4', 'm8') and a valid SEW (one of e8, e16, e32, e64)}} */
#pragma clang rvv lmul_sew(m1, 32) /* expected-error {{unexpected keyword for '#pragma clang rvv lmul_sew', expects a legal LMUL identifier(one of 'mf8', 'mf4', 'mf2, 'm1', 'm2', 'm4', 'm8') and a valid SEW (one of e8, e16, e32, e64)}} */
#pragma clang rvv lmul_sew(m1, e32, extra_token) /* expected-warning {{extra tokens at end of '#pragma clang rvv lmul_sew' - ignored}}*/
  DUMMY_LOOP;
}
