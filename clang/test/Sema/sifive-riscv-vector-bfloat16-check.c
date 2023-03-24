// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -disable-O0-optnone -o - \
// RUN:   -fsyntax-only %s -verify
// REQUIRES: riscv-registered-target
#include <riscv_vector.h>

vbfloat16mf4_t foo() { /* expected-error {{RISC-V type 'vbfloat16mf4_t' (aka '__rvv_bfloat16mf4_t') requires the 'xsfvfhbfmin' or 'xsfvfwmaccqqq' extension}} */
} /* expected-warning {{non-void function does not return a value}}*/
