// RUN: %clang_cc1 %s -triple riscv64 -fsyntax-only -verify

typedef __attribute__((neon_vector_type(2))) int int32x2_t; // expected-error{{'neon_vector_type' attribute is not supported on targets missing 'v', 'zfh' and 'zvfh'; specify an appropriate -march= or -mcpu=}}
