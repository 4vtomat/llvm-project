// RUN: %clang_cc1 %s -triple riscv64 -fsyntax-only -verify
// expected-no-diagnostics

typedef _Float16 float16_t;
typedef __attribute__((neon_vector_type(4))) float16_t float16x4_t;

float16x4_t test(const float16_t *in_0)
{
  return __builtin_neon_vld1_v(in_0, 8);
}
