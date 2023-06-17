// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -verify %s
// expected-no-diagnostics

#include <riscv_ntlh.h>
#include <riscv_vector.h>

// v0.11 segment load
void test_vlseg2e32_v_i32m1(vint32m1_t *v0, vint32m1_t *v1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_v_i32m1(v0, v1, base, vl);
}

void test_vlseg2e32_v_i32m1_m(vint32m1_t *v0, vint32m1_t *v1, vbool32_t mask, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_v_i32m1_m(v0, v1, mask, base, vl);
}

void test_vlseg2e32_v_i32m1_m_overloaded(vint32m1_t *v0, vint32m1_t *v1, vbool32_t mask, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32(v0, v1, mask, base, vl);
}

void test_vlseg2e32_v_i32m1_tum(vint32m1_t *v0, vint32m1_t *v1, vbool32_t mask, vint32m1_t maskedoff0, vint32m1_t maskedoff1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_v_i32m1_tum(v0, v1, mask, maskedoff0, maskedoff1, base, vl);
}

void test_vlseg2e32_v_i32m1_tumu(vint32m1_t *v0, vint32m1_t *v1, vbool32_t mask, vint32m1_t maskedoff0, vint32m1_t maskedoff1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_v_i32m1_tumu(v0, v1, mask, maskedoff0, maskedoff1, base, vl);
}

void test_vlseg2e32_v_i32m1_mu(vint32m1_t *v0, vint32m1_t *v1, vbool32_t mask, vint32m1_t maskedoff0, vint32m1_t maskedoff1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_v_i32m1_mu(v0, v1, mask, maskedoff0, maskedoff1, base, vl);
}

void test_vlseg2e32_v_i32m1_tu(vint32m1_t *v0, vint32m1_t *v1, vint32m1_t maskedoff0, vint32m1_t maskedoff1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_v_i32m1_tu(v0, v1, maskedoff0, maskedoff1, base, vl);
}

void test_vlseg2e32_v_i32m1_tu_overloaded(vint32m1_t *v0, vint32m1_t *v1, vint32m1_t maskedoff0, vint32m1_t maskedoff1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_tu(v0, v1, maskedoff0, maskedoff1, base, vl);
}

void test_vlseg2e32_v_i32m1_tum_overloaded(vint32m1_t *v0, vint32m1_t *v1, vbool32_t mask, vint32m1_t maskedoff0, vint32m1_t maskedoff1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_tum(v0, v1, mask, maskedoff0, maskedoff1, base, vl);
}

void test_vlseg2e32_v_i32m1_tumu_overloaded(vint32m1_t *v0, vint32m1_t *v1, vbool32_t mask, vint32m1_t maskedoff0, vint32m1_t maskedoff1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_tumu(v0, v1, mask, maskedoff0, maskedoff1, base, vl);
}

void test_vlseg2e32_v_i32m1_mu_overloaded(vint32m1_t *v0, vint32m1_t *v1, vbool32_t mask, vint32m1_t maskedoff0, vint32m1_t maskedoff1, const int32_t *base, size_t vl) {
  return __riscv_vlseg2e32_mu(v0, v1, mask, maskedoff0, maskedoff1, base, vl);
}

// v0.11 ntlh segment load
void test_vlseg2e32_v_i32m1_tu_ntl_PALL(vint32m1_t *v0, vint32m1_t *v1, vint32m1_t maskedoff0, vint32m1_t maskedoff1, const int32_t *base, size_t vl, int domain) {
  return __riscv_vlseg2e32_v_i32m1_tu_ntl(v0, v1, maskedoff0, maskedoff1, base, vl, __RISCV_NTLH_ALL_PRIVATE);
}
