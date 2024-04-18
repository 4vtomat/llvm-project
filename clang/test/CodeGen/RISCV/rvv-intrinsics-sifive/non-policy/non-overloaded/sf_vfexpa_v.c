// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zve64f -target-feature +zvfh \
// RUN:   -disable-O0-optnone -emit-llvm %s -o - | opt -S -passes=mem2reg | \
// RUN:   FileCheck --check-prefix=CHECK-RV64 %s

#include <riscv_vector.h>

vfloat16mf4_t test_sf_vfexpa_v_vfloat16mf4(vuint16mf4_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16mf4(src, vl);
}

vfloat16mf2_t test_sf_vfexpa_v_vfloat16mf2(vuint16mf2_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16mf2(src, vl);
}

vfloat16m1_t test_sf_vfexpa_v_vfloat16m1(vuint16m1_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16m1(src, vl);
}

vfloat16m2_t test_sf_vfexpa_v_vfloat16m2(vuint16m2_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16m2(src, vl);
}

vfloat16m4_t test_sf_vfexpa_v_vfloat16m4(vuint16m4_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16m4(src, vl);
}

vfloat16m8_t test_sf_vfexpa_v_vfloat16m8(vuint16m8_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16m8(src, vl);
}

vfloat32mf2_t test_sf_vfexpa_v_vfloat32mf2(vuint32mf2_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32mf2(src, vl);
}

vfloat32m1_t test_sf_vfexpa_v_vfloat32m1(vuint32m1_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32m1(src, vl);
}

vfloat32m2_t test_sf_vfexpa_v_vfloat32m2(vuint32m2_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32m2(src, vl);
}

vfloat32m4_t test_sf_vfexpa_v_vfloat32m4(vuint32m4_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32m4(src, vl);
}

vfloat32m8_t test_sf_vfexpa_v_vfloat32m8(vuint32m8_t src, size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32m8(src, vl);
}

vfloat16mf4_t test_sf_vfexpa_v_vfloat16mf4_m(vbool64_t vm, vuint16mf4_t src,
                                             size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16mf4_m(vm, src, vl);
}

vfloat16mf2_t test_sf_vfexpa_v_vfloat16mf2_m(vbool32_t vm, vuint16mf2_t src,
                                             size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16mf2_m(vm, src, vl);
}

vfloat16m1_t test_sf_vfexpa_v_vfloat16m1_m(vbool16_t vm, vuint16m1_t src,
                                           size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16m1_m(vm, src, vl);
}

vfloat16m2_t test_sf_vfexpa_v_vfloat16m2_m(vbool8_t vm, vuint16m2_t src,
                                           size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16m2_m(vm, src, vl);
}

vfloat16m4_t test_sf_vfexpa_v_vfloat16m4_m(vbool4_t vm, vuint16m4_t src,
                                           size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16m4_m(vm, src, vl);
}

vfloat16m8_t test_sf_vfexpa_v_vfloat16m8_m(vbool2_t vm, vuint16m8_t src,
                                           size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat16m8_m(vm, src, vl);
}

vfloat32mf2_t test_sf_vfexpa_v_vfloat32mf2_m(vbool64_t vm, vuint32mf2_t src,
                                             size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32mf2_m(vm, src, vl);
}

vfloat32m1_t test_sf_vfexpa_v_vfloat32m1_m(vbool32_t vm, vuint32m1_t src,
                                           size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32m1_m(vm, src, vl);
}

vfloat32m2_t test_sf_vfexpa_v_vfloat32m2_m(vbool16_t vm, vuint32m2_t src,
                                           size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32m2_m(vm, src, vl);
}

vfloat32m4_t test_sf_vfexpa_v_vfloat32m4_m(vbool8_t vm, vuint32m4_t src,
                                           size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32m4_m(vm, src, vl);
}

vfloat32m8_t test_sf_vfexpa_v_vfloat32m8_m(vbool4_t vm, vuint32m8_t src,
                                           size_t vl) {
  return __riscv_sf_vfexpa_v_vfloat32m8_m(vm, src, vl);
}
