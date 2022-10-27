// RUN: %clang_cc1  -triple riscv32 -target-feature +experimental-zihintntl -emit-llvm %s -o - \
// RUN:     | FileCheck %s

#include <riscv_ntlh.h>

signed char sc;
unsigned char uc;
signed short ss;
unsigned short us;
signed int si;
unsigned int ui;
signed long long sll;
unsigned long long ull;
_Float16 h1, h2;
float f1, f2;
double d1, d2;

// clang-format off
void ntl_all_sizes() {                                       // CHECK-LABEL: ntl_all_sizes
  uc = __rv_ntl_load(&sc, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i8{{.*}}align 1, !nontemporal !4
  sc = __rv_ntl_load(&uc, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i8{{.*}}align 1, !nontemporal !4
  us = __rv_ntl_load(&ss, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i16{{.*}}align 2, !nontemporal !4
  ss = __rv_ntl_load(&us, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i16{{.*}}align 2, !nontemporal !4
  ui = __rv_ntl_load(&si, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i32{{.*}}align 4, !nontemporal !4
  si = __rv_ntl_load(&ui, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load i32{{.*}}align 4, !nontemporal !4
  ull = __rv_ntl_load(&sll, __RISCV_NTLH_INNERMOST_PRIVATE); // CHECK: load i64{{.*}}align 8, !nontemporal !4
  sll = __rv_ntl_load(&ull, __RISCV_NTLH_INNERMOST_PRIVATE); // CHECK: load i64{{.*}}align 8, !nontemporal !4
  h1 = __rv_ntl_load(&h2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load half{{.*}}align 2, !nontemporal !4
  f1 = __rv_ntl_load(&f2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load float{{.*}}align 4, !nontemporal !4
  d1 = __rv_ntl_load(&d2, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: load double{{.*}}align 8, !nontemporal !4

  uc = __rv_ntl_load(&sc, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i8{{.*}}align 1, !nontemporal !5
  sc = __rv_ntl_load(&uc, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i8{{.*}}align 1, !nontemporal !5
  us = __rv_ntl_load(&ss, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i16{{.*}}align 2, !nontemporal !5
  ss = __rv_ntl_load(&us, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i16{{.*}}align 2, !nontemporal !5
  ui = __rv_ntl_load(&si, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i32{{.*}}align 4, !nontemporal !5
  si = __rv_ntl_load(&ui, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load i32{{.*}}align 4, !nontemporal !5
  ull = __rv_ntl_load(&sll, __RISCV_NTLH_ALL_PRIVATE); // CHECK: load i64{{.*}}align 8, !nontemporal !5
  sll = __rv_ntl_load(&ull, __RISCV_NTLH_ALL_PRIVATE); // CHECK: load i64{{.*}}align 8, !nontemporal !5
  h1 = __rv_ntl_load(&h2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load half{{.*}}align 2, !nontemporal !5
  f1 = __rv_ntl_load(&f2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load float{{.*}}align 4, !nontemporal !5
  d1 = __rv_ntl_load(&d2, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: load double{{.*}}align 8, !nontemporal !5

  uc = __rv_ntl_load(&sc, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i8{{.*}}align 1, !nontemporal !6
  sc = __rv_ntl_load(&uc, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i8{{.*}}align 1, !nontemporal !6
  us = __rv_ntl_load(&ss, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i16{{.*}}align 2, !nontemporal !6
  ss = __rv_ntl_load(&us, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i16{{.*}}align 2, !nontemporal !6
  ui = __rv_ntl_load(&si, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i32{{.*}}align 4, !nontemporal !6
  si = __rv_ntl_load(&ui, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load i32{{.*}}align 4, !nontemporal !6
  ull = __rv_ntl_load(&sll, __RISCV_NTLH_INNERMOST_SHARED); // CHECK: load i64{{.*}}align 8, !nontemporal !6
  sll = __rv_ntl_load(&ull, __RISCV_NTLH_INNERMOST_SHARED); // CHECK: load i64{{.*}}align 8, !nontemporal !6
  h1 = __rv_ntl_load(&h2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load half{{.*}}align 2, !nontemporal !6
  f1 = __rv_ntl_load(&f2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load float{{.*}}align 4, !nontemporal !6
  d1 = __rv_ntl_load(&d2, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: load double{{.*}}align 8, !nontemporal !6

  uc = __rv_ntl_load(&sc, __RISCV_NTLH_ALL);   // CHECK: load i8{{.*}}align 1, !nontemporal !7
  sc = __rv_ntl_load(&uc, __RISCV_NTLH_ALL);   // CHECK: load i8{{.*}}align 1, !nontemporal !7
  us = __rv_ntl_load(&ss, __RISCV_NTLH_ALL);   // CHECK: load i16{{.*}}align 2, !nontemporal !7
  ss = __rv_ntl_load(&us, __RISCV_NTLH_ALL);   // CHECK: load i16{{.*}}align 2, !nontemporal !7
  ui = __rv_ntl_load(&si, __RISCV_NTLH_ALL);   // CHECK: load i32{{.*}}align 4, !nontemporal !7
  si = __rv_ntl_load(&ui, __RISCV_NTLH_ALL);   // CHECK: load i32{{.*}}align 4, !nontemporal !7
  ull = __rv_ntl_load(&sll, __RISCV_NTLH_ALL); // CHECK: load i64{{.*}}align 8, !nontemporal !7
  sll = __rv_ntl_load(&ull, __RISCV_NTLH_ALL); // CHECK: load i64{{.*}}align 8, !nontemporal !7
  h1 = __rv_ntl_load(&h2, __RISCV_NTLH_ALL);   // CHECK: load half{{.*}}align 2, !nontemporal !7
  f1 = __rv_ntl_load(&f2, __RISCV_NTLH_ALL);   // CHECK: load float{{.*}}align 4, !nontemporal !7
  d1 = __rv_ntl_load(&d2, __RISCV_NTLH_ALL);   // CHECK: load double{{.*}}align 8, !nontemporal !7

  __rv_ntl_store(&uc, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i8{{.*}}align 1, !nontemporal !4
  __rv_ntl_store(&sc, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i8{{.*}}align 1, !nontemporal !4
  __rv_ntl_store(&us, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i16{{.*}}align 2, !nontemporal !4
  __rv_ntl_store(&ss, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i16{{.*}}align 2, !nontemporal !4
  __rv_ntl_store(&ui, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i32{{.*}}align 4, !nontemporal !4
  __rv_ntl_store(&si, 1, __RISCV_NTLH_INNERMOST_PRIVATE);    // CHECK: store i32{{.*}}align 4, !nontemporal !4
  __rv_ntl_store(&ull, 1, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: store i64{{.*}}align 8, !nontemporal !4
  __rv_ntl_store(&sll, 1, __RISCV_NTLH_INNERMOST_PRIVATE);   // CHECK: store i64{{.*}}align 8, !nontemporal !4
  __rv_ntl_store(&h1, 1.0, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store half{{.*}}align 2, !nontemporal !4
  __rv_ntl_store(&f1, 1.0, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store float{{.*}}align 4, !nontemporal !4
  __rv_ntl_store(&d1, 1.0, __RISCV_NTLH_INNERMOST_PRIVATE);  // CHECK: store double{{.*}}align 8, !nontemporal !4

  __rv_ntl_store(&uc, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i8{{.*}}align 1, !nontemporal !5
  __rv_ntl_store(&sc, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i8{{.*}}align 1, !nontemporal !5
  __rv_ntl_store(&us, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i16{{.*}}align 2, !nontemporal !5
  __rv_ntl_store(&ss, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i16{{.*}}align 2, !nontemporal !5
  __rv_ntl_store(&ui, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i32{{.*}}align 4, !nontemporal !5
  __rv_ntl_store(&si, 1, __RISCV_NTLH_ALL_PRIVATE);    // CHECK: store i32{{.*}}align 4, !nontemporal !5
  __rv_ntl_store(&ull, 1, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: store i64{{.*}}align 8, !nontemporal !5
  __rv_ntl_store(&sll, 1, __RISCV_NTLH_ALL_PRIVATE);   // CHECK: store i64{{.*}}align 8, !nontemporal !5
  __rv_ntl_store(&h1, 1.0, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store half{{.*}}align 2, !nontemporal !5
  __rv_ntl_store(&f1, 1.0, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store float{{.*}}align 4, !nontemporal !5
  __rv_ntl_store(&d1, 1.0, __RISCV_NTLH_ALL_PRIVATE);  // CHECK: store double{{.*}}align 8, !nontemporal !5

  __rv_ntl_store(&uc, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i8{{.*}}align 1, !nontemporal !6
  __rv_ntl_store(&sc, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i8{{.*}}align 1, !nontemporal !6
  __rv_ntl_store(&us, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i16{{.*}}align 2, !nontemporal !6
  __rv_ntl_store(&ss, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i16{{.*}}align 2, !nontemporal !6
  __rv_ntl_store(&ui, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i32{{.*}}align 4, !nontemporal !6
  __rv_ntl_store(&si, 1, __RISCV_NTLH_INNERMOST_SHARED);    // CHECK: store i32{{.*}}align 4, !nontemporal !6
  __rv_ntl_store(&ull, 1, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: store i64{{.*}}align 8, !nontemporal !6
  __rv_ntl_store(&sll, 1, __RISCV_NTLH_INNERMOST_SHARED);   // CHECK: store i64{{.*}}align 8, !nontemporal !6
  __rv_ntl_store(&h1, 1.0, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store half{{.*}}align 2, !nontemporal !6
  __rv_ntl_store(&f1, 1.0, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store float{{.*}}align 4, !nontemporal !6
  __rv_ntl_store(&d1, 1.0, __RISCV_NTLH_INNERMOST_SHARED);  // CHECK: store double{{.*}}align 8, !nontemporal !6

  __rv_ntl_store(&uc, 1, __RISCV_NTLH_ALL);    // CHECK: store i8{{.*}}align 1, !nontemporal !7
  __rv_ntl_store(&sc, 1, __RISCV_NTLH_ALL);    // CHECK: store i8{{.*}}align 1, !nontemporal !7
  __rv_ntl_store(&us, 1, __RISCV_NTLH_ALL);    // CHECK: store i16{{.*}}align 2, !nontemporal !7
  __rv_ntl_store(&ss, 1, __RISCV_NTLH_ALL);    // CHECK: store i16{{.*}}align 2, !nontemporal !7
  __rv_ntl_store(&ui, 1, __RISCV_NTLH_ALL);    // CHECK: store i32{{.*}}align 4, !nontemporal !7
  __rv_ntl_store(&si, 1, __RISCV_NTLH_ALL);    // CHECK: store i32{{.*}}align 4, !nontemporal !7
  __rv_ntl_store(&ull, 1, __RISCV_NTLH_ALL);   // CHECK: store i64{{.*}}align 8, !nontemporal !7
  __rv_ntl_store(&sll, 1, __RISCV_NTLH_ALL);   // CHECK: store i64{{.*}}align 8, !nontemporal !7
  __rv_ntl_store(&h1, 1.0, __RISCV_NTLH_ALL);  // CHECK: store half{{.*}}align 2, !nontemporal !7
  __rv_ntl_store(&f1, 1.0, __RISCV_NTLH_ALL);  // CHECK: store float{{.*}}align 4, !nontemporal !7
  __rv_ntl_store(&d1, 1.0, __RISCV_NTLH_ALL);  // CHECK: store double{{.*}}align 8, !nontemporal !7

}
// clang-format on

// CHECK: !4 = !{i32 2}
// CHECK: !5 = !{i32 3}
// CHECK: !6 = !{i32 4}
// CHECK: !7 = !{i32 5}
