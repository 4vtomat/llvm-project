// RUN: %clang_cc1 %s -O1 -disable-llvm-passes -emit-llvm -o - | FileCheck %s
#include <stdint.h>

int32_t foo(const int32_t n, const int32_t *__restrict a)
{
  int32_t red = 0;
// CHECK: !{!"llvm.loop.vectorize.enable", i1 false}
    #pragma clang loop vectorize(disable)
    for (int32_t i = 0; i < n; ++i) {
        red += a[i];
    }
  return red;
}

