// RUN: %clang_cc1 -triple riscv32 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// Test RISC-V specific inline assembly constraints.

void test_I(void) {
// CHECK-LABEL: define{{.*}} void @test_I()
// CHECK: call void asm sideeffect "", "I,~{vl},~{vtype}"(i32 2047)
  asm volatile ("" :: "I"(2047));
// CHECK: call void asm sideeffect "", "I,~{vl},~{vtype}"(i32 -2048)
  asm volatile ("" :: "I"(-2048));
}

void test_J(void) {
// CHECK-LABEL: define{{.*}} void @test_J()
// CHECK: call void asm sideeffect "", "J,~{vl},~{vtype}"(i32 0)
  asm volatile ("" :: "J"(0));
}

void test_K(void) {
// CHECK-LABEL: define{{.*}} void @test_K()
// CHECK: call void asm sideeffect "", "K,~{vl},~{vtype}"(i32 31)
  asm volatile ("" :: "K"(31));
// CHECK: call void asm sideeffect "", "K,~{vl},~{vtype}"(i32 0)
  asm volatile ("" :: "K"(0));
}

float f;
double d;
void test_f(void) {
// CHECK-LABEL: define{{.*}} void @test_f()
// CHECK: [[FLT_ARG:%[a-zA-Z_0-9]+]] = load float, ptr @f
// CHECK: call void asm sideeffect "", "f,~{vl},~{vtype}"(float [[FLT_ARG]])
  asm volatile ("" :: "f"(f));
// CHECK: [[FLT_ARG:%[a-zA-Z_0-9]+]] = load double, ptr @d
// CHECK: call void asm sideeffect "", "f,~{vl},~{vtype}"(double [[FLT_ARG]])
  asm volatile ("" :: "f"(d));
}

void test_A(int *p) {
// CHECK-LABEL: define{{.*}} void @test_A(ptr noundef %p)
// CHECK: call void asm sideeffect "", "*A,~{vl},~{vtype}"(ptr elementtype(i32) %p)
  asm volatile("" :: "A"(*p));
}

extern int var, arr[2][2];
struct Pair { int a, b; } pair;

// CHECK-LABEL: test_s(
<<<<<<< HEAD
// CHECK:         call void asm sideeffect "// $0 $1 $2", "s,s,s,~{vl},~{vtype}"(ptr nonnull @var, ptr nonnull getelementptr inbounds (i8, ptr @arr, {{.*}}), ptr nonnull @test_s)
// CHECK:         call void asm sideeffect "// $0", "s,~{vl},~{vtype}"(ptr nonnull getelementptr inbounds (i8, ptr @pair, {{.*}}))
// CHECK:         call void asm sideeffect "// $0 $1 $2", "S,S,S,~{vl},~{vtype}"(ptr nonnull @var, ptr nonnull getelementptr inbounds (i8, ptr @arr, {{.*}}), ptr nonnull @test_s)
=======
// CHECK:         call void asm sideeffect "// $0 $1 $2", "s,s,s"(ptr nonnull @var, ptr nonnull getelementptr inbounds (i8, ptr @arr, {{.*}}), ptr nonnull @test_s)
// CHECK:         call void asm sideeffect "// $0", "s"(ptr nonnull getelementptr inbounds nuw (i8, ptr @pair, {{.*}}))
// CHECK:         call void asm sideeffect "// $0 $1 $2", "S,S,S"(ptr nonnull @var, ptr nonnull getelementptr inbounds (i8, ptr @arr, {{.*}}), ptr nonnull @test_s)
>>>>>>> ddda37a
void test_s(void) {
  asm("// %0 %1 %2" :: "s"(&var), "s"(&arr[1][1]), "s"(test_s));
  asm("// %0" :: "s"(&pair.b));

  asm("// %0 %1 %2" :: "S"(&var), "S"(&arr[1][1]), "S"(test_s));
}
