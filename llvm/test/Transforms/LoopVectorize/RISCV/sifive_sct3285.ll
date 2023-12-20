; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mcpu=sifive-x280 -mtriple riscv64 -debug-only=loop-vectorize -disable-output %s 2>&1| FileCheck %s

; CHECK: LV: The target has no vector registers.

%struct.b2SimplexCache = type { float, i16, [3 x i8], [3 x i8] }

define void @test_small_distance_in_bytes(ptr %cache, i64 %smax) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %cache2 = getelementptr %struct.b2SimplexCache, ptr %cache, i64 0, i32 2, i64 %iv
  store i8 0, ptr %cache2, align 1
  %0 = load i32, ptr null, align 4
  %cache3 = getelementptr %struct.b2SimplexCache, ptr %cache, i64 0, i32 3, i64 %iv
  store i8 0, ptr %cache3, align 1
  %iv.next = add i64 %iv, 1
  %.not = icmp eq i64 %iv, %smax
  br i1 %.not, label %exit, label %for.body

exit:
  ret void
}
