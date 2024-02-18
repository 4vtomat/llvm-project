; RUN: opt -S -mtriple=riscv64 -mattr=+d,+v -riscv-v-vector-bits-min=512  -passes=loop-vectorize -debug-only=loop-vectorize %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; CHECK: LV: Not vectorizing: Runtime ptr check is required with -Os/-Oz

define void @test(ptr %a, ptr %b) #0 {
entry:
  br label %for.body

for.body:
  br label %for.body19

for.body19:
  %indvars.iv1 = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body19 ]
  store double 0.000000e+00, ptr %b, align 8
  store double 0.000000e+00, ptr %a, align 8
  %indvars.iv.next = add i64 %indvars.iv1, 1
  %exitcond.not = icmp eq i64 %indvars.iv1, 12
  br i1 %exitcond.not, label %for.body, label %for.body19
}

attributes #0 = { optsize }
