; RUN: opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 -mcpu=sifive-x280 -debug-only=vplan-cost-model -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 -mcpu=sifive-p470 -debug-only=vplan-cost-model -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 -mcpu=sifive-p670 -debug-only=vplan-cost-model -disable-output %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; CHECK: VPlanCM: cost 1 for RVL (m1, i64) for VPInstruction: CLONE ir<%iv.next49> = add nsw
; CHECK: VPlanCM: cost 1 for RVL (m2, i64) for VPInstruction: CLONE ir<%iv.next49> = add nsw
; CHECK: VPlanCM: cost 1 for RVL (m4, i64) for VPInstruction: CLONE ir<%iv.next49> = add nsw
; CHECK: VPlanCM: cost 1 for RVL (m8, i64) for VPInstruction: CLONE ir<%iv.next49> = add nsw

@flat_2d_array = global [65536 x float] zeroinitializer, align 64

define float @test() {
entry:
  br label %for.body9

cleanup8:
  ret float 0.0

for.body9:
  %iv48 = phi i64 [ 0, %entry ], [ %iv.next49, %for.body9 ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body9 ]
  %iv.next49 = add nsw i64 %iv48, 1
  %arrayidx21 = getelementptr inbounds [65536 x float], ptr @flat_2d_array, i64 0, i64 %iv.next49
  %0 = load float, ptr %arrayidx21, align 4
  store float %0, ptr %arrayidx21, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %not = icmp eq i64 %iv.next, 256
  br i1 %not, label %cleanup8, label %for.body9
}
