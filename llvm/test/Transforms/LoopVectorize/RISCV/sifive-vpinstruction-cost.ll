; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 -mcpu=sifive-x280 -debug-only=vplan-cost-model,vplan -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 -mcpu=sifive-p470 -debug-only=vplan-cost-model,vplan -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 -mcpu=sifive-p670 -debug-only=vplan-cost-model,vplan -disable-output %s 2>&1 | FileCheck %s -check-prefix=P670

; check cost of VPInstruction Not and LogicalAnd
define void @foo(ptr %x, ptr %y, float %alpha, i32 %N) {
; CHECK: VPlanCM: cost 2 for RVL (m1, i32) for VPInstruction: EMIT vp<%7> = not ir<%cmp1>
; CHECK: VPlanCM: cost 2 for RVL (m1, i32) for VPInstruction: EMIT vp<%9> = logical-and vp<%7>, ir<%cmp8>
; P670: VPlanCM: cost 1 for RVL (m1, i32) for VPInstruction: EMIT vp<%7> = not ir<%cmp1>
; P670: VPlanCM: cost 1 for RVL (m1, i32) for VPInstruction: EMIT vp<%9> = logical-and vp<%7>, ir<%cmp8>

entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.body.preheader, label %exit

for.body.preheader:
  %wide.trip.count = zext nneg i32 %N to i64
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %for.body.preheader ], [ %iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %y, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp eq i32 %0, 4
  br i1 %cmp1, label %for.inc.sink.split, label %if.else

if.else:
  %arrayidx7 = getelementptr inbounds i32, ptr %x, i64 %iv
  %1 = load i32, ptr %arrayidx7, align 4
  %add = add nsw i32 %1, 8
  %cmp8 = icmp eq i32 %0, %add
  br i1 %cmp8, label %for.inc.sink.split, label %for.inc

for.inc.sink.split:
  %.sink = phi i32 [ 70, %for.body ], [ 80, %if.else ]
  store i32 %.sink, ptr %arrayidx, align 4
  br label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %cond = icmp eq i64 %iv.next, %wide.trip.count
  br i1 %cond, label %exit, label %for.body

exit:
  ret void
}
