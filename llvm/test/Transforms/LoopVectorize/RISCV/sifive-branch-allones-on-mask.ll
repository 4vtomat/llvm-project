; RUN: opt < %s -riscv-use-vla-vectorizer=true -passes=loop-vectorize -mtriple riscv64-linux-gnu -mattr=+v,+f -S -debug-only=loop-vectorize 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: LV: Scalarizing:  %exitcond.not = icmp eq i64 %iv.next, 1024
; CHECK-NEXT: VPlan 'Initial VPlan for VF={vscale x 1},UF>=1' {

; If this test sees the LV: Scalarizing debug message, then a call to
; createReplicateRegion was made. Previously, createReplicateRegion was
; crashing this test case because there was a mask of all ones. If the
; VPlan debug message is made, that means that there was no crash.

define void @vector_srem(ptr noalias nocapture %a, i64 %v, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %elem = load i64, ptr %arrayidx
  %divrem = srem i64 %elem, %v
  store i64 %divrem, ptr %arrayidx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret void
}


