; RUN: opt < %s -riscv-use-vla-vectorizer=true -passes=loop-vectorize -mtriple riscv64-linux-gnu -mattr=+v -S 2>&1 -debug-only=loop-vectorize,vplan-cost-model | FileCheck %s

; Check that illegal sew won't crash the vplan cost model

define void @foo1(ptr %val) {
; CHECK-LABEL: LV: Checking a loop in 'foo1'
; CHECK: VPlanCM: unsupported Runtime VL = (unsupported, i40)
;
entry:
  br label %for.cond

for.cond:
  %P.0 = phi ptr [ null, %entry ], [ %incdec.ptr, %for.body ]
  %cond = icmp eq ptr %P.0, %val
  br i1 %cond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void

for.body:
  store i40 0, ptr %P.0, align 4
  %incdec.ptr = getelementptr inbounds i8, ptr %P.0, i64 8
  br label %for.cond
}

; Check that illegal recipe returns invalid cost
define void @foo2(ptr %val) {
; CHECK-LABEL: LV: Checking a loop in 'foo2'
; CHECK: VPlanCM: cost 4 for RVL (m1, i32) for VPInstruction: WIDEN store ir<%P.0>, ir<0>
; CHECK: VPlanCM: cost Invalid for RVL (m1, i32) for VPInstruction: WIDEN store ir<%P.1>, ir<0>
;
entry:
  br label %for.cond

for.cond:
  %P.0 = phi ptr [ null, %entry ], [ %incdec.ptr, %for.body ]
  %P.1 = phi ptr [ null, %entry ], [ %incdec2.ptr, %for.body ]
  %cond = icmp eq ptr %P.0, %val
  br i1 %cond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void

for.body:
  store i32 0, ptr %P.0, align 4
  store i17 0, ptr %P.1, align 4
  %incdec.ptr = getelementptr inbounds i8, ptr %P.0, i64 8
  %incdec2.ptr = getelementptr inbounds i8, ptr %P.1, i64 8
  br label %for.cond
}
