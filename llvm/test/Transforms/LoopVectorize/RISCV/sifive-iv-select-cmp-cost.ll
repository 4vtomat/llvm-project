; REQUIRES: asserts
; RUN: opt < %s -mtriple riscv64-linux-gnu -mattr=+v -riscv-use-vla-vectorizer=true -passes=loop-vectorize -S -debug-only=loop-vectorize,vplan 2>&1 | FileCheck %s

; CHECK-LABEL: LV: Checking a loop in 'select_icmp'
; CHECK: LV: Found an estimated overhead of 5 for VF vscale x 1 For recipe: WIDEN-REDUCTION-PHI ir<%idx.09> = phi ir<%ii>, vp<%18>

define i64 @select_icmp(ptr %a, ptr %b, i64 %ii, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %idx.09 = phi i64 [ %cond, %for.body ], [ %ii, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %i.010
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %i.010
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %i.010, i64 %idx.09
  %inc = add nuw nsw i64 %i.010, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_fcmp'
; CHECK: LV: Found an estimated overhead of 5 for VF vscale x 1 For recipe: WIDEN-REDUCTION-PHI ir<%idx.09> = phi ir<%ii>, vp<%18>

define i64 @select_fcmp(ptr %a, ptr %b, i64 %ii, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %idx.09 = phi i64 [ %cond, %for.body ], [ %ii, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %i.010
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %b, i64 %i.010
  %1 = load float, ptr %arrayidx1, align 4
  %cmp2 = fcmp ogt float %0, %1
  %cond = select i1 %cmp2, i64 %i.010, i64 %idx.09
  %inc = add nuw nsw i64 %i.010, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}
