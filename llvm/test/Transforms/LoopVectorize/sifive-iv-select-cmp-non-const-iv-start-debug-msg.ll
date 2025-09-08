; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize,iv-descriptors -scalar-evolution-use-expensive-range-sharpening -disable-output %s 2>&1 | FileCheck %s

; CHECK-LABEL: LV: Checking a loop in 'select_non_const_iv_start_signed_guard'
; CHECK: LV: FindLastIV valid range is [-9223372036854775808,9223372036854775807), and the signed range of {%iv_start,+,1}<nsw><%for.body> is [-9223372036854775808,9223372036854775807)
; CHECK: Found a FindLastIV reduction PHI.
;
define i64 @select_non_const_iv_start_signed_guard(ptr %a, i64 %ii, i64 %iv_start ,i64 %n) {
entry:
  %cmp6 = icmp slt i64 %iv_start, %n
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %iv_start, %entry ], [ %indvars.iv.next, %for.body ]
  %rdx.07 = phi i64 [ %ii, %entry ], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %1 = load i64, ptr %arrayidx, align 4
  %cmp1 = icmp sgt i64 %1, 3
  %cond = select i1 %cmp1, i64 %indvars.iv, i64 %rdx.07
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %idx.0.lcssa = phi i64 [ %ii, %entry ], [ %cond, %for.body ]
  ret i64 %idx.0.lcssa
}

; CHECK-LABEL: LV: Checking a loop in 'select_trunc_non_const_iv_start_signed_guard'
; CHECK: LV: FindLastIV valid range is [-2147483648,2147483647), and the signed range of {%iv_start,+,1}<%for.body> is full-set
; CHECK-NOT: Found a FindLastIV reduction PHI.
;
define i32 @select_trunc_non_const_iv_start_signed_guard(ptr %a, i32 %ii, i32 %iv_start ,i32 %n) {
entry:
  %cmp6 = icmp slt i32 %iv_start, %n
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = sext i32 %iv_start to i64
  %wide.trip.count = sext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %rdx.07 = phi i32 [ %ii, %for.body.preheader ], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp sgt i32 %1, 3
  %2 = trunc i64 %indvars.iv to i32
  %cond = select i1 %cmp1, i32 %2, i32 %rdx.07
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %idx.0.lcssa = phi i32 [ %ii, %entry ], [ %cond, %for.body ]
  ret i32 %idx.0.lcssa
}
