; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize,iv-descriptors -disable-output %s 2>&1 | FileCheck %s

; CHECK-LABEL: LV: Checking a loop in 'select_trunc_iv_icmp_signed_guard'
; CHECK: LV: FindLastIV valid range is [-2147483647,-2147483648), and the range of {0,+,1}<%for.body> is [0,2147483647)
;
define i32 @select_trunc_iv_icmp_signed_guard(ptr %a, ptr %b, i32 %ii, i32 %n) {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %idx.010 = phi i32 [ %ii, %for.body.preheader ], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp3 = icmp sgt i32 %0, %1
  %2 = trunc i64 %indvars.iv to i32
  %cond = select i1 %cmp3, i32 %2, i32 %idx.010
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %idx.0.lcssa = phi i32 [ %ii, %entry ], [ %cond, %for.body ]
  ret i32 %idx.0.lcssa
}

; CHECK-LABEL: LV: Checking a loop in 'select_trunc_iv_icmp_const_rdx_start_lt_const_iv_start_signed_guard'
; CHECK: LV: FindLastIV valid range is [-2147483647,-2147483648), and the range of {0,+,1}<%for.body> is [0,2147483647)
;
define i32 @select_trunc_iv_icmp_const_rdx_start_lt_const_iv_start_signed_guard(ptr %a, ptr %b, i32 %n) {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %idx.010 = phi i32 [ -1, %for.body.preheader ], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp3 = icmp sgt i32 %0, %1
  %2 = trunc i64 %indvars.iv to i32
  %cond = select i1 %cmp3, i32 %2, i32 %idx.010
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %idx.0.lcssa = phi i32 [ -1, %entry ], [ %cond, %for.body ]
  ret i32 %idx.0.lcssa
}

; CHECK-LABEL: LV: Checking a loop in 'select_trunc_iv_icmp_const_rdx_start_eq_const_iv_start_signed_guard'
; CHECK: LV: FindLastIV valid range is [-2147483647,-2147483648), and the range of {0,+,1}<%for.body> is [0,2147483647)
;
define i32 @select_trunc_iv_icmp_const_rdx_start_eq_const_iv_start_signed_guard(ptr %a, ptr %b, i32 %n) {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %idx.010 = phi i32 [ 0, %for.body.preheader ], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp3 = icmp sgt i32 %0, %1
  %2 = trunc i64 %indvars.iv to i32
  %cond = select i1 %cmp3, i32 %2, i32 %idx.010
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %idx.0.lcssa = phi i32 [ 0, %entry ], [ %cond, %for.body ]
  ret i32 %idx.0.lcssa
}

; CHECK-LABEL: LV: Checking a loop in 'select_trunc_iv_icmp_const_rdx_start_gt_const_iv_start_signed_guard'
; CHECK: LV: FindLastIV valid range is [-2147483647,-2147483648), and the range of {0,+,1}<%for.body> is [0,2147483647)
;
define i32 @select_trunc_iv_icmp_const_rdx_start_gt_const_iv_start_signed_guard(ptr %a, ptr %b, i32 %n) {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %idx.010 = phi i32 [ 3, %for.body.preheader ], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp3 = icmp sgt i32 %0, %1
  %2 = trunc i64 %indvars.iv to i32
  %cond = select i1 %cmp3, i32 %2, i32 %idx.010
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %idx.0.lcssa = phi i32 [ 3, %entry ], [ %cond, %for.body ]
  ret i32 %idx.0.lcssa
}

;
; From TSVC/s331
;
; CHECK-LABEL: LV: Checking a loop in 'select_trunc_fcmp_const_tripcount'
; CHECK: LV: FindLastIV valid range is [-2147483647,-2147483648), and the range of {0,+,1}<%for.body> is [0,32000)
;
define i32 @select_trunc_fcmp_const_tripcount(ptr %a) {
entry:
  br label %for.body

for.body:                                        ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %j.119 = phi i32 [ -1, %entry ], [ %j.2, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp6 = fcmp fast olt float %0, 0.000000e+00
  %1 = trunc i64 %indvars.iv to i32
  %j.2 = select i1 %cmp6, i32 %1, i32 %j.119
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 32000
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                            ; preds = %for.body
  ret i32 %j.2
}

; CHECK-LABEL: LV: Checking a loop in 'select_trunc_fcmp_max_valid_const_ub'
; CHECK: LV: FindLastIV valid range is [-2147483647,-2147483648), and the range of {0,+,1}<%for.body> is [0,2147483647)
;
define i32 @select_trunc_fcmp_max_valid_const_ub(ptr %a) {
entry:
  br label %for.body

for.body:                                        ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %j.119 = phi i32 [ -1, %entry ], [ %j.2, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp6 = fcmp fast olt float %0, 0.000000e+00
  %1 = trunc i64 %indvars.iv to i32
  %j.2 = select i1 %cmp6, i32 %1, i32 %j.119
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 2147483647
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                            ; preds = %for.body
  ret i32 %j.2
}
