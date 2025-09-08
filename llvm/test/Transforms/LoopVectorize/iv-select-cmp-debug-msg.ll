; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize,iv-descriptors -disable-output %s 2>&1 | FileCheck %s

; CHECK-LABEL: LV: Checking a loop in 'select_icmp_const_1'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_icmp_const_1(ptr %a, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 3, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %cmp2 = icmp eq i64 %0, 3
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_icmp_const_2'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_icmp_const_2(ptr %a, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 3, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %cmp2 = icmp eq i64 %0, 3
  %cond = select i1 %cmp2, i64 %rdx, i64 %iv
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_icmp_const_3_variable_rdx_start'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_icmp_const_3_variable_rdx_start(ptr %a, i64 %rdx.start, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %cmp2 = icmp eq i64 %0, 3
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_fcmp_const_fast'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_fcmp_const_fast(ptr %a, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 2, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp2 = fcmp fast ueq float %0, 3.0
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_fcmp_const'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_fcmp_const(ptr %a, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 2, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp2 = fcmp ueq float %0, 3.0
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_icmp'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_icmp(ptr %a, ptr %b, i64 %rdx.start, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_fcmp'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_fcmp(ptr %a, ptr %b, i64 %rdx.start, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %b, i64 %iv
  %1 = load float, ptr %arrayidx1, align 4
  %cmp2 = fcmp ogt float %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_icmp_const_rdx_start_lt_const_iv_start'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_icmp_const_rdx_start_lt_const_iv_start(ptr %a, ptr %b, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ -1, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_icmp_const_rdx_start_eq_const_iv_start'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_icmp_const_rdx_start_eq_const_iv_start(ptr %a, ptr %b, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_icmp_const_rdx_start_gt_const_iv_start'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {0,+,1}<nuw><nsw><%for.body> is [0,-9223372036854775808)
;
define i64 @select_icmp_const_rdx_start_gt_const_iv_start(ptr %a, ptr %b, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 3, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; CHECK-LABEL: LV: Checking a loop in 'select_icmp_min_valid_iv_start'
; CHECK: LV: FindLastIV valid range is [-9223372036854775807,-9223372036854775808), and the range of {-9223372036854775807,+,1}<nsw><%for.body> is [-9223372036854775807,-9223372036854775808)
;
define i64 @select_icmp_min_valid_iv_start(ptr %a, ptr %b, i64 %rdx.start, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv.j = phi i64 [ %inc3, %for.body ], [ -9223372036854775807, %entry]
  %iv.i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv.i
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv.i
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv.j, i64 %rdx
  %inc = add nuw nsw i64 %iv.i, 1
  %inc3 = add nsw i64 %iv.j, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; Negative tests

; CHECK-LABEL: LV: Checking a loop in 'not_vectorized_select_icmp_iv_out_of_bound'
; CHECK: LV: Not vectorizing: Found an unidentified PHI   %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
;
define i64 @not_vectorized_select_icmp_iv_out_of_bound(ptr %a, ptr %b, i64 %rdx.start, i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv.j = phi i64 [ %inc3, %for.body ], [ -9223372036854775808, %entry]
  %iv.i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv.i
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv.i
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv.j, i64 %rdx
  %inc = add nuw nsw i64 %iv.i, 1
  %inc3 = add nsw i64 %iv.j, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}
