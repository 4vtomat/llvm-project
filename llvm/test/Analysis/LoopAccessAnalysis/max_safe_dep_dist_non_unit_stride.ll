; RUN: opt -S -disable-output -passes='print<access-info>' < %s 2>&1 | FileCheck %s

; Generated from following C program:
; void foo(int len, int *a) {
;   for (int k = 0; k < len; k+=3) {
;     a[k] = a[k + 4];
;     a[k+2] = a[k+6];
;   }
; }
define void @foo(i32 noundef signext %len, ptr nocapture noundef %a) {
; CHECK-LABEL: Loop access info in function 'foo':
; CHECK-NEXT:  for.body:
; CHECK-NEXT:    Memory dependences are safe with a maximum dependence distance of 8 bytes
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      BackwardVectorizable:
; CHECK-NEXT:          store i32 %2, ptr %arrayidx2, align 4 ->
; CHECK-NEXT:          %4 = load i32, ptr %arrayidx5, align 4
; CHECK-EMPTY:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:
;
entry:
  %cmp18 = icmp sgt i32 %len, 0
  br i1 %cmp18, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = zext i32 %len to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %1 = add nuw nsw i64 %indvars.iv, 4
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %1
  %2 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %2, ptr %arrayidx2, align 4
  %3 = add nuw nsw i64 %indvars.iv, 6
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %3
  %4 = load i32, ptr %arrayidx5, align 4
  %5 = add nuw nsw i64 %indvars.iv, 2
  %arrayidx8 = getelementptr inbounds i32, ptr %a, i64 %5
  store i32 %4, ptr %arrayidx8, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 3
  %cmp = icmp ult i64 %indvars.iv.next, %0
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit
}
