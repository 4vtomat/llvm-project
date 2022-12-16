; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -force-vectorization \
; RUN:   -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN:   -scalable-vectorization=only -riscv-use-vla-vectorizer \
; RUN:   -mtriple riscv64 -riscv-v-vector-bits-min=128 \
; RUN:   -vector-primary-lmul-max=0 -mattr="+v" -debug-only=loop-vectorize -S \
; RUN:   2>&1 | FileCheck %s

; CHECK: LV: Loop hints: force=enabled
; CHECK: LV: Changed scalar cost to Inf as user forced vectorization.
; CHECK: LV: Vectorization seems to be not beneficial, but was forced by a user.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown"

;
; double total;
; int n;
; int x;
; double *m_1;
; double *m_2;
; double *m_3;
; double *m_4;
;
; void foo() {
;   total = 0;
;   for (int i = n - 1; i >= (n - x); i--)
;     total += m_1[i] + m_2[i] + m_3[i] + m_4[i];
; }
;

@total = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@n = dso_local local_unnamed_addr global i32 0, align 4
@x = dso_local local_unnamed_addr global i32 0, align 4
@m_1 = dso_local local_unnamed_addr global ptr null, align 8
@m_2 = dso_local local_unnamed_addr global ptr null, align 8
@m_3 = dso_local local_unnamed_addr global ptr null, align 8
@m_4 = dso_local local_unnamed_addr global ptr null, align 8

define void @no_profit() {
entry:
  store double 0.000000e+00, ptr @total, align 8
  %0 = load i32, ptr @x, align 4
  %cmp.not.not16 = icmp sgt i32 %0, 0
  br i1 %cmp.not.not16, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %1 = load i32, ptr @n, align 4
  %sub1 = sub nsw i32 %1, %0
  %2 = load ptr, ptr @m_1, align 8
  %3 = load ptr, ptr @m_2, align 8
  %4 = load ptr, ptr @m_3, align 8
  %5 = load ptr, ptr @m_4, align 8
  %6 = sext i32 %1 to i64
  %7 = sext i32 %sub1 to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %8 = phi double [ 0.000000e+00, %for.body.lr.ph ], [ %add10, %for.body ]
  %indvars.iv = phi i64 [ %6, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds double, ptr %2, i64 %indvars.iv.next
  %9 = load double, ptr %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds double, ptr %3, i64 %indvars.iv.next
  %10 = load double, ptr %arrayidx3, align 8
  %add = fadd fast double %10, %9
  %arrayidx5 = getelementptr inbounds double, ptr %4, i64 %indvars.iv.next
  %11 = load double, ptr %arrayidx5, align 8
  %add6 = fadd fast double %add, %11
  %arrayidx8 = getelementptr inbounds double, ptr %5, i64 %indvars.iv.next
  %12 = load double, ptr %arrayidx8, align 8
  %add9 = fadd fast double %add6, %12
  %add10 = fadd fast double %add9, %8
  store double %add10, ptr @total, align 8
  %cmp.not.not = icmp sgt i64 %indvars.iv.next, %7
  br i1 %cmp.not.not, label %for.body, label %for.cond.cleanup
}


; The following IR is generated from the following code:
; int foo(int* a, int n) {
;     int res = 0;
; #pragma clang loop vectorize(disable)
;     for (int i = 0; i < n; ++i)
;         res += a[i];
;     return res;
; }

; CHECK: LV: Not vectorizing, Respects #pragma vectorize(disable) over option 'force-vectorization'
define signext i32 @foo(ptr %a, i32 %n) {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %res.05 = phi i32 [ 0, %for.body.preheader ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %res.05
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 false}
