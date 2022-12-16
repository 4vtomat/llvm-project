; RUN: opt < %s -passes=loop-vectorize \
; RUN:   -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN:   -scalable-vectorization=only -riscv-use-vla-vectorizer \
; RUN:   -mtriple riscv64 -riscv-v-vector-bits-min=128 \
; RUN:   -vector-primary-lmul-max=0 -mattr="+v" -S | \
; RUN:   FileCheck %s --check-prefix=CHECK-NO-FORCE-VEC
; RUN: opt < %s -passes=loop-vectorize -force-vectorization \
; RUN:   -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN:   -scalable-vectorization=only -riscv-use-vla-vectorizer \
; RUN:   -mtriple riscv64 -riscv-v-vector-bits-min=128 \
; RUN:   -vector-primary-lmul-max=0 -mattr="+v" -S | \
; RUN:   FileCheck %s --check-prefix=CHECK-FORCE-VEC

; CHECK-NO-FORCE-VEC-NOT: <vscale x 1 x double>
; CHECK-FORCE-VEC: <vscale x 1 x double>

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
