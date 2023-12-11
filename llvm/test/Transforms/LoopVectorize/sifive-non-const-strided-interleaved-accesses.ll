; REQUIRES: asserts
; RUN: opt -S -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses -enable-mem-access-versioning=false -sifive-loop-vectorizer-enable-interleaved-access=invariant-stride -max-interleave-group-factor=8 -debug-only=loop-vectorize,vectorutils -disable-output < %s 2>&1 | FileCheck %s

; CHECK: LV: Checking a loop in 'load_factor_1'
; CHECK: LV: Analyzing interleaved accesses...
; CHECK-NEXT: LV: Creating an interleave group with:  %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT: LV: Invalidate candidate strided interleaved group due to invalid interleave factor.
;
define i32 @load_factor_1(i64 %n, ptr %a, i64 %stride) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.069 = phi i32 [ 0, %entry ], [ %add1, %for.body ]
  %0 = mul nsw i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %1, %red.069
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i32 %add1
}

; CHECK: LV: Checking a loop in 'load_factor_2'
; CHECK: LV: Analyzing interleaved accesses...
; CHECK-NEXT: LV: Creating an interleave group with:  %3 = load i32, ptr %arrayidx5, align 4
; CHECK-NEXT: LV: Inserted:  %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:     into the interleave group with  %3 = load i32, ptr %arrayidx5, align 4
; CHECK-NEXT: LV: Reassign the factor to 2
;
define i32 @load_factor_2(i64 %n, ptr %a, i64 %stride) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.069 = phi i32 [ 0, %entry ], [ %add6, %for.body ]
  %0 = mul nsw i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %1, %red.069
  %2 = add nsw i64 %0, 1
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  %3 = load i32, ptr %arrayidx5, align 4
  %add6 = add nsw i32 %add1, %3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i32 %add6
}

; CHECK: LV: Checking a loop in 'load_factor_3'
; CHECK: LV: Analyzing interleaved accesses...
; CHECK-NEXT: LV: Creating an interleave group with:  %5 = load i32, ptr %arrayidx10, align 4
; CHECK-NEXT: LV: Inserted:  %3 = load i32, ptr %arrayidx5, align 4
; CHECK-NEXT:     into the interleave group with  %5 = load i32, ptr %arrayidx10, align 4
; CHECK-NEXT: LV: Inserted:  %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:     into the interleave group with  %5 = load i32, ptr %arrayidx10, align 4
; CHECK-NEXT: LV: Reassign the factor to 3
;
define i32 @load_factor_3(i64 %n, ptr %a, i64 %stride) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.069 = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  %0 = mul nsw i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %1, %red.069
  %2 = add nsw i64 %0, 1
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  %3 = load i32, ptr %arrayidx5, align 4
  %add6 = add nsw i32 %add1, %3
  %4 = add nsw i64 %0, 2
  %arrayidx10 = getelementptr inbounds i32, ptr %a, i64 %4
  %5 = load i32, ptr %arrayidx10, align 4
  %add11 = add nsw i32 %add6, %5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i32 %add11
}

; CHECK: LV: Checking a loop in 'load_factor_4'
; CHECK: LV: Analyzing interleaved accesses...
; CHECK-NEXT: LV: Creating an interleave group with:  %7 = load i32, ptr %arrayidx15, align 4
; CHECK-NEXT: LV: Inserted:  %5 = load i32, ptr %arrayidx10, align 4
; CHECK-NEXT:     into the interleave group with  %7 = load i32, ptr %arrayidx15, align 4
; CHECK-NEXT: LV: Inserted:  %3 = load i32, ptr %arrayidx5, align 4
; CHECK-NEXT:     into the interleave group with  %7 = load i32, ptr %arrayidx15, align 4
; CHECK-NEXT: LV: Inserted:  %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:     into the interleave group with  %7 = load i32, ptr %arrayidx15, align 4
; CHECK-NEXT: LV: Reassign the factor to 4
;
define i32 @load_factor_4(i64 %n, ptr %a, i64 %stride) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.069 = phi i32 [ 0, %entry ], [ %add16, %for.body ]
  %0 = mul nsw i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %1, %red.069
  %2 = add nsw i64 %0, 1
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  %3 = load i32, ptr %arrayidx5, align 4
  %add6 = add nsw i32 %add1, %3
  %4 = add nsw i64 %0, 2
  %arrayidx10 = getelementptr inbounds i32, ptr %a, i64 %4
  %5 = load i32, ptr %arrayidx10, align 4
  %add11 = add nsw i32 %add6, %5
  %6 = add nsw i64 %0, 3
  %arrayidx15 = getelementptr inbounds i32, ptr %a, i64 %6
  %7 = load i32, ptr %arrayidx15, align 4
  %add16 = add nsw i32 %add11, %7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i32 %add16
}

; CHECK: LV: Checking a loop in 'load_factor_5'
; CHECK: LV: Analyzing interleaved accesses...
; CHECK-NEXT: LV: Creating an interleave group with:  %9 = load i32, ptr %arrayidx20, align 4
; CHECK-NEXT: LV: Inserted:  %7 = load i32, ptr %arrayidx15, align 4
; CHECK-NEXT:     into the interleave group with  %9 = load i32, ptr %arrayidx20, align 4
; CHECK-NEXT: LV: Inserted:  %5 = load i32, ptr %arrayidx10, align 4
; CHECK-NEXT:     into the interleave group with  %9 = load i32, ptr %arrayidx20, align 4
; CHECK-NEXT: LV: Inserted:  %3 = load i32, ptr %arrayidx5, align 4
; CHECK-NEXT:     into the interleave group with  %9 = load i32, ptr %arrayidx20, align 4
; CHECK-NEXT: LV: Inserted:  %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:     into the interleave group with  %9 = load i32, ptr %arrayidx20, align 4
; CHECK-NEXT: LV: Reassign the factor to 5
;
define i32 @load_factor_5(i64 %n, ptr %a, i64 %stride) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.069 = phi i32 [ 0, %entry ], [ %add21, %for.body ]
  %0 = mul nsw i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %1, %red.069
  %2 = add nsw i64 %0, 1
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  %3 = load i32, ptr %arrayidx5, align 4
  %add6 = add nsw i32 %add1, %3
  %4 = add nsw i64 %0, 2
  %arrayidx10 = getelementptr inbounds i32, ptr %a, i64 %4
  %5 = load i32, ptr %arrayidx10, align 4
  %add11 = add nsw i32 %add6, %5
  %6 = add nsw i64 %0, 3
  %arrayidx15 = getelementptr inbounds i32, ptr %a, i64 %6
  %7 = load i32, ptr %arrayidx15, align 4
  %add16 = add nsw i32 %add11, %7
  %8 = add nsw i64 %0, 4
  %arrayidx20 = getelementptr inbounds i32, ptr %a, i64 %8
  %9 = load i32, ptr %arrayidx20, align 4
  %add21 = add nsw i32 %add16, %9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i32 %add21
}

; CHECK: LV: Checking a loop in 'load_factor_6'
; CHECK: LV: Analyzing interleaved accesses...
; CHECK-NEXT: LV: Creating an interleave group with:  %11 = load i32, ptr %arrayidx25, align 4
; CHECK-NEXT: LV: Inserted:  %9 = load i32, ptr %arrayidx20, align 4
; CHECK-NEXT:     into the interleave group with  %11 = load i32, ptr %arrayidx25, align 4
; CHECK-NEXT: LV: Inserted:  %7 = load i32, ptr %arrayidx15, align 4
; CHECK-NEXT:     into the interleave group with  %11 = load i32, ptr %arrayidx25, align 4
; CHECK-NEXT: LV: Inserted:  %5 = load i32, ptr %arrayidx10, align 4
; CHECK-NEXT:     into the interleave group with  %11 = load i32, ptr %arrayidx25, align 4
; CHECK-NEXT: LV: Inserted:  %3 = load i32, ptr %arrayidx5, align 4
; CHECK-NEXT:     into the interleave group with  %11 = load i32, ptr %arrayidx25, align 4
; CHECK-NEXT: LV: Inserted:  %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:     into the interleave group with  %11 = load i32, ptr %arrayidx25, align 4
; CHECK-NEXT: LV: Reassign the factor to 6
;
define i32 @load_factor_6(i64 %n, ptr %a, i64 %stride) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.069 = phi i32 [ 0, %entry ], [ %add26, %for.body ]
  %0 = mul nsw i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %1, %red.069
  %2 = add nsw i64 %0, 1
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  %3 = load i32, ptr %arrayidx5, align 4
  %add6 = add nsw i32 %add1, %3
  %4 = add nsw i64 %0, 2
  %arrayidx10 = getelementptr inbounds i32, ptr %a, i64 %4
  %5 = load i32, ptr %arrayidx10, align 4
  %add11 = add nsw i32 %add6, %5
  %6 = add nsw i64 %0, 3
  %arrayidx15 = getelementptr inbounds i32, ptr %a, i64 %6
  %7 = load i32, ptr %arrayidx15, align 4
  %add16 = add nsw i32 %add11, %7
  %8 = add nsw i64 %0, 4
  %arrayidx20 = getelementptr inbounds i32, ptr %a, i64 %8
  %9 = load i32, ptr %arrayidx20, align 4
  %add21 = add nsw i32 %add16, %9
  %10 = add nsw i64 %0, 5
  %arrayidx25 = getelementptr inbounds i32, ptr %a, i64 %10
  %11 = load i32, ptr %arrayidx25, align 4
  %add26 = add nsw i32 %add21, %11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i32 %add26
}

; CHECK: LV: Checking a loop in 'load_factor_7'
; CHECK: LV: Analyzing interleaved accesses...
; CHECK-NEXT: LV: Creating an interleave group with:  %13 = load i32, ptr %arrayidx30, align 4
; CHECK-NEXT: LV: Inserted:  %11 = load i32, ptr %arrayidx25, align 4
; CHECK-NEXT:     into the interleave group with  %13 = load i32, ptr %arrayidx30, align 4
; CHECK-NEXT: LV: Inserted:  %9 = load i32, ptr %arrayidx20, align 4
; CHECK-NEXT:     into the interleave group with  %13 = load i32, ptr %arrayidx30, align 4
; CHECK-NEXT: LV: Inserted:  %7 = load i32, ptr %arrayidx15, align 4
; CHECK-NEXT:     into the interleave group with  %13 = load i32, ptr %arrayidx30, align 4
; CHECK-NEXT: LV: Inserted:  %5 = load i32, ptr %arrayidx10, align 4
; CHECK-NEXT:     into the interleave group with  %13 = load i32, ptr %arrayidx30, align 4
; CHECK-NEXT: LV: Inserted:  %3 = load i32, ptr %arrayidx5, align 4
; CHECK-NEXT:     into the interleave group with  %13 = load i32, ptr %arrayidx30, align 4
; CHECK-NEXT: LV: Inserted:  %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:     into the interleave group with  %13 = load i32, ptr %arrayidx30, align 4
; CHECK-NEXT: LV: Reassign the factor to 7
;
define i32 @load_factor_7(i64 %n, ptr %a, i64 %stride) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.069 = phi i32 [ 0, %entry ], [ %add31, %for.body ]
  %0 = mul nsw i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %1, %red.069
  %2 = add nsw i64 %0, 1
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  %3 = load i32, ptr %arrayidx5, align 4
  %add6 = add nsw i32 %add1, %3
  %4 = add nsw i64 %0, 2
  %arrayidx10 = getelementptr inbounds i32, ptr %a, i64 %4
  %5 = load i32, ptr %arrayidx10, align 4
  %add11 = add nsw i32 %add6, %5
  %6 = add nsw i64 %0, 3
  %arrayidx15 = getelementptr inbounds i32, ptr %a, i64 %6
  %7 = load i32, ptr %arrayidx15, align 4
  %add16 = add nsw i32 %add11, %7
  %8 = add nsw i64 %0, 4
  %arrayidx20 = getelementptr inbounds i32, ptr %a, i64 %8
  %9 = load i32, ptr %arrayidx20, align 4
  %add21 = add nsw i32 %add16, %9
  %10 = add nsw i64 %0, 5
  %arrayidx25 = getelementptr inbounds i32, ptr %a, i64 %10
  %11 = load i32, ptr %arrayidx25, align 4
  %add26 = add nsw i32 %add21, %11
  %12 = add nsw i64 %0, 6
  %arrayidx30 = getelementptr inbounds i32, ptr %a, i64 %12
  %13 = load i32, ptr %arrayidx30, align 4
  %add31 = add nsw i32 %add26, %13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i32 %add31
}

; CHECK: LV: Checking a loop in 'load_factor_8'
; CHECK: LV: Analyzing interleaved accesses...
; CHECK-NEXT: LV: Creating an interleave group with:  %15 = load i32, ptr %arrayidx35, align 4
; CHECK-NEXT: LV: Inserted:  %13 = load i32, ptr %arrayidx30, align 4
; CHECK-NEXT:     into the interleave group with  %15 = load i32, ptr %arrayidx35, align 4
; CHECK-NEXT: LV: Inserted:  %11 = load i32, ptr %arrayidx25, align 4
; CHECK-NEXT:     into the interleave group with  %15 = load i32, ptr %arrayidx35, align 4
; CHECK-NEXT: LV: Inserted:  %9 = load i32, ptr %arrayidx20, align 4
; CHECK-NEXT:     into the interleave group with  %15 = load i32, ptr %arrayidx35, align 4
; CHECK-NEXT: LV: Inserted:  %7 = load i32, ptr %arrayidx15, align 4
; CHECK-NEXT:     into the interleave group with  %15 = load i32, ptr %arrayidx35, align 4
; CHECK-NEXT: LV: Inserted:  %5 = load i32, ptr %arrayidx10, align 4
; CHECK-NEXT:     into the interleave group with  %15 = load i32, ptr %arrayidx35, align 4
; CHECK-NEXT: LV: Inserted:  %3 = load i32, ptr %arrayidx5, align 4
; CHECK-NEXT:     into the interleave group with  %15 = load i32, ptr %arrayidx35, align 4
; CHECK-NEXT: LV: Inserted:  %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:     into the interleave group with  %15 = load i32, ptr %arrayidx35, align 4
; CHECK-NOT: LV: Reassign the factor to 8
;
define i32 @load_factor_8(i64 %n, ptr %a, i64 %stride) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %red.069 = phi i32 [ 0, %entry ], [ %add36, %for.body ]
  %0 = mul nsw i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %1, %red.069
  %2 = add nsw i64 %0, 1
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  %3 = load i32, ptr %arrayidx5, align 4
  %add6 = add nsw i32 %add1, %3
  %4 = add nsw i64 %0, 2
  %arrayidx10 = getelementptr inbounds i32, ptr %a, i64 %4
  %5 = load i32, ptr %arrayidx10, align 4
  %add11 = add nsw i32 %add6, %5
  %6 = add nsw i64 %0, 3
  %arrayidx15 = getelementptr inbounds i32, ptr %a, i64 %6
  %7 = load i32, ptr %arrayidx15, align 4
  %add16 = add nsw i32 %add11, %7
  %8 = add nsw i64 %0, 4
  %arrayidx20 = getelementptr inbounds i32, ptr %a, i64 %8
  %9 = load i32, ptr %arrayidx20, align 4
  %add21 = add nsw i32 %add16, %9
  %10 = add nsw i64 %0, 5
  %arrayidx25 = getelementptr inbounds i32, ptr %a, i64 %10
  %11 = load i32, ptr %arrayidx25, align 4
  %add26 = add nsw i32 %add21, %11
  %12 = add nsw i64 %0, 6
  %arrayidx30 = getelementptr inbounds i32, ptr %a, i64 %12
  %13 = load i32, ptr %arrayidx30, align 4
  %add31 = add nsw i32 %add26, %13
  %14 = add nsw i64 %0, 7
  %arrayidx35 = getelementptr inbounds i32, ptr %a, i64 %14
  %15 = load i32, ptr %arrayidx35, align 4
  %add36 = add nsw i32 %add31, %15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i32 %add36
}
