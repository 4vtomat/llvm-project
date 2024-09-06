; RUN: opt -S -mtriple riscv64 -mattr=+v -debug-only=loop-vectorize -passes=loop-vectorize < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Check that the scalar cost of the predicated block is divided
define void @foo(ptr %y, ptr readonly %b, i32 %N) {
; CHECK: LV: Checking a loop in 'foo' from <stdin>
; CHECK: LV: Found an estimated cost of 0 for VF 1 For instruction:   %iv = phi i64
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   %addry = getelementptr inbounds i32, ptr %y, i64 %iv
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %0 = load i32, ptr %addry, align 4
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cmp1 = icmp eq i32 %0, 50
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %cmp1, label %for.split, label %if.else
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   %addrb = getelementptr inbounds i32, ptr %b, i64 %iv
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load i32, ptr %addrb, align 4
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cmp2 = icmp eq i32 %0, %1
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %cmp2, label %if.then, label %for.inc
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %add = add nsw i32 %0, 5
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   br label %for.split
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   %val = phi i32 [ %add, %if.then ], [ 10003, %for.body ]
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %val, ptr %addry, align 4
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   br label %for.inc
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 1 For instruction:   %cond = icmp eq i64 %iv.next, %wide.trip.count
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 1 For instruction:   br i1 %cond, label %exit.loopexit, label %for.body
; CHECK-NEXT: LV: Scalar loop costs: 5.
;
entry:
  %wide.trip.count = zext nneg i32 %N to i64
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.body, label %exit

for.body:
  %iv = phi i64 [ 0, %entry], [ %iv.next, %for.inc ]
  %addry = getelementptr inbounds i32, ptr %y, i64 %iv
  %0 = load i32, ptr %addry, align 4
  %cmp1 = icmp eq i32 %0, 50
  br i1 %cmp1, label %for.split, label %if.else

if.else:
  %addrb = getelementptr inbounds i32, ptr %b, i64 %iv
  %1 = load i32, ptr %addrb, align 4
  %cmp2 = icmp eq i32 %0, %1
  br i1 %cmp2, label %if.then, label %for.inc

if.then:
  %add = add nsw i32 %0, 5
  br label %for.split

for.split:
  %val = phi i32 [ %add, %if.then ], [ 10003, %for.body ]
  store i32 %val, ptr %addry, align 4
  br label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %cond = icmp eq i64 %iv.next, %wide.trip.count
  br i1 %cond, label %exit, label %for.body

exit:
  ret void

}
