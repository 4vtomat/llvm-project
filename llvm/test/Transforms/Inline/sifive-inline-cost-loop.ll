; RUN: opt --passes='print<inline-cost>' %s -disable-output 2>&1 | FileCheck %s
; RUN: opt --passes='print<inline-cost>' %s -enable-loop-inline-bonus=false -disable-output 2>&1 | FileCheck %s --check-prefix=NO-LOOP-BONUS
; RUN: opt --passes='print<inline-cost>' %s -loop-concat-canonicalize -inline-leaf-threshold-limit=1 -disable-output 2>&1 | FileCheck %s --check-prefix=NO-LOOP-BONUS
; RUN: opt --passes='print<inline-cost>' %s -inverse-caller-nested-loop-inline-bonus=true -disable-output 2>&1 | FileCheck %s --check-prefix=INVERSE-CALLER
; RUN: opt --passes='print<inline-cost>' %s -inverse-callee-nested-loop-inline-bonus=false -disable-output 2>&1 | FileCheck %s --check-prefix=NO-INVERSE-CALLEE

; The LoopConcatenation switch with an altered leaf threshold limit is equivalent to disabling
; the loop bonus feature as it does the same thing in this case.

define void @callee(ptr %dst) {
  store i64 87, ptr %dst
  ret void
}

define i64 @callee_with_loop(i64 %x, ptr %dst) {
entry:
  %p0 = icmp slt i64 0, %x
  br i1 %p0, label %header, label %exit

header:
  %iv = phi i64 [0, %entry], [%iv.next, %header]
  store i64 %x, ptr %dst
  %iv.next = add i64 %iv, 2
  %p = icmp slt i64 %iv.next, %x
  br i1 %p, label %header, label %exit

exit:
  ret i64 %x
}

define i64 @callee_with_nested_loop(i64 %x, ptr %dst) {
entry:
  %p0 = icmp slt i64 0, %x
  br i1 %p0, label %header, label %exit

header:
  %iv = phi i64 [0, %entry], [%iv.next, %preheader2], [%iv.next, %header2]
  %iv.next = add i64 %iv, 2
  %p = icmp slt i64 %iv.next, %x
  br i1 %p, label %preheader2, label %exit

preheader2:
  %p0.0 = icmp slt i64 0, %iv.next
  br i1 %p0.0, label %header2, label %header

header2:
  %iv2 = phi i64 [0, %preheader2], [%iv.next2, %header2]
  store i64 %x, ptr %dst
  %iv.next2 = add i64 %iv2, 1
  %p1 = icmp slt i64 %iv.next2, %iv.next
  br i1 %p1, label %header2, label %header

exit:
  ret i64 %x
}

; CHECK-LABEL: Analyzing call of callee_with_loop... (caller:caller_on_loop)
; CHECK: Cost: -18
; CHECK-NEXT: Threshold: 225
; NO-LOOP-BONUS-LABEL: Analyzing call of callee_with_loop... (caller:caller_on_loop)
; NO-LOOP-BONUS: Cost: -10
; NO-LOOP-BONUS-NEXT: Threshold: 225
; NO-INVERSE-CALLEE-LABEL: Analyzing call of callee_with_loop... (caller:caller_on_loop)
; NO-INVERSE-CALLEE: Cost: -18
; NO-INVERSE-CALLEE-NEXT: Threshold: 225
define i64 @caller_on_loop(i64 %x, ptr %dst) {
  %a = add i64 %x, 87
  %r = call i64 @callee_with_loop(i64 %a, ptr %dst)
  ret i64 %r
}

; CHECK-LABEL: Analyzing call of callee_with_nested_loop... (caller:caller_on_nested_loop)
; CHECK: Cost: 1
; CHECK-NEXT: Threshold: 225
; NO-LOOP-BONUS-LABEL: Analyzing call of callee_with_nested_loop... (caller:caller_on_nested_loop)
; NO-LOOP-BONUS: Cost: 15
; NO-LOOP-BONUS-NEXT: Threshold: 225
; NO-INVERSE-CALLEE-LABEL: Analyzing call of callee_with_nested_loop... (caller:caller_on_nested_loop)
; NO-INVERSE-CALLEE: Cost: -7
; NO-INVERSE-CALLEE-NEXT: Threshold: 225
define i64 @caller_on_nested_loop(i64 %x, ptr %dst) {
  %a = add i64 %x, 87
  %r = call i64 @callee_with_nested_loop(i64 %a, ptr %dst)
  ret i64 %r
}

; CHECK-LABEL: Analyzing call of callee... (caller:caller_with_loop)
; CHECK: Cost: -30
; CHECK-NEXT: Threshold: 382
; NO-LOOP-BONUS-LABEL: Analyzing call of callee... (caller:caller_with_loop)
; NO-LOOP-BONUS: Cost: -30
; NO-LOOP-BONUS-NEXT: Threshold: 337
; INVERSE-CALLER-LABEL: Analyzing call of callee... (caller:caller_with_loop)
; INVERSE-CALLER: Cost: -30
; INVERSE-CALLER-NEXT: Threshold: 382
define i64 @caller_with_loop(i64 %x, ptr %dst) {
entry:
  %p0 = icmp slt i64 0, %x
  br i1 %p0, label %header, label %exit

header:
  %iv = phi i64 [0, %entry], [%iv.next, %header]
  call void @callee(ptr %dst)
  %iv.next = add i64 %iv, 2
  %p = icmp slt i64 %iv.next, %x
  br i1 %p, label %header, label %exit

exit:
  ret i64 %x
}

; CHECK-LABEL: Analyzing call of callee... (caller:caller_with_nested_loop)
; CHECK: Cost: -30
; CHECK-NEXT: Threshold: 427
; NO-LOOP-BONUS-LABEL: Analyzing call of callee... (caller:caller_with_nested_loop)
; NO-LOOP-BONUS: Cost: -30
; NO-LOOP-BONUS-NEXT: Threshold: 337
; INVERSE-CALLER-LABEL: Analyzing call of callee... (caller:caller_with_nested_loop)
; INVERSE-CALLER: Cost: -30
; INVERSE-CALLER-NEXT: Threshold: 359
define i64 @caller_with_nested_loop(i64 %x, ptr %dst) {
entry:
  %p0 = icmp slt i64 0, %x
  br i1 %p0, label %header, label %exit

header:
  %iv = phi i64 [0, %entry], [%iv.next, %preheader2], [%iv.next, %header2]
  %iv.next = add i64 %iv, 2
  %p = icmp slt i64 %iv.next, %x
  br i1 %p, label %preheader2, label %exit

preheader2:
  %p0.0 = icmp slt i64 0, %iv.next
  br i1 %p0.0, label %header2, label %header

header2:
  %iv2 = phi i64 [0, %preheader2], [%iv.next2, %header2]
  call void @callee(ptr %dst)
  %iv.next2 = add i64 %iv2, 1
  %p1 = icmp slt i64 %iv.next2, %iv.next
  br i1 %p1, label %header2, label %header

exit:
  ret i64 %x
}
