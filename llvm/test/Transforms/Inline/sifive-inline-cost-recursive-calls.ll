; REQUIRES: asserts
; RUN: opt --passes=inline -debug-only=inline %s -disable-output -allow-inline-recursive-funcs 2>&1 | FileCheck %s

define dso_local signext i32 @user(i32 noundef signext %n) local_unnamed_addr {
entry:
  %call = tail call fastcc signext i32 @fib(i32 noundef signext 10)
  %call1 = tail call fastcc signext i32 @fib(i32 noundef signext %n)
  %add = add i32 %call1, %call
  ret i32 %add
}

; CHECK: Inlining calls in: fib
; CHECK: Inlining (cost={{[0-9]*}}, threshold={{[0-9]*}}), Call:   %call = tail call fastcc signext i32 @fib(i32 noundef signext %sub)
; CHECK-NOT: NOT Inlining (cost=never): recursive, Call:   %call = tail call fastcc signext i32 @fib(i32 noundef signext %sub)
define internal fastcc signext i32 @fib(i32 noundef signext %n) unnamed_addr {
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.end, %entry
  %accumulator.tr = phi i32 [ 0, %entry ], [ %add, %if.end ]
  %n.tr = phi i32 [ %n, %entry ], [ %sub1, %if.end ]
  %cmp = icmp ult i32 %n.tr, 2
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %tailrecurse
  %sub = add i32 %n.tr, -1
  %call = tail call fastcc signext i32 @fib(i32 noundef signext %sub)
  %sub1 = add i32 %n.tr, -2
  %add = add i32 %accumulator.tr, %call
  br label %tailrecurse

return:                                           ; preds = %tailrecurse
  %accumulator.ret.tr = add i32 %accumulator.tr, %n.tr
  ret i32 %accumulator.ret.tr
}
