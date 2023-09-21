; This file verifies the behavior of the OptBisect class, which is used to
; diagnose optimization related failures.  The tests check various
; invocations that result in different sets of optimization passes that
; are run in different ways.

; RUN: opt -riscv-disable-all-loop-idiom=false -passes=riscv-loop-idiom -mtriple=riscv64-unknown-linux-gnu -mattr=+v -opt-bisect-limit=-1 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-ALL-PASS
; CHECK-ALL-PASS: BISECT: running pass (1) LoopSimplifyPass on compare_bytes_simple
; CHECK-ALL-PASS: BISECT: running pass (2) LCSSAPass on compare_bytes_simple
; CHECK-ALL-PASS: BISECT: running pass (3) RISCVLoopIdiomRecognizePass on while.cond

; RUN: opt -riscv-disable-all-loop-idiom=false -passes=riscv-loop-idiom -mtriple=riscv64-unknown-linux-gnu -mattr=+v -opt-bisect-limit=0 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NO-PASS
; CHECK-NO-PASS: BISECT: NOT running pass (1) LoopSimplifyPass on compare_bytes_simple
; CHECK-NO-PASS: BISECT: NOT running pass (2) LCSSAPass on compare_bytes_simple
; CHECK-NO-PASS: BISECT: NOT running pass (3) RISCVLoopIdiomRecognizePass on while.cond

; RUN: opt -riscv-disable-all-loop-idiom=false -passes=riscv-loop-idiom -mtriple=riscv64-unknown-linux-gnu -mattr=+v -opt-bisect-limit=1 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-ONE-PASS
; CHECK-ONE-PASS: BISECT: running pass (1) LoopSimplifyPass on compare_bytes_simple
; CHECK-ONE-PASS: BISECT: NOT running pass (2) LCSSAPass on compare_bytes_simple
; CHECK-ONE-PASS: BISECT: NOT running pass (3) RISCVLoopIdiomRecognizePass on while.cond

; RUN: opt -riscv-disable-all-loop-idiom=false -passes=riscv-loop-idiom -mtriple=riscv64-unknown-linux-gnu -mattr=+v -opt-bisect-limit=2 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-PASS
; CHECK-TWO-PASS: BISECT: running pass (1) LoopSimplifyPass on compare_bytes_simple
; CHECK-TWO-PASS: BISECT: running pass (2) LCSSAPass on compare_bytes_simple
; CHECK-TWO-PASS: BISECT: NOT running pass (3) RISCVLoopIdiomRecognizePass on while.cond

; RUN: opt -riscv-disable-all-loop-idiom=false -passes=riscv-loop-idiom -mtriple=riscv64-unknown-linux-gnu -mattr=+v -opt-bisect-limit=3 %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-THREE-PASS
; CHECK-THREE-PASS: BISECT: running pass (1) LoopSimplifyPass on compare_bytes_simple
; CHECK-THREE-PASS: BISECT: running pass (2) LCSSAPass on compare_bytes_simple
; CHECK-THREE-PASS: BISECT: running pass (3) RISCVLoopIdiomRecognizePass on while.cond

define i32 @compare_bytes_simple(ptr %a, ptr %b, i32 %len, i32 %n) {
entry:
  br label %while.cond

while.cond:
  %len.addr = phi i32 [ %len, %entry ], [ %inc, %while.body ]
  %inc = add i32 %len.addr, 1
  %cmp.not = icmp eq i32 %inc, %n
  br i1 %cmp.not, label %while.end, label %while.body

while.body:
  %idxprom = zext i32 %inc to i64
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %idxprom
  %0 = load i8, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %idxprom
  %1 = load i8, ptr %arrayidx2
  %cmp.not2 = icmp eq i8 %0, %1
  br i1 %cmp.not2, label %while.cond, label %while.end

while.end:
  %inc.lcssa = phi i32 [ %inc, %while.body ], [ %inc, %while.cond ]
  ret i32 %inc.lcssa
}
