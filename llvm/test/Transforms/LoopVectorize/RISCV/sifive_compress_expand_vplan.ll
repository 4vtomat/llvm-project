; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mcpu=sifive-x280 -disable-output -debug-only=loop-vectorize %s -mtriple riscv64 2>&1 | FileCheck %s

; CHECK: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4,vscale x 8},UF>=1' {
; CHECK-NEXT: Live-in vp<%0> = vector-trip-count
; CHECK-NEXT: vp<%1> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ph:
; CHECK-NEXT:   EMIT vp<%1> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%16>
; CHECK-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%3> = phi ir<0>, vp<%16>
; CHECK-NEXT:     EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; CHECK-NEXT:     vp<%5> = SCALAR-STEPS vp<%2>, ir<1>
; CHECK-NEXT:     EMIT vp<%6> = EXPLICIT-VECTOR-LENGTH vp<%3>, vp<%1>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%5>
; CHECK-NEXT:     vp<%8> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<%0> = load vp<%8>
; CHECK-NEXT:     WIDEN ir<%tobool.not> = icmp eq ir<%0>, ir<0>
; CHECK-NEXT:     EMIT vp<%11> = not ir<%tobool.not>
; CHECK-NEXT:     CLONE ir<%idx.ext> = sext ir<%ret.011>
; CHECK-NEXT:     CLONE ir<%add.ptr> = getelementptr ir<%a>, ir<%idx.ext>
; CHECK-NEXT:     vp<%14> = vector-pointer ir<%add.ptr>
; CHECK-NEXT:     WIDEN store vp<%14>, ir<%0>, vp<%11>
; CHECK-NEXT:     monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; CHECK-NEXT:     EMIT vp<%16> = EXPLICIT-VECTOR-LENGTH + vp<%3>, vp<%6>
; CHECK-NEXT:     EMIT vp<%17> = add vp<%3>, vp<%6>
; CHECK-NEXT:     EMIT branch-on-count vp<%16>, vp<%0>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: Live-out i32 %ret.1.lcssa = ir<%inc>
; CHECK-NEXT: }

define i32 @compress_store(i32 %n, ptr noalias %a, ptr noalias %b) {
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:
  %ret.0.lcssa = phi i32 [ 0, %entry ], [ %ret.1, %for.inc ]
  ret i32 %ret.0.lcssa

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %ret.011 = phi i32 [ 0, %for.body.preheader ], [ %ret.1, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:
  %idx.ext = sext i32 %ret.011 to i64
  %add.ptr = getelementptr inbounds i32, ptr %a, i64 %idx.ext
  store i32 %0, ptr %add.ptr, align 4
  %inc = add nsw i32 %ret.011, 1
  br label %for.inc

for.inc:
  %ret.1 = phi i32 [ %inc, %if.then ], [ %ret.011, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
