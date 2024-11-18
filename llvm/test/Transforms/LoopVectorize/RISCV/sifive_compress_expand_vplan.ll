; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mcpu=sifive-x280 -disable-output -debug-only=loop-vectorize %s -mtriple riscv64 2>&1 | FileCheck %s

; CHECK-LABEL: LV: Checking a loop in 'compress_store'
; CHECK: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4,vscale x 8},UF={1}' {
; CHECK-NEXT: Live-in vp<%0> = vector-trip-count
; CHECK-NEXT: vp<%1> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body.preheader>:
; CHECK-NEXT:   IR  %wide.trip.count = zext i32 %n to i64
; CHECK-NEXT:   EMIT vp<%1> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%11>
; CHECK-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%3> = phi ir<0>, vp<%11>
; CHECK-NEXT:     EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; CHECK-NEXT:     EMIT vp<%4> = sub vp<%0>, vp<%3>
; CHECK-NEXT:     EMIT vp<%5> = EXPLICIT-VECTOR-LENGTH vp<%4>
; CHECK-NEXT:     vp<%6> = SCALAR-STEPS vp<%3>, ir<1>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%6>
; CHECK-NEXT:     vp<%7> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<%0> = vp.load vp<%7>, vp<%5>	unit-strided
; CHECK-NEXT:     WIDEN ir<%tobool.not> = icmp eq ir<%0>, ir<0>
; CHECK-NEXT:     EMIT vp<%8> = not ir<%tobool.not>
; CHECK-NEXT:     CLONE ir<%idx.ext> = sext ir<%ret.011>
; CHECK-NEXT:     CLONE ir<%add.ptr> = getelementptr ir<%a>, ir<%idx.ext>
; CHECK-NEXT:     vp<%9> = vector-pointer ir<%add.ptr>
; CHECK-NEXT:     WIDEN vp.store vp<%9>, ir<%0>, vp<%5>, vp<%8>	unit-strided
; CHECK-NEXT:     monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%8>
; CHECK-NEXT:     SCALAR-CAST vp<%10> = zext vp<%5> to i64
; CHECK-NEXT:     EMIT vp<%11> = add vp<%10>, vp<%3>
; CHECK-NEXT:     EMIT branch-on-count vp<%11>, vp<%0>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT branch-on-cond ir<true>
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT:   IR   %ret.1.lcssa = phi i32 [ %ret.1, %for.inc ] (extra operand: ir<%inc>)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: No successors
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

; CHECK-LABEL: LV: Checking a loop in 'expand_load'
; CHECK: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4,vscale x 8},UF={1}' {
; CHECK-NEXT: Live-in vp<%0> = vector-trip-count
; CHECK-NEXT: vp<%1> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.body.preheader>:
; CHECK-NEXT:   IR  %wide.trip.count = zext nneg i32 %n to i64
; CHECK-NEXT:   EMIT vp<%1> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:   vector.body:
; CHECK-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%12>
; CHECK-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%3> = phi ir<0>, vp<%12>
; CHECK-NEXT:     EMIT ir<%ret.013> = monotonic-phi ir<0>, ir<%inc>
; CHECK-NEXT:     EMIT vp<%4> = sub vp<%0>, vp<%3>
; CHECK-NEXT:     EMIT vp<%5> = EXPLICIT-VECTOR-LENGTH vp<%4>
; CHECK-NEXT:     vp<%6> = SCALAR-STEPS vp<%3>, ir<1>
; CHECK-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds ir<%b>, vp<%6>
; CHECK-NEXT:     vp<%7> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:     WIDEN ir<%0> = vp.load vp<%7>, vp<%5>	unit-strided
; CHECK-NEXT:     WIDEN ir<%tobool.not> = icmp eq ir<%0>, ir<0>
; CHECK-NEXT:     EMIT vp<%8> = not ir<%tobool.not>
; CHECK-NEXT:     CLONE ir<%idxprom1> = sext ir<%ret.013>
; CHECK-NEXT:     CLONE ir<%arrayidx2> = getelementptr ir<%b>, ir<%idxprom1>
; CHECK-NEXT:     vp<%9> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:     WIDEN ir<%1> = vp.load vp<%9>, vp<%5>, vp<%8>	unit-strided
; CHECK-NEXT:     CLONE ir<%arrayidx4> = getelementptr ir<%a>, vp<%6>
; CHECK-NEXT:     vp<%10> = vector-pointer ir<%arrayidx4>
; CHECK-NEXT:     WIDEN vp.store vp<%10>, ir<%1>, vp<%5>, vp<%8>	unit-strided
; CHECK-NEXT:     monotonic-update ir<%inc> = add ir<%ret.013>, ir<1> @vp<%8>
; CHECK-NEXT:     SCALAR-CAST vp<%11> = zext vp<%5> to i64
; CHECK-NEXT:     EMIT vp<%12> = add vp<%11>, vp<%3>
; CHECK-NEXT:     EMIT branch-on-count vp<%12>, vp<%0>
; CHECK-NEXT:   No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT branch-on-cond ir<true>
; CHECK-NEXT: Successor(s): ir-bb<for.cond.cleanup.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT:   IR   %ret.1.lcssa = phi i32 [ %ret.1, %for.inc ] (extra operand: ir<%inc>)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: No successors
; CHECK-NEXT: }

define i32 @expand_load(i32 %n, ptr noalias %a, ptr noalias %b) {
entry:
  %cmp12 = icmp sgt i32 %n, 0
  br i1 %cmp12, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:
  %ret.1.lcssa = phi i32 [ %ret.1, %for.inc ]
  br label %for.cond.cleanup

for.cond.cleanup:
  %ret.0.lcssa = phi i32 [ 0, %entry ], [ %ret.1.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %ret.0.lcssa

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %ret.013 = phi i32 [ 0, %for.body.preheader ], [ %ret.1, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:
  %idxprom1 = sext i32 %ret.013 to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %idxprom1
  %1 = load i32, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %1, ptr %arrayidx4, align 4
  %inc = add nsw i32 %ret.013, 1
  br label %for.inc

for.inc:
  %ret.1 = phi i32 [ %inc, %if.then ], [ %ret.013, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
