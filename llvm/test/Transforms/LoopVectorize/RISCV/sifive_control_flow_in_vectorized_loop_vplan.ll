; RUN: opt -passes=loop-vectorize -mtriple riscv64 -mcpu=sifive-p470 -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck --check-prefix VPLAN %s

; REQUIRES: asserts

; VPLAN:  VPlan 'Final VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF={1}' {
; VPLAN-NEXT:  Live-in vp<%0> = vector-trip-count
; VPLAN-NEXT:  vp<%1> = original trip-count
; VPLAN-EMPTY:
; VPLAN-NEXT:  ir-bb<for.body.lr.ph>:
; VPLAN-NEXT:    EMIT vp<%1> = EXPAND SCEV (zext i32 %reg.4.val to i64)
; VPLAN-NEXT:  No successors
; VPLAN-EMPTY:
; VPLAN-NEXT:  vector.ph:
; VPLAN-NEXT:  Successor(s): vector loop
; VPLAN-EMPTY:
; VPLAN-NEXT:  <x1> vector loop: {
; VPLAN-NEXT:    vector.body:
; VPLAN-NEXT:      EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%9>
; VPLAN-NEXT:      EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%3> = phi ir<0>, vp<%9>
; VPLAN-NEXT:      EMIT vp<%4> = EXPLICIT-VECTOR-LENGTH vp<%3>, vp<%0>
; VPLAN-NEXT:      vp<%5> = SCALAR-STEPS vp<%3>, ir<1>
; VPLAN-NEXT:      CLONE ir<%arrayidx> = getelementptr ir<%reg.24.val>, vp<%5>
; VPLAN-NEXT:      vp<%6> = vector-pointer ir<%arrayidx>
; VPLAN-NEXT:      WIDEN ir<%2> = vp.load vp<%6>, vp<%4>       unit-strided
; VPLAN-NEXT:      WIDEN ir<%3> = and ir<%2>, ir<%1>
; VPLAN-NEXT:      WIDEN ir<%or.cond.not> = icmp eq ir<%3>, ir<%1>
; VPLAN-NEXT:    Successor(s): 
; VPLAN-EMPTY:
; VPLAN-NEXT:    if (ir<%or.cond.not> != 0) {
; VPLAN-NEXT:      vector.if.bb:
; VPLAN-NEXT:        WIDEN ir<%xor> = xor ir<%2>, ir<%shl11>
; VPLAN-NEXT:        vp<%7> = vector-pointer ir<%arrayidx>
; VPLAN-NEXT:        WIDEN vp.store vp<%7>, ir<%xor>, vp<%4>, ir<%or.cond.not> unit-strided
; VPLAN-NEXT:        BRANCH-ON-MASK  All-One, vector.body.split
; VPLAN-NEXT:      No successors
; VPLAN-NEXT:    }
; VPLAN-NEXT:    Successor(s): vector.body.split
; VPLAN-EMPTY:
; VPLAN-NEXT:    vector.body.split:
; VPLAN-NEXT:      SCALAR-CAST vp<%8> = zext vp<%4> to i64
; VPLAN-NEXT:      EMIT vp<%9> = add vp<%8>, vp<%3>
; VPLAN-NEXT:      EMIT vp<%10> = add vp<%3>, vp<%4>
; VPLAN-NEXT:      EMIT branch-on-count vp<%9>, vp<%0>
; VPLAN-NEXT:    No successors
; VPLAN-NEXT:  }
; VPLAN-NEXT:  Successor(s): middle.block
; VPLAN-EMPTY:
; VPLAN-NEXT:  middle.block:
; VPLAN-NEXT:    EMIT branch-on-cond ir<true>
; VPLAN-NEXT:  Successor(s): ir-bb<for.end.loopexit>, scalar.ph
; VPLAN-EMPTY:
; VPLAN-NEXT:  ir-bb<for.end.loopexit>:
; VPLAN-NEXT:  No successors
; VPLAN-EMPTY:
; VPLAN-NEXT:  scalar.ph:
; VPLAN-NEXT:  No successors
; VPLAN-NEXT:  }

define void @test(i32 %control1, i32 %control2, i32 %target, i32 %reg.4.val, ptr %reg.24.val) {
entry:
  %cmp1 = icmp sgt i32 %reg.4.val, 0
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %sh_prom = zext nneg i32 %control1 to i64
  %shl = shl nuw i64 1, %sh_prom
  %sh_prom5 = zext nneg i32 %control2 to i64
  %shl6 = shl nuw i64 1, %sh_prom5
  %sh_prom10 = zext nneg i32 %target to i64
  %shl11 = shl nuw nsw i64 1, %sh_prom10
  %wide.trip.count = zext nneg i32 %reg.4.val to i64
  %0 = freeze i64 %shl6
  %1 = or i64 %shl, %0
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i64, ptr %reg.24.val, i64 %indvars.iv
  %2 = load i64, ptr %arrayidx, align 8
  %3 = and i64 %2, %1
  %or.cond.not = icmp eq i64 %3, %1
  br i1 %or.cond.not, label %if.then9, label %for.inc

if.then9:
  %xor = xor i64 %2, %shl11
  store i64 %xor, ptr %arrayidx, align 8
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}
