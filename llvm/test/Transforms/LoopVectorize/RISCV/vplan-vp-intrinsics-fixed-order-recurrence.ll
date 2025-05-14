; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

define void @first_order_recurrence(ptr noalias %A, ptr noalias %B, i64 %TC) {
; IF-EVL: VPlan 'Initial VPlan for VF={1},UF={1}'
;
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}' { 
; IF-EVL-NEXT: Live-in vp<[[VF:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%TC> = original trip-count
; IF-EVL-EMPTY:
; IF-EVL: ir-bb<entry>:
; IF-EVL-NEXT: Successor(s): vector.ph
; IF-EVL-EMPTY:
; IF-EVL: vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop
; IF-EVL-EMPTY:
; IF-EVL: <x1> vector loop: {
; IF-EVL-NEXT:   vector.body:
; IF-EVL-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%index.evl.next>
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%3> = phi ir<0>, vp<%index.evl.next>
; IF-EVL-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%4> = phi vp<%1>, vp<%5>
; IF-EVL-NEXT:     FIRST-ORDER-RECURRENCE-PHI ir<%for1> = phi ir<33>, ir<%0>
; IF-EVL-NEXT:     EMIT vp<%avl> = sub vp<%0>, vp<%3>
; IF-EVL-NEXT:     EMIT vp<%5> = EXPLICIT-VECTOR-LENGTH vp<%avl>
; IF-EVL-NEXT:     vp<%6> = SCALAR-STEPS vp<%3>, ir<1>
; IF-EVL-NEXT:     CLONE ir<%arrayidx> = getelementptr inbounds nuw ir<%A>, vp<%6>
; IF-EVL-NEXT:     vp<%7> = vector-pointer ir<%arrayidx>
; IF-EVL-NEXT:     WIDEN ir<%0> = vp.load vp<%7>, vp<%5>       unit-strided
; IF-EVL-NEXT:     EMIT vp<%8> = first-order splice ir<%for1>, ir<%0>
; IF-EVL-NEXT:     WIDEN ir<%add> = add nsw vp<%8>, ir<%0>
; IF-EVL-NEXT:     CLONE ir<%arrayidx2> = getelementptr inbounds nuw ir<%B>, vp<%6>
; IF-EVL-NEXT:     vp<%9> = vector-pointer ir<%arrayidx2>
; IF-EVL-NEXT:     WIDEN vp.store vp<%9>, ir<%add>, vp<%5>     unit-strided
; IF-EVL-NEXT:     SCALAR-CAST vp<%10> = zext vp<%5> to i64
; IF-EVL-NEXT:     EMIT vp<%index.evl.next> = add nuw vp<%10>, vp<%3>
; IF-EVL-NEXT:     EMIT branch-on-count vp<%index.evl.next>, vp<%0>
; IF-EVL-NEXT:   No successors
; IF-EVL-NEXT: }
; IF-EVL-NEXT: Successor(s): middle.block
; IF-EVL-EMPTY:
; IF-EVL: middle.block:
; IF-EVL-NEXT:   EMIT vp<%vector.recur.extract> = extract-from-end ir<%0>, ir<1>
; IF-EVL-NEXT:   EMIT branch-on-cond ir<true>
; IF-EVL-NEXT: Successor(s): ir-bb<for.end>, scalar.ph

entry:
  br label %for.body

for.body:
  %indvars = phi i64 [ 0, %entry ], [ %indvars.next, %for.body ]
  %for1 = phi i32 [ 33, %entry ], [ %0, %for.body ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %A, i64 %indvars
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %for1, %0
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %B, i64 %indvars
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.next = add nuw nsw i64 %indvars, 1
  %exitcond.not = icmp eq i64 %indvars.next, %TC
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
