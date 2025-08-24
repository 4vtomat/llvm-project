; RUN: opt -S -riscv-use-vla-vectorizer -passes=loop-vectorize -mtriple=riscv64-unknown-linux-gnu -mattr=+v -sifive-uncountable-loop-vectorization=stress -debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=VPLANS
; REQUIRES: asserts

define i64 @strlen_i8(ptr %start) {
; VPLANS-LABEL: Checking a loop in 'strlen_i8'
; VPLANS: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4,vscale x 8},UF={1}' {
; VPLANS-NEXT: Live-in vp<[[VF:%.+]]> = VF
; VPLANS-EMPTY:
; VPLANS-NEXT: ir-bb<entry>:
; VPLANS-NEXT: Successor(s): scalar.ph, vector.ph
; VPLANS-EMPTY:
; VPLANS-NEXT: vector.ph:
; VPLANS-NEXT: Successor(s): vector loop
; VPLANS-EMPTY:
; VPLANS-NEXT: <x1> vector loop: {
; VPLANS-NEXT:   vector.body:
; VPLANS-NEXT:     EMIT vp<[[IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<[[IV_NEXT:%.+]]>
; VPLANS-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVLPHI:%.+]]> = phi ir<0>, vp<%index.evl.next>
; VPLANS-NEXT:     EMIT vp<[[VL:%.+]]> = EXPLICIT-VECTOR-LENGTH
; VPLANS-NEXT:     vp<[[DERIV:%.+]]> = DERIVED-IV ir<0> + vp<[[EVLPHI]]> * ir<1>
; VPLANS-NEXT:     vp<[[SC:%.+]]> = SCALAR-STEPS vp<[[DERIV]]>, ir<1>
; VPLANS-NEXT:     EMIT vp<%next.gep> = ptradd ir<%start>, vp<[[SC]]>
; VPLANS-NEXT:     vp<[[PTR:%.+]]> = vector-pointer vp<%next.gep>
; VPLANS-NEXT:     WIDEN-SPECULATIVE-INSTRUCTION ir<%0>, vp<[[NEWEVL:%.+]]> = vp.load vp<[[PTR]]>, vp<[[VL]]>	unit-strided
; VPLANS-NEXT:     WIDEN ir<%cmp.not> = icmp eq ir<%0>, ir<0>
; VPLANS-NEXT:     EMIT vp<[[VPFIRST:%.+]]> = vp-first ir<%cmp.not>, vp<[[NEWEVL]]>
; VPLANS-NEXT:     EMIT vp<[[EXITCOND:%.+]]> = icmp sge vp<[[VPFIRST]]>, ir<0>
; VPLANS-NEXT:     EMIT vp<[[CAST:%.+]]> = zext vp<[[NEWEVL]]> to i64
; VPLANS-NEXT:     EMIT vp<[[IV_NEXT]]> = add nuw vp<[[CAST]]>, vp<[[EVLPHI]]>
; VPLANS-NEXT:     EMIT branch-on-cond vp<[[EXITCOND]]>
; VPLANS-NEXT:   No successors
; VPLANS-NEXT: }
; VPLANS-NEXT: Successor(s): middle.block
; VPLANS-EMPTY:
; VPLANS-NEXT: middle.block:
; VPLANS-NEXT:   EMIT vp<[[VPFIRST64:%.+]]> = zext vp<[[VPFIRST]]> to i64
; VPLANS-NEXT:   EMIT vp<[[TC:%.+]]> = add vp<[[EVLPHI]]>, vp<[[VPFIRST64]]>
; VPLANS-NEXT:   vp<[[DERIV2:%.+]]> = DERIVED-IV ir<%start> + vp<[[TC]]> * ir<1>
; VPLANS-NEXT:   EMIT branch-on-cond ir<true>
; VPLANS-NEXT: Successor(s): ir-bb<for.end>, scalar.ph
; VPLANS-EMPTY:
; VPLANS-NEXT: ir-bb<for.end>:
; VPLANS-NEXT:   IR   %end.0.lcssa = phi ptr [ %end.0, %for.cond ] (extra operand: vp<[[DERIV2]]> from middle.block)
; VPLANS-NEXT:   IR   %sub.ptr.lhs.cast = ptrtoint ptr %end.0.lcssa to i64
; VPLANS-NEXT:   IR   %sub.ptr.rhs.cast = ptrtoint ptr %start to i64
; VPLANS-NEXT:   IR   %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
; VPLANS-NEXT: No successors
; VPLANS-EMPTY:
; VPLANS-NEXT: scalar.ph:
; VPLANS-NEXT: Successor(s): ir-bb<for.cond>
; VPLANS-EMPTY:
; VPLANS-NEXT: ir-bb<for.cond>:
; VPLANS-NEXT:   IR   %end.0 = phi ptr [ %start, %entry ], [ %incdec.ptr, %for.cond ]
; VPLANS-NEXT:   IR   %0 = load i8, ptr %end.0, align 1
; VPLANS-NEXT:   IR   %cmp.not = icmp eq i8 %0, 0
; VPLANS-NEXT:   IR   %incdec.ptr = getelementptr inbounds i8, ptr %end.0, i64 1
; VPLANS-NEXT: No successors
; VPLANS-NEXT: }
entry:
  br label %for.cond

for.cond:
  %end.0 = phi ptr [ %start, %entry ], [ %incdec.ptr, %for.cond ]
  %0 = load i8, ptr %end.0, align 1
  %cmp.not = icmp eq i8 %0, 0
  %incdec.ptr = getelementptr inbounds i8, ptr %end.0, i64 1
  br i1 %cmp.not, label %for.end, label %for.cond

for.end:
  %end.0.lcssa = phi ptr [ %end.0, %for.cond ]
  %sub.ptr.lhs.cast = ptrtoint ptr %end.0.lcssa to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %start to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  ret i64 %sub.ptr.sub
}
