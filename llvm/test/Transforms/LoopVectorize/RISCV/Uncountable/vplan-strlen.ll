; RUN: opt -S -riscv-use-vla-vectorizer -passes=loop-vectorize -mtriple=riscv64-unknown-linux-gnu -mattr=+v -sifive-uncountable-loop-vectorization=stress -debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=VPLANS
; REQUIRES: asserts

define i64 @strlen_i8(ptr %start) {
; VPLANS-LABEL: Checking a loop in 'strlen_i8'
; VPLANS: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4,vscale x 8},UF={1}' {
; VPLANS-EMPTY:
; VPLANS-NEXT: vector.ph:
; VPLANS-NEXT: Successor(s): vector loop
; VPLANS-EMPTY:
; VPLANS-NEXT: <x1> vector loop: {
; VPLANS-NEXT:   vector.body:
; VPLANS-NEXT:     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%11>
; VPLANS-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<%3> = phi ir<0>, vp<%11>
; VPLANS-NEXT:     EMIT vp<%4> = EXPLICIT-VECTOR-LENGTH
; VPLANS-NEXT:     vp<%5> = SCALAR-STEPS vp<%3>, ir<1>
; VPLANS-NEXT:     EMIT vp<%6> = ptradd ir<%start>, vp<%5>
; VPLANS-NEXT:     vp<%7> = vector-pointer vp<%6>
; VPLANS-NEXT:     WIDEN-SPECULATIVE-INSTRUCTION ir<%0>, vp<%8> = vp.load vp<%7>, vp<%4>	unit-strided
; VPLANS-NEXT:     WIDEN ir<%cmp.not> = icmp eq ir<%0>, ir<0>
; VPLANS-NEXT:     EMIT vp<%9> = exiting-cond ir<%cmp.not>
; VPLANS-NEXT:     SCALAR-CAST vp<%10> = zext vp<%8> to i64
; VPLANS-NEXT:     EMIT vp<%11> = add nuw vp<%10>, vp<%3>
; VPLANS-NEXT:     EMIT branch-on-cond vp<%9>
; VPLANS-NEXT:   No successors
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
