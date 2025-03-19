; RUN: opt -S -riscv-use-vla-vectorizer -passes=loop-vectorize -mtriple=riscv64-unknown-linux-gnu -mattr=+v -sifive-uncountable-loop-vectorization=stress -debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=VPLANS
; REQUIRES: asserts

define ptr @find(ptr %first, ptr %last, ptr %value) {
; VPLANS-LABEL: Checking a loop in 'find'
; VPLANS: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
; VPLANS-NEXT: Live-in vp<[[VTC:%.+]]> = vector-trip-count
; VPLANS-NEXT: vp<[[TC:%.+]]> = original trip-count
; VPLANS-EMPTY:
; VPLANS-NEXT: ir-bb<for.body.lr.ph>:
; VPLANS-NEXT:   IR   %0 = load i32, ptr %value, align 4
; VPLANS-NEXT:   EMIT vp<[[TC]]> = EXPAND SCEV (1 + ((-4 + (-1 * (ptrtoint ptr %first to i64)) + (ptrtoint ptr %last to i64)) /u 4))<nuw><nsw>
; VPLANS-NEXT: Successor(s): vector.ph
; VPLANS-EMPTY:
; VPLANS-NEXT: vector.ph:
; VPLANS-NEXT:   vp<[[ENDV:%.+]]> = DERIVED-IV ir<%first> + vp<[[VTC]]> * ir<4>
; VPLANS-NEXT: Successor(s): vector loop
; VPLANS-EMPTY:
; VPLANS-NEXT: <x1> vector loop: {
; VPLANS-NEXT:   vector.body:
; VPLANS-NEXT:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>
; VPLANS-NEXT:     EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_IV:%.+]]> = phi ir<0>
; VPLANS-NEXT:     EMIT ir<%first.addr.07> = WIDEN-POINTER-INDUCTION ir<%first>, ir<4>
; VPLANS-NEXT:     EMIT vp<[[AVL:%.+]]> = sub vp<[[TC]]>, vp<[[EVL_IV]]>
; VPLANS-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<%avl>
; VPLANS-NEXT:     vp<[[VEC_PTR:%.+]]> = vector-pointer ir<%first.addr.07>
; VPLANS-NEXT:     WIDEN-SPECULATIVE-INSTRUCTION ir<[[DATA:%.+]]>, vp<[[EVL2:%.+]]> = vp.load vp<[[VEC_PTR]]>, vp<[[EVL]]>	unit-strided
; VPLANS-NEXT:     WIDEN ir<%cmp1> = icmp eq ir<[[DATA]]>, ir<%0>
; VPLANS-NEXT:     EMIT vp<[[CMP:%.+]]> = exiting-cond ir<%cmp1>
; VPLANS-NEXT:     EMIT branch-on-cond vp<[[CMP]]>
; VPLANS-NEXT:   Successor(s): vector.early.exit, for.inc
; VPLANS-EMPTY:
; VPLANS-NEXT:   vector.early.exit:
; VPLANS-NEXT:   No successors
; VPLANS-EMPTY:
; VPLANS-NEXT:   for.inc:
; VPLANS-NEXT:     SCALAR-CAST vp<[[EVL2_I64:%.+]]> = zext vp<[[EVL2]]> to i64
; VPLANS-NEXT:     EMIT vp<[[EVL_IV_NEXT:%.+]]> = add nuw vp<[[EVL2_I64]]>, vp<[[EVL_IV]]>
; VPLANS-NEXT:     EMIT branch-on-count vp<[[EVL_IV_NEXT]]>, vp<[[TC]]>
; VPLANS-NEXT:   No successors
; VPLANS-NEXT: }
; VPLANS-NEXT: Successor(s): middle.block
; VPLANS-EMPTY:
; VPLANS-NEXT: middle.block:
; VPLANS-NEXT:   EMIT branch-on-cond ir<true>
; VPLANS-NEXT: Successor(s): ir-bb<return.loopexit>, scalar.ph
; VPLANS-EMPTY:
; VPLANS-NEXT: scalar.ph:
; VPLANS-NEXT:   EMIT vp<%bc.resume.val> = resume-phi vp<[[ENDV]]>, ir<%first>
; VPLANS-NEXT: Successor(s): ir-bb<for.body>
; VPLANS-EMPTY:
; VPLANS-NEXT: ir-bb<for.body>:
; VPLANS-NEXT:   IR   %first.addr.07 = phi ptr [ %first, %for.body.lr.ph ], [ %incdec.ptr, %for.inc ] (extra operand: vp<%bc.resume.val> from scalar.ph)
; VPLANS-NEXT:   IR   %1 = load i32, ptr %first.addr.07, align 4
; VPLANS-NEXT:   IR   %cmp1 = icmp eq i32 %1, %0
; VPLANS-NEXT: No successors
; VPLANS-EMPTY:
; VPLANS-NEXT: ir-bb<return.loopexit>:
; VPLANS-NEXT:   IR   %retval.0.ph = phi ptr [ %first.addr.07, %for.body ], [ %last, %for.inc ] (extra operand: ir<%last> from middle.block)
; VPLANS-NEXT: No successors
; VPLAN-NEXT: }
entry:
  %cmp.not6 = icmp eq ptr %first, %last
  br i1 %cmp.not6, label %return, label %for.body.lr.ph

for.body.lr.ph:
  %0 = load i32, ptr %value, align 4
  br label %for.body

for.body:
  %first.addr.07 = phi ptr [ %first, %for.body.lr.ph ], [ %incdec.ptr, %for.inc ]
  %1 = load i32, ptr %first.addr.07, align 4
  %cmp1 = icmp eq i32 %1, %0
  br i1 %cmp1, label %return.loopexit, label %for.inc

for.inc:
  %incdec.ptr = getelementptr inbounds i32, ptr %first.addr.07, i64 1
  %cmp.not = icmp eq ptr %incdec.ptr, %last
  br i1 %cmp.not, label %return.loopexit, label %for.body

return.loopexit:
  %retval.0.ph = phi ptr [ %first.addr.07, %for.body ], [ %last, %for.inc ]
  br label %return

return:
  %retval.0 = phi ptr [ %first, %entry ], [ %retval.0.ph, %return.loopexit ]
  ret ptr %retval.0
}
