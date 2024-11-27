; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-vector-width=4 -scalar-evolution-use-expensive-range-sharpening \
; RUN: -disable-output < %s 2>&1 | FileCheck %s

define i64 @findlastiv(ptr %a, ptr %b, i64 %ii, i64 %n) {
; CHECK: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%[0-9]+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<[[OTC:%n]]> = original trip-count
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:    WIDEN-INDUCTION [[IV_PHI:%.+]] = phi
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<[[RDX_PHI:%.+]]> = phi ir<%ii>, ir<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:    vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[IV]]>, ir<1>
; CHECK-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; CHECK-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; CHECK-NEXT:    WIDEN ir<[[LD1:%.+]]> = load vp<[[PTR1]]>
; CHECK-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; CHECK-NEXT:    vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; CHECK-NEXT:    WIDEN ir<[[LD2:%.+]]> = load vp<[[PTR2]]>
; CHECK-NEXT:    WIDEN ir<[[CMP:%.+]]> = icmp sgt ir<[[LD1]]>, ir<[[LD2]]>
; CHECK-NEXT:    WIDEN-SELECT ir<[[SELECT:%.+]]> = select ir<[[CMP]]>, ir<[[IV_PHI]]>, ir<[[RDX_PHI]]>
; CHECK-NEXT:    EMIT vp<[[IV_NEXT:%.+]]> = add nuw vp<[[IV]]>, vp<[[VFUF]]>
; CHECK-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RDX:%.+]]> = compute-reduction-result ir<[[RDX_PHI]]>, ir<[[SELECT]]>
; CHECK-NEXT:   EMIT vp<[[EXT:%[0-9]+]]> = extract-from-end vp<[[RDX]]>, ir<1>
; CHECK-NEXT:   EMIT vp<[[EXIT_COND:%.+]]> = icmp eq ir<[[OTC]]>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[EXIT_COND]]>
; CHECK-NEXT: Successor(s): ir-bb<exit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT: IR   %cond.lcssa = phi i64 [ %cond, %for.body ] (extra operand: vp<[[EXT]]>)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %ii, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @findlastiv_need_mask(ptr %a, ptr %b, i64 %ii, i64 %iv_start, i64 %n) {
; CHECK: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%[0-9]+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; CHECK-NEXT: vp<[[OTC:%[0-9]+]]> = original trip-count
; CHECK-EMPTY:
; CHECK:      ir-bb<for.body.preheader>:
; CHECK-NEXT:   EMIT vp<[[OTC]]> = EXPAND SCEV ((-1 * %iv_start) + %n)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:    WIDEN-INDUCTION [[IV_PHI:%.+]] = phi
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<[[RDX_PHI:%.+]]> = phi ir<%ii>, ir<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:    vp<[[DIV:%[0-9]+]]>    = DERIVED-IV ir<%iv_start> + vp<[[IV]]> * ir<1>
; CHECK-NEXT:    vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[DIV]]>, ir<1>
; CHECK-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; CHECK-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; CHECK-NEXT:    WIDEN ir<[[LD1:%.+]]> = load vp<[[PTR1]]>
; CHECK-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; CHECK-NEXT:    vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; CHECK-NEXT:    WIDEN ir<[[LD2:%.+]]> = load vp<[[PTR2]]>
; CHECK-NEXT:    WIDEN ir<[[CMP:%.+]]> = icmp sgt ir<[[LD1]]>, ir<[[LD2]]>
; CHECK-NEXT:    WIDEN-SELECT ir<[[SELECT:%.+]]> = select ir<[[CMP]]>, ir<[[IV_PHI]]>, ir<[[RDX_PHI]]>
; CHECK-NEXT:    EMIT vp<[[IV_NEXT:%.+]]> = add nuw vp<[[IV]]>, vp<[[VFUF]]>
; CHECK-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RDX_MASK:%.+]]> = icmp ne ir<[[SELECT]]>, ir<9223372036854775807>
; CHECK-NEXT:   EMIT vp<[[RDX:%.+]]> = compute-reduction-result-with-mask ir<[[RDX_PHI]]>, ir<[[SELECT]]>, vp<[[RDX_MASK]]>
; CHECK-NEXT:   EMIT vp<[[EXT:%[0-9]+]]> = extract-from-end vp<[[RDX]]>, ir<1>
; CHECK-NEXT:   EMIT vp<[[EXIT_COND:%.+]]> = icmp eq vp<[[OTC]]>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[EXIT_COND]]>
; CHECK-NEXT: Successor(s): ir-bb<exit.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit.loopexit>:
; CHECK-NEXT:   IR   %cond.lcssa1 = phi i64 [ %cond, %for.body ] (extra operand: vp<[[EXT]]>)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  %guard = icmp slt i64 %iv_start, %n
  br i1 %guard, label %for.body, label %exit

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ %iv_start, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %ii, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  %cond.lcssa = phi i64 [ %ii, %entry ], [ %cond, %for.body ]
  ret i64 %cond.lcssa
}

define i64 @findlastiv_need_mask_with_intermediate_store(ptr %a, ptr %b, i64 %ii, i64 %iv_start, i64 %n, ptr %dst) {
; CHECK: VPlan 'Initial VPlan for VF={4},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VF:%[0-9]+]]> = VF
; CHECK-NEXT: Live-in vp<[[VFUF:%[0-9]+]]> = VF * UF
; CHECK-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; CHECK-NEXT: vp<[[OTC:%[0-9]+]]> = original trip-count
; CHECK-EMPTY:
; CHECK:      ir-bb<for.body.preheader>:
; CHECK-NEXT:   EMIT vp<[[OTC]]> = EXPAND SCEV ((-1 * %iv_start) + %n)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:    WIDEN-INDUCTION [[IV_PHI:%.+]] = phi
; CHECK-NEXT:    WIDEN-REDUCTION-PHI ir<[[RDX_PHI:%.+]]> = phi ir<%ii>, ir<[[RDX_NEXT:%.+]]>
; CHECK-NEXT:    vp<[[DIV:%[0-9]+]]>    = DERIVED-IV ir<%iv_start> + vp<[[IV]]> * ir<1>
; CHECK-NEXT:    vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[DIV]]>, ir<1>
; CHECK-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; CHECK-NEXT:    vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
; CHECK-NEXT:    WIDEN ir<[[LD1:%.+]]> = load vp<[[PTR1]]>
; CHECK-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; CHECK-NEXT:    vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
; CHECK-NEXT:    WIDEN ir<[[LD2:%.+]]> = load vp<[[PTR2]]>
; CHECK-NEXT:    WIDEN ir<[[CMP:%.+]]> = icmp sgt ir<[[LD1]]>, ir<[[LD2]]>
; CHECK-NEXT:    WIDEN-SELECT ir<[[SELECT:%.+]]> = select ir<[[CMP]]>, ir<[[IV_PHI]]>, ir<[[RDX_PHI]]>
; CHECK-NEXT:    EMIT vp<[[IV_NEXT:%.+]]> = add nuw vp<[[IV]]>, vp<[[VFUF]]>
; CHECK-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT: }
; CHECK-NEXT: Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT: middle.block:
; CHECK-NEXT:   EMIT vp<[[RDX_MASK:%.+]]> = icmp ne ir<[[SELECT]]>, ir<9223372036854775807>
; CHECK-NEXT:   EMIT vp<[[RDX:%.+]]> = compute-reduction-result-with-mask ir<[[RDX_PHI]]>, ir<[[SELECT]]>, vp<[[RDX_MASK]]>
; CHECK-NEXT:   EMIT vp<[[EXT:%[0-9]+]]> = extract-from-end vp<[[RDX]]>, ir<1>
; CHECK-NEXT:   CLONE store vp<[[RDX]]>, ir<%dst>
; CHECK-NEXT:   EMIT vp<[[EXIT_COND:%.+]]> = icmp eq vp<[[OTC]]>, vp<[[VTC]]>
; CHECK-NEXT:   EMIT branch-on-cond vp<[[EXIT_COND]]>
; CHECK-NEXT: Successor(s): ir-bb<exit.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit.loopexit>:
; CHECK-NEXT:   IR   %cond.lcssa1 = phi i64 [ %cond, %for.body ] (extra operand: vp<[[EXT]]>)
; CHECK-NEXT: No successors
; CHECK-EMPTY:
; CHECK-NEXT: scalar.ph:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
;
entry:
  %guard = icmp slt i64 %iv_start, %n
  br i1 %guard, label %for.body, label %exit

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ %iv_start, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %ii, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  store i64 %cond, ptr %dst, align 8
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  %cond.lcssa = phi i64 [ %ii, %entry ], [ %cond, %for.body ]
  ret i64 %cond.lcssa
}
