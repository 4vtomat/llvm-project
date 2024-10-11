; RUN: opt -passes=loop-vectorize -mcpu=sifive-x280 -disable-output -debug-only=vplan-cost-model -mtriple riscv64 %s 2>&1 | FileCheck --check-prefix X280 %s
; RUN: opt -passes=loop-vectorize -mcpu=sifive-x390 -disable-output -debug-only=vplan-cost-model -mtriple riscv64 %s 2>&1 | FileCheck --check-prefix X390 %s
; RUN: opt -passes=loop-vectorize -mcpu=sifive-p470 -disable-output -debug-only=vplan-cost-model -mtriple riscv64 %s 2>&1 | FileCheck --check-prefix P470 %s
; RUN: opt -passes=loop-vectorize -mcpu=sifive-p670 -disable-output -debug-only=vplan-cost-model -mtriple riscv64 %s 2>&1 | FileCheck --check-prefix P670 %s

; REQUIRES: asserts

; X280: VPlanCM: cost 256 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X280: VPlanCM: cost 256 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X280: VPlanCM: cost 256 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X280: VPlanCM: cost 512 for RVL (m2, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X280: VPlanCM: cost 1024 for RVL (m4, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X280: VPlanCM: cost 2048 for RVL (m8, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>

; X390: VPlanCM: cost 512 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X390: VPlanCM: cost 512 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X390: VPlanCM: cost 512 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X390: VPlanCM: cost 1024 for RVL (m2, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X390: VPlanCM: cost 2048 for RVL (m4, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; X390: VPlanCM: cost 4096 for RVL (m8, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>

; P470: VPlanCM: cost 64 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P470: VPlanCM: cost 64 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P470: VPlanCM: cost 64 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P470: VPlanCM: cost 128 for RVL (m2, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P470: VPlanCM: cost 256 for RVL (m4, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P470: VPlanCM: Spill and Reload of 4 registers is required. Cost increased by 64
; P470: VPlanCM: cost 512 for RVL (m8, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>

; P670: VPlanCM: cost 32 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P670: VPlanCM: cost 32 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P670: VPlanCM: cost 32 for RVL (m1, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P670: VPlanCM: cost 64 for RVL (m2, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P670: VPlanCM: cost 128 for RVL (m4, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>
; P670: VPlanCM: Spill and Reload of 4 registers is required. Cost increased by 64
; P670: VPlanCM: cost 256 for RVL (m8, i16) for VPInstruction: WIDEN ir<%div.us> = vp.udiv ir<%add.us>, ir<255>, vp<%5>

define void @test(i32 %width, i32 %channels, ptr noalias %other_row, ptr noalias %this_row) {
entry:
  %cmp28 = icmp sgt i32 %width, 0
  br i1 %cmp28, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %arrayidx = getelementptr inbounds i8, ptr %other_row, i64 3
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i16
  %sub = xor i16 %conv, 255
  %cmp226 = icmp sgt i32 %channels, 0
  br i1 %cmp226, label %for.body.us.preheader, label %for.cond.cleanup

for.body.us.preheader:                            ; preds = %for.body.lr.ph
  %wide.trip.count = zext nneg i32 %channels to i64
  br label %for.body.us

for.body.us:                                      ; preds = %for.body.us.preheader, %for.cond1.for.cond.cleanup4_crit_edge.us
  %x.029.us = phi i32 [ %inc16.us, %for.cond1.for.cond.cleanup4_crit_edge.us ], [ 0, %for.body.us.preheader ]
  br label %for.body5.us

for.body5.us:                                     ; preds = %for.body.us, %for.body5.us
  %indvars.iv = phi i64 [ 0, %for.body.us ], [ %indvars.iv.next, %for.body5.us ]
  %arrayidx6.us = getelementptr inbounds i8, ptr %other_row, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx6.us, align 1
  %conv7.us = zext i8 %1 to i16
  %mul.us = mul nuw i16 %conv7.us, %conv
  %arrayidx9.us = getelementptr inbounds i8, ptr %this_row, i64 %indvars.iv
  %2 = load i8, ptr %arrayidx9.us, align 1
  %conv10.us = zext i8 %2 to i16
  %mul11.us = mul nuw i16 %sub, %conv10.us
  %add.us = add i16 %mul11.us, %mul.us
  %div.us = udiv i16 %add.us, 255
  %conv12.us = trunc i16 %div.us to i8
  store i8 %conv12.us, ptr %arrayidx9.us, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond1.for.cond.cleanup4_crit_edge.us, label %for.body5.us

for.cond1.for.cond.cleanup4_crit_edge.us:         ; preds = %for.body5.us
  %inc16.us = add nuw nsw i32 %x.029.us, 1
  %exitcond32.not = icmp eq i32 %inc16.us, %width
  br i1 %exitcond32.not, label %for.cond.cleanup.loopexit, label %for.body.us

for.cond.cleanup.loopexit:                        ; preds = %for.cond1.for.cond.cleanup4_crit_edge.us
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %for.body.lr.ph, %entry
  ret void
}
