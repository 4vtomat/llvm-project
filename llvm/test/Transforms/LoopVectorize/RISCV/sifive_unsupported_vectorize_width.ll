; RUN: opt -passes=loop-vectorize -mcpu=sifive-x280  -mtriple riscv64 %s -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s

; Make sure that given vectorize_width is ignored by the VPlan
;
; CHECK: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {

define void @_Z69benchForTruncOrZextVecWithAddInLoopWithVW16From_uint32_t_To_uint64_t_RN9benchmark5StateE(ptr %0) personality ptr null {
entry:
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i1, %entry
  br label %for.body.i1

for.body.i1:                                      ; preds = %for.body.i1, %for.body.i
  %indvars.iv.i3 = phi i64 [ 0, %for.body.i ], [ %indvars.iv.next.i, %for.body.i1 ]
  %arrayidx2.i = getelementptr i64, ptr %0, i64 %indvars.iv.i3
  store i64 1, ptr %arrayidx2.i, align 8
  %indvars.iv.next.i = add i64 %indvars.iv.i3, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.i3, 10000
  br i1 %exitcond.not.i, label %for.body.i, label %for.body.i1, !llvm.loop !0
}

; uselistorder directives
uselistorder ptr null, { 1, 2, 0 }

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 16}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
!4 = !{!"llvm.loop.interleave.count", i32 4}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}
