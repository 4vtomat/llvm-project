; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 -mcpu=sifive-x280 -debug-only=vplan-cost-model,vplan -disable-output %s 2>&1 | FileCheck %s --check-prefix=CHECK-X280
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 -mcpu=sifive-p470 -debug-only=vplan-cost-model,vplan -disable-output %s 2>&1 | FileCheck %s --check-prefix=CHECK-P470
; RUN: opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 -mcpu=sifive-p670 -debug-only=vplan-cost-model,vplan -disable-output %s 2>&1 | FileCheck %s --check-prefix=CHECK-P670

; CHECK-X280: VPlanCM: cost 6 for RVL (mf2, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-X280: VPlanCM: cost 47 for RVL (mf2, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-X280: LV: Found an estimated overhead of 0 for VF vscale x 1 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-X280: VPlanCM: cost 12 for RVL (m1, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-X280: VPlanCM: cost 48 for RVL (m1, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-X280: LV: Found an estimated overhead of 0 for VF vscale x 2 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-X280: VPlanCM: cost 24 for RVL (m2, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-X280: VPlanCM: cost 50 for RVL (m2, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-X280: LV: Found an estimated overhead of 0 for VF vscale x 4 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-X280: VPlanCM: cost 48 for RVL (m4, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-X280: VPlanCM: cost 54 for RVL (m4, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-X280: LV: Found an estimated overhead of 0 for VF vscale x 8 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-X280: VPlanCM: cost 96 for RVL (m8, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-X280: VPlanCM: cost 62 for RVL (m8, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-X280: LV: Found an estimated overhead of 0 for VF vscale x 16 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>


; CHECK-P470: VPlanCM: cost 4 for RVL (mf2, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P470: VPlanCM: cost 16 for RVL (mf2, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P470: LV: Found an estimated overhead of 0 for VF vscale x 1 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-P470: VPlanCM: cost 4 for RVL (m1, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P470: VPlanCM: cost 18 for RVL (m1, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P470: LV: Found an estimated overhead of 0 for VF vscale x 2 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-P470: VPlanCM: cost 8 for RVL (m2, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P470: VPlanCM: cost 22 for RVL (m2, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P470: LV: Found an estimated overhead of 0 for VF vscale x 4 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-P470: VPlanCM: cost 16 for RVL (m4, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P470: VPlanCM: cost 30 for RVL (m4, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P470: LV: Found an estimated overhead of 0 for VF vscale x 8 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-P470: VPlanCM: cost 32 for RVL (m8, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P470: VPlanCM: cost 46 for RVL (m8, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P470: LV: Found an estimated overhead of 0 for VF vscale x 16 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>


; CHECK-P670: VPlanCM: cost 6 for RVL (mf2, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P670: VPlanCM: cost 16 for RVL (mf2, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P670: LV: Found an estimated overhead of 0 for VF vscale x 1 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-P670: VPlanCM: cost 6 for RVL (m1, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P670: VPlanCM: cost 18 for RVL (m1, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P670: LV: Found an estimated overhead of 0 for VF vscale x 2 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-P670: VPlanCM: cost 12 for RVL (m2, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P670: VPlanCM: cost 22 for RVL (m2, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P670: LV: Found an estimated overhead of 0 for VF vscale x 4 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-P670: VPlanCM: cost 24 for RVL (m4, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P670: VPlanCM: cost 30 for RVL (m4, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P670: LV: Found an estimated overhead of 0 for VF vscale x 8 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

; CHECK-P670: VPlanCM: cost 48 for RVL (m8, float) for VPInstruction: EMIT vp<%12> = fmul ir<%sub>, ir<%sub>
; CHECK-P670: VPlanCM: cost 46 for RVL (m8, float) for VPInstruction: REDUCE ir<%2> = ir<%weighted_MSE.07> + reduce.fadd (vp<%12>, vp<%8>)
; CHECK-P670: LV: Found an estimated overhead of 0 for VF vscale x 16 For recipe: WIDEN-REDUCTION-PHI ir<%weighted_MSE.07> = phi ir<0.000000e+00>, ir<%2>

define float @fmuladd(i32 %profile_size, ptr %all_shifted_test_db, i64 %current_shift, ptr %template_copy) {
entry:
  %add.ptr = getelementptr inbounds float, ptr %all_shifted_test_db, i64 %current_shift
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %fptr.08 = phi ptr [ %incdec.ptr, %for.body ], [ %add.ptr, %entry ]
  %weighted_MSE.07 = phi float [ %2, %for.body ], [ 0.000000e+00, %entry ]
  %fptr2.06 = phi ptr [ %incdec.ptr1, %for.body ], [ %template_copy, %entry ]
  %incdec.ptr = getelementptr inbounds float, ptr %fptr.08, i64 1
  %0 = load float, ptr %fptr.08, align 4
  %incdec.ptr1 = getelementptr inbounds float, ptr %fptr2.06, i64 1
  %1 = load float, ptr %fptr2.06, align 4
  %sub = fsub float %0, %1
  %2 = tail call float @llvm.fmuladd.f32(float %sub, float %sub, float %weighted_MSE.07)
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %profile_size
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret float %2
}

declare float @llvm.fmuladd.f32(float, float, float)
