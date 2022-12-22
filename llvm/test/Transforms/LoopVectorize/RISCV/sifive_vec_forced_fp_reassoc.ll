; RUN: opt -S -mtriple=riscv64 -mattr=+d,+v -riscv-v-vector-bits-min=512 -passes=loop-vectorize -debug-only=loop-vectorize %s -o - 2>&1 | FileCheck %s

;
; float reassociate_on_vec_forced(const int32_t n, float *a)
; {
;   float red = 0.0f;
;     #pragma clang loop vectorize(enable)
;     for (int32_t i = 0; i < n; ++i) {
;        #pragma clang fp reassociate(on)
;         red += a[i] * a[i];
;     }
;   return red;
; }
;
; CHECK: LV: Checking a loop in 'reassociate_on_vec_forced'
; CHECK: Executing best plan with
;
define dso_local float @reassociate_on_vec_forced(i32 noundef signext %n, ptr nocapture noundef readonly %a) local_unnamed_addr {
entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %.lcssa = phi float [ %1, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %red.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %.lcssa, %for.cond.cleanup.loopexit ]
  ret float %red.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %red.09 = phi float [ 0.000000e+00, %for.body.preheader ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = tail call reassoc float @llvm.fmuladd.f32(float %0, float %0, float %red.09)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !10
}

;
;float reassociate_off_vec_forced(const int32_t n, float *a)
;{
;  float red = 0.0f;
;    #pragma clang loop vectorize(enable)
;    for (int32_t i = 0; i < n; ++i) {
;       #pragma clang fp reassociate(off)
;        red += a[i] * a[i];
;    }
;  return red;
;}
;
; CHECK: LV: Checking a loop in 'reassociate_off_vec_forced'
; CHECK: LV: loop not vectorized: cannot prove it is safe to reorder floating-point operations
;
define dso_local float @reassociate_off_vec_forced(i32 noundef signext %n, ptr nocapture noundef readonly %a) local_unnamed_addr {
entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %red.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %1, %for.body ]
  ret float %red.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %red.09 = phi float [ 0.000000e+00, %for.body.preheader ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = tail call float @llvm.fmuladd.f32(float %0, float %0, float %red.09)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !13
}

declare float @llvm.fmuladd.f32(float, float, float) #1

!10 = distinct !{!10, !11, !12}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.vectorize.enable", i1 true}
!13 = distinct !{!13, !11, !12}
