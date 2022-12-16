; RUN: opt -passes=loop-vectorize -S -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-linux-gnu"

; CHECK: remark: <unknown>:0:0: loop remark: ignoring invalid vectorize_width value
; Function Attrs: nofree norecurse nosync nounwind readonly
define dso_local float @dot_ref(float* nocapture noundef readonly %x, float* nocapture noundef readonly %y, i64 noundef %N) local_unnamed_addr #0 align 8 {
entry:
  %cmp8.not = icmp eq i64 %N, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add.lcssa = phi float [ %add, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %s.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret float %s.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.010 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %s.09 = phi float [ %add, %for.body ], [ 0.000000e+00, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %i.010
  %0 = load float, float* %arrayidx, align 4, !tbaa !4
  %arrayidx1 = getelementptr inbounds float, float* %y, i64 %i.010
  %1 = load float, float* %arrayidx1, align 4, !tbaa !4
  %mul = fmul float %0, %1
  %add = fadd float %s.09, %mul
  %inc = add nuw i64 %i.010, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !8
}

attributes #0 = { nofree norecurse nosync nounwind readonly "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sifive-x280" "target-features"="+64bit,+a,+c,+d,+experimental-zvfh,+f,+m,+relax,+v,+xsfvfhbfmin,+xsfvqmaccqoq,+zba,+zbb,+zfh,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl256b,+zvl32b,+zvl512b,+zvl64b,-save-restore" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64d"}
!2 = !{i32 1, !"SmallDataLimit", i32 8}
!3 = !{!"clang version 15.0.0 (git@github.com:sifive/riscv-llvm-internal.git b047e055b0cc629c65a60ee290d927d8a7c06e8d)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9, !10, !11, !12}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.vectorize.width", i32 16384}
!11 = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
!12 = !{!"llvm.loop.vectorize.enable", i1 true}
