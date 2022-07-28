; RUN: opt -S -mtriple=riscv64-unknown-elf -march=rv64gcv1p0 -mattr=+v,+f \
; RUN: -force-vector-width=16 -use-vla-vectorizer -passes=loop-vectorize \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize \
; RUN: -pass-remarks-analysis=loop-vectorize < %s  2>%t 
; RUN: cat %t | FileCheck %s -check-prefix=CHECK-REMARK

; CHECK-REMARK: Ignoring UserVF=16 because VLA was enabled.
define dso_local void @axpy_ref(double noundef %0, ptr nocapture noundef readonly %1, ptr nocapture noundef %2, i32 noundef signext %3) local_unnamed_addr {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %19

6:                                                ; preds = %4
  %7 = zext i32 %3 to i64
  br label %8

8:                                                ; preds = %6, %8
  %9 = phi i64 [ 0, %6 ], [ %16, %8 ]
  %10 = getelementptr inbounds double, ptr %1, i64 %9
  %11 = load double, ptr %10, align 8, !tbaa !4
  %12 = fmul fast double %11, %0
  %13 = getelementptr inbounds double, ptr %2, i64 %9
  %14 = load double, ptr %13, align 8, !tbaa !4
  %15 = fadd fast double %14, %12
  store double %15, ptr %13, align 8, !tbaa !4
  %16 = add nuw nsw i64 %9, 1
  %17 = icmp eq i64 %16, %7
  br i1 %17, label %18, label %8, !llvm.loop !8

18:                                               ; preds = %8
  br label %19

19:                                               ; preds = %18, %4
  ret void
}

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64d"}
!2 = !{i32 1, !"SmallDataLimit", i32 8}
!3 = !{!"clang version 14.9.0 (git@github.com:sifive/riscv-llvm-internal.git e96983a7e723ab2754bbb421b91307ca5048d67e)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"double", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
