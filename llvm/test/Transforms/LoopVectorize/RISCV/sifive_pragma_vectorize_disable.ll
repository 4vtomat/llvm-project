; RUN: opt -S -passes=loop-vectorize -mtriple riscv64-unknown-linux-gnu -prefer-predicate-over-epilogue=predicate-dont-vectorize -scalable-vectorization=only -riscv-use-vla-vectorizer -riscv-v-vector-bits-min=-1 -mattr="+64bit,+a,+c,+d,+experimental-zvfh,+f,+m,+relax,+v,+xsfvfhbfmin,+xsfvqmaccqoq,+zba,+zbb,+zfh,+zicsr,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl256b,+zvl32b,+zvl512b,+zvl64b,-save-restore" %s -vector-primary-lmul-max=1 -pass-remarks-missed='loop-vectorize' 2>&1 | FileCheck %s

; CHECK: loop not vectorized: vectorization is explicitly disabled

define dso_local signext i32 @foo(i32 noundef signext %n, ptr noalias nocapture noundef readonly %a) local_unnamed_addr {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %red.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %red.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %red.05 = phi i32 [ 0, %for.body.preheader ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4, !tbaa !4
  %add = add nsw i32 %0, %red.05
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !8
}

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64d"}
!2 = !{i32 1, !"SmallDataLimit", i32 8}
!3 = !{!"clang version 14.9.0 (git@github.com:sifive/riscv-llvm-internal.git fb26e18a44833d11ca823b36d9344d2b5f35d80c)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.vectorize.enable", i1 false}
