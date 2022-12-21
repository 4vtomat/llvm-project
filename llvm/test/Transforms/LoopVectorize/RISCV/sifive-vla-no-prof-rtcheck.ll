; RUN:  sed 's/METADATA//' < %s | opt -passes=loop-vectorize -mtriple riscv64 -mattr=+v -S \
; RUN:  | FileCheck -check-prefix=CHECK-HAS-PROF-RTCHECK %s
; RUN:  sed 's/METADATA//' < %s | opt -passes=loop-vectorize -mtriple riscv64 -mattr=+v -S -force-vectorization \
; RUN:  | FileCheck -check-prefix=CHECK-NO-PROF-RTCHECK %s
; RUN:  sed 's/METADATA/, !llvm.loop !0/' < %s | opt -passes=loop-vectorize -mtriple riscv64 -mattr=+v -S \
; RUN:  | FileCheck -check-prefix=CHECK-NO-PROF-RTCHECK %s

define i32 @foo(i32 %n, ptr %a) {
; CHECK-HAS-PROF-RTCHECK-LABEL: @foo(
; CHECK-HAS-PROF-RTCHECK-NEXT:  entry:
; CHECK-HAS-PROF-RTCHECK-NEXT:    [[CMP4:%.*]] = icmp sgt i32 [[N:%.*]], 0
; CHECK-HAS-PROF-RTCHECK-NEXT:    br i1 [[CMP4]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_COND_CLEANUP:%.*]]
; CHECK-HAS-PROF-RTCHECK:       for.body.preheader:
; CHECK-HAS-PROF-RTCHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[N]] to i64
; CHECK-HAS-PROF-RTCHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[WIDE_TRIP_COUNT]], i64 2, i64 0)
; CHECK-HAS-PROF-RTCHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 3
; CHECK-HAS-PROF-RTCHECK-NEXT:    [[PROF_MIN_ITERS_CHECK:%.*]] = icmp ule i64 [[WIDE_TRIP_COUNT]], [[TMP1]]
; CHECK-HAS-PROF-RTCHECK-NEXT:    [[TMP2:%.*]] = or i1 false, [[PROF_MIN_ITERS_CHECK]]
; CHECK-HAS-PROF-RTCHECK-NEXT:    br i1 [[TMP2]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
;
; CHECK-NO-PROF-RTCHECK-LABEL: @foo(
; CHECK-NO-PROF-RTCHECK-NEXT:  entry:
; CHECK-NO-PROF-RTCHECK-NEXT:    [[CMP4:%.*]] = icmp sgt i32 [[N:%.*]], 0
; CHECK-NO-PROF-RTCHECK-NEXT:    br i1 [[CMP4]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_COND_CLEANUP:%.*]]
; CHECK-NO-PROF-RTCHECK:       for.body.preheader:
; CHECK-NO-PROF-RTCHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[N]] to i64
; CHECK-NO-PROF-RTCHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
;
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %sum.05 = phi i32 [ 0, %for.body.preheader ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %sum.05
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body METADATA
}
!0 = !{!0, !{!"llvm.loop.vectorize.enable", i1 true}}
