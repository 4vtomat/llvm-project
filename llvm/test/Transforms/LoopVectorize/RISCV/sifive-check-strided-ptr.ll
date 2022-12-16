; RUN: opt -passes=loop-vectorize -S -mtriple riscv64-unknown-linux-gnu -mattr=+v -vectorizer-use-vp-strided-load-store < %s | FileCheck %s

@e = global [400 x i32] zeroinitializer
@board = global [400 x i8] zeroinitializer

define void @foo(i32 %n) {
; CHECK-LABEL: foo(
; CHECK-NOT: vp.strided.load.nxv2i8
entry:
  %cmp19 = icmp sgt i32 %n, 0
  br i1 %cmp19, label %for.cond1.preheader.preheader, label %for.cond.cleanup

for.cond1.preheader.preheader:                    ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.cond.cleanup3
  %indvars.iv22 = phi i64 [ 0, %for.cond1.preheader.preheader ], [ %indvars.iv.next23, %for.cond.cleanup3 ]
  %arrayidx6 = getelementptr inbounds [400 x i8], ptr @board, i64 0, i64 %indvars.iv22
  br label %for.body4

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  ret void

for.cond.cleanup3:                                ; preds = %for.inc
  %indvars.iv.next23 = add nuw nsw i64 %indvars.iv22, 1
  %exitcond25.not = icmp eq i64 %indvars.iv.next23, %wide.trip.count
  br i1 %exitcond25.not, label %for.cond.cleanup, label %for.cond1.preheader

for.body4:                                        ; preds = %for.cond1.preheader, %for.inc
  %indvars.iv = phi i64 [ 21, %for.cond1.preheader ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds [400 x i32], ptr @e, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %land.lhs.true

land.lhs.true:                                    ; preds = %for.body4
  %1 = load i8, ptr %arrayidx6, align 1
  %tobool7.not = icmp eq i8 %1, 0
  br i1 %tobool7.not, label %for.inc, label %if.then

if.then:                                          ; preds = %land.lhs.true
  store i32 2, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body4, %land.lhs.true, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 400
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4
}
