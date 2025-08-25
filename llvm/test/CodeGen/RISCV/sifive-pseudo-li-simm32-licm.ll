; REQUIRES: asserts
;
; Test to show that there is an ADDIW that is not hoisted without +pseudo-li-simm32
; RUN: [ $(llc -mtriple=riscv64 -debug-only=machinelicm < %s 2>&1 \
; RUN:     | grep -c "Won't hoist.*ADDI") \
; RUN:   -gt 0 ]

; Test to show that ADDIW is hoisted with +pseudo-li-simm32
; RUN: [ $(llc -mtriple=riscv64 -debug-only=machinelicm -mattr=+pseudo-li-simm32 < %s 2>&1 \
; RUN:     | grep -c "Won't hoist.*ADDI") \
; RUN:   -eq 0 ]

@a = external local_unnamed_addr global [0 x i32], align 4
@b = external local_unnamed_addr global [0 x i32], align 4

; Function Attrs: nofree norecurse nounwind
define dso_local void @foo(i32 noundef signext %n, ptr noundef %c) local_unnamed_addr #0 {
entry:
  store i32 -987654321, ptr @a, align 4
  %0 = load volatile i32, ptr %c, align 4
  %1 = load volatile i32, ptr %c, align 4
  %2 = load volatile i32, ptr %c, align 4
  %3 = load volatile i32, ptr %c, align 4
  %4 = load volatile i32, ptr %c, align 4
  %5 = load volatile i32, ptr %c, align 4
  %6 = load volatile i32, ptr %c, align 4
  %7 = load volatile i32, ptr %c, align 4
  %8 = load volatile i32, ptr %c, align 4
  %9 = load volatile i32, ptr %c, align 4
  %10 = load volatile i32, ptr %c, align 4
  %11 = load volatile i32, ptr %c, align 4
  %12 = load volatile i32, ptr %c, align 4
  %13 = load volatile i32, ptr %c, align 4
  %14 = load volatile i32, ptr %c, align 4
  %15 = load volatile i32, ptr %c, align 4
  %16 = load volatile i32, ptr %c, align 4
  %17 = load volatile i32, ptr %c, align 4
  %18 = load volatile i32, ptr %c, align 4
  %19 = load volatile i32, ptr %c, align 4
  %20 = load volatile i32, ptr %c, align 4
  %cmp.not71 = icmp slt i32 %n, 1
  br i1 %cmp.not71, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %21 = zext i32 %n to i64
  %22 = add nuw i32 %n, 1
  %wide.trip.count = zext i32 %22 to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  store volatile i32 %0, ptr %c, align 4
  store volatile i32 %1, ptr %c, align 4
  store volatile i32 %2, ptr %c, align 4
  store volatile i32 %3, ptr %c, align 4
  store volatile i32 %4, ptr %c, align 4
  store volatile i32 %5, ptr %c, align 4
  store volatile i32 %6, ptr %c, align 4
  store volatile i32 %7, ptr %c, align 4
  store volatile i32 %8, ptr %c, align 4
  store volatile i32 %9, ptr %c, align 4
  store volatile i32 %10, ptr %c, align 4
  store volatile i32 %11, ptr %c, align 4
  store volatile i32 %12, ptr %c, align 4
  store volatile i32 %13, ptr %c, align 4
  store volatile i32 %14, ptr %c, align 4
  store volatile i32 %15, ptr %c, align 4
  store volatile i32 %16, ptr %c, align 4
  store volatile i32 %17, ptr %c, align 4
  store volatile i32 %18, ptr %c, align 4
  store volatile i32 %19, ptr %c, align 4
  store volatile i32 %20, ptr %c, align 4
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds [0 x i32], ptr @a, i64 0, i64 %indvars.iv
  %23 = load i32, ptr %arrayidx, align 4
  %24 = tail call i32 @llvm.smax.i32(i32 %23, i32 -987654321)
  store i32 %24, ptr %arrayidx, align 4
  %cmp7 = icmp ult i64 %indvars.iv, %21
  br i1 %cmp7, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx9 = getelementptr inbounds [0 x i32], ptr @b, i64 0, i64 %indvars.iv
  %25 = load i32, ptr %arrayidx9, align 4
  %26 = tail call i32 @llvm.smax.i32(i32 %25, i32 -987654321)
  store i32 %26, ptr %arrayidx9, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.smax.i32(i32, i32) #1
