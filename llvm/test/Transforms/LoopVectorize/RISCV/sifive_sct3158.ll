; RUN: opt -passes=loop-vectorize -mcpu=sifive-p470 -mtriple riscv64 -sifive-loop-vectorizer-enable-interleaved-access=invariant-stride -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK-LABLE: LV: Checking a loop in 'test_16x16
; CHECK: LV: loop body cost with SLP = {{.*}}
; CHECK: LV: Vectorization is possible but not beneficial.
define i32 @test_16x16(ptr  %pix1, i32  %i_stride_pix1, ptr  %pix2, i32  %i_stride_pix2) {
entry:
  %pix1_ext = sext i32 %i_stride_pix1 to i64
  %pix2_ext = sext i32 %i_stride_pix2 to i64
  br label %preheader

preheader:
  %y.025 = phi i32 [ 0, %entry ], [ %inc11, %preheader ]
  %i_sum.024 = phi i32 [ 0, %entry ], [ %add.15, %preheader ]
  %pix1.addr.023 = phi ptr [ %pix1, %entry ], [ %add.ptr, %preheader ]
  %pix2.addr.022 = phi ptr [ %pix2, %entry ], [ %add.ptr9, %preheader ]
  %0 = load i8, ptr %pix1.addr.023, align 1
  %conv = zext i8 %0 to i32
  %1 = load i8, ptr %pix2.addr.022, align 1
  %conv7 = zext i8 %1 to i32
  %sub = sub nsw i32 %conv, %conv7
  %2 = tail call i32 @llvm.abs.i32(i32 %sub, i1 true)
  %add = add nsw i32 %2, %i_sum.024
  %arrayidx.1 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 1
  %3 = load i8, ptr %arrayidx.1, align 1
  %conv.1 = zext i8 %3 to i32
  %arrayidx6.1 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 1
  %4 = load i8, ptr %arrayidx6.1, align 1
  %conv7.1 = zext i8 %4 to i32
  %sub.1 = sub nsw i32 %conv.1, %conv7.1
  %5 = tail call i32 @llvm.abs.i32(i32 %sub.1, i1 true)
  %add.1 = add nsw i32 %5, %add
  %arrayidx.2 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 2
  %6 = load i8, ptr %arrayidx.2, align 1
  %conv.2 = zext i8 %6 to i32
  %arrayidx6.2 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 2
  %7 = load i8, ptr %arrayidx6.2, align 1
  %conv7.2 = zext i8 %7 to i32
  %sub.2 = sub nsw i32 %conv.2, %conv7.2
  %8 = tail call i32 @llvm.abs.i32(i32 %sub.2, i1 true)
  %add.2 = add nsw i32 %8, %add.1
  %arrayidx.3 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 3
  %9 = load i8, ptr %arrayidx.3, align 1
  %conv.3 = zext i8 %9 to i32
  %arrayidx6.3 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 3
  %10 = load i8, ptr %arrayidx6.3, align 1
  %conv7.3 = zext i8 %10 to i32
  %sub.3 = sub nsw i32 %conv.3, %conv7.3
  %11 = tail call i32 @llvm.abs.i32(i32 %sub.3, i1 true)
  %add.3 = add nsw i32 %11, %add.2
  %arrayidx.4 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 4
  %12 = load i8, ptr %arrayidx.4, align 1
  %conv.4 = zext i8 %12 to i32
  %arrayidx6.4 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 4
  %13 = load i8, ptr %arrayidx6.4, align 1
  %conv7.4 = zext i8 %13 to i32
  %sub.4 = sub nsw i32 %conv.4, %conv7.4
  %14 = tail call i32 @llvm.abs.i32(i32 %sub.4, i1 true)
  %add.4 = add nsw i32 %14, %add.3
  %arrayidx.5 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 5
  %15 = load i8, ptr %arrayidx.5, align 1
  %conv.5 = zext i8 %15 to i32
  %arrayidx6.5 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 5
  %16 = load i8, ptr %arrayidx6.5, align 1
  %conv7.5 = zext i8 %16 to i32
  %sub.5 = sub nsw i32 %conv.5, %conv7.5
  %17 = tail call i32 @llvm.abs.i32(i32 %sub.5, i1 true)
  %add.5 = add nsw i32 %17, %add.4
  %arrayidx.6 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 6
  %18 = load i8, ptr %arrayidx.6, align 1
  %conv.6 = zext i8 %18 to i32
  %arrayidx6.6 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 6
  %19 = load i8, ptr %arrayidx6.6, align 1
  %conv7.6 = zext i8 %19 to i32
  %sub.6 = sub nsw i32 %conv.6, %conv7.6
  %20 = tail call i32 @llvm.abs.i32(i32 %sub.6, i1 true)
  %add.6 = add nsw i32 %20, %add.5
  %arrayidx.7 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 7
  %21 = load i8, ptr %arrayidx.7, align 1
  %conv.7 = zext i8 %21 to i32
  %arrayidx6.7 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 7
  %22 = load i8, ptr %arrayidx6.7, align 1
  %conv7.7 = zext i8 %22 to i32
  %sub.7 = sub nsw i32 %conv.7, %conv7.7
  %23 = tail call i32 @llvm.abs.i32(i32 %sub.7, i1 true)
  %add.7 = add nsw i32 %23, %add.6
  %arrayidx.8 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 8
  %24 = load i8, ptr %arrayidx.8, align 1
  %conv.8 = zext i8 %24 to i32
  %arrayidx6.8 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 8
  %25 = load i8, ptr %arrayidx6.8, align 1
  %conv7.8 = zext i8 %25 to i32
  %sub.8 = sub nsw i32 %conv.8, %conv7.8
  %26 = tail call i32 @llvm.abs.i32(i32 %sub.8, i1 true)
  %add.8 = add nsw i32 %26, %add.7
  %arrayidx.9 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 9
  %27 = load i8, ptr %arrayidx.9, align 1
  %conv.9 = zext i8 %27 to i32
  %arrayidx6.9 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 9
  %28 = load i8, ptr %arrayidx6.9, align 1
  %conv7.9 = zext i8 %28 to i32
  %sub.9 = sub nsw i32 %conv.9, %conv7.9
  %29 = tail call i32 @llvm.abs.i32(i32 %sub.9, i1 true)
  %add.9 = add nsw i32 %29, %add.8
  %arrayidx.10 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 10
  %30 = load i8, ptr %arrayidx.10, align 1
  %conv.10 = zext i8 %30 to i32
  %arrayidx6.10 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 10
  %31 = load i8, ptr %arrayidx6.10, align 1
  %conv7.10 = zext i8 %31 to i32
  %sub.10 = sub nsw i32 %conv.10, %conv7.10
  %32 = tail call i32 @llvm.abs.i32(i32 %sub.10, i1 true)
  %add.10 = add nsw i32 %32, %add.9
  %arrayidx.11 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 11
  %33 = load i8, ptr %arrayidx.11, align 1
  %conv.11 = zext i8 %33 to i32
  %arrayidx6.11 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 11
  %34 = load i8, ptr %arrayidx6.11, align 1
  %conv7.11 = zext i8 %34 to i32
  %sub.11 = sub nsw i32 %conv.11, %conv7.11
  %35 = tail call i32 @llvm.abs.i32(i32 %sub.11, i1 true)
  %add.11 = add nsw i32 %35, %add.10
  %arrayidx.12 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 12
  %36 = load i8, ptr %arrayidx.12, align 1
  %conv.12 = zext i8 %36 to i32
  %arrayidx6.12 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 12
  %37 = load i8, ptr %arrayidx6.12, align 1
  %conv7.12 = zext i8 %37 to i32
  %sub.12 = sub nsw i32 %conv.12, %conv7.12
  %38 = tail call i32 @llvm.abs.i32(i32 %sub.12, i1 true)
  %add.12 = add nsw i32 %38, %add.11
  %arrayidx.13 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 13
  %39 = load i8, ptr %arrayidx.13, align 1
  %conv.13 = zext i8 %39 to i32
  %arrayidx6.13 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 13
  %40 = load i8, ptr %arrayidx6.13, align 1
  %conv7.13 = zext i8 %40 to i32
  %sub.13 = sub nsw i32 %conv.13, %conv7.13
  %41 = tail call i32 @llvm.abs.i32(i32 %sub.13, i1 true)
  %add.13 = add nsw i32 %41, %add.12
  %arrayidx.14 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 14
  %42 = load i8, ptr %arrayidx.14, align 1
  %conv.14 = zext i8 %42 to i32
  %arrayidx6.14 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 14
  %43 = load i8, ptr %arrayidx6.14, align 1
  %conv7.14 = zext i8 %43 to i32
  %sub.14 = sub nsw i32 %conv.14, %conv7.14
  %44 = tail call i32 @llvm.abs.i32(i32 %sub.14, i1 true)
  %add.14 = add nsw i32 %44, %add.13
  %arrayidx.15 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 15
  %45 = load i8, ptr %arrayidx.15, align 1
  %conv.15 = zext i8 %45 to i32
  %arrayidx6.15 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 15
  %46 = load i8, ptr %arrayidx6.15, align 1
  %conv7.15 = zext i8 %46 to i32
  %sub.15 = sub nsw i32 %conv.15, %conv7.15
  %47 = tail call i32 @llvm.abs.i32(i32 %sub.15, i1 true)
  %add.15 = add nsw i32 %47, %add.14
  %add.ptr = getelementptr inbounds i8, ptr %pix1.addr.023, i64 %pix1_ext
  %add.ptr9 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 %pix2_ext
  %inc11 = add nuw nsw i32 %y.025, 1
  %exitcond.not = icmp eq i32 %inc11, 16
  br i1 %exitcond.not, label %cleanup, label %preheader

cleanup:
  %add.15.lcssa = phi i32 [ %add.15, %preheader ]
  ret i32 %add.15.lcssa
}

; CHECK-LABLE: LV: Checking a loop in 'test_16x8
; CHECK: LV: loop body cost with SLP = {{.*}}
; CHECK: LV: Vectorization is possible but not beneficial.
define i32 @test_8x16(ptr  %pix1, i32  %i_stride_pix1, ptr  %pix2, i32  %i_stride_pix2) {
entry:
  %pix1_ext = sext i32 %i_stride_pix1 to i64
  %pix2_ext = sext i32 %i_stride_pix2 to i64
  br label %preheader

preheader:
  %y.025 = phi i32 [ 0, %entry ], [ %inc11, %preheader ]
  %i_sum.024 = phi i32 [ 0, %entry ], [ %add.7, %preheader ]
  %pix1.addr.023 = phi ptr [ %pix1, %entry ], [ %add.ptr, %preheader ]
  %pix2.addr.022 = phi ptr [ %pix2, %entry ], [ %add.ptr9, %preheader ]
  %0 = load i8, ptr %pix1.addr.023, align 1
  %conv = zext i8 %0 to i32
  %1 = load i8, ptr %pix2.addr.022, align 1
  %conv7 = zext i8 %1 to i32
  %sub = sub nsw i32 %conv, %conv7
  %2 = tail call i32 @llvm.abs.i32(i32 %sub, i1 true)
  %add = add nsw i32 %2, %i_sum.024
  %arrayidx.1 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 1
  %3 = load i8, ptr %arrayidx.1, align 1
  %conv.1 = zext i8 %3 to i32
  %arrayidx6.1 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 1
  %4 = load i8, ptr %arrayidx6.1, align 1
  %conv7.1 = zext i8 %4 to i32
  %sub.1 = sub nsw i32 %conv.1, %conv7.1
  %5 = tail call i32 @llvm.abs.i32(i32 %sub.1, i1 true)
  %add.1 = add nsw i32 %5, %add
  %arrayidx.2 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 2
  %6 = load i8, ptr %arrayidx.2, align 1
  %conv.2 = zext i8 %6 to i32
  %arrayidx6.2 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 2
  %7 = load i8, ptr %arrayidx6.2, align 1
  %conv7.2 = zext i8 %7 to i32
  %sub.2 = sub nsw i32 %conv.2, %conv7.2
  %8 = tail call i32 @llvm.abs.i32(i32 %sub.2, i1 true)
  %add.2 = add nsw i32 %8, %add.1
  %arrayidx.3 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 3
  %9 = load i8, ptr %arrayidx.3, align 1
  %conv.3 = zext i8 %9 to i32
  %arrayidx6.3 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 3
  %10 = load i8, ptr %arrayidx6.3, align 1
  %conv7.3 = zext i8 %10 to i32
  %sub.3 = sub nsw i32 %conv.3, %conv7.3
  %11 = tail call i32 @llvm.abs.i32(i32 %sub.3, i1 true)
  %add.3 = add nsw i32 %11, %add.2
  %arrayidx.4 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 4
  %12 = load i8, ptr %arrayidx.4, align 1
  %conv.4 = zext i8 %12 to i32
  %arrayidx6.4 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 4
  %13 = load i8, ptr %arrayidx6.4, align 1
  %conv7.4 = zext i8 %13 to i32
  %sub.4 = sub nsw i32 %conv.4, %conv7.4
  %14 = tail call i32 @llvm.abs.i32(i32 %sub.4, i1 true)
  %add.4 = add nsw i32 %14, %add.3
  %arrayidx.5 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 5
  %15 = load i8, ptr %arrayidx.5, align 1
  %conv.5 = zext i8 %15 to i32
  %arrayidx6.5 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 5
  %16 = load i8, ptr %arrayidx6.5, align 1
  %conv7.5 = zext i8 %16 to i32
  %sub.5 = sub nsw i32 %conv.5, %conv7.5
  %17 = tail call i32 @llvm.abs.i32(i32 %sub.5, i1 true)
  %add.5 = add nsw i32 %17, %add.4
  %arrayidx.6 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 6
  %18 = load i8, ptr %arrayidx.6, align 1
  %conv.6 = zext i8 %18 to i32
  %arrayidx6.6 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 6
  %19 = load i8, ptr %arrayidx6.6, align 1
  %conv7.6 = zext i8 %19 to i32
  %sub.6 = sub nsw i32 %conv.6, %conv7.6
  %20 = tail call i32 @llvm.abs.i32(i32 %sub.6, i1 true)
  %add.6 = add nsw i32 %20, %add.5
  %arrayidx.7 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 7
  %21 = load i8, ptr %arrayidx.7, align 1
  %conv.7 = zext i8 %21 to i32
  %arrayidx6.7 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 7
  %22 = load i8, ptr %arrayidx6.7, align 1
  %conv7.7 = zext i8 %22 to i32
  %sub.7 = sub nsw i32 %conv.7, %conv7.7
  %23 = tail call i32 @llvm.abs.i32(i32 %sub.7, i1 true)
  %add.7 = add nsw i32 %23, %add.6
  %add.ptr = getelementptr inbounds i8, ptr %pix1.addr.023, i64 %pix1_ext
  %add.ptr9 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 %pix2_ext
  %inc11 = add nuw nsw i32 %y.025, 1
  %exitcond.not = icmp eq i32 %inc11, 16
  br i1 %exitcond.not, label %cleanup, label %preheader

cleanup:
  %add.7.lcssa = phi i32 [ %add.7, %preheader ]
  ret i32 %add.7.lcssa
}

; CHECK-LABLE: LV: Checking a loop in 'test_8x8
; CHECK: LV: loop body cost with SLP = {{.*}}
; CHECK: LV: Vectorization is possible but not beneficial.
define i32 @test_8x8(ptr  %pix1, i32  %i_stride_pix1, ptr  %pix2, i32  %i_stride_pix2) {
entry:
  %pix1_ext = sext i32 %i_stride_pix1 to i64
  %pix2_ext = sext i32 %i_stride_pix2 to i64
  br label %preheader

preheader:
  %y.025 = phi i32 [ 0, %entry ], [ %inc11, %preheader ]
  %i_sum.024 = phi i32 [ 0, %entry ], [ %add.7, %preheader ]
  %pix1.addr.023 = phi ptr [ %pix1, %entry ], [ %add.ptr, %preheader ]
  %pix2.addr.022 = phi ptr [ %pix2, %entry ], [ %add.ptr9, %preheader ]
  %0 = load i8, ptr %pix1.addr.023, align 1
  %conv = zext i8 %0 to i32
  %1 = load i8, ptr %pix2.addr.022, align 1
  %conv7 = zext i8 %1 to i32
  %sub = sub nsw i32 %conv, %conv7
  %2 = tail call i32 @llvm.abs.i32(i32 %sub, i1 true)
  %add = add nsw i32 %2, %i_sum.024
  %arrayidx.1 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 1
  %3 = load i8, ptr %arrayidx.1, align 1
  %conv.1 = zext i8 %3 to i32
  %arrayidx6.1 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 1
  %4 = load i8, ptr %arrayidx6.1, align 1
  %conv7.1 = zext i8 %4 to i32
  %sub.1 = sub nsw i32 %conv.1, %conv7.1
  %5 = tail call i32 @llvm.abs.i32(i32 %sub.1, i1 true)
  %add.1 = add nsw i32 %5, %add
  %arrayidx.2 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 2
  %6 = load i8, ptr %arrayidx.2, align 1
  %conv.2 = zext i8 %6 to i32
  %arrayidx6.2 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 2
  %7 = load i8, ptr %arrayidx6.2, align 1
  %conv7.2 = zext i8 %7 to i32
  %sub.2 = sub nsw i32 %conv.2, %conv7.2
  %8 = tail call i32 @llvm.abs.i32(i32 %sub.2, i1 true)
  %add.2 = add nsw i32 %8, %add.1
  %arrayidx.3 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 3
  %9 = load i8, ptr %arrayidx.3, align 1
  %conv.3 = zext i8 %9 to i32
  %arrayidx6.3 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 3
  %10 = load i8, ptr %arrayidx6.3, align 1
  %conv7.3 = zext i8 %10 to i32
  %sub.3 = sub nsw i32 %conv.3, %conv7.3
  %11 = tail call i32 @llvm.abs.i32(i32 %sub.3, i1 true)
  %add.3 = add nsw i32 %11, %add.2
  %arrayidx.4 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 4
  %12 = load i8, ptr %arrayidx.4, align 1
  %conv.4 = zext i8 %12 to i32
  %arrayidx6.4 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 4
  %13 = load i8, ptr %arrayidx6.4, align 1
  %conv7.4 = zext i8 %13 to i32
  %sub.4 = sub nsw i32 %conv.4, %conv7.4
  %14 = tail call i32 @llvm.abs.i32(i32 %sub.4, i1 true)
  %add.4 = add nsw i32 %14, %add.3
  %arrayidx.5 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 5
  %15 = load i8, ptr %arrayidx.5, align 1
  %conv.5 = zext i8 %15 to i32
  %arrayidx6.5 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 5
  %16 = load i8, ptr %arrayidx6.5, align 1
  %conv7.5 = zext i8 %16 to i32
  %sub.5 = sub nsw i32 %conv.5, %conv7.5
  %17 = tail call i32 @llvm.abs.i32(i32 %sub.5, i1 true)
  %add.5 = add nsw i32 %17, %add.4
  %arrayidx.6 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 6
  %18 = load i8, ptr %arrayidx.6, align 1
  %conv.6 = zext i8 %18 to i32
  %arrayidx6.6 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 6
  %19 = load i8, ptr %arrayidx6.6, align 1
  %conv7.6 = zext i8 %19 to i32
  %sub.6 = sub nsw i32 %conv.6, %conv7.6
  %20 = tail call i32 @llvm.abs.i32(i32 %sub.6, i1 true)
  %add.6 = add nsw i32 %20, %add.5
  %arrayidx.7 = getelementptr inbounds i8, ptr %pix1.addr.023, i64 7
  %21 = load i8, ptr %arrayidx.7, align 1
  %conv.7 = zext i8 %21 to i32
  %arrayidx6.7 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 7
  %22 = load i8, ptr %arrayidx6.7, align 1
  %conv7.7 = zext i8 %22 to i32
  %sub.7 = sub nsw i32 %conv.7, %conv7.7
  %23 = tail call i32 @llvm.abs.i32(i32 %sub.7, i1 true)
  %add.7 = add nsw i32 %23, %add.6
  %add.ptr = getelementptr inbounds i8, ptr %pix1.addr.023, i64 %pix1_ext
  %add.ptr9 = getelementptr inbounds i8, ptr %pix2.addr.022, i64 %pix2_ext
  %inc11 = add nuw nsw i32 %y.025, 1
  %exitcond.not = icmp eq i32 %inc11, 8
  br i1 %exitcond.not, label %cleanup, label %preheader

cleanup:
  ret i32 %add.7
}
