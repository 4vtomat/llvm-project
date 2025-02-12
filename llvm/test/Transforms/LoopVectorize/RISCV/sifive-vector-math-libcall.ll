; RUN: opt -passes=inject-tli-mappings,loop-vectorize -vector-library=SiFive_NF \
; RUN:   -S -mtriple=riscv64 -mattr=+v -vector-primary-lmul-max=0 < %s | FileCheck %s --check-prefix=M1
; RUN: opt -passes=inject-tli-mappings,loop-vectorize -vector-library=SiFive_NF \
; RUN:   -S -mtriple=riscv64 -mattr=+v -vector-primary-lmul-max=1 < %s | FileCheck %s --check-prefix=M2
; RUN: opt -passes=inject-tli-mappings,loop-vectorize -vector-library=SiFive_NF \
; RUN:   -S -mtriple=riscv64 -mattr=+v -vector-primary-lmul-max=2 < %s | FileCheck %s --check-prefix=M4
; RUN: opt -passes=inject-tli-mappings,loop-vectorize -vector-library=SiFive_NF \
; RUN:   -S -mtriple=riscv64 -mattr=+v -vector-primary-lmul-max=3 < %s | FileCheck %s --check-prefix=M8

declare float @cosf(float)
define void @test_cosf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_cosf(
; M1:  call <vscale x 2 x float> @skl_vfcos_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_cosf(
; M2:  call <vscale x 4 x float> @skl_vfcos_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_cosf(
; M4:  call <vscale x 8 x float> @skl_vfcos_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_cosf(
; M8:  call <vscale x 16 x float> @skl_vfcos_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @cosf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @cos(double)
define void @test_cos(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_cos(
; M1:  call <vscale x 1 x double> @skl_vfcos_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_cos(
; M2:  call <vscale x 2 x double> @skl_vfcos_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_cos(
; M4:  call <vscale x 4 x double> @skl_vfcos_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_cos(
; M8:  call <vscale x 8 x double> @skl_vfcos_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @cos(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @coshf(float)
define void @test_coshf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_coshf(
; M1:  call <vscale x 2 x float> @skl_vfcosh_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_coshf(
; M2:  call <vscale x 4 x float> @skl_vfcosh_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_coshf(
; M4:  call <vscale x 8 x float> @skl_vfcosh_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_coshf(
; M8:  call <vscale x 16 x float> @skl_vfcosh_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @coshf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @cosh(double)
define void @test_cosh(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_cosh(
; M1:  call <vscale x 1 x double> @skl_vfcosh_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_cosh(
; M2:  call <vscale x 2 x double> @skl_vfcosh_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_cosh(
; M4:  call <vscale x 4 x double> @skl_vfcosh_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_cosh(
; M8:  call <vscale x 8 x double> @skl_vfcosh_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @cosh(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @sinf(float)
define void @test_sinf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_sinf(
; M1:  call <vscale x 2 x float> @skl_vfsin_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_sinf(
; M2:  call <vscale x 4 x float> @skl_vfsin_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_sinf(
; M4:  call <vscale x 8 x float> @skl_vfsin_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_sinf(
; M8:  call <vscale x 16 x float> @skl_vfsin_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @sinf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @sin(double)
define void @test_sin(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_sin(
; M1:  call <vscale x 1 x double> @skl_vfsin_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_sin(
; M2:  call <vscale x 2 x double> @skl_vfsin_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_sin(
; M4:  call <vscale x 4 x double> @skl_vfsin_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_sin(
; M8:  call <vscale x 8 x double> @skl_vfsin_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @sin(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @sinhf(float)
define void @test_sinhf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_sinhf(
; M1:  call <vscale x 2 x float> @skl_vfsinh_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_sinhf(
; M2:  call <vscale x 4 x float> @skl_vfsinh_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_sinhf(
; M4:  call <vscale x 8 x float> @skl_vfsinh_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_sinhf(
; M8:  call <vscale x 16 x float> @skl_vfsinh_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @sinhf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @sinh(double)
define void @test_sinh(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_sinh(
; M1:  call <vscale x 1 x double> @skl_vfsinh_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_sinh(
; M2:  call <vscale x 2 x double> @skl_vfsinh_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_sinh(
; M4:  call <vscale x 4 x double> @skl_vfsinh_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_sinh(
; M8:  call <vscale x 8 x double> @skl_vfsinh_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @sinh(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @tanf(float)
define void @test_tanf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_tanf(
; M1:  call <vscale x 2 x float> @skl_vftan_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_tanf(
; M2:  call <vscale x 4 x float> @skl_vftan_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_tanf(
; M4:  call <vscale x 8 x float> @skl_vftan_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_tanf(
; M8:  call <vscale x 16 x float> @skl_vftan_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @tanf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @tan(double)
define void @test_tan(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_tan(
; M1:  call <vscale x 1 x double> @skl_vftan_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_tan(
; M2:  call <vscale x 2 x double> @skl_vftan_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_tan(
; M4:  call <vscale x 4 x double> @skl_vftan_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_tan(
; M8:  call <vscale x 8 x double> @skl_vftan_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @tan(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @tanhf(float)
define void @test_tanhf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_tanhf(
; M1:  call <vscale x 2 x float> @skl_vftanh_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_tanhf(
; M2:  call <vscale x 4 x float> @skl_vftanh_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_tanhf(
; M4:  call <vscale x 8 x float> @skl_vftanh_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_tanhf(
; M8:  call <vscale x 16 x float> @skl_vftanh_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @tanhf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @tanh(double)
define void @test_tanh(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_tanh(
; M1:  call <vscale x 1 x double> @skl_vftanh_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_tanh(
; M2:  call <vscale x 2 x double> @skl_vftanh_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_tanh(
; M4:  call <vscale x 4 x double> @skl_vftanh_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_tanh(
; M8:  call <vscale x 8 x double> @skl_vftanh_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @tanh(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @acosf(float)
define void @test_acosf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_acosf(
; M1:  call <vscale x 2 x float> @skl_vfacos_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_acosf(
; M2:  call <vscale x 4 x float> @skl_vfacos_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_acosf(
; M4:  call <vscale x 8 x float> @skl_vfacos_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_acosf(
; M8:  call <vscale x 16 x float> @skl_vfacos_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @acosf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @acos(double)
define void @test_acos(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_acos(
; M1:  call <vscale x 1 x double> @skl_vfacos_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_acos(
; M2:  call <vscale x 2 x double> @skl_vfacos_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_acos(
; M4:  call <vscale x 4 x double> @skl_vfacos_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_acos(
; M8:  call <vscale x 8 x double> @skl_vfacos_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @acos(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @acoshf(float)
define void @test_acoshf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_acoshf(
; M1:  call <vscale x 2 x float> @skl_vfacosh_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_acoshf(
; M2:  call <vscale x 4 x float> @skl_vfacosh_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_acoshf(
; M4:  call <vscale x 8 x float> @skl_vfacosh_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_acoshf(
; M8:  call <vscale x 16 x float> @skl_vfacosh_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @acoshf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @acosh(double)
define void @test_acosh(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_acosh(
; M1:  call <vscale x 1 x double> @skl_vfacosh_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_acosh(
; M2:  call <vscale x 2 x double> @skl_vfacosh_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_acosh(
; M4:  call <vscale x 4 x double> @skl_vfacosh_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_acosh(
; M8:  call <vscale x 8 x double> @skl_vfacosh_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @acosh(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @asinf(float)
define void @test_asinf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_asinf(
; M1:  call <vscale x 2 x float> @skl_vfasin_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_asinf(
; M2:  call <vscale x 4 x float> @skl_vfasin_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_asinf(
; M4:  call <vscale x 8 x float> @skl_vfasin_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_asinf(
; M8:  call <vscale x 16 x float> @skl_vfasin_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @asinf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @asin(double)
define void @test_asin(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_asin(
; M1:  call <vscale x 1 x double> @skl_vfasin_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_asin(
; M2:  call <vscale x 2 x double> @skl_vfasin_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_asin(
; M4:  call <vscale x 4 x double> @skl_vfasin_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_asin(
; M8:  call <vscale x 8 x double> @skl_vfasin_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @asin(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @asinhf(float)
define void @test_asinhf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_asinhf(
; M1:  call <vscale x 2 x float> @skl_vfasinh_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_asinhf(
; M2:  call <vscale x 4 x float> @skl_vfasinh_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_asinhf(
; M4:  call <vscale x 8 x float> @skl_vfasinh_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_asinhf(
; M8:  call <vscale x 16 x float> @skl_vfasinh_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @asinhf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @asinh(double)
define void @test_asinh(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_asinh(
; M1:  call <vscale x 1 x double> @skl_vfasinh_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_asinh(
; M2:  call <vscale x 2 x double> @skl_vfasinh_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_asinh(
; M4:  call <vscale x 4 x double> @skl_vfasinh_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_asinh(
; M8:  call <vscale x 8 x double> @skl_vfasinh_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @asinh(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @atanf(float)
define void @test_atanf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_atanf(
; M1:  call <vscale x 2 x float> @skl_vfatan_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_atanf(
; M2:  call <vscale x 4 x float> @skl_vfatan_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_atanf(
; M4:  call <vscale x 8 x float> @skl_vfatan_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_atanf(
; M8:  call <vscale x 16 x float> @skl_vfatan_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @atanf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @atan(double)
define void @test_atan(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_atan(
; M1:  call <vscale x 1 x double> @skl_vfatan_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_atan(
; M2:  call <vscale x 2 x double> @skl_vfatan_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_atan(
; M4:  call <vscale x 4 x double> @skl_vfatan_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_atan(
; M8:  call <vscale x 8 x double> @skl_vfatan_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @atan(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @atan2f(float, float)
define void @test_atan2f(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_atan2f(
; M1:  call <vscale x 2 x float> @skl_vfatan2_vv_f32m1({{(<vscale x 2 x float> %.*){2}, i32 %.*}})
; M2-LABEL: @test_atan2f(
; M2:  call <vscale x 4 x float> @skl_vfatan2_vv_f32m2({{(<vscale x 4 x float> %.*){2}, i32 %.*}})
; M4-LABEL: @test_atan2f(
; M4:  call <vscale x 8 x float> @skl_vfatan2_vv_f32m4({{(<vscale x 8 x float> %.*){2}, i32 %.*}})
; M8-LABEL: @test_atan2f(
; M8:  call <vscale x 16 x float> @skl_vfatan2_vv_f32m8({{(<vscale x 16 x float> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = load float, ptr %arrayidx2, align 4
  %2 = call float @atan2f(float %0, float %1)
  store float %2, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @atan2(double, double)
define void @test_atan2(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_atan2(
; M1:  call <vscale x 1 x double> @skl_vfatan2_vv_f64m1({{(<vscale x 1 x double> %.*){2}, i32 %.*}})
; M2-LABEL: @test_atan2(
; M2:  call <vscale x 2 x double> @skl_vfatan2_vv_f64m2({{(<vscale x 2 x double> %.*){2}, i32 %.*}})
; M4-LABEL: @test_atan2(
; M4:  call <vscale x 4 x double> @skl_vfatan2_vv_f64m4({{(<vscale x 4 x double> %.*){2}, i32 %.*}})
; M8-LABEL: @test_atan2(
; M8:  call <vscale x 8 x double> @skl_vfatan2_vv_f64m8({{(<vscale x 8 x double> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = load double, ptr %arrayidx2, align 8
  %2 = call double @atan2(double %0, double %1)
  store double %2, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @atanhf(float)
define void @test_atanhf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_atanhf(
; M1:  call <vscale x 2 x float> @skl_vfatanh_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_atanhf(
; M2:  call <vscale x 4 x float> @skl_vfatanh_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_atanhf(
; M4:  call <vscale x 8 x float> @skl_vfatanh_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_atanhf(
; M8:  call <vscale x 16 x float> @skl_vfatanh_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @atanhf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @atanh(double)
define void @test_atanh(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_atanh(
; M1:  call <vscale x 1 x double> @skl_vfatanh_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_atanh(
; M2:  call <vscale x 2 x double> @skl_vfatanh_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_atanh(
; M4:  call <vscale x 4 x double> @skl_vfatanh_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_atanh(
; M8:  call <vscale x 8 x double> @skl_vfatanh_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @atanh(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @cbrtf(float)
define void @test_cbrtf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_cbrtf(
; M1:  call <vscale x 2 x float> @skl_vfcbrt_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_cbrtf(
; M2:  call <vscale x 4 x float> @skl_vfcbrt_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_cbrtf(
; M4:  call <vscale x 8 x float> @skl_vfcbrt_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_cbrtf(
; M8:  call <vscale x 16 x float> @skl_vfcbrt_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @cbrtf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @cbrt(double)
define void @test_cbrt(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_cbrt(
; M1:  call <vscale x 1 x double> @skl_vfcbrt_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_cbrt(
; M2:  call <vscale x 2 x double> @skl_vfcbrt_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_cbrt(
; M4:  call <vscale x 4 x double> @skl_vfcbrt_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_cbrt(
; M8:  call <vscale x 8 x double> @skl_vfcbrt_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @cbrt(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @erff(float)
define void @test_erff(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_erff(
; M1:  call <vscale x 2 x float> @skl_vferf_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_erff(
; M2:  call <vscale x 4 x float> @skl_vferf_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_erff(
; M4:  call <vscale x 8 x float> @skl_vferf_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_erff(
; M8:  call <vscale x 16 x float> @skl_vferf_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @erff(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @erf(double)
define void @test_erf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_erf(
; M1:  call <vscale x 1 x double> @skl_vferf_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_erf(
; M2:  call <vscale x 2 x double> @skl_vferf_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_erf(
; M4:  call <vscale x 4 x double> @skl_vferf_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_erf(
; M8:  call <vscale x 8 x double> @skl_vferf_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @erf(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @erfcf(float)
define void @test_erfcf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_erfcf(
; M1:  call <vscale x 2 x float> @skl_vferfc_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_erfcf(
; M2:  call <vscale x 4 x float> @skl_vferfc_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_erfcf(
; M4:  call <vscale x 8 x float> @skl_vferfc_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_erfcf(
; M8:  call <vscale x 16 x float> @skl_vferfc_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @erfcf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @erfc(double)
define void @test_erfc(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_erfc(
; M1:  call <vscale x 1 x double> @skl_vferfc_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_erfc(
; M2:  call <vscale x 2 x double> @skl_vferfc_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_erfc(
; M4:  call <vscale x 4 x double> @skl_vferfc_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_erfc(
; M8:  call <vscale x 8 x double> @skl_vferfc_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @erfc(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @expf(float)
define void @test_expf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_expf(
; M1:  call <vscale x 2 x float> @skl_vfexp_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_expf(
; M2:  call <vscale x 4 x float> @skl_vfexp_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_expf(
; M4:  call <vscale x 8 x float> @skl_vfexp_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_expf(
; M8:  call <vscale x 16 x float> @skl_vfexp_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @expf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @exp(double)
define void @test_exp(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_exp(
; M1:  call <vscale x 1 x double> @skl_vfexp_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_exp(
; M2:  call <vscale x 2 x double> @skl_vfexp_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_exp(
; M4:  call <vscale x 4 x double> @skl_vfexp_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_exp(
; M8:  call <vscale x 8 x double> @skl_vfexp_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @exp(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @exp10f(float)
define void @test_exp10f(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_exp10f(
; M1:  call <vscale x 2 x float> @skl_vfexp10_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_exp10f(
; M2:  call <vscale x 4 x float> @skl_vfexp10_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_exp10f(
; M4:  call <vscale x 8 x float> @skl_vfexp10_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_exp10f(
; M8:  call <vscale x 16 x float> @skl_vfexp10_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @exp10f(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @exp10(double)
define void @test_exp10(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_exp10(
; M1:  call <vscale x 1 x double> @skl_vfexp10_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_exp10(
; M2:  call <vscale x 2 x double> @skl_vfexp10_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_exp10(
; M4:  call <vscale x 4 x double> @skl_vfexp10_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_exp10(
; M8:  call <vscale x 8 x double> @skl_vfexp10_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @exp10(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @exp2f(float)
define void @test_exp2f(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_exp2f(
; M1:  call <vscale x 2 x float> @skl_vfexp2_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_exp2f(
; M2:  call <vscale x 4 x float> @skl_vfexp2_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_exp2f(
; M4:  call <vscale x 8 x float> @skl_vfexp2_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_exp2f(
; M8:  call <vscale x 16 x float> @skl_vfexp2_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @exp2f(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @exp2(double)
define void @test_exp2(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_exp2(
; M1:  call <vscale x 1 x double> @skl_vfexp2_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_exp2(
; M2:  call <vscale x 2 x double> @skl_vfexp2_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_exp2(
; M4:  call <vscale x 4 x double> @skl_vfexp2_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_exp2(
; M8:  call <vscale x 8 x double> @skl_vfexp2_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @exp2(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @expm1f(float)
define void @test_expm1f(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_expm1f(
; M1:  call <vscale x 2 x float> @skl_vfexpm1_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_expm1f(
; M2:  call <vscale x 4 x float> @skl_vfexpm1_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_expm1f(
; M4:  call <vscale x 8 x float> @skl_vfexpm1_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_expm1f(
; M8:  call <vscale x 16 x float> @skl_vfexpm1_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @expm1f(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @expm1(double)
define void @test_expm1(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_expm1(
; M1:  call <vscale x 1 x double> @skl_vfexpm1_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_expm1(
; M2:  call <vscale x 2 x double> @skl_vfexpm1_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_expm1(
; M4:  call <vscale x 4 x double> @skl_vfexpm1_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_expm1(
; M8:  call <vscale x 8 x double> @skl_vfexpm1_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @expm1(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @fmodf(float, float)
define void @test_fmodf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_fmodf(
; M1:  call <vscale x 2 x float> @skl_vffmod_vv_f32m1({{(<vscale x 2 x float> %.*){2}, i32 %.*}})
; M2-LABEL: @test_fmodf(
; M2:  call <vscale x 4 x float> @skl_vffmod_vv_f32m2({{(<vscale x 4 x float> %.*){2}, i32 %.*}})
; M4-LABEL: @test_fmodf(
; M4:  call <vscale x 8 x float> @skl_vffmod_vv_f32m4({{(<vscale x 8 x float> %.*){2}, i32 %.*}})
; M8-LABEL: @test_fmodf(
; M8:  call <vscale x 16 x float> @skl_vffmod_vv_f32m8({{(<vscale x 16 x float> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = load float, ptr %arrayidx2, align 4
  %2 = call float @fmodf(float %0, float %1)
  store float %2, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @fmod(double, double)
define void @test_fmod(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_fmod(
; M1:  call <vscale x 1 x double> @skl_vffmod_vv_f64m1({{(<vscale x 1 x double> %.*){2}, i32 %.*}})
; M2-LABEL: @test_fmod(
; M2:  call <vscale x 2 x double> @skl_vffmod_vv_f64m2({{(<vscale x 2 x double> %.*){2}, i32 %.*}})
; M4-LABEL: @test_fmod(
; M4:  call <vscale x 4 x double> @skl_vffmod_vv_f64m4({{(<vscale x 4 x double> %.*){2}, i32 %.*}})
; M8-LABEL: @test_fmod(
; M8:  call <vscale x 8 x double> @skl_vffmod_vv_f64m8({{(<vscale x 8 x double> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = load double, ptr %arrayidx2, align 8
  %2 = call double @fmod(double %0, double %1)
  store double %2, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @hypotf(float, float)
define void @test_hypotf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_hypotf(
; M1:  call <vscale x 2 x float> @skl_vfhypot_vv_f32m1({{(<vscale x 2 x float> %.*){2}, i32 %.*}})
; M2-LABEL: @test_hypotf(
; M2:  call <vscale x 4 x float> @skl_vfhypot_vv_f32m2({{(<vscale x 4 x float> %.*){2}, i32 %.*}})
; M4-LABEL: @test_hypotf(
; M4:  call <vscale x 8 x float> @skl_vfhypot_vv_f32m4({{(<vscale x 8 x float> %.*){2}, i32 %.*}})
; M8-LABEL: @test_hypotf(
; M8:  call <vscale x 16 x float> @skl_vfhypot_vv_f32m8({{(<vscale x 16 x float> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = load float, ptr %arrayidx, align 4
  %2 = call float @hypotf(float %0, float %1)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @hypot(double, double)
define void @test_hypot(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_hypot(
; M1:  call <vscale x 1 x double> @skl_vfhypot_vv_f64m1({{(<vscale x 1 x double> %.*){2}, i32 %.*}})
; M2-LABEL: @test_hypot(
; M2:  call <vscale x 2 x double> @skl_vfhypot_vv_f64m2({{(<vscale x 2 x double> %.*){2}, i32 %.*}})
; M4-LABEL: @test_hypot(
; M4:  call <vscale x 4 x double> @skl_vfhypot_vv_f64m4({{(<vscale x 4 x double> %.*){2}, i32 %.*}})
; M8-LABEL: @test_hypot(
; M8:  call <vscale x 8 x double> @skl_vfhypot_vv_f64m8({{(<vscale x 8 x double> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = load double, ptr %arrayidx2, align 8
  %2 = call double @hypot(double %0, double %1)
  store double %2, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare signext i32 @ilogbf(float)
define void @test_ilogbf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_ilogbf(
; M1:  call <vscale x 2 x i32> @skl_vfilogb_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_ilogbf(
; M2:  call <vscale x 4 x i32> @skl_vfilogb_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_ilogbf(
; M4:  call <vscale x 8 x i32> @skl_vfilogb_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_ilogbf(
; M8:  call <vscale x 16 x i32> @skl_vfilogb_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call signext i32 @ilogbf(float %0)
  store i32 %1, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare signext i32 @ilogb(double)
define void @test_ilogb(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_ilogb(
; M1:  call <vscale x 1 x i32> @skl_vfilogb_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_ilogb(
; M2:  call <vscale x 2 x i32> @skl_vfilogb_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_ilogb(
; M4:  call <vscale x 4 x i32> @skl_vfilogb_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_ilogb(
; M8:  call <vscale x 8 x i32> @skl_vfilogb_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call signext i32 @ilogb(double %0)
  store i32 %1, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @ldexpf(float, i32 signext)
define void @test_ldexpf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_ldexpf(
; M1:  call <vscale x 2 x float> @skl_vfldexp_vv_f32m1({{<vscale x 2 x float> %.*, <vscale x 2 x i32> %.*, i32 %.*}})
; M2-LABEL: @test_ldexpf(
; M2:  call <vscale x 4 x float> @skl_vfldexp_vv_f32m2({{<vscale x 4 x float> %.*, <vscale x 4 x i32> %.*, i32 %.*}})
; M4-LABEL: @test_ldexpf(
; M4:  call <vscale x 8 x float> @skl_vfldexp_vv_f32m4({{<vscale x 8 x float> %.*, <vscale x 8 x i32> %.*, i32 %.*}})
; M8-LABEL: @test_ldexpf(
; M8:  call <vscale x 16 x float> @skl_vfldexp_vv_f32m8({{<vscale x 16 x float> %.*, <vscale x 16 x i32> %.*, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = load i32, ptr %arrayidx2, align 4
  %2 = call float @ldexpf(float %0, i32 %1)
  store float %2, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @ldexp(double, i32 signext)
define void @test_ldexp(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_ldexp(
; M1:  call <vscale x 1 x double> @skl_vfldexp_vv_f64m1({{<vscale x 1 x double> %.*, <vscale x 1 x i32> %.*, i32 %.*}})
; M2-LABEL: @test_ldexp(
; M2:  call <vscale x 2 x double> @skl_vfldexp_vv_f64m2({{<vscale x 2 x double> %.*, <vscale x 2 x i32> %.*, i32 %.*}})
; M4-LABEL: @test_ldexp(
; M4:  call <vscale x 4 x double> @skl_vfldexp_vv_f64m4({{<vscale x 4 x double> %.*, <vscale x 4 x i32> %.*, i32 %.*}})
; M8-LABEL: @test_ldexp(
; M8:  call <vscale x 8 x double> @skl_vfldexp_vv_f64m8({{<vscale x 8 x double> %.*, <vscale x 8 x i32> %.*, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = load i32, ptr %arrayidx2, align 4
  %2 = call double @ldexp(double %0, i32 %1)
  store double %2, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @logf(float)
define void @test_logf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_logf(
; M1:  call <vscale x 2 x float> @skl_vflog_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_logf(
; M2:  call <vscale x 4 x float> @skl_vflog_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_logf(
; M4:  call <vscale x 8 x float> @skl_vflog_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_logf(
; M8:  call <vscale x 16 x float> @skl_vflog_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @logf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @log(double)
define void @test_log(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_log(
; M1:  call <vscale x 1 x double> @skl_vflog_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_log(
; M2:  call <vscale x 2 x double> @skl_vflog_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_log(
; M4:  call <vscale x 4 x double> @skl_vflog_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_log(
; M8:  call <vscale x 8 x double> @skl_vflog_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @log(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @log10f(float)
define void @test_log10f(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_log10f(
; M1:  call <vscale x 2 x float> @skl_vflog10_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_log10f(
; M2:  call <vscale x 4 x float> @skl_vflog10_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_log10f(
; M4:  call <vscale x 8 x float> @skl_vflog10_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_log10f(
; M8:  call <vscale x 16 x float> @skl_vflog10_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @log10f(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @log10(double)
define void @test_log10(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_log10(
; M1:  call <vscale x 1 x double> @skl_vflog10_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_log10(
; M2:  call <vscale x 2 x double> @skl_vflog10_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_log10(
; M4:  call <vscale x 4 x double> @skl_vflog10_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_log10(
; M8:  call <vscale x 8 x double> @skl_vflog10_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @log10(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @log1pf(float)
define void @test_log1pf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_log1pf(
; M1:  call <vscale x 2 x float> @skl_vflog1p_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_log1pf(
; M2:  call <vscale x 4 x float> @skl_vflog1p_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_log1pf(
; M4:  call <vscale x 8 x float> @skl_vflog1p_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_log1pf(
; M8:  call <vscale x 16 x float> @skl_vflog1p_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @log1pf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @log1p(double)
define void @test_log1p(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_log1p(
; M1:  call <vscale x 1 x double> @skl_vflog1p_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_log1p(
; M2:  call <vscale x 2 x double> @skl_vflog1p_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_log1p(
; M4:  call <vscale x 4 x double> @skl_vflog1p_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_log1p(
; M8:  call <vscale x 8 x double> @skl_vflog1p_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @log1p(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @log2f(float)
define void @test_log2f(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_log2f(
; M1:  call <vscale x 2 x float> @skl_vflog2_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_log2f(
; M2:  call <vscale x 4 x float> @skl_vflog2_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_log2f(
; M4:  call <vscale x 8 x float> @skl_vflog2_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_log2f(
; M8:  call <vscale x 16 x float> @skl_vflog2_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @log2f(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @log2(double)
define void @test_log2(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_log2(
; M1:  call <vscale x 1 x double> @skl_vflog2_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_log2(
; M2:  call <vscale x 2 x double> @skl_vflog2_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_log2(
; M4:  call <vscale x 4 x double> @skl_vflog2_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_log2(
; M8:  call <vscale x 8 x double> @skl_vflog2_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @log2(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
declare float @logbf(float)
define void @test_logbf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_logbf(
; M1:  call <vscale x 2 x float> @skl_vflogb_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_logbf(
; M2:  call <vscale x 4 x float> @skl_vflogb_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_logbf(
; M4:  call <vscale x 8 x float> @skl_vflogb_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_logbf(
; M8:  call <vscale x 16 x float> @skl_vflogb_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @logbf(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @logb(double)
define void @test_logb(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_logb(
; M1:  call <vscale x 1 x double> @skl_vflogb_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_logb(
; M2:  call <vscale x 2 x double> @skl_vflogb_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_logb(
; M4:  call <vscale x 4 x double> @skl_vflogb_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_logb(
; M8:  call <vscale x 8 x double> @skl_vflogb_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @logb(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @nextafterf(float, float)
define void @test_nextafterf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_nextafterf(
; M1:  call <vscale x 2 x float> @skl_vfnextafter_vv_f32m1({{(<vscale x 2 x float> %.*){2}, i32 %.*}})
; M2-LABEL: @test_nextafterf(
; M2:  call <vscale x 4 x float> @skl_vfnextafter_vv_f32m2({{(<vscale x 4 x float> %.*){2}, i32 %.*}})
; M4-LABEL: @test_nextafterf(
; M4:  call <vscale x 8 x float> @skl_vfnextafter_vv_f32m4({{(<vscale x 8 x float> %.*){2}, i32 %.*}})
; M8-LABEL: @test_nextafterf(
; M8:  call <vscale x 16 x float> @skl_vfnextafter_vv_f32m8({{(<vscale x 16 x float> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = load float, ptr %arrayidx2, align 4
  %2 = call float @nextafterf(float %0, float %1)
  store float %2, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @nextafter(double, double)
define void @test_nextafter(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_nextafter(
; M1:  call <vscale x 1 x double> @skl_vfnextafter_vv_f64m1({{(<vscale x 1 x double> %.*){2}, i32 %.*}})
; M2-LABEL: @test_nextafter(
; M2:  call <vscale x 2 x double> @skl_vfnextafter_vv_f64m2({{(<vscale x 2 x double> %.*){2}, i32 %.*}})
; M4-LABEL: @test_nextafter(
; M4:  call <vscale x 4 x double> @skl_vfnextafter_vv_f64m4({{(<vscale x 4 x double> %.*){2}, i32 %.*}})
; M8-LABEL: @test_nextafter(
; M8:  call <vscale x 8 x double> @skl_vfnextafter_vv_f64m8({{(<vscale x 8 x double> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = load double, ptr %arrayidx2, align 8
  %2 = call double @nextafter(double %0, double %1)
  store double %2, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare float @powf(float, float)
define void @test_powf(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_powf(
; M1:  call <vscale x 2 x float> @skl_vfpow_vv_f32m1({{(<vscale x 2 x float> %.*){2}, i32 %.*}})
; M2-LABEL: @test_powf(
; M2:  call <vscale x 4 x float> @skl_vfpow_vv_f32m2({{(<vscale x 4 x float> %.*){2}, i32 %.*}})
; M4-LABEL: @test_powf(
; M4:  call <vscale x 8 x float> @skl_vfpow_vv_f32m4({{(<vscale x 8 x float> %.*){2}, i32 %.*}})
; M8-LABEL: @test_powf(
; M8:  call <vscale x 16 x float> @skl_vfpow_vv_f32m8({{(<vscale x 16 x float> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = load float, ptr %arrayidx2, align 4
  %2 = call float @powf(float %0, float %1)
  store float %2, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @pow(double, double)
define void @test_pow(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_pow(
; M1:  call <vscale x 1 x double> @skl_vfpow_vv_f64m1({{(<vscale x 1 x double> %.*){2}, i32 %.*}})
; M2-LABEL: @test_pow(
; M2:  call <vscale x 2 x double> @skl_vfpow_vv_f64m2({{(<vscale x 2 x double> %.*){2}, i32 %.*}})
; M4-LABEL: @test_pow(
; M4:  call <vscale x 4 x double> @skl_vfpow_vv_f64m4({{(<vscale x 4 x double> %.*){2}, i32 %.*}})
; M8-LABEL: @test_pow(
; M8:  call <vscale x 8 x double> @skl_vfpow_vv_f64m8({{(<vscale x 8 x double> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = load double, ptr %arrayidx, align 8
  %2 = call double @pow(double %0, double %1)
  store double %2, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_cosf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_cosf32(
; M1:  call <vscale x 2 x float> @skl_vfcos_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_cosf32(
; M2:  call <vscale x 4 x float> @skl_vfcos_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_cosf32(
; M4:  call <vscale x 8 x float> @skl_vfcos_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_cosf32(
; M8:  call <vscale x 16 x float> @skl_vfcos_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.cos.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_cosf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_cosf64(
; M1:  call <vscale x 1 x double> @skl_vfcos_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_cosf64(
; M2:  call <vscale x 2 x double> @skl_vfcos_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_cosf64(
; M4:  call <vscale x 4 x double> @skl_vfcos_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_cosf64(
; M8:  call <vscale x 8 x double> @skl_vfcos_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.cos.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_sinf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_sinf32(
; M1:  call <vscale x 2 x float> @skl_vfsin_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_sinf32(
; M2:  call <vscale x 4 x float> @skl_vfsin_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_sinf32(
; M4:  call <vscale x 8 x float> @skl_vfsin_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_sinf32(
; M8:  call <vscale x 16 x float> @skl_vfsin_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.sin.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_sinf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_sinf64(
; M1:  call <vscale x 1 x double> @skl_vfsin_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_sinf64(
; M2:  call <vscale x 2 x double> @skl_vfsin_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_sinf64(
; M4:  call <vscale x 4 x double> @skl_vfsin_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_sinf64(
; M8:  call <vscale x 8 x double> @skl_vfsin_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.sin.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_expf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_expf32(
; M1:  call <vscale x 2 x float> @skl_vfexp_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_expf32(
; M2:  call <vscale x 4 x float> @skl_vfexp_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_expf32(
; M4:  call <vscale x 8 x float> @skl_vfexp_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_expf32(
; M8:  call <vscale x 16 x float> @skl_vfexp_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.exp.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_expf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_expf64(
; M1:  call <vscale x 1 x double> @skl_vfexp_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_expf64(
; M2:  call <vscale x 2 x double> @skl_vfexp_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_expf64(
; M4:  call <vscale x 4 x double> @skl_vfexp_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_expf64(
; M8:  call <vscale x 8 x double> @skl_vfexp_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.exp.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_exp10f32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_exp10f32(
; M1:  call <vscale x 2 x float> @skl_vfexp10_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_exp10f32(
; M2:  call <vscale x 4 x float> @skl_vfexp10_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_exp10f32(
; M4:  call <vscale x 8 x float> @skl_vfexp10_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_exp10f32(
; M8:  call <vscale x 16 x float> @skl_vfexp10_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.exp10.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_exp10f64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_exp10f64(
; M1:  call <vscale x 1 x double> @skl_vfexp10_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_exp10f64(
; M2:  call <vscale x 2 x double> @skl_vfexp10_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_exp10f64(
; M4:  call <vscale x 4 x double> @skl_vfexp10_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_exp10f64(
; M8:  call <vscale x 8 x double> @skl_vfexp10_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.exp10.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_exp2f32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_exp2f32(
; M1:  call <vscale x 2 x float> @skl_vfexp2_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_exp2f32(
; M2:  call <vscale x 4 x float> @skl_vfexp2_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_exp2f32(
; M4:  call <vscale x 8 x float> @skl_vfexp2_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_exp2f32(
; M8:  call <vscale x 16 x float> @skl_vfexp2_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.exp2.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_exp2f64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_exp2f64(
; M1:  call <vscale x 1 x double> @skl_vfexp2_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_exp2f64(
; M2:  call <vscale x 2 x double> @skl_vfexp2_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_exp2f64(
; M4:  call <vscale x 4 x double> @skl_vfexp2_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_exp2f64(
; M8:  call <vscale x 8 x double> @skl_vfexp2_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.exp2.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_ldexpf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_ldexpf32(
; M1:  call <vscale x 2 x float> @skl_vfldexp_vv_f32m1({{<vscale x 2 x float> %.*, <vscale x 2 x i32> %.*, i32 %.*}})
; M2-LABEL: @test_llvm_ldexpf32(
; M2:  call <vscale x 4 x float> @skl_vfldexp_vv_f32m2({{<vscale x 4 x float> %.*, <vscale x 4 x i32> %.*, i32 %.*}})
; M4-LABEL: @test_llvm_ldexpf32(
; M4:  call <vscale x 8 x float> @skl_vfldexp_vv_f32m4({{<vscale x 8 x float> %.*, <vscale x 8 x i32> %.*, i32 %.*}})
; M8-LABEL: @test_llvm_ldexpf32(
; M8:  call <vscale x 16 x float> @skl_vfldexp_vv_f32m8({{<vscale x 16 x float> %.*, <vscale x 16 x i32> %.*, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = load i32, ptr %arrayidx2, align 4
  %2 = call float @llvm.ldexp.f32.i32(float %0, i32 %1)
  store float %2, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_ldexpf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_ldexpf64(
; M1:  call <vscale x 1 x double> @skl_vfldexp_vv_f64m1({{<vscale x 1 x double> %.*, <vscale x 1 x i32> %.*, i32 %.*}})
; M2-LABEL: @test_llvm_ldexpf64(
; M2:  call <vscale x 2 x double> @skl_vfldexp_vv_f64m2({{<vscale x 2 x double> %.*, <vscale x 2 x i32> %.*, i32 %.*}})
; M4-LABEL: @test_llvm_ldexpf64(
; M4:  call <vscale x 4 x double> @skl_vfldexp_vv_f64m4({{<vscale x 4 x double> %.*, <vscale x 4 x i32> %.*, i32 %.*}})
; M8-LABEL: @test_llvm_ldexpf64(
; M8:  call <vscale x 8 x double> @skl_vfldexp_vv_f64m8({{<vscale x 8 x double> %.*, <vscale x 8 x i32> %.*, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = load i32, ptr %arrayidx2, align 4
  %2 = call double @llvm.ldexp.f64.i32(double %0, i32 %1)
  store double %2, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
define void @test_llvm_logf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_logf32(
; M1:  call <vscale x 2 x float> @skl_vflog_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_logf32(
; M2:  call <vscale x 4 x float> @skl_vflog_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_logf32(
; M4:  call <vscale x 8 x float> @skl_vflog_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_logf32(
; M8:  call <vscale x 16 x float> @skl_vflog_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.log.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_logf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_logf64(
; M1:  call <vscale x 1 x double> @skl_vflog_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_logf64(
; M2:  call <vscale x 2 x double> @skl_vflog_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_logf64(
; M4:  call <vscale x 4 x double> @skl_vflog_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_logf64(
; M8:  call <vscale x 8 x double> @skl_vflog_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.log.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_log10f32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_log10f32(
; M1:  call <vscale x 2 x float> @skl_vflog10_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_log10f32(
; M2:  call <vscale x 4 x float> @skl_vflog10_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_log10f32(
; M4:  call <vscale x 8 x float> @skl_vflog10_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_log10f32(
; M8:  call <vscale x 16 x float> @skl_vflog10_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.log10.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_log10f64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_log10f64(
; M1:  call <vscale x 1 x double> @skl_vflog10_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_log10f64(
; M2:  call <vscale x 2 x double> @skl_vflog10_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_log10f64(
; M4:  call <vscale x 4 x double> @skl_vflog10_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_log10f64(
; M8:  call <vscale x 8 x double> @skl_vflog10_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.log10.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_log2f32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_log2f32(
; M1:  call <vscale x 2 x float> @skl_vflog2_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_log2f32(
; M2:  call <vscale x 4 x float> @skl_vflog2_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_log2f32(
; M4:  call <vscale x 8 x float> @skl_vflog2_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_log2f32(
; M8:  call <vscale x 16 x float> @skl_vflog2_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.log2.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_log2f64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_log2f64(
; M1:  call <vscale x 1 x double> @skl_vflog2_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_log2f64(
; M2:  call <vscale x 2 x double> @skl_vflog2_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_log2f64(
; M4:  call <vscale x 4 x double> @skl_vflog2_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_log2f64(
; M8:  call <vscale x 8 x double> @skl_vflog2_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.log2.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_powf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_powf32(
; M1:  call <vscale x 2 x float> @skl_vfpow_vv_f32m1({{(<vscale x 2 x float> %.*){2}, i32 %.*}})
; M2-LABEL: @test_llvm_powf32(
; M2:  call <vscale x 4 x float> @skl_vfpow_vv_f32m2({{(<vscale x 4 x float> %.*){2}, i32 %.*}})
; M4-LABEL: @test_llvm_powf32(
; M4:  call <vscale x 8 x float> @skl_vfpow_vv_f32m4({{(<vscale x 8 x float> %.*){2}, i32 %.*}})
; M8-LABEL: @test_llvm_powf32(
; M8:  call <vscale x 16 x float> @skl_vfpow_vv_f32m8({{(<vscale x 16 x float> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = load float, ptr %arrayidx2, align 4
  %2 = call float @llvm.pow.f32(float %0, float %1)
  store float %2, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_powf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_powf64(
; M1:  call <vscale x 1 x double> @skl_vfpow_vv_f64m1({{(<vscale x 1 x double> %.*){2}, i32 %.*}})
; M2-LABEL: @test_llvm_powf64(
; M2:  call <vscale x 2 x double> @skl_vfpow_vv_f64m2({{(<vscale x 2 x double> %.*){2}, i32 %.*}})
; M4-LABEL: @test_llvm_powf64(
; M4:  call <vscale x 4 x double> @skl_vfpow_vv_f64m4({{(<vscale x 4 x double> %.*){2}, i32 %.*}})
; M8-LABEL: @test_llvm_powf64(
; M8:  call <vscale x 8 x double> @skl_vfpow_vv_f64m8({{(<vscale x 8 x double> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = load double, ptr %arrayidx2, align 8
  %2 = call double @llvm.pow.f64(double %0, double %1)
  store double %2, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_tanf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_tanf32(
; M1:  call <vscale x 2 x float> @skl_vftan_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_tanf32(
; M2:  call <vscale x 4 x float> @skl_vftan_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_tanf32(
; M4:  call <vscale x 8 x float> @skl_vftan_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_tanf32(
; M8:  call <vscale x 16 x float> @skl_vftan_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.tan.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_tanf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_tanf64(
; M1:  call <vscale x 1 x double> @skl_vftan_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_tanf64(
; M2:  call <vscale x 2 x double> @skl_vftan_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_tanf64(
; M4:  call <vscale x 4 x double> @skl_vftan_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_tanf64(
; M8:  call <vscale x 8 x double> @skl_vftan_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.tan.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_asinf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_asinf32(
; M1:  call <vscale x 2 x float> @skl_vfasin_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_asinf32(
; M2:  call <vscale x 4 x float> @skl_vfasin_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_asinf32(
; M4:  call <vscale x 8 x float> @skl_vfasin_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_asinf32(
; M8:  call <vscale x 16 x float> @skl_vfasin_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.asin.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_asinf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_asinf64(
; M1:  call <vscale x 1 x double> @skl_vfasin_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_asinf64(
; M2:  call <vscale x 2 x double> @skl_vfasin_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_asinf64(
; M4:  call <vscale x 4 x double> @skl_vfasin_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_asinf64(
; M8:  call <vscale x 8 x double> @skl_vfasin_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.asin.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_acosf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_acosf32(
; M1:  call <vscale x 2 x float> @skl_vfacos_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_acosf32(
; M2:  call <vscale x 4 x float> @skl_vfacos_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_acosf32(
; M4:  call <vscale x 8 x float> @skl_vfacos_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_acosf32(
; M8:  call <vscale x 16 x float> @skl_vfacos_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.acos.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_acosf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_acosf64(
; M1:  call <vscale x 1 x double> @skl_vfacos_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_acosf64(
; M2:  call <vscale x 2 x double> @skl_vfacos_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_acosf64(
; M4:  call <vscale x 4 x double> @skl_vfacos_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_acosf64(
; M8:  call <vscale x 8 x double> @skl_vfacos_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.acos.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_atanf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_atanf32(
; M1:  call <vscale x 2 x float> @skl_vfatan_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_atanf32(
; M2:  call <vscale x 4 x float> @skl_vfatan_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_atanf32(
; M4:  call <vscale x 8 x float> @skl_vfatan_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_atanf32(
; M8:  call <vscale x 16 x float> @skl_vfatan_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.atan.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_atanf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_atanf64(
; M1:  call <vscale x 1 x double> @skl_vfatan_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_atanf64(
; M2:  call <vscale x 2 x double> @skl_vfatan_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_atanf64(
; M4:  call <vscale x 4 x double> @skl_vfatan_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_atanf64(
; M8:  call <vscale x 8 x double> @skl_vfatan_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.atan.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_sinhf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_sinhf32(
; M1:  call <vscale x 2 x float> @skl_vfsinh_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_sinhf32(
; M2:  call <vscale x 4 x float> @skl_vfsinh_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_sinhf32(
; M4:  call <vscale x 8 x float> @skl_vfsinh_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_sinhf32(
; M8:  call <vscale x 16 x float> @skl_vfsinh_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.sinh.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_sinhf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_sinhf64(
; M1:  call <vscale x 1 x double> @skl_vfsinh_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_sinhf64(
; M2:  call <vscale x 2 x double> @skl_vfsinh_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_sinhf64(
; M4:  call <vscale x 4 x double> @skl_vfsinh_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_sinhf64(
; M8:  call <vscale x 8 x double> @skl_vfsinh_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.sinh.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_coshf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_coshf32(
; M1:  call <vscale x 2 x float> @skl_vfcosh_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_coshf32(
; M2:  call <vscale x 4 x float> @skl_vfcosh_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_coshf32(
; M4:  call <vscale x 8 x float> @skl_vfcosh_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_coshf32(
; M8:  call <vscale x 16 x float> @skl_vfcosh_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.cosh.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_coshf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_coshf64(
; M1:  call <vscale x 1 x double> @skl_vfcosh_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_coshf64(
; M2:  call <vscale x 2 x double> @skl_vfcosh_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_coshf64(
; M4:  call <vscale x 4 x double> @skl_vfcosh_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_coshf64(
; M8:  call <vscale x 8 x double> @skl_vfcosh_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.cosh.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_tanhf32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_tanhf32(
; M1:  call <vscale x 2 x float> @skl_vftanh_v_f32m1(<vscale x 2 x float> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_tanhf32(
; M2:  call <vscale x 4 x float> @skl_vftanh_v_f32m2(<vscale x 4 x float> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_tanhf32(
; M4:  call <vscale x 8 x float> @skl_vftanh_v_f32m4(<vscale x 8 x float> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_tanhf32(
; M8:  call <vscale x 16 x float> @skl_vftanh_v_f32m8(<vscale x 16 x float> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = call float @llvm.tanh.f32(float %0)
  store float %1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_tanhf64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_tanhf64(
; M1:  call <vscale x 1 x double> @skl_vftanh_v_f64m1(<vscale x 1 x double> {{%.*}}, i32 {{%.*}})
; M2-LABEL: @test_llvm_tanhf64(
; M2:  call <vscale x 2 x double> @skl_vftanh_v_f64m2(<vscale x 2 x double> {{%.*}}, i32 {{%.*}})
; M4-LABEL: @test_llvm_tanhf64(
; M4:  call <vscale x 4 x double> @skl_vftanh_v_f64m4(<vscale x 4 x double> {{%.*}}, i32 {{%.*}})
; M8-LABEL: @test_llvm_tanhf64(
; M8:  call <vscale x 8 x double> @skl_vftanh_v_f64m8(<vscale x 8 x double> {{%.*}}, i32 {{%.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = call double @llvm.tanh.f64(double %0)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_atan2f32(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_atan2f32(
; M1:  call <vscale x 2 x float> @skl_vfatan2_vv_f32m1({{(<vscale x 2 x float> %.*){2}, i32 %.*}})
; M2-LABEL: @test_llvm_atan2f32(
; M2:  call <vscale x 4 x float> @skl_vfatan2_vv_f32m2({{(<vscale x 4 x float> %.*){2}, i32 %.*}})
; M4-LABEL: @test_llvm_atan2f32(
; M4:  call <vscale x 8 x float> @skl_vfatan2_vv_f32m4({{(<vscale x 8 x float> %.*){2}, i32 %.*}})
; M8-LABEL: @test_llvm_atan2f32(
; M8:  call <vscale x 16 x float> @skl_vfatan2_vv_f32m8({{(<vscale x 16 x float> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %1 = load float, ptr %arrayidx2, align 4
  %2 = call float @llvm.atan2.f32(float %0, float %1)
  store float %2, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @test_llvm_atan2f64(i32 %n, ptr noundef %a, ptr noundef %b) {
; M1-LABEL: @test_llvm_atan2f64(
; M1:  call <vscale x 1 x double> @skl_vfatan2_vv_f64m1({{(<vscale x 1 x double> %.*){2}, i32 %.*}})
; M2-LABEL: @test_llvm_atan2f64(
; M2:  call <vscale x 2 x double> @skl_vfatan2_vv_f64m2({{(<vscale x 2 x double> %.*){2}, i32 %.*}})
; M4-LABEL: @test_llvm_atan2f64(
; M4:  call <vscale x 4 x double> @skl_vfatan2_vv_f64m4({{(<vscale x 4 x double> %.*){2}, i32 %.*}})
; M8-LABEL: @test_llvm_atan2f64(
; M8:  call <vscale x 8 x double> @skl_vfatan2_vv_f64m8({{(<vscale x 8 x double> %.*){2}, i32 %.*}})
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %indvars.iv
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = load double, ptr %arrayidx2, align 8
  %2 = call double @llvm.atan2.f64(double %0, double %1)
  store double %2, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
