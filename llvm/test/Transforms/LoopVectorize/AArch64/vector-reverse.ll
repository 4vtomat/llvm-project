; Test VLA for reverse with fixed size vector
; This is the loop in c++ being vectorize in this file with
; shuffle reverse
;  #pragma clang loop vectorize_width(8, fixed)
;  for (int i = N-1; i >= 0; --i)
;    a[i] = b[i] + 1.0;

; RUN: opt -passes=loop-vectorize,dce  -mtriple aarch64-linux-gnu -S \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue < %s | FileCheck %s

define void @vector_reverse_f64(i64 %N, ptr %a, ptr %b) #0 {
; CHECK-LABEL: vector_reverse_f64
; CHECK-LABEL: vector.body
; CHECK: %[[GEP:.*]] = getelementptr inbounds double, ptr %{{.*}}, i32 0
; CHECK-NEXT: %[[GEP1:.*]] = getelementptr inbounds double, ptr %[[GEP]], i32 -7
; CHECK-NEXT: %[[WIDE:.*]] = load <8 x double>, ptr %[[GEP1]], align 8
; CHECK-NEXT: %[[FADD:.*]] = fadd <8 x double> %[[WIDE]]
; CHECK-NEXT: %[[GEP2:.*]] = getelementptr inbounds double, ptr {{.*}}, i64 {{.*}}
; CHECK-NEXT: %[[GEP3:.*]] = getelementptr inbounds double, ptr %[[GEP2]], i32 0
; CHECK-NEXT: %[[GEP4:.*]] = getelementptr inbounds double, ptr %[[GEP3]], i32 -7
; CHECK-NEXT:  store <8 x double> %[[FADD]], ptr %[[GEP4]], align 8

entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.08.in = phi i64 [ %i.08, %for.body ], [ %N, %entry ]
  %i.08 = add nsw i64 %i.08.in, -1
  %arrayidx = getelementptr inbounds double, ptr %b, i64 %i.08
  %0 = load double, ptr %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx1 = getelementptr inbounds double, ptr %a, i64 %i.08
  store double %add, ptr %arrayidx1, align 8
  %cmp = icmp sgt i64 %i.08.in, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

define void @vector_reverse_i64(i64 %N, ptr %a, ptr %b) #0 {
; CHECK-LABEL: vector_reverse_i64
; CHECK-LABEL: vector.body
; CHECK: %[[GEP:.*]] = getelementptr inbounds i64, ptr %{{.*}}, i32 0
; CHECK-NEXT: %[[GEP1:.*]] = getelementptr inbounds i64, ptr %[[GEP]], i32 -7
; CHECK-NEXT: %[[WIDE:.*]] = load <8 x i64>, ptr %[[GEP1]], align 8
; CHECK-NEXT: %[[FADD:.*]] = add <8 x i64> %[[WIDE]]
; CHECK-NEXT: %[[GEP2:.*]] = getelementptr inbounds i64, ptr {{.*}}, i64 {{.*}}
; CHECK-NEXT: %[[GEP3:.*]] = getelementptr inbounds i64, ptr %[[GEP2]], i32 0
; CHECK-NEXT: %[[GEP4:.*]] = getelementptr inbounds i64, ptr %[[GEP3]], i32 -7
; CHECK-NEXT:  store <8 x i64> %[[FADD]], ptr %[[GEP4]], align 8

entry:
  %cmp8 = icmp sgt i64 %N, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.09.in = phi i64 [ %i.09, %for.body ], [ %N, %entry ]
  %i.09 = add nsw i64 %i.09.in, -1
  %arrayidx = getelementptr inbounds i64, ptr %b, i64 %i.09
  %0 = load i64, ptr %arrayidx, align 8
  %add = add i64 %0, 1
  %arrayidx2 = getelementptr inbounds i64, ptr %a, i64 %i.09
  store i64 %add, ptr %arrayidx2, align 8
  %cmp = icmp sgt i64 %i.09.in, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

attributes #0 = { "target-cpu"="generic" "target-features"="+neon,+sve" }

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 8}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
