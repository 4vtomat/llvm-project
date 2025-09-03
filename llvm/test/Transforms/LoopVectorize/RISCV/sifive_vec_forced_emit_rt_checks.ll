; RUN: opt -mtriple=riscv64-unknown-linux-gnu -passes="loop-vectorize" -vector-primary-lmul-max=1 -mattr="+v" -debug -S -vectorize-memory-check-threshold=0 -pass-remarks-missed='loop-vectorize'  %s 2>&1 | FileCheck %s --check-prefix=VEC-NOT-FORCED
; RUN: opt -mtriple=riscv64-unknown-linux-gnu -passes="loop-vectorize" -vector-primary-lmul-max=1 -mattr="+v" -debug -S -vectorize-memory-check-threshold=0 -pass-remarks='loop-vectorize' -force-vectorization %s 2>&1 | FileCheck %s --check-prefix=VEC-FORCED
; REQUIRES: asserts

; VEC-NOT-FORCED: number of checks exceeded threshold
; VEC-NOT-FORCED: remark: <unknown>:0:0: loop not vectorized

; VEC-FORCED: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m2, i32))
; VEC-FORCED:       vector.memcheck:
; VEC-FORCED-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; VEC-FORCED-NEXT:    [[TMP1:%.*]] = mul nuw i64 [[TMP0]], 4
; VEC-FORCED-NEXT:    [[TMP2:%.*]] = mul i64 [[TMP1]], 4
; VEC-FORCED-NEXT:    [[TMP3:%.*]] = sub i64 {{.*}}, {{.*}}
; VEC-FORCED-NEXT:    [[DIFF_CHECK:%.*]] = icmp ult i64 [[TMP3]], [[TMP2]]
; VEC-FORCED-NEXT:    br i1 [[DIFF_CHECK]], label {{.*}}, label [[VECTOR_PH:%.*]]

define void @test(i32 %n, ptr %a, ptr %b) {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
