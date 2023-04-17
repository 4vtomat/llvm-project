; RUN: opt < %s -mtriple riscv64 -riscv-v-vector-bits-min=256 -mattr="+v" -passes=loop-vectorize -riscv-use-vla-vectorizer -force-vector-width=4 -scalable-vectorization=off -S -pass-remarks-analysis=loop-vectorize  2>&1 | FileCheck %s

; void test(int *A, int Length) {
;   for (int i = 0; i < Length; i++)
;     A[i] = i;
; }
; CHECK: warning: <unknown>:0:0: ignoring user-specified vector width because RVV VLA vectorization was enabled. Consider using '#pragma clang rvv lmul_sew(LMUL, SEW)' instead
define void @test(ptr nocapture %A, i32 %Length) {
entry:
  %cmp4 = icmp sgt i32 %Length, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %Length
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

