; RUN: opt -passes='loop-vectorize' -S -mtriple riscv64-unknown-linux-gnu -riscv-use-vla-vectorizer -mcpu=sifive-x280n %s -pass-remarks-missed='loop-vectorize' 2>&1 | FileCheck %s

; NOTE: May need to be removed or updated after SCT-1963

; CHECK: remark: <unknown>:0:0: the cost-model indicates that vectorization is not beneficial
; CHECK: remark: <unknown>:0:0: the cost-model indicates that interleaving is not beneficial

define dso_local void @_Z3fooijPhS_(i32 %channels, i32 %alpha, ptr %other_row, ptr %this_row) {
entry:
  %sub = sub i32 255, %alpha
  %cmp14 = icmp sgt i32 %channels, 0
  br i1 %cmp14, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %channels to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %other_row, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %mul = mul i32 %conv, %alpha
  %arrayidx2 = getelementptr inbounds i8, ptr %this_row, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  %mul4 = mul i32 %sub, %conv3
  %add = add i32 %mul4, %mul
  %div = udiv i32 %add, 255
  %conv5 = trunc i32 %div to i8
  store i8 %conv5, ptr %arrayidx2, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
