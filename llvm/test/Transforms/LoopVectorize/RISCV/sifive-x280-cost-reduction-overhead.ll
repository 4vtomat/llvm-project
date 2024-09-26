; RUN:  sed 's/TC/2/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-2 %s
; RUN:  sed 's/TC/8/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-8 %s
; RUN:  sed 's/TC/16/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-16 %s
; RUN:  sed 's/TC/32/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-32 %s
; RUN:  sed 's/TC/64/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-64 %s
; RUN:  sed 's/TC/128/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-128 %s

; FIXME: Need to tune the cost later to match the experiment result.
; In current experiment the LMUL choice for each trip count on x280 :
; Use Scalar for trip count between 2~5
; Use LMUL1 for trip count between 6~16
; Use LMUL2 for trip count between 24~64
; Use LMUL4 for trip count between 56~512
define internal float @foo(ptr nocapture noundef readonly %ptr, ptr nocapture noundef readonly %ptr2) {
; CHECK-REMARK-2: remark: <unknown>:0:0: loop not vectorized
; CHECK-REMARK-8: remark: <unknown>:0:0: loop not vectorized
; CHECK-REMARK-16: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m1, float))
; CHECK-REMARK-32: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m2, float))
; CHECK-REMARK-64: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m4, float))
; CHECK-REMARK-128: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m8, float))
entry:
  %profile_size = add i32 0, TC
  %cmp5 = icmp sgt i32 %profile_size, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add.lcssa = phi float [ %add, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %weighted_MSE.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret float %weighted_MSE.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %ptr.addr.09 = phi ptr [ %incdec.ptr, %for.body ], [ %ptr, %for.body.preheader ]
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %weighted_MSE.07 = phi float [ %add, %for.body ], [ 0.000000e+00, %for.body.preheader ]
  %ptr2.addr.06 = phi ptr [ %incdec.ptr1, %for.body ], [ %ptr2, %for.body.preheader ]
  %incdec.ptr = getelementptr inbounds float, ptr %ptr.addr.09, i64 1
  %0 = load float, ptr %ptr.addr.09, align 4
  %incdec.ptr1 = getelementptr inbounds float, ptr %ptr2.addr.06, i64 1
  %1 = load float, ptr %ptr2.addr.06, align 4
  %sub = fsub fast float %0, %1
  %mul = fmul fast float %sub, %sub
  %add = fadd fast float %mul, %weighted_MSE.07
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond.not = icmp eq i32 %inc, %profile_size
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
