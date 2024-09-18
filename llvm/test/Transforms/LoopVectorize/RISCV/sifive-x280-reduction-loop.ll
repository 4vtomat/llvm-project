; RUN:  sed 's/TC/9/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-9 %s
; RUN:  sed 's/TC/10/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-10 %s
; RUN:  sed 's/TC/48/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-48 %s
; RUN:  sed 's/TC/56/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-56 %s
; RUN:  sed 's/TC/128/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-128 %s
; RUN:  sed 's/TC/160/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-160 %s
; RUN:  sed 's/TC/512/g' %s | opt -passes=loop-vectorize -mtriple riscv64 -vector-primary-lmul-max=3 --mcpu=sifive-x280 -S \
; RUN: -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1 | FileCheck -check-prefix=CHECK-REMARK-512 %s

; FIXME: Need to tune the cost later to match the experiment result.

; In current experiment the LMUL choice for each trip count on x280 :
; Use Scalar for trip count between 2~9
; Use LMUL1 for trip count between 10~48
; Use LMUL2 for trip count between 56~128
; Use LMUL4 for trip count between 160~512
define float @foo(ptr nocapture noundef readonly %ptr) {
; CHECK-REMARK-9: remark: <unknown>:0:0: the cost-model indicates that vectorization is not beneficial
; CHECK-REMARK-10: remark: <unknown>:0:0: the cost-model indicates that vectorization is not beneficial
; CHECK-REMARK-48: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m1, float))
; CHECK-REMARK-56: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m4, float))
; CHECK-REMARK-128: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m8, float))
; CHECK-REMARK-160: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m4, float))
; CHECK-REMARK-512: remark: <unknown>:0:0: vectorized loop ((lmul, type): (m8, float))
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %add.lcssa = phi float [ %add, %for.body ]
  ret float %add.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %weighted_MSE.04 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %ptr.addr.03 = phi ptr [ %ptr, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds float, ptr %ptr.addr.03, i64 1
  %0 = load float, ptr %ptr.addr.03, align 4
  %add = fadd fast float %0, %weighted_MSE.04
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, TC
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

