; RUN: opt -S -loop-vectorize -mtriple=riscv64 -riscv-use-vla-vectorizer -mattr=+v -o - 2>&1 \
; RUN: -pass-remarks=loop-vectorize -vector-primary-lmul-max=3 < %s \
; RUN: | FileCheck %s --check-prefixes=CHECK

; Reduced case from 403.gcc
define void @convert_to_ssa(ptr %0, ptr %elms.i159, ptr %1) {
; CHECK: remark: <unknown>:0:0: vectorized loop ((lmul, type): (2, i64))
entry:
  br label %for.body.i172

for.body.i172:                                    ; preds = %for.body.i172, %entry
  %cp.010.i = phi ptr [ %incdec.ptr8.7.i, %for.body.i172 ], [ %0, %entry ]
  %bp.09.i2 = phi ptr [ %incdec.ptr7.7.i, %for.body.i172 ], [ null, %entry ]
  %ap.08.i = phi ptr [ %incdec.ptr.7.i, %for.body.i172 ], [ %1, %entry ]
  %niter.i = phi i32 [ %niter.next.7.i, %for.body.i172 ], [ 0, %entry ]
  %incdec.ptr.5.i3 = getelementptr inbounds i64, ptr %ap.08.i, i64 6
  %incdec.ptr.6.i = getelementptr inbounds i64, ptr %ap.08.i, i64 7
  %2 = load i64, ptr %elms.i159, align 8
  %incdec.ptr7.6.i = getelementptr inbounds i64, ptr %bp.09.i2, i64 7
  %3 = load i64, ptr %0, align 8
  %4 = load i64, ptr %cp.010.i, align 8
  %and.6.i = and i64 %3, %4
  %or.6.i = or i64 %and.6.i, %2
  store i64 %or.6.i, ptr %incdec.ptr.5.i3, align 8
  %incdec.ptr.7.i = getelementptr inbounds i64, ptr %ap.08.i, i64 8
  %incdec.ptr7.7.i = getelementptr inbounds i64, ptr %bp.09.i2, i64 8
  %5 = load i64, ptr %incdec.ptr7.6.i, align 8
  %incdec.ptr8.7.i = getelementptr inbounds i64, ptr %cp.010.i, i64 8
  store i64 %5, ptr %incdec.ptr.6.i, align 8
  %niter.next.7.i = add i32 %niter.i, 1
  %niter.ncmp.7.not.i = icmp eq i32 %niter.next.7.i, 0
  br i1 %niter.ncmp.7.not.i, label %for.end.loopexit.unr-lcssa.i177.loopexit, label %for.body.i172

for.end.loopexit.unr-lcssa.i177.loopexit:         ; preds = %for.body.i172
  ret void
}
