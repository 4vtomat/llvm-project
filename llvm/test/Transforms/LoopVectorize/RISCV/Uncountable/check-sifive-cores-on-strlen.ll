; RUN: opt -S -riscv-use-vla-vectorizer -passes=loop-vectorize  -mtriple=riscv64-unknown-linux-gnu -mcpu=sifive-x280 < %s -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1| FileCheck %s --check-prefix=NOVEC
; RUN: opt -S -riscv-use-vla-vectorizer -passes=loop-vectorize  -mtriple=riscv64-unknown-linux-gnu -mcpu=sifive-p470 < %s -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1| FileCheck %s --check-prefix=VEC
; RUN: opt -S -riscv-use-vla-vectorizer -passes=loop-vectorize  -mtriple=riscv64-unknown-linux-gnu -mcpu=sifive-p670 < %s -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>&1| FileCheck %s --check-prefix=VEC
define i64 @strlen_i8(ptr %start) {
; NOVEC: remark: <unknown>:0:0: loop not vectorized
; VEC: remark: <unknown>:0:0: vectorized uncountable loop
entry:
  br label %for.cond

for.cond:
  %end.0 = phi ptr [ %start, %entry ], [ %incdec.ptr, %for.cond ]
  %0 = load i8, ptr %end.0, align 1
  %cmp.not = icmp eq i8 %0, 0
  %incdec.ptr = getelementptr inbounds i8, ptr %end.0, i64 1
  br i1 %cmp.not, label %for.end, label %for.cond

for.end:
  %end.0.lcssa = phi ptr [ %end.0, %for.cond ]
  %sub.ptr.lhs.cast = ptrtoint ptr %end.0.lcssa to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %start to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  ret i64 %sub.ptr.sub
}

