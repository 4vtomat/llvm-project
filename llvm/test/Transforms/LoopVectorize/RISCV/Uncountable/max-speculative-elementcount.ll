; RUN: opt < %s -S -riscv-use-vla-vectorizer -passes=loop-vectorize -mtriple=riscv64-unknown-linux-gnu -mattr=+v,+zvl512b -sifive-uncountable-loop-vectorization=stress 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt < %s -S -riscv-use-vla-vectorizer -passes=loop-vectorize -mtriple=riscv64-unknown-linux-gnu -mattr=+v,+zvl512b -sifive-uncountable-loop-vectorization=stress -sifive-loop-vectorizer-clamp-speculative-vl=0 2>&1 | FileCheck %s --check-prefix=VLMAX
; RUN: opt < %s -S -riscv-use-vla-vectorizer -passes=loop-vectorize -mtriple=riscv64-unknown-linux-gnu -mattr=+v,+zvl512b -sifive-uncountable-loop-vectorization=stress -sifive-loop-vectorizer-clamp-speculative-vl=8 2>&1 | FileCheck %s --check-prefix=VL8
; RUN: opt < %s -S -riscv-use-vla-vectorizer -passes=loop-vectorize -mtriple=riscv64-unknown-linux-gnu -mattr=+v,+zvl512b -sifive-uncountable-loop-vectorization=stress -sifive-loop-vectorizer-clamp-speculative-vl=8 -sifive-loop-vectorizer-clamp-vl=4 2>&1 | FileCheck %s --check-prefix=VL4

define i64 @strlen_i8(ptr %start) {
; DEFAULT-LABEL: define i64 @strlen_i8(
; DEFAULT:    [[TMP1:%.*]] = call i32 @llvm.experimental.get.vector.length.i64(i64 16, i32 8, i1 true)
;
; VLMAX-LABEL: define i64 @strlen_i8(
; VLMAX:    [[TMP1:%.*]] = call i32 @llvm.experimental.get.vector.length.i64(i64 -1, i32 8, i1 true)
;
; VL8-LABEL: define i64 @strlen_i8(
; VL8:    [[TMP1:%.*]] = call i32 @llvm.experimental.get.vector.length.i64(i64 8, i32 8, i1 true)
;
; VL4-LABEL: define i64 @strlen_i8(
; VL4:    [[TMP1:%.*]] = call i32 @llvm.experimental.get.vector.length.i64(i64 8, i32 8, i1 true)
;
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
