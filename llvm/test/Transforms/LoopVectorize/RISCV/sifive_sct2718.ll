; RUN: opt -S -passes=loop-vectorize -mtriple riscv64-unknown-linux-gnu -riscv-use-vla-vectorizer -mcpu=sifive-x280n -debug-only=loop-vectorize %s 2>&1 | FileCheck %s --check-prefix=DEBUG-CHECK

; REQUIRES: asserts

; DEBUG-CHECK: LV: Cannot adjust ordered recurrences. Constructed VPlan is rejected

define i32 @test() {
entry:
  br label %do.body

do.body:
  %g.0 = phi i32 [ 0, %entry ], [ %or7, %do.body ]
  %f.0 = phi i32 [ 0, %entry ], [ %xor96, %do.body ]
  %iters.0 = phi i64 [ 1, %entry ], [ %dec, %do.body ]
  %s.addr.0 = phi ptr [ null, %entry ], [ %add.ptr104, %do.body ]
  %add.ptr78 = getelementptr i8, ptr %s.addr.0, i64 16
  %add.ptr78.val = load i32, ptr %add.ptr78, align 1
  %xor96 = xor i32 %g.0, %add.ptr78.val
  %or7 = tail call i32 @llvm.bswap.i32(i32 %f.0)
  %add.ptr104 = getelementptr i8, ptr %s.addr.0, i64 20
  %dec = add i64 %iters.0, 1
  %cmp106.not = icmp eq i64 %iters.0, 0
  br i1 %cmp106.not, label %do.end107, label %do.body

do.end107:
  ret i32 %g.0
}

declare i32 @llvm.bswap.i32(i32) #0
