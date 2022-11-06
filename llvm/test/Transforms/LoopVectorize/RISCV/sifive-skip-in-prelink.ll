; REQUIRES: asserts
; RUN: opt -passes="lto-pre-link<O3>" -mtriple=riscv64 -mattr=+v \
; RUN: -riscv-v-vector-bits-min=128 -sifive-vectorize-assume-optimizable-strided-accesses -debug -S %s 2>&1 | FileCheck %s
; RUN: opt -passes="lto-pre-link<O2>" -mtriple=riscv64 -mattr=+v \
; RUN: -riscv-v-vector-bits-min=128 -sifive-vectorize-assume-optimizable-strided-accesses -debug -S %s 2>&1 | FileCheck %s

; Skipping unroll in pre-link is controlled by option "sifive-vectorize-assume-optimizable-strided-accesses".
; This file tests `LoopVectorize.cpp::hasOnlyNonUnitStrideOrMemoryAccesses`.
; Please checkout SCT-1716 and the actual function for more detail.

; The IR is compiled by the following code:
;   void foo (int *A, int *B, int *C, int N) {
;     for (int I = 0; I < N; I += 2) {
;       C[I] = A[I] + B[I];
;     }
;   }

define void @foo(ptr %A, ptr %B, ptr %C, i32 %N) {
; CHECK: LV: Bail out in pre-link stage when there is only non-unit stride memory accesses.
entry:
  %cmp10 = icmp sgt i32 %N, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = sext i32 %N to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %2, %1
  %arrayidx4 = getelementptr inbounds i32, ptr %C, i64 %indvars.iv
  store i32 %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp slt i64 %indvars.iv.next, %0
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; The following IR is from compiling quantum_toffoli.

%struct.quantum_reg_struct = type { i32, i32, i32, ptr, ptr }
%struct.quantum_reg_node_struct = type { { float, float }, i64 }

define void @quantum_toffoli(i32%control1, i32 %control2, i32 %target, ptr %reg) {
; CHECK: LV: Bail out in pre-link stage when there is only non-unit stride memory accesses.
entry:
  %size = getelementptr inbounds %struct.quantum_reg_struct, ptr %reg, i64 0, i32 1
  %0 = load i32, ptr %size, align 4
  %cmp24 = icmp sgt i32 %0, 0
  br i1 %cmp24, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %node = getelementptr inbounds %struct.quantum_reg_struct, ptr %reg, i64 0, i32 3
  %1 = load ptr, ptr %node, align 8
  %sh_prom = zext i32 %control1 to i64
  %shl = shl i64 1, %sh_prom
  %sh_prom5 = zext i32 %control2 to i64
  %shl6 = shl i64 1, %sh_prom5
  %sh_prom10 = zext i32 %target to i64
  %shl11 = shl i64 1, %sh_prom10
  %wide.trip.count = zext i32 %0 to i64
  %2 = freeze i64 %shl6
  %3 = or i64 %shl, %2
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.inc ]
  %state = getelementptr inbounds %struct.quantum_reg_node_struct, ptr %1, i64 %indvars.iv, i32 1
  %4 = load i64, ptr %state, align 8
  %5 = and i64 %4, %3
  %.not = icmp eq i64 %5, %3
  br i1 %.not, label %if.then9, label %for.inc

if.then9:                                         ; preds = %for.body
  %xor = xor i64 %4, %shl11
  store i64 %xor, ptr %state, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}
