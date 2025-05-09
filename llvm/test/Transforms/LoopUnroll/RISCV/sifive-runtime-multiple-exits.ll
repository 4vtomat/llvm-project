; RUN: opt -mtriple=riscv64 -mcpu=sifive-x280 -p loop-unroll -disable-output %s -debug-only=loop-unroll 2>&1 | FileCheck %s --check-prefixes=ALL,UNROLL
; RUN: opt -mtriple=riscv64 -mcpu=sifive-p470 -p loop-unroll -disable-output %s -debug-only=loop-unroll 2>&1 | FileCheck %s --check-prefixes=ALL,NOUNROLL
; REQUIRES: asserts

; We should unroll loops with multiple exits on in-order cores.

; ALL: Using prolog remainder.
; NOUNROLL: Multiple exit/exiting blocks in loop and multi-exit unrolling not enabled!
; UNROLL: COMPLETELY UNROLLING
; ALL: UNROLLING loop

define i1 @multi_exit_loop(i64 %coerce.val.pi.i) {
entry:
  br i1 false, label %entry.for.body.i.i.i_crit_edge, label %cleanup

entry.for.body.i.i.i_crit_edge:                   ; preds = %entry
  br label %for.body.i.i.i

for.body.i.i.i:                                   ; preds = %if.end.i.i.i, %entry.for.body.i.i.i_crit_edge
  %__trip_count.06.i.i.i = phi i64 [ %dec.i.i.i, %if.end.i.i.i ], [ %coerce.val.pi.i, %entry.for.body.i.i.i_crit_edge ]
  br i1 false, label %for.body.i.i.i.cleanup_crit_edge, label %if.end.i.i.i

for.body.i.i.i.cleanup_crit_edge:                 ; preds = %for.body.i.i.i
  br label %cleanup

if.end.i.i.i:                                     ; preds = %for.body.i.i.i
  %dec.i.i.i = add i64 %__trip_count.06.i.i.i, -1
  %cmp.i.i.i = icmp sgt i64 %__trip_count.06.i.i.i, 0
  br i1 %cmp.i.i.i, label %for.body.i.i.i, label %if.end.i.i.i.cleanup_crit_edge

if.end.i.i.i.cleanup_crit_edge:                   ; preds = %if.end.i.i.i
  br label %cleanup

cleanup:                                          ; preds = %if.end.i.i.i.cleanup_crit_edge, %for.body.i.i.i.cleanup_crit_edge, %entry
  ret i1 false
}

