; REQUIRES: asserts
; RUN: opt -passes='loop-simplify,loop-distribute' -enable-loop-distribute \
; RUN:   -debug-only=loop-distribute -disable-output 2>&1 %s | FileCheck %s

; Do not pass this test through the auto gen scripts, hand modify instead.

%union.31.33 = type { %struct.30.32 }
%struct.30.32 = type { i64 }

@d = external global i16, align 1

define void @f(%union.31.33* %agg.result) {
; CHECK-LABEL: LDist: Checking a loop
; CHECK: LDist: Populated partitions:
; CHECK: LDist: Partition 0: (cycle)
; CHECK: LDist: Partition 1:

entry:
  %0 = bitcast %union.31.33* %agg.result to i32*
  br i1 undef, label %entry.split.us, label %entry.entry.split_crit_edge

entry.entry.split_crit_edge:                      ; preds = %entry
  br label %entry.split

entry.split.us:                                   ; preds = %entry
  br label %lbl1.us

lbl1.us:                                          ; preds = %cleanup4.us, %entry.split.us
  br label %cleanup4.loopexit.us

cleanup.cont.us:                                  ; No predecessors!
  br label %for.end.us

for.end.us:                                       ; preds = %cleanup.cont.us
  br label %cleanup4.us

cleanup4.loopexit.us:                             ; preds = %lbl1.us
  br label %cleanup4.us

cleanup4.us:                                      ; preds = %cleanup4.loopexit.us, %for.end.us
  br i1 undef, label %lbl1.us, label %return.us-lcssa.us

return.us-lcssa.us:                               ; preds = %cleanup4.us
  br label %return

entry.split:                                      ; preds = %entry.entry.split_crit_edge
  br label %lbl1

lbl1:                                             ; preds = %cleanup4, %entry.split
  %d.promoted = load i16, i16* @d, align 1
  br label %for.end

cleanup.cont:                                     ; No predecessors!
  br label %for.end

for.end:                                          ; preds = %cleanup.cont, %lbl1
  store i16 undef, i16* @d, align 1
  store i32 undef, i32* %0, align 1
  br label %cleanup4

cleanup4.loopexit:                                ; No predecessors!
  br label %cleanup4

cleanup4:                                         ; preds = %cleanup4.loopexit, %for.end
  %1 = phi i16 [ undef, %for.end ], [ 0, %cleanup4.loopexit ]
  br i1 false, label %lbl1, label %return.us-lcssa

return.us-lcssa:                                  ; preds = %cleanup4
  br label %return

return:                                           ; preds = %return.us-lcssa, %return.us-lcssa.us
  ret void
}
