; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -pass-remarks-analysis=loop-vectorize -S < %s 2>&1 | FileCheck %s

; CHECK: remark: <unknown>:0:0: loop remark: ignoring invalid vectorize_width value
define void @test1(i32* %a, i32* %b) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %0 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %0, 1
  %1 = add nuw nsw i64 %iv, 4
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %1
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 8
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.width", i32 5}
