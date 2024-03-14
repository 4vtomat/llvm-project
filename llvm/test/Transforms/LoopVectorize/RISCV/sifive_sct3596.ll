; RUN: opt --passes='require<profile-summary>,loop-vectorize' -mcpu=sifive-p470 -mtriple=riscv64 -debug-only=loop-vectorize %s -disable-output 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: LV: Not vectorizing: Runtime ptr check is required with -Os/-Oz.

define i32 @test(ptr %coeff_cost, ptr %0) !prof !29 {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ %inc413, %for.body ], [ 0, %entry ]
  store i32 0, ptr %0, align 4
  store i32 0, ptr %coeff_cost, align 4
  %inc413 = add i32 %i, 1
  %cmp22412 = icmp ugt i32 %i, 6
  br i1 %cmp22412, label %for.end, label %for.body

for.end:
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 192109567147}
!4 = !{!"MaxCount", i64 29840623362}
!5 = !{!"MaxInternalCount", i64 10683596552}
!6 = !{!"MaxFunctionCount", i64 29840623362}
!7 = !{!"NumCounts", i64 7902}
!8 = !{!"NumFunctions", i64 519}
!9 = !{!"IsPartialProfile", i64 0}
!10 = !{!"PartialProfileRatio", double 0.000000e+00}
!11 = !{!"DetailedSummary", !12}
!12 = !{!13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28}
!13 = !{i32 10000, i64 29840623362, i32 1}
!14 = !{i32 100000, i64 29840623362, i32 1}
!15 = !{i32 200000, i64 17173614368, i32 2}
!16 = !{i32 300000, i64 16472354288, i32 3}
!17 = !{i32 400000, i64 10683596552, i32 5}
!18 = !{i32 500000, i64 10326023776, i32 6}
!19 = !{i32 600000, i64 1422314144, i32 19}
!20 = !{i32 700000, i64 727820082, i32 53}
!21 = !{i32 800000, i64 511088757, i32 61}
!22 = !{i32 900000, i64 179554560, i32 125}
!23 = !{i32 950000, i64 72423243, i32 215}
!24 = !{i32 990000, i64 8809264, i32 488}
!25 = !{i32 999000, i64 879760, i32 999}
!26 = !{i32 999900, i64 125258, i32 1456}
!27 = !{i32 999990, i64 29756, i32 1688}
!28 = !{i32 999999, i64 2258, i32 1864}
!29 = !{!"function_entry_count", i64 0}
