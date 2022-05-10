; RUN: opt -S -loop-vectorize -mtriple riscv64 -force-vector-width=8 -force-vector-interleave=2 -riscv-v-vector-bits-min=128 -mattr="+v" <%s | FileCheck %s
;
; CHECK: %{{.*}}= call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> {{.*}}), !dbg ![[DL:[0-9]+]]
; CHECK-NEXT: %{{.*}}= trunc i32{{.*}}, !dbg ![[DL]]
; CHECK-NEXT: %{{.*}}= add i16{{.*}}, !dbg ![[DL]]
; CHECK-NEXT: %{{.*}}= zext i16{{.*}}, !dbg ![[DL]]
; CHECK: ![[DL]] = !DILocation(line: 6, column: 5

; short foo(unsigned short *ptr, unsigned int start, int count) {
;    unsigned int iv = 0;
;    unsigned int sum = start;
;    unsigned short tmp;
;
;    while (iv != count) {
;       tmp = ptr[iv];
;       sum = (sum & 65535) + tmp;
;       iv = iv + 1;
;    }
;
;    return (unsigned short) sum;
;}

define dso_local signext i16 @foo(i16* nocapture readonly %ptr, i32 signext %start, i32 signext %count) local_unnamed_addr !dbg !8 {
entry:
  %cmp.not9 = icmp eq i32 %count, 0, !dbg !11
  %extract.t = trunc i32 %start to i16, !dbg !12
  br i1 %cmp.not9, label %while.end, label %while.body.preheader, !dbg !12

while.body.preheader:                             ; preds = %entry
  %0 = zext i32 %count to i64, !dbg !12
  br label %while.body, !dbg !12

while.body:                                       ; preds = %while.body.preheader, %while.body
  %indvars.iv = phi i64 [ 0, %while.body.preheader ], [ %indvars.iv.next, %while.body ]
  %sum.010 = phi i32 [ %start, %while.body.preheader ], [ %add, %while.body ]
  %arrayidx = getelementptr inbounds i16, i16* %ptr, i64 %indvars.iv, !dbg !13
  %1 = load i16, i16* %arrayidx, align 2, !dbg !13, !tbaa !14
  %and = and i32 %sum.010, 65535, !dbg !18
  %conv = zext i16 %1 to i32, !dbg !19
  %add = add nuw nsw i32 %and, %conv, !dbg !20
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !21
  %cmp.not = icmp eq i64 %indvars.iv.next, %0, !dbg !11
  br i1 %cmp.not, label %while.end.loopexit, label %while.body, !dbg !12, !llvm.loop !22

while.end.loopexit:                               ; preds = %while.body
  %extract.t12 = trunc i32 %add to i16, !dbg !12
  br label %while.end, !dbg !25

while.end:                                        ; preds = %while.end.loopexit, %entry
  %sum.0.lcssa.off0 = phi i16 [ %extract.t, %entry ], [ %extract.t12, %while.end.loopexit ]
  ret i16 %sum.0.lcssa.off0, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"SmallDataLimit", i32 8}
!7 = !{!""}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!9 = !DISubroutineType(types: !10)
!10 = !{}
!11 = !DILocation(line: 6, column: 15, scope: !8)
!12 = !DILocation(line: 6, column: 5, scope: !8)
!13 = !DILocation(line: 7, column: 8, scope: !8)
!14 = !{!15, !15, i64 0}
!15 = !{!"short", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C/C++ TBAA"}
!18 = !DILocation(line: 8, column: 13, scope: !8)
!19 = !DILocation(line: 8, column: 24, scope: !8)
!20 = !DILocation(line: 8, column: 22, scope: !8)
!21 = !DILocation(line: 9, column: 10, scope: !8)
!22 = distinct !{!22, !12, !23, !24}
!23 = !DILocation(line: 10, column: 5, scope: !8)
!24 = !{!"llvm.loop.mustprogress"}
!25 = !DILocation(line: 12, column: 5, scope: !8)
