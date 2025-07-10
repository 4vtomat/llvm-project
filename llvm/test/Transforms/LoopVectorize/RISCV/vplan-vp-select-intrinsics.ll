; REQUIRES: asserts

 ; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
 ; RUN: -force-tail-folding-style=data-with-evl \
 ; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
 ; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output < %s 2>&1 | FileCheck --check-prefix=IF-EVL %s

 define void @vp_select(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
 ; IF-EVL: VPlan 'Initial VPlan for VF={1},UF={1}'
 ; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF={1}'
 ; IF-EVL: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI
 ;
 ; IF-EVL: VPlan 'Final VPlan for VF={vscale x 1,vscale x 2},UF={1}' {
 ; IF-EVL-NEXT: Live-in ir<[[VTC:%.+]]> = vector-trip-count
 ; IF-EVL-NEXT: Live-in ir<%N>.1 = original trip-count

 ; IF-EVL:      ir-bb<entry>:
 ; IF-EVL-NEXT: Successor(s): ir-bb<scalar.ph>, ir-bb<vector.ph>
 ; IF-EVL:      ir-bb<vector.ph>:
 ; IF-EVL-NEXT:   IR   %0 = call i32 @llvm.experimental.get.vector.length.i64(i64 %N, i32 2, i1 true)
 ; IF-EVL-NEXT: Successor(s): vector loop

 ; IF-EVL: <x1> vector loop: {
 ; IF-EVL-NEXT:   vector.body:
 ; IF-EVL-NEXT:     EMIT vp<[[IV:%.+]]> = phi ir<0>, vp<[[IV_NEXT_EXIT:%.+]]>
 ; IF-EVL-NEXT:     EMIT vp<[[EVL_PHI:%.+]]>  = phi ir<0>, vp<[[IV_NEX:%.+]]>
 ; IF-EVL-NEXT:     EMIT vp<[[AVL:%.+]]> = sub ir<%N>, vp<[[EVL_PHI]]>
 ; IF-EVL-NEXT:     EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[AVL]]>
 ; IF-EVL-NEXT:     CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[EVL_PHI]]>
 ; IF-EVL-NEXT:     vp<[[PTR1:%[0-9]+]]> = vector-pointer ir<[[GEP1]]>
 ; IF-EVL-NEXT:     WIDEN ir<[[LD1:%.+]]> = vp.load vp<[[PTR1]]>, vp<[[EVL]]>
 ; IF-EVL-NEXT:     CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[EVL_PHI]]>
 ; IF-EVL-NEXT:     vp<[[PTR2:%[0-9]+]]> = vector-pointer ir<[[GEP2]]>
 ; IF-EVL-NEXT:     WIDEN ir<[[LD2:%.+]]> = vp.load vp<[[PTR2]]>, vp<[[EVL]]>
 ; IF-EVL-NEXT:     WIDEN ir<[[CMP:%.+]]> = icmp sgt ir<[[LD1]]>, ir<[[LD2]]>
 ; IF-EVL-NEXT:     WIDEN ir<[[SUB:%.+]]> = sub ir<0>, ir<[[LD2]]>
 ; IF-EVL-NEXT:     WIDEN-SELECT ir<%cond.p> = select  ir<%cmp4>, ir<%2>, ir<%3>
 ; IF-EVL-NEXT:     WIDEN ir<%cond> = add ir<%cond.p>, ir<%1>
 ; IF-EVL-NEXT:     CLONE ir<%arrayidx15> = getelementptr inbounds ir<%a>, vp<%evl.based.iv>
 ; IF-EVL-NEXT:     vp<%4> = vector-pointer ir<%arrayidx15>
 ; IF-EVL-NEXT:     WIDEN vp.store vp<%4>, ir<%cond>, vp<%1>    unit-strided
 ; IF-EVL-NEXT:     EMIT vp<%5> = zext vp<%1> to i64
 ; IF-EVL-NEXT:     EMIT vp<%index.evl.next> = add nuw vp<%5>, vp<%evl.based.iv>
 ; IF-EVL-NEXT:     EMIT branch-on-count vp<%index.evl.next>, ir<%N> 
 ; IF-EVL-NEXT:   No successors
 ; IF-EVL-NEXT: }

 entry:
   br label %for.body

 for.body:
   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
   %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
   %0 = load i32, ptr %arrayidx, align 4
   %arrayidx3 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
   %1 = load i32, ptr %arrayidx3, align 4
   %cmp4 = icmp sgt i32 %0, %1
   %2 = sub i32 0, %1
   %cond.p = select i1 %cmp4, i32 %1, i32 %2
   %cond = add i32 %cond.p, %0
   %arrayidx15 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
   store i32 %cond, ptr %arrayidx15, align 4
   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
   %exitcond.not = icmp eq i64 %indvars.iv.next, %N
   br i1 %exitcond.not, label %exit, label %for.body

 exit:
   ret void
 }
