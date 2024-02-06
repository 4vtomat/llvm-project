; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mcpu=sifive-x280 -disable-output -debug-only=vplan-cost-model %s -mtriple riscv64 2>&1 | FileCheck %s --check-prefix VPLAN-CM-X280
; RUN: opt -passes=loop-vectorize -mcpu=sifive-p470 -disable-output -debug-only=vplan-cost-model %s -mtriple riscv64 2>&1 | FileCheck %s --check-prefix VPLAN-CM-P470
; RUN: opt -passes=loop-vectorize -mcpu=sifive-p670 -disable-output -debug-only=vplan-cost-model %s -mtriple riscv64 2>&1 | FileCheck %s --check-prefix VPLAN-CM-P670

; VPLAN-CM-X280: VPlanCM: cost 1 for RVL (mf2, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-X280: VPlanCM: cost 10 for RVL (mf2, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-X280: VPlanCM: cost 9 for RVL (mf2, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; VPLAN-CM-X280: VPlanCM: cost 1 for RVL (m1, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-X280: VPlanCM: cost 19 for RVL (m1, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-X280: VPlanCM: cost 9 for RVL (m1, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; VPLAN-CM-X280: VPlanCM: cost 1 for RVL (m2, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-X280: VPlanCM: cost 37 for RVL (m2, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-X280: VPlanCM: cost 9 for RVL (m2, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; VPLAN-CM-X280: VPlanCM: cost 1 for RVL (m4, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-X280: VPlanCM: cost 73 for RVL (m4, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-X280: VPlanCM: cost 9 for RVL (m4, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>

; VPLAN-CM-P470: VPlanCM: cost 1 for RVL (mf2, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-P470: VPlanCM: cost 3 for RVL (mf2, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-P470: VPlanCM: cost 9 for RVL (mf2, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; VPLAN-CM-P470: VPlanCM: cost 1 for RVL (m1, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-P470: VPlanCM: cost 3 for RVL (m1, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-P470: VPlanCM: cost 9 for RVL (m1, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; VPLAN-CM-P470: VPlanCM: cost 1 for RVL (m2, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-P470: VPlanCM: cost 5 for RVL (m2, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-P470: VPlanCM: cost 9 for RVL (m2, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; VPLAN-CM-P470: VPlanCM: cost 1 for RVL (m4, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-P470: VPlanCM: cost 9 for RVL (m4, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-P470: VPlanCM: cost 9 for RVL (m4, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>

; VPLAN-CM-P670: VPlanCM: cost 1 for RVL (mf2, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-P670: VPlanCM: cost 3 for RVL (mf2, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-P670: VPlanCM: cost 9 for RVL (mf2, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; VPLAN-CM-P670: VPlanCM: cost 1 for RVL (m1, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-P670: VPlanCM: cost 3 for RVL (m1, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-P670: VPlanCM: cost 9 for RVL (m1, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; VPLAN-CM-P670: VPlanCM: cost 1 for RVL (m2, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-P670: VPlanCM: cost 5 for RVL (m2, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-P670: VPlanCM: cost 9 for RVL (m2, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>
; VPLAN-CM-P670: VPlanCM: cost 1 for RVL (m4, i32) for VPInstruction: EMIT ir<%ret.011> = monotonic-phi ir<0>, ir<%inc>
; VPLAN-CM-P670: VPlanCM: cost 9 for RVL (m4, i32) for VPInstruction: WIDEN store vp<%14>, ir<%0>, vp<%11>
; VPLAN-CM-P670: VPlanCM: cost 9 for RVL (m4, i32) for VPInstruction: monotonic-update ir<%inc> = add ir<%ret.011>, ir<1> @vp<%11>

define i32 @compress_store(i32 %n, ptr noalias %a, ptr noalias %b) {
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:
  %ret.0.lcssa = phi i32 [ 0, %entry ], [ %ret.1, %for.inc ]
  ret i32 %ret.0.lcssa

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %ret.011 = phi i32 [ 0, %for.body.preheader ], [ %ret.1, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:
  %idx.ext = sext i32 %ret.011 to i64
  %add.ptr = getelementptr inbounds i32, ptr %a, i64 %idx.ext
  store i32 %0, ptr %add.ptr, align 4
  %inc = add nsw i32 %ret.011, 1
  br label %for.inc

for.inc:
  %ret.1 = phi i32 [ %inc, %if.then ], [ %ret.011, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
