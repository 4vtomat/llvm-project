; RUN: opt -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 \
; RUN:   -scalable-vectorization=off -S < %s | FileCheck %s
; RUN: opt -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 \
; RUN:   -scalable-vectorization=on -S < %s | FileCheck %s -check-prefix=SCALABLE

target triple = "riscv64"

define i32 @select_icmp(i32 %x, i32 %y, i32* nocapture readonly %c, i64 %n) #0 {
; CHECK-LABEL: @select_icmp
; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[X:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <4 x i32> poison, i32 [[Y:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT1]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 1)
; CHECK-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, i32* [[C:%.*]], i64 [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, i32* [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i32* [[TMP8]] to <4 x i32>*
; CHECK-NEXT:    [[VP_OP_LOAD:%.*]] = call <4 x i32> @llvm.vp.load.v4i32.p0v4i32(<4 x i32>* [[TMP9]], <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_ICMP:%.*]] = call <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32> [[VP_OP_LOAD]], <4 x i32> [[BROADCAST_SPLAT]], metadata !"slt", <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <4 x i32> @llvm.vp.select.v4i32(<4 x i1> [[VP_OP_ICMP]], <4 x i32> [[VEC_PHI]], <4 x i32> [[BROADCAST_SPLAT2]], i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_SELECT]] = call <4 x i32> @llvm.vp.merge.v4i32(<4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> [[VP_WIDEN_SELECT]], <4 x i32> [[VEC_PHI]], i32 [[TMP5]])
; CHECK-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[VP_OP_SELECT]], zeroinitializer
; CHECK-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 [[Y]], i32 0
;
; SCALABLE-LABEL: @select_icmp
; SCALABLE:       vector.ph:
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i32> poison, i32 [[X:%.*]], i32 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 2 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <vscale x 2 x i32> poison, i32 [[Y:%.*]], i32 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <vscale x 2 x i32> [[BROADCAST_SPLATINSERT1]], <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer
; SCALABLE-NEXT:    br label [[VECTOR_BODY:%.*]]
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 2 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; SCALABLE-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 0)
; SCALABLE-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; SCALABLE-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, i32* [[C:%.*]], i64 [[TMP6]]
; SCALABLE-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, i32* [[TMP7]], i32 0
; SCALABLE-NEXT:    [[TMP9:%.*]] = bitcast i32* [[TMP8]] to <vscale x 2 x i32>*
; SCALABLE-NEXT:    [[VP_OP_LOAD:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* [[TMP9]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_ICMP:%.*]] = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> [[VP_OP_LOAD]], <vscale x 2 x i32> [[BROADCAST_SPLAT]], metadata !"slt", <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <vscale x 2 x i32> @llvm.vp.select.nxv2i32(<vscale x 2 x i1> [[VP_OP_ICMP]], <vscale x 2 x i32> [[VEC_PHI]], <vscale x 2 x i32> [[BROADCAST_SPLAT2]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_SELECT]] = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), <vscale x 2 x i32> [[VP_WIDEN_SELECT]], <vscale x 2 x i32> [[VEC_PHI]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; SCALABLE-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; SCALABLE-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 2 x i32> [[VP_OP_SELECT]], zeroinitializer
; SCALABLE-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.nxv2i1(<vscale x 2 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 [[Y]], i32 0
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %a = phi i32 [ 0, %entry], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp slt i32 %0, %x
  %cond = select i1 %cmp1, i32 %a, i32 %y
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 %cond
}

define i32 @select_fcmp(float %x, i32 %y, float* nocapture readonly %c, i64 %n) #0 {
; CHECK-LABEL: @select_fcmp
; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x float> poison, float [[X:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x float> [[BROADCAST_SPLATINSERT]], <4 x float> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <4 x i32> poison, i32 [[Y:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT1]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 1)
; CHECK-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds float, float* [[C:%.*]], i64 [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds float, float* [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast float* [[TMP8]] to <4 x float>*
; CHECK-NEXT:    [[VP_OP_LOAD:%.*]] = call <4 x float> @llvm.vp.load.v4f32.p0v4f32(<4 x float>* [[TMP9]], <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_FCMP:%.*]] = call <4 x i1> @llvm.vp.fcmp.v4f32(<4 x float> [[VP_OP_LOAD]], <4 x float> [[BROADCAST_SPLAT]], metadata !"olt", <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <4 x i32> @llvm.vp.select.v4i32(<4 x i1> [[VP_OP_FCMP]], <4 x i32> [[VEC_PHI]], <4 x i32> [[BROADCAST_SPLAT2]], i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_SELECT]] = call <4 x i32> @llvm.vp.merge.v4i32(<4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> [[VP_WIDEN_SELECT]], <4 x i32> [[VEC_PHI]], i32 [[TMP5]])
; CHECK-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[VP_OP_SELECT]], zeroinitializer
; CHECK-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 [[Y]], i32 0
;
; SCALABLE-LABEL: @select_fcmp
; SCALABLE:       vector.ph:
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 2 x float> poison, float [[X:%.*]], i32 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 2 x float> [[BROADCAST_SPLATINSERT]], <vscale x 2 x float> poison, <vscale x 2 x i32> zeroinitializer
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <vscale x 2 x i32> poison, i32 [[Y:%.*]], i32 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <vscale x 2 x i32> [[BROADCAST_SPLATINSERT1]], <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer
; SCALABLE-NEXT:    br label [[VECTOR_BODY:%.*]]
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 2 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; SCALABLE-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 0)
; SCALABLE-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; SCALABLE-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP7:%.*]] = getelementptr inbounds float, float* [[C:%.*]], i64 [[TMP6]]
; SCALABLE-NEXT:    [[TMP8:%.*]] = getelementptr inbounds float, float* [[TMP7]], i32 0
; SCALABLE-NEXT:    [[TMP9:%.*]] = bitcast float* [[TMP8]] to <vscale x 2 x float>*
; SCALABLE-NEXT:    [[VP_OP_LOAD:%.*]] = call <vscale x 2 x float> @llvm.vp.load.nxv2f32.p0nxv2f32(<vscale x 2 x float>* [[TMP9]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_FCMP:%.*]] = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> [[VP_OP_LOAD]], <vscale x 2 x float> [[BROADCAST_SPLAT]], metadata !"olt", <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <vscale x 2 x i32> @llvm.vp.select.nxv2i32(<vscale x 2 x i1> [[VP_OP_FCMP]], <vscale x 2 x i32> [[VEC_PHI]], <vscale x 2 x i32> [[BROADCAST_SPLAT2]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_SELECT]] = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), <vscale x 2 x i32> [[VP_WIDEN_SELECT]], <vscale x 2 x i32> [[VEC_PHI]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; SCALABLE-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; SCALABLE-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 2 x i32> [[VP_OP_SELECT]], zeroinitializer
; SCALABLE-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.nxv2i1(<vscale x 2 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 [[Y]], i32 0
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %a = phi i32 [ 0, %entry], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %c, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp fast olt float %0, %x
  %cond = select i1 %cmp1, i32 %a, i32 %y
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret i32 %cond
}

define i32 @select_const_i32_from_icmp(i32* nocapture readonly %v, i64 %n) #0 {
; CHECK-LABEL: @select_const_i32_from_icmp
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ <i32 3, i32 3, i32 3, i32 3>, [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 1)
; CHECK-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, i32* [[V:%.*]], i64 [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, i32* [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i32* [[TMP8]] to <4 x i32>*
; CHECK-NEXT:    [[VP_OP_LOAD:%.*]] = call <4 x i32> @llvm.vp.load.v4i32.p0v4i32(<4 x i32>* [[TMP9]], <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_ICMP:%.*]] = call <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32> [[VP_OP_LOAD]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>, metadata !"eq", <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <4 x i32> @llvm.vp.select.v4i32(<4 x i1> [[VP_OP_ICMP]], <4 x i32> [[VEC_PHI]], <4 x i32> <i32 7, i32 7, i32 7, i32 7>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_SELECT]] = call <4 x i32> @llvm.vp.merge.v4i32(<4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> [[VP_WIDEN_SELECT]], <4 x i32> [[VEC_PHI]], i32 [[TMP5]])
; CHECK-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[VP_OP_SELECT]], <i32 3, i32 3, i32 3, i32 3>
; CHECK-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 7, i32 3
;
; SCALABLE-LABEL: @select_const_i32_from_icmp
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 2 x i32> [ shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 3, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer), [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; SCALABLE-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 0)
; SCALABLE-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; SCALABLE-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, i32* [[V:%.*]], i64 [[TMP6]]
; SCALABLE-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, i32* [[TMP7]], i32 0
; SCALABLE-NEXT:    [[TMP9:%.*]] = bitcast i32* [[TMP8]] to <vscale x 2 x i32>*
; SCALABLE-NEXT:    [[VP_OP_LOAD:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* [[TMP9]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_ICMP:%.*]] = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> [[VP_OP_LOAD]], <vscale x 2 x i32> shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 3, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer), metadata !"eq", <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <vscale x 2 x i32> @llvm.vp.select.nxv2i32(<vscale x 2 x i1> [[VP_OP_ICMP]], <vscale x 2 x i32> [[VEC_PHI]], <vscale x 2 x i32> shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 7, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_SELECT]] = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), <vscale x 2 x i32> [[VP_WIDEN_SELECT]], <vscale x 2 x i32> [[VEC_PHI]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; SCALABLE-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; SCALABLE-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP6:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 2 x i32> [[VP_OP_SELECT]], shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 3, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.nxv2i1(<vscale x 2 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 7, i32 3
;
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 3, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, i32* %v, i64 %0
  %3 = load i32, i32* %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 %1, i32 7
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}

define i32 @select_i32_from_icmp(i32* nocapture readonly %v, i32 %a, i32 %b, i64 %n) #0 {
; CHECK-LABEL: @select_i32_from_icmp
; CHECK:       vector.ph:
; CHECK-NEXT:    [[MINMAX_IDENT_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[A:%.*]], i32 0
; CHECK-NEXT:    [[MINMAX_IDENT_SPLAT:%.*]] = shufflevector <4 x i32> [[MINMAX_IDENT_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[B:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ [[MINMAX_IDENT_SPLAT]], [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 1)
; CHECK-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, i32* [[V:%.*]], i64 [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, i32* [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i32* [[TMP8]] to <4 x i32>*
; CHECK-NEXT:    [[VP_OP_LOAD:%.*]] = call <4 x i32> @llvm.vp.load.v4i32.p0v4i32(<4 x i32>* [[TMP9]], <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_ICMP:%.*]] = call <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32> [[VP_OP_LOAD]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>, metadata !"eq", <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <4 x i32> @llvm.vp.select.v4i32(<4 x i1> [[VP_OP_ICMP]], <4 x i32> [[VEC_PHI]], <4 x i32> [[BROADCAST_SPLAT]], i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_SELECT]] = call <4 x i32> @llvm.vp.merge.v4i32(<4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> [[VP_WIDEN_SELECT]], <4 x i32> [[VEC_PHI]], i32 [[TMP5]])
; CHECK-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP8:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[A]], i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <4 x i32> [[DOTSPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[VP_OP_SELECT]], [[DOTSPLAT]]
;
; SCALABLE-LABEL: @select_i32_from_icmp
; SCALABLE:       vector.ph:
; SCALABLE-NEXT:    [[MINMAX_IDENT_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i32> poison, i32 [[A:%.*]], i32 0
; SCALABLE-NEXT:    [[MINMAX_IDENT_SPLAT:%.*]] = shufflevector <vscale x 2 x i32> [[MINMAX_IDENT_SPLATINSERT]], <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer
; SCALABLE-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i32> poison, i32 [[B:%.*]], i32 0
; SCALABLE-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 2 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer
; SCALABLE-NEXT:    br label [[VECTOR_BODY:%.*]]
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 2 x i32> [ [[MINMAX_IDENT_SPLAT]], [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; SCALABLE-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 0)
; SCALABLE-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; SCALABLE-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, i32* [[V:%.*]], i64 [[TMP6]]
; SCALABLE-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, i32* [[TMP7]], i32 0
; SCALABLE-NEXT:    [[TMP9:%.*]] = bitcast i32* [[TMP8]] to <vscale x 2 x i32>*
; SCALABLE-NEXT:    [[VP_OP_LOAD:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* [[TMP9]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_ICMP:%.*]] = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> [[VP_OP_LOAD]], <vscale x 2 x i32> shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 3, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer), metadata !"eq", <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <vscale x 2 x i32> @llvm.vp.select.nxv2i32(<vscale x 2 x i1> [[VP_OP_ICMP]], <vscale x 2 x i32> [[VEC_PHI]], <vscale x 2 x i32> [[BROADCAST_SPLAT]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_SELECT]] = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), <vscale x 2 x i32> [[VP_WIDEN_SELECT]], <vscale x 2 x i32> [[VEC_PHI]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; SCALABLE-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; SCALABLE-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP8:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 2 x i32> poison, i32 [[A]], i32 0
; SCALABLE-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 2 x i32> [[DOTSPLATINSERT]], <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 2 x i32> [[VP_OP_SELECT]], [[DOTSPLAT]]
; SCALABLE-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.nxv2i1(<vscale x 2 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 [[B]], i32 [[A]]
;
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, i32* %v, i64 %0
  %3 = load i32, i32* %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select i1 %4, i32 %1, i32 %b
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}

define i32 @select_const_i32_from_fcmp(float* nocapture readonly %v, i64 %n) #0 {
; CHECK-LABEL: @select_const_i32_from_fcmp
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ <i32 2, i32 2, i32 2, i32 2>, [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 1)
; CHECK-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds float, float* [[V:%.*]], i64 [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds float, float* [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast float* [[TMP8]] to <4 x float>*
; CHECK-NEXT:    [[VP_OP_LOAD:%.*]] = call <4 x float> @llvm.vp.load.v4f32.p0v4f32(<4 x float>* [[TMP9]], <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_FCMP:%.*]] = call <4 x i1> @llvm.vp.fcmp.v4f32(<4 x float> [[VP_OP_LOAD]], <4 x float> <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>, metadata !"ueq", <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <4 x i32> @llvm.vp.select.v4i32(<4 x i1> [[VP_OP_FCMP]], <4 x i32> [[VEC_PHI]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_SELECT]] = call <4 x i32> @llvm.vp.merge.v4i32(<4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> [[VP_WIDEN_SELECT]], <4 x i32> [[VEC_PHI]], i32 [[TMP5]])
; CHECK-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP10:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[VP_OP_SELECT]], <i32 2, i32 2, i32 2, i32 2>
; CHECK-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 1, i32 2
;
; SCALABLE-LABEL: @select_const_i32_from_fcmp
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 2 x i32> [ shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 2, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer), [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; SCALABLE-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 0)
; SCALABLE-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; SCALABLE-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP7:%.*]] = getelementptr inbounds float, float* [[V:%.*]], i64 [[TMP6]]
; SCALABLE-NEXT:    [[TMP8:%.*]] = getelementptr inbounds float, float* [[TMP7]], i32 0
; SCALABLE-NEXT:    [[TMP9:%.*]] = bitcast float* [[TMP8]] to <vscale x 2 x float>*
; SCALABLE-NEXT:    [[VP_OP_LOAD:%.*]] = call <vscale x 2 x float> @llvm.vp.load.nxv2f32.p0nxv2f32(<vscale x 2 x float>* [[TMP9]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_FCMP:%.*]] = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> [[VP_OP_LOAD]], <vscale x 2 x float> shufflevector (<vscale x 2 x float> insertelement (<vscale x 2 x float> poison, float 3.000000e+00, i32 0), <vscale x 2 x float> poison, <vscale x 2 x i32> zeroinitializer), metadata !"ueq", <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <vscale x 2 x i32> @llvm.vp.select.nxv2i32(<vscale x 2 x i1> [[VP_OP_FCMP]], <vscale x 2 x i32> [[VEC_PHI]], <vscale x 2 x i32> shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 1, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_SELECT]] = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), <vscale x 2 x i32> [[VP_WIDEN_SELECT]], <vscale x 2 x i32> [[VEC_PHI]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[TMP10:%.*]] = zext i32 [[TMP5]] to i64
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP10]]
; SCALABLE-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; SCALABLE-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP10:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 2 x i32> [[VP_OP_SELECT]], shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 2, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer)
; SCALABLE-NEXT:    [[TMP12:%.*]] = call i1 @llvm.vector.reduce.or.nxv2i1(<vscale x 2 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP12]], i32 1, i32 2
;
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ 2, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds float, float* %v, i64 %0
  %3 = load float, float* %2, align 4
  %4 = fcmp fast ueq float %3, 3.0
  %5 = select i1 %4, i32 %1, i32 1
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}

define float @select_const_f32_from_icmp(i32* nocapture readonly %v, i64 %n) #0 {
; CHECK-LABEL: @select_const_f32_from_icmp
; CHECK-NOT: vector.body
;
; SCALABLE-LABEL: @select_const_f32_from_icmp
; SCALABLE-NOT: vector.body
;
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi fast float [ 3.0, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, i32* %v, i64 %0
  %3 = load i32, i32* %2, align 4
  %4 = icmp eq i32 %3, 3
  %5 = select fast i1 %4, float %1, float 7.0
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret float %5
}

define i32 @pred_select_const_i32_from_icmp(i32* noalias nocapture readonly %src1, i32* noalias nocapture readonly %src2, i64 %n) #0 {
; CHECK-LABEL: @pred_select_const_i32_from_icmp
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 1)
; CHECK-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, i32* [[SRC1:%.*]], i64 [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, i32* [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i32* [[TMP8]] to <4 x i32>*
; CHECK-NEXT:    [[VP_OP_LOAD:%.*]] = call <4 x i32> @llvm.vp.load.v4i32.p0v4i32(<4 x i32>* [[TMP9]], <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_ICMP:%.*]] = call <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32> [[VP_OP_LOAD]], <4 x i32> <i32 35, i32 35, i32 35, i32 35>, metadata !"sgt", <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[SRC2:%.*]], i64 [[TMP6]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <4 x i32>*
; CHECK-NEXT:    [[VP_OP_LOAD1:%.*]] = call <4 x i32> @llvm.vp.load.v4i32.p0v4i32(<4 x i32>* [[TMP12]], <4 x i1> [[VP_OP_ICMP]], i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_ICMP2:%.*]] = call <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32> [[VP_OP_LOAD1]], <4 x i32> <i32 2, i32 2, i32 2, i32 2>, metadata !"eq", <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <4 x i32> @llvm.vp.select.v4i32(<4 x i1> [[VP_OP_ICMP2]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32> [[VEC_PHI]], i32 [[TMP5]])
; CHECK-NEXT:    [[PRED_NOT:%.*]] = call <4 x i1> @llvm.vp.xor.v4i1(<4 x i1> [[VP_OP_ICMP]], <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 [[TMP5]])
; CHECK-NEXT:    [[PREDPHI:%.*]] = call <4 x i32> @llvm.vp.merge.v4i32(<4 x i1> [[VP_OP_ICMP]], <4 x i32> [[VP_WIDEN_SELECT]], <4 x i32> [[VEC_PHI]], i32 [[TMP5]])
; CHECK-NEXT:    [[VP_OP_SELECT]] = call <4 x i32> @llvm.vp.merge.v4i32(<4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> [[PREDPHI]], <4 x i32> [[VEC_PHI]], i32 [[TMP5]])
; CHECK-NEXT:    [[TMP13:%.*]] = zext i32 [[TMP5]] to i64
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP13]]
; CHECK-NEXT:    [[TMP14:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP14]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP12:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <4 x i32> [[VP_OP_SELECT]], zeroinitializer
; CHECK-NEXT:    [[TMP15:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[RDX_SELECT_CMP]])
; CHECK-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP15]], i32 1, i32 0
;
; SCALABLE-LABEL: @pred_select_const_i32_from_icmp
; SCALABLE:       vector.body:
; SCALABLE-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 2 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[VP_OP_SELECT:%.*]], [[VECTOR_BODY]] ]
; SCALABLE-NEXT:    [[TMP3:%.*]] = sub i64 [[N:%.*]], [[INDEX]]
; SCALABLE-NEXT:    [[TMP4:%.*]] = call i64 @llvm.riscv.vsetvli.i64(i64 [[TMP3]], i64 2, i64 0)
; SCALABLE-NEXT:    [[TMP5:%.*]] = trunc i64 [[TMP4]] to i32
; SCALABLE-NEXT:    [[TMP6:%.*]] = add i64 [[INDEX]], 0
; SCALABLE-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, i32* [[SRC1:%.*]], i64 [[TMP6]]
; SCALABLE-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, i32* [[TMP7]], i32 0
; SCALABLE-NEXT:    [[TMP9:%.*]] = bitcast i32* [[TMP8]] to <vscale x 2 x i32>*
; SCALABLE-NEXT:    [[VP_OP_LOAD:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* [[TMP9]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_ICMP:%.*]] = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> [[VP_OP_LOAD]], <vscale x 2 x i32> shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 35, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer), metadata !"sgt", <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[SRC2:%.*]], i64 [[TMP6]]
; SCALABLE-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[TMP10]], i32 0
; SCALABLE-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 2 x i32>*
; SCALABLE-NEXT:    [[VP_OP_LOAD1:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* [[TMP12]], <vscale x 2 x i1> [[VP_OP_ICMP]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_ICMP2:%.*]] = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> [[VP_OP_LOAD1]], <vscale x 2 x i32> shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 2, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer), metadata !"eq", <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_WIDEN_SELECT:%.*]] = call <vscale x 2 x i32> @llvm.vp.select.nxv2i32(<vscale x 2 x i1> [[VP_OP_ICMP2]], <vscale x 2 x i32> shufflevector (<vscale x 2 x i32> insertelement (<vscale x 2 x i32> poison, i32 1, i32 0), <vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer), <vscale x 2 x i32> [[VEC_PHI]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[PRED_NOT:%.*]] = call <vscale x 2 x i1> @llvm.vp.xor.nxv2i1(<vscale x 2 x i1> [[VP_OP_ICMP]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[TMP5]])
; SCALABLE-NEXT:    [[PREDPHI:%.*]] = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> [[VP_OP_ICMP]], <vscale x 2 x i32> [[VP_WIDEN_SELECT]], <vscale x 2 x i32> [[VEC_PHI]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[VP_OP_SELECT]] = call <vscale x 2 x i32> @llvm.vp.merge.nxv2i32(<vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), <vscale x 2 x i32> [[PREDPHI]], <vscale x 2 x i32> [[VEC_PHI]], i32 [[TMP5]])
; SCALABLE-NEXT:    [[TMP13:%.*]] = zext i32 [[TMP5]] to i64
; SCALABLE-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP13]]
; SCALABLE-NEXT:    [[TMP14:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; SCALABLE-NEXT:    br i1 [[TMP14]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP12:![0-9]+]]
; SCALABLE:       middle.block:
; SCALABLE-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <vscale x 2 x i32> [[VP_OP_SELECT]], zeroinitializer
; SCALABLE-NEXT:    [[TMP15:%.*]] = call i1 @llvm.vector.reduce.or.nxv2i1(<vscale x 2 x i1> [[RDX_SELECT_CMP]])
; SCALABLE-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP15]], i32 1, i32 0
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.013 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %r.012 = phi i32 [ %r.1, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %src1, i64 %i.013
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 35
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %src2, i64 %i.013
  %1 = load i32, i32* %arrayidx2, align 4
  %cmp3 = icmp eq i32 %1, 2
  %spec.select = select i1 %cmp3, i32 1, i32 %r.012
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %r.1 = phi i32 [ %r.012, %for.body ], [ %spec.select, %if.then ]
  %inc = add nuw nsw i64 %i.013, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.inc
  %r.1.lcssa = phi i32 [ %r.1, %for.inc ]
  ret i32 %r.1.lcssa
}

attributes #0 = { "target-features"="+f,+v" }
