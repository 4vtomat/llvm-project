; REQUIRES: asserts
; RUN: opt < %s -passes=slp-vectorizer -S -mtriple=riscv64-unknown-linux-gnu \
; RUN: -mcpu=sifive-x280o -debug-only=SLP 2>&1 | FileCheck %s

; CHECK-NOT: SLP: vectorized
%struct.ImageParameters = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [9 x [16 x [16 x i16]]], [5 x [16 x [16 x i16]]], [9 x [8 x [8 x i16]]], [2 x [4 x [16 x [16 x i16]]]], [16 x [16 x i16]], [16 x [16 x i32]], ptr, ptr, ptr, ptr, ptr, [1200 x %struct.syntaxelement], ptr, ptr, i32, i32, i32, i32, [4 x [4 x i32]], i32, i32, i32, i32, i32, double, i32, i32, i32, i32, ptr, ptr, ptr, ptr, [15 x i16], i32, i32, i32, i32, i32, i32, i32, i32, [6 x [15 x i32]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [1 x i32], i32, i32, [2 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, double, i32, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, i32, i32 }
%struct.syntaxelement = type { i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr }
%struct.macroblock = type { i32, i32, i32, i32, i32, [8 x i32], ptr, ptr, i32, [2 x [4 x [4 x [2 x i32]]]], [16 x i32], [16 x i32], i32, i64, [4 x i32], [4 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, double, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.pic_parameter_set_rbsp_t = type { i32, i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, [8 x i32], [8 x i32], [8 x i32], i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.InputParameters = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [8 x [2 x i32]], [8 x [2 x i32]], i32, i32, i32, i32, [200 x i8], [200 x i8], [200 x i8], [200 x i8], [200 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [1024 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, double, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [200 x i8], [200 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [200 x i8], i32, i32, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [6 x double], [200 x i8], i32 }
%struct.storable_picture = type { i32, i32, i32, i32, i32, i32, [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32 }

@input = external global ptr
@BlockSAD = external global ptr
@img = external global ptr
@search_setup_done = external global ptr
@search_center_x = external global ptr
@search_center_y = external global ptr
@pos_00 = external global ptr
@max_search_range = external global ptr
@active_pps = external global ptr
@listX = external global [6 x ptr]
@enc_picture = external global ptr
@imgY_org = external global ptr
@PelYline_11 = external global ptr
@spiral_search_x = external global ptr
@spiral_search_y = external global ptr
@byte_abs = external global ptr

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @SetupLargerBlocks(i32 signext, i32 signext, i32 signext)

define void @SetupFastFullPelSearch(i16 signext %ref, i32 signext %list) {
entry:
  %pmv = alloca [2 x i16]
  %orig_blocks = alloca [256 x i16]
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %pmv)
  call void @llvm.lifetime.start.p0(i64 512, ptr nonnull %orig_blocks)
  %0 = load ptr, ptr @BlockSAD
  %idxprom = sext i32 %list to i64
  %arrayidx = getelementptr inbounds ptr, ptr %0, i64 %idxprom
  %1 = load ptr, ptr %arrayidx
  %idxprom1 = sext i16 %ref to i64
  %arrayidx2 = getelementptr inbounds ptr, ptr %1, i64 %idxprom1
  %2 = load ptr, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds ptr, ptr %2, i64 7
  %3 = load ptr, ptr %arrayidx3
  %4 = load ptr, ptr @max_search_range
  %arrayidx5 = getelementptr inbounds ptr, ptr %4, i64 %idxprom
  %5 = load ptr, ptr %arrayidx5
  %arrayidx7 = getelementptr inbounds i32, ptr %5, i64 %idxprom1
  %6 = load i32, ptr %arrayidx7
  %mul = shl nsw i32 %6, 1
  %add = or i32 %mul, 1
  %mul10 = mul i32 %add, %add
  %7 = load ptr, ptr @img
  %MbaffFrameFlag = getelementptr inbounds %struct.ImageParameters, ptr %7, i64 0, i32 90
  %8 = load i32, ptr %MbaffFrameFlag
  %tobool.not = icmp eq i32 %8, 0
  br i1 %tobool.not, label %cond.end, label %land.lhs.true

land.lhs.true:
  %mb_data = getelementptr inbounds %struct.ImageParameters, ptr %7, i64 0, i32 51
  %9 = load ptr, ptr %mb_data
  %current_mb_nr = getelementptr inbounds %struct.ImageParameters, ptr %7, i64 0, i32 3
  %10 = load i32, ptr %current_mb_nr
  %idxprom11 = sext i32 %10 to i64
  %mb_field = getelementptr inbounds %struct.macroblock, ptr %9, i64 %idxprom11, i32 22
  %11 = load i32, ptr %mb_field
  %tobool13.not = icmp eq i32 %11, 0
  br i1 %tobool13.not, label %cond.end, label %cond.true

cond.true:
  %12 = and i32 %10, 1
  %tobool15.not = icmp eq i32 %12, 0
  %cond = select i1 %tobool15.not, i32 2, i32 4
  br label %cond.end

cond.end:
  %cond16 = phi i32 [ %cond, %cond.true ], [ 0, %land.lhs.true ], [ 0, %entry ]
  %13 = load ptr, ptr @active_pps
  %weighted_pred_flag = getelementptr inbounds %struct.pic_parameter_set_rbsp_t, ptr %13, i64 0, i32 19
  %14 = load i32, ptr %weighted_pred_flag
  %tobool17.not = icmp eq i32 %14, 0
  br i1 %tobool17.not, label %lor.rhs, label %land.lhs.true18

land.lhs.true18:
  %type = getelementptr inbounds %struct.ImageParameters, ptr %7, i64 0, i32 6
  %15 = load i32, ptr %type
  switch i32 %15, label %lor.rhs [
    i32 0, label %lor.end.thread
    i32 3, label %lor.end.thread
  ]

lor.rhs:
  %weighted_bipred_idc = getelementptr inbounds %struct.pic_parameter_set_rbsp_t, ptr %13, i64 0, i32 20
  %16 = load i32, ptr %weighted_bipred_idc
  %tobool21.not = icmp eq i32 %16, 0
  br i1 %tobool21.not, label %lor.end.thread171, label %lor.end

lor.end.thread171:
  %add24172 = add nsw i32 %cond16, %list
  %idxprom25173 = sext i32 %add24172 to i64
  %arrayidx26174 = getelementptr inbounds [6 x ptr], ptr @listX, i64 0, i64 %idxprom25173
  %17 = load ptr, ptr %arrayidx26174
  %arrayidx28175 = getelementptr inbounds ptr, ptr %17, i64 %idxprom1
  %18 = load ptr, ptr %arrayidx28175
  br label %if.else

lor.end.thread:
  %add24167 = add nsw i32 %cond16, %list
  %idxprom25168 = sext i32 %add24167 to i64
  %arrayidx26169 = getelementptr inbounds [6 x ptr], ptr @listX, i64 0, i64 %idxprom25168
  %19 = load ptr, ptr %arrayidx26169
  %arrayidx28170 = getelementptr inbounds ptr, ptr %19, i64 %idxprom1
  %20 = load ptr, ptr %arrayidx28170
  br label %land.lhs.true30

lor.end:
  %type22 = getelementptr inbounds %struct.ImageParameters, ptr %7, i64 0, i32 6
  %21 = load i32, ptr %type22
  %cmp23 = icmp eq i32 %21, 1
  %add24 = add nsw i32 %cond16, %list
  %idxprom25 = sext i32 %add24 to i64
  %arrayidx26 = getelementptr inbounds [6 x ptr], ptr @listX, i64 0, i64 %idxprom25
  %22 = load ptr, ptr %arrayidx26
  %arrayidx28 = getelementptr inbounds ptr, ptr %22, i64 %idxprom1
  %23 = load ptr, ptr %arrayidx28
  br i1 %cmp23, label %land.lhs.true30, label %if.else

land.lhs.true30:
  %24 = phi ptr [ %20, %lor.end.thread ], [ %23, %lor.end ]
  %25 = load ptr, ptr @input
  %UseWeightedReferenceME = getelementptr inbounds %struct.InputParameters, ptr %25, i64 0, i32 48
  %26 = load i32, ptr %UseWeightedReferenceME
  %tobool31.not = icmp eq i32 %26, 0
  br i1 %tobool31.not, label %if.else, label %if.then

if.then:
  %imgY_11_w = getelementptr inbounds %struct.storable_picture, ptr %24, i64 0, i32 27
  br label %if.end

if.else:
  %27 = phi ptr [ %24, %land.lhs.true30 ], [ %23, %lor.end ], [ %18, %lor.end.thread171 ]
  %imgY_11 = getelementptr inbounds %struct.storable_picture, ptr %27, i64 0, i32 26
  br label %if.end

if.end:
  %28 = phi ptr [ %24, %if.then ], [ %27, %if.else ]
  %ref_pic.0.in = phi ptr [ %imgY_11_w, %if.then ], [ %imgY_11, %if.else ]
  %ref_pic.0 = load ptr, ptr %ref_pic.0.in
  %size_x = getelementptr inbounds %struct.storable_picture, ptr %28, i64 0, i32 18
  %29 = load i32, ptr %size_x
  %sub = add nsw i32 %29, -17
  %size_y = getelementptr inbounds %struct.storable_picture, ptr %28, i64 0, i32 19
  %30 = load i32, ptr %size_y
  %sub32 = add nsw i32 %30, -17
  %31 = load ptr, ptr @enc_picture
  %ref_idx = getelementptr inbounds %struct.storable_picture, ptr %31, i64 0, i32 32
  %32 = load ptr, ptr %ref_idx
  %mv = getelementptr inbounds %struct.storable_picture, ptr %31, i64 0, i32 35
  %33 = load ptr, ptr %mv
  call void @SetMotionVectorPredictor(ptr nonnull %pmv, ptr %32, ptr %33, i16 signext %ref, i32 signext %list, i32 signext 0, i32 signext 0, i32 signext 16, i32 16)
  %34 = load i16, ptr %pmv
  %35 = sdiv i16 %34, 4
  %div = sext i16 %35 to i32
  %36 = load ptr, ptr @search_center_x
  %arrayidx38 = getelementptr inbounds ptr, ptr %36, i64 %idxprom
  %37 = load ptr, ptr %arrayidx38
  %arrayidx40 = getelementptr inbounds i32, ptr %37, i64 %idxprom1
  store i32 %div, ptr %arrayidx40
  %arrayidx41 = getelementptr inbounds [2 x i16], ptr %pmv, i64 0, i64 1
  %38 = load i16, ptr %arrayidx41
  %39 = sdiv i16 %38, 4
  %div43 = sext i16 %39 to i32
  %40 = load ptr, ptr @search_center_y
  %arrayidx45 = getelementptr inbounds ptr, ptr %40, i64 %idxprom
  %41 = load ptr, ptr %arrayidx45
  %arrayidx47 = getelementptr inbounds i32, ptr %41, i64 %idxprom1
  store i32 %div43, ptr %arrayidx47
  %42 = load ptr, ptr @input
  %rdopt = getelementptr inbounds %struct.InputParameters, ptr %42, i64 0, i32 85
  %43 = load i32, ptr %rdopt
  %tobool48.not = icmp eq i32 %43, 0
  br i1 %tobool48.not, label %if.then49, label %if.end130

if.then49:
  %sub50 = sub nsw i32 0, %6
  %44 = load i32, ptr %arrayidx40
  %45 = tail call i32 @llvm.smin.i32(i32 %6, i32 %44)
  %46 = tail call i32 @llvm.smax.i32(i32 %45, i32 %sub50)
  store i32 %46, ptr %arrayidx40
  %47 = load i32, ptr %arrayidx47
  %48 = tail call i32 @llvm.smin.i32(i32 %6, i32 %47)
  %49 = tail call i32 @llvm.smax.i32(i32 %48, i32 %sub50)
  store i32 %49, ptr %arrayidx47
  br label %if.end130

if.end130:
  %50 = load ptr, ptr @img
  %opix_x = getelementptr inbounds %struct.ImageParameters, ptr %50, i64 0, i32 37
  %51 = load i32, ptr %opix_x
  %52 = load i32, ptr %arrayidx40
  %add135 = add nsw i32 %52, %51
  store i32 %add135, ptr %arrayidx40
  %opix_y = getelementptr inbounds %struct.ImageParameters, ptr %50, i64 0, i32 38
  %53 = load i32, ptr %opix_y
  %54 = load i32, ptr %arrayidx47
  %add140 = add nsw i32 %54, %53
  store i32 %add140, ptr %arrayidx47
  %55 = load i32, ptr %arrayidx40
  %56 = load i32, ptr %opix_y
  %57 = load i32, ptr %opix_x
  %58 = load ptr, ptr @imgY_org
  %59 = sext i32 %57 to i64
  %60 = sext i32 %56 to i64
  %61 = add nsw i32 %57, 15
  %62 = sext i32 %61 to i64
  %63 = add nsw i32 %56, 15
  %64 = sext i32 %63 to i64
  br label %for.body

for.body:
  %indvars.iv196 = phi i64 [ %60, %if.end130 ], [ %indvars.iv.next197, %for.inc165 ]
  %orgptr.0180 = phi ptr [ %orig_blocks, %if.end130 ], [ %incdec.ptr, %for.inc165 ]
  %arrayidx162 = getelementptr inbounds ptr, ptr %58, i64 %indvars.iv196
  %65 = load ptr, ptr %arrayidx162
  br label %for.body160

for.body160:
  %indvars.iv = phi i64 [ %59, %for.body ], [ %indvars.iv.next, %for.body160 ]
  %orgptr.1178 = phi ptr [ %orgptr.0180, %for.body ], [ %incdec.ptr, %for.body160 ]
  %arrayidx164 = getelementptr inbounds i16, ptr %65, i64 %indvars.iv
  %66 = load i16, ptr %arrayidx164
  %incdec.ptr = getelementptr inbounds i16, ptr %orgptr.1178, i64 1
  store i16 %66, ptr %orgptr.1178
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %cmp158 = icmp slt i64 %indvars.iv, %62
  br i1 %cmp158, label %for.body160, label %for.inc165

for.inc165:
  %indvars.iv.next197 = add nsw i64 %indvars.iv196, 1
  %cmp152 = icmp slt i64 %indvars.iv196, %64
  br i1 %cmp152, label %for.body, label %for.end167

for.end167:
  %cmp168.not = icmp slt i32 %55, %6
  %sub171 = sub nsw i32 %sub, %6
  %cmp172.not = icmp sgt i32 %55, %sub171
  %or.cond163 = select i1 %cmp168.not, i1 true, i1 %cmp172.not
  %cmp175.not = icmp slt i32 %add140, %6
  %or.cond164 = select i1 %or.cond163, i1 true, i1 %cmp175.not
  %sub178 = sub nsw i32 %sub32, %6
  %cmp179.not = icmp sgt i32 %add140, %sub178
  %or.cond165 = select i1 %or.cond164, i1 true, i1 %cmp179.not
  br i1 %or.cond165, label %if.end183, label %if.then181

if.then181:
  store ptr @FastLine16Y_11, ptr @PelYline_11
  br label %if.end183

if.end183:
  %67 = load i32, ptr %rdopt
  %tobool185.not = icmp eq i32 %67, 0
  br i1 %tobool185.not, label %if.then186, label %if.end183.if.end213_crit_edge

if.end183.if.end213_crit_edge:
  %.pre = call i32 @llvm.umax.i32(i32 %mul10, i32 1)
  %.pre216 = zext i32 %.pre to i64
  br label %if.end213

if.then186:
  %sub188 = sub nsw i32 %57, %55
  %sub190 = sub nsw i32 %56, %add140
  %68 = load ptr, ptr @spiral_search_x
  %69 = load ptr, ptr @spiral_search_y
  %umax = call i32 @llvm.umax.i32(i32 %mul10, i32 1)
  %wide.trip.count = zext i32 %umax to i64
  br label %for.body194

for.body194:
  %indvars.iv199 = phi i64 [ 0, %if.then186 ], [ %indvars.iv.next200, %for.inc210 ]
  %arrayidx196 = getelementptr inbounds i32, ptr %68, i64 %indvars.iv199
  %70 = load i32, ptr %arrayidx196
  %cmp197 = icmp eq i32 %sub188, %70
  br i1 %cmp197, label %land.lhs.true199, label %for.inc210

land.lhs.true199:
  %arrayidx201 = getelementptr inbounds i32, ptr %69, i64 %indvars.iv199
  %71 = load i32, ptr %arrayidx201
  %cmp202 = icmp eq i32 %sub190, %71
  br i1 %cmp202, label %if.then204, label %for.inc210

if.then204:
  %72 = trunc i64 %indvars.iv199 to i32
  %73 = load ptr, ptr @pos_00
  %arrayidx206 = getelementptr inbounds ptr, ptr %73, i64 %idxprom
  %74 = load ptr, ptr %arrayidx206
  %arrayidx208 = getelementptr inbounds i32, ptr %74, i64 %idxprom1
  store i32 %72, ptr %arrayidx208
  br label %if.end213

for.inc210:
  %indvars.iv.next200 = add nuw nsw i64 %indvars.iv199, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next200, %wide.trip.count
  br i1 %exitcond.not, label %if.end213, label %for.body194

if.end213:
  %wide.trip.count214.pre-phi = phi i64 [ %.pre216, %if.end183.if.end213_crit_edge ], [ %wide.trip.count, %if.then204 ], [ %wide.trip.count, %for.inc210 ]
  br label %for.body217

for.body217:
  %indvars.iv210 = phi i64 [ 0, %if.end213 ], [ %indvars.iv.next211, %for.inc405 ]
  %75 = load ptr, ptr @spiral_search_y
  %arrayidx219 = getelementptr inbounds i32, ptr %75, i64 %indvars.iv210
  %76 = load i32, ptr %arrayidx219
  %add220 = add nsw i32 %76, %add140
  %77 = load ptr, ptr @spiral_search_x
  %arrayidx222 = getelementptr inbounds i32, ptr %77, i64 %indvars.iv210
  %78 = load i32, ptr %arrayidx222
  %add223 = add nsw i32 %78, %55
  br i1 %or.cond165, label %if.then225, label %for.cond246.preheader

if.then225:
  %cmp226 = icmp sgt i32 %add220, -1
  br i1 %cmp226, label %land.lhs.true228, label %if.else238

land.lhs.true228:
  %cmp229 = icmp sgt i32 %add220, %sub32
  %cmp232 = icmp slt i32 %add223, 0
  %or.cond = select i1 %cmp229, i1 true, i1 %cmp232
  %cmp235.not = icmp sgt i32 %add223, %sub
  %or.cond166 = select i1 %or.cond, i1 true, i1 %cmp235.not
  br i1 %or.cond166, label %if.else238, label %if.end240.sink.split

if.else238:
  br label %if.end240.sink.split

if.end240.sink.split:
  %FastLine16Y_11.sink = phi ptr [ @UMVLine16Y_11, %if.else238 ], [ @FastLine16Y_11, %land.lhs.true228 ]
  store ptr %FastLine16Y_11.sink, ptr @PelYline_11
  br label %for.cond246.preheader

for.cond246.preheader:
  %indvars.iv203 = phi i64 [ %indvars.iv.next204, %for.end381 ], [ 0, %if.end240.sink.split ], [ 0, %for.body217 ]
  %orgptr.2192 = phi ptr [ %uglygep, %for.end381 ], [ %orig_blocks, %if.end240.sink.split ], [ %orig_blocks, %for.body217 ]
  %abs_y.0191 = phi i32 [ %inc250, %for.end381 ], [ %add220, %if.end240.sink.split ], [ %add220, %for.body217 ]
  %blky.0189 = phi i32 [ %inc403, %for.end381 ], [ 0, %if.end240.sink.split ], [ 0, %for.body217 ]
  br label %for.body249

for.body249:
  %orgptr.3188 = phi ptr [ %orgptr.2192, %for.cond246.preheader ], [ %incdec.ptr373, %for.body249 ]
  %y.1187 = phi i32 [ 0, %for.cond246.preheader ], [ %inc380, %for.body249 ]
  %abs_y.1186 = phi i32 [ %abs_y.0191, %for.cond246.preheader ], [ %inc250, %for.body249 ]
  %LineSadBlk0.0185 = phi i32 [ 0, %for.cond246.preheader ], [ %add282, %for.body249 ]
  %LineSadBlk1.0184 = phi i32 [ 0, %for.cond246.preheader ], [ %add314, %for.body249 ]
  %LineSadBlk3.0183 = phi i32 [ 0, %for.cond246.preheader ], [ %add378, %for.body249 ]
  %LineSadBlk2.0182 = phi i32 [ 0, %for.cond246.preheader ], [ %add346, %for.body249 ]
  %79 = load ptr, ptr @PelYline_11
  %inc250 = add nsw i32 %abs_y.1186, 1
  %call = tail call ptr %79(ptr %ref_pic.0, i32 signext %abs_y.1186, i32 signext %add223, i32 signext %30, i32 signext %29)
  %80 = load ptr, ptr @byte_abs
  %incdec.ptr251 = getelementptr inbounds i16, ptr %call, i64 1
  %81 = load i16, ptr %call
  %conv252 = zext i16 %81 to i64
  %incdec.ptr253 = getelementptr inbounds i16, ptr %orgptr.3188, i64 1
  %82 = load i16, ptr %orgptr.3188
  %conv254 = zext i16 %82 to i64
  %sub255 = sub nsw i64 %conv252, %conv254
  %arrayidx257 = getelementptr inbounds i32, ptr %80, i64 %sub255
  %83 = load i32, ptr %arrayidx257
  %add258 = add nsw i32 %83, %LineSadBlk0.0185
  %incdec.ptr259 = getelementptr inbounds i16, ptr %call, i64 2
  %84 = load i16, ptr %incdec.ptr251
  %conv260 = zext i16 %84 to i64
  %incdec.ptr261 = getelementptr inbounds i16, ptr %orgptr.3188, i64 2
  %85 = load i16, ptr %incdec.ptr253
  %conv262 = zext i16 %85 to i64
  %sub263 = sub nsw i64 %conv260, %conv262
  %arrayidx265 = getelementptr inbounds i32, ptr %80, i64 %sub263
  %86 = load i32, ptr %arrayidx265
  %add266 = add nsw i32 %add258, %86
  %incdec.ptr267 = getelementptr inbounds i16, ptr %call, i64 3
  %87 = load i16, ptr %incdec.ptr259
  %conv268 = zext i16 %87 to i64
  %incdec.ptr269 = getelementptr inbounds i16, ptr %orgptr.3188, i64 3
  %88 = load i16, ptr %incdec.ptr261
  %conv270 = zext i16 %88 to i64
  %sub271 = sub nsw i64 %conv268, %conv270
  %arrayidx273 = getelementptr inbounds i32, ptr %80, i64 %sub271
  %89 = load i32, ptr %arrayidx273
  %add274 = add nsw i32 %add266, %89
  %incdec.ptr275 = getelementptr inbounds i16, ptr %call, i64 4
  %90 = load i16, ptr %incdec.ptr267
  %conv276 = zext i16 %90 to i64
  %incdec.ptr277 = getelementptr inbounds i16, ptr %orgptr.3188, i64 4
  %91 = load i16, ptr %incdec.ptr269
  %conv278 = zext i16 %91 to i64
  %sub279 = sub nsw i64 %conv276, %conv278
  %arrayidx281 = getelementptr inbounds i32, ptr %80, i64 %sub279
  %92 = load i32, ptr %arrayidx281
  %add282 = add nsw i32 %add274, %92
  %incdec.ptr283 = getelementptr inbounds i16, ptr %call, i64 5
  %93 = load i16, ptr %incdec.ptr275
  %conv284 = zext i16 %93 to i64
  %incdec.ptr285 = getelementptr inbounds i16, ptr %orgptr.3188, i64 5
  %94 = load i16, ptr %incdec.ptr277
  %conv286 = zext i16 %94 to i64
  %sub287 = sub nsw i64 %conv284, %conv286
  %arrayidx289 = getelementptr inbounds i32, ptr %80, i64 %sub287
  %95 = load i32, ptr %arrayidx289
  %add290 = add nsw i32 %95, %LineSadBlk1.0184
  %incdec.ptr291 = getelementptr inbounds i16, ptr %call, i64 6
  %96 = load i16, ptr %incdec.ptr283
  %conv292 = zext i16 %96 to i64
  %incdec.ptr293 = getelementptr inbounds i16, ptr %orgptr.3188, i64 6
  %97 = load i16, ptr %incdec.ptr285
  %conv294 = zext i16 %97 to i64
  %sub295 = sub nsw i64 %conv292, %conv294
  %arrayidx297 = getelementptr inbounds i32, ptr %80, i64 %sub295
  %98 = load i32, ptr %arrayidx297
  %add298 = add nsw i32 %add290, %98
  %incdec.ptr299 = getelementptr inbounds i16, ptr %call, i64 7
  %99 = load i16, ptr %incdec.ptr291
  %conv300 = zext i16 %99 to i64
  %incdec.ptr301 = getelementptr inbounds i16, ptr %orgptr.3188, i64 7
  %100 = load i16, ptr %incdec.ptr293
  %conv302 = zext i16 %100 to i64
  %sub303 = sub nsw i64 %conv300, %conv302
  %arrayidx305 = getelementptr inbounds i32, ptr %80, i64 %sub303
  %101 = load i32, ptr %arrayidx305
  %add306 = add nsw i32 %add298, %101
  %incdec.ptr307 = getelementptr inbounds i16, ptr %call, i64 8
  %102 = load i16, ptr %incdec.ptr299
  %conv308 = zext i16 %102 to i64
  %incdec.ptr309 = getelementptr inbounds i16, ptr %orgptr.3188, i64 8
  %103 = load i16, ptr %incdec.ptr301
  %conv310 = zext i16 %103 to i64
  %sub311 = sub nsw i64 %conv308, %conv310
  %arrayidx313 = getelementptr inbounds i32, ptr %80, i64 %sub311
  %104 = load i32, ptr %arrayidx313
  %add314 = add nsw i32 %add306, %104
  %incdec.ptr315 = getelementptr inbounds i16, ptr %call, i64 9
  %105 = load i16, ptr %incdec.ptr307
  %conv316 = zext i16 %105 to i64
  %incdec.ptr317 = getelementptr inbounds i16, ptr %orgptr.3188, i64 9
  %106 = load i16, ptr %incdec.ptr309
  %conv318 = zext i16 %106 to i64
  %sub319 = sub nsw i64 %conv316, %conv318
  %arrayidx321 = getelementptr inbounds i32, ptr %80, i64 %sub319
  %107 = load i32, ptr %arrayidx321
  %add322 = add nsw i32 %107, %LineSadBlk2.0182
  %incdec.ptr323 = getelementptr inbounds i16, ptr %call, i64 10
  %108 = load i16, ptr %incdec.ptr315
  %conv324 = zext i16 %108 to i64
  %incdec.ptr325 = getelementptr inbounds i16, ptr %orgptr.3188, i64 10
  %109 = load i16, ptr %incdec.ptr317
  %conv326 = zext i16 %109 to i64
  %sub327 = sub nsw i64 %conv324, %conv326
  %arrayidx329 = getelementptr inbounds i32, ptr %80, i64 %sub327
  %110 = load i32, ptr %arrayidx329
  %add330 = add nsw i32 %add322, %110
  %incdec.ptr331 = getelementptr inbounds i16, ptr %call, i64 11
  %111 = load i16, ptr %incdec.ptr323
  %conv332 = zext i16 %111 to i64
  %incdec.ptr333 = getelementptr inbounds i16, ptr %orgptr.3188, i64 11
  %112 = load i16, ptr %incdec.ptr325
  %conv334 = zext i16 %112 to i64
  %sub335 = sub nsw i64 %conv332, %conv334
  %arrayidx337 = getelementptr inbounds i32, ptr %80, i64 %sub335
  %113 = load i32, ptr %arrayidx337
  %add338 = add nsw i32 %add330, %113
  %incdec.ptr339 = getelementptr inbounds i16, ptr %call, i64 12
  %114 = load i16, ptr %incdec.ptr331
  %conv340 = zext i16 %114 to i64
  %incdec.ptr341 = getelementptr inbounds i16, ptr %orgptr.3188, i64 12
  %115 = load i16, ptr %incdec.ptr333
  %conv342 = zext i16 %115 to i64
  %sub343 = sub nsw i64 %conv340, %conv342
  %arrayidx345 = getelementptr inbounds i32, ptr %80, i64 %sub343
  %116 = load i32, ptr %arrayidx345
  %add346 = add nsw i32 %add338, %116
  %incdec.ptr347 = getelementptr inbounds i16, ptr %call, i64 13
  %117 = load i16, ptr %incdec.ptr339
  %conv348 = zext i16 %117 to i64
  %incdec.ptr349 = getelementptr inbounds i16, ptr %orgptr.3188, i64 13
  %118 = load i16, ptr %incdec.ptr341
  %conv350 = zext i16 %118 to i64
  %sub351 = sub nsw i64 %conv348, %conv350
  %arrayidx353 = getelementptr inbounds i32, ptr %80, i64 %sub351
  %119 = load i32, ptr %arrayidx353
  %add354 = add nsw i32 %119, %LineSadBlk3.0183
  %incdec.ptr355 = getelementptr inbounds i16, ptr %call, i64 14
  %120 = load i16, ptr %incdec.ptr347
  %conv356 = zext i16 %120 to i64
  %incdec.ptr357 = getelementptr inbounds i16, ptr %orgptr.3188, i64 14
  %121 = load i16, ptr %incdec.ptr349
  %conv358 = zext i16 %121 to i64
  %sub359 = sub nsw i64 %conv356, %conv358
  %arrayidx361 = getelementptr inbounds i32, ptr %80, i64 %sub359
  %122 = load i32, ptr %arrayidx361
  %add362 = add nsw i32 %add354, %122
  %incdec.ptr363 = getelementptr inbounds i16, ptr %call, i64 15
  %123 = load i16, ptr %incdec.ptr355
  %conv364 = zext i16 %123 to i64
  %incdec.ptr365 = getelementptr inbounds i16, ptr %orgptr.3188, i64 15
  %124 = load i16, ptr %incdec.ptr357
  %conv366 = zext i16 %124 to i64
  %sub367 = sub nsw i64 %conv364, %conv366
  %arrayidx369 = getelementptr inbounds i32, ptr %80, i64 %sub367
  %125 = load i32, ptr %arrayidx369
  %add370 = add nsw i32 %add362, %125
  %126 = load i16, ptr %incdec.ptr363
  %conv372 = zext i16 %126 to i64
  %incdec.ptr373 = getelementptr inbounds i16, ptr %orgptr.3188, i64 16
  %127 = load i16, ptr %incdec.ptr365
  %conv374 = zext i16 %127 to i64
  %sub375 = sub nsw i64 %conv372, %conv374
  %arrayidx377 = getelementptr inbounds i32, ptr %80, i64 %sub375
  %128 = load i32, ptr %arrayidx377
  %add378 = add nsw i32 %add370, %128
  %inc380 = add nuw nsw i32 %y.1187, 1
  %exitcond202.not = icmp eq i32 %inc380, 4
  br i1 %exitcond202.not, label %for.end381, label %for.body249

for.end381:
  %uglygep = getelementptr i8, ptr %orgptr.2192, i64 128
  %129 = or i64 %indvars.iv203, 1
  %arrayidx384 = getelementptr inbounds ptr, ptr %3, i64 %indvars.iv203
  %130 = load ptr, ptr %arrayidx384
  %arrayidx386 = getelementptr inbounds i32, ptr %130, i64 %indvars.iv210
  store i32 %add282, ptr %arrayidx386
  %131 = or i64 %indvars.iv203, 2
  %arrayidx389 = getelementptr inbounds ptr, ptr %3, i64 %129
  %132 = load ptr, ptr %arrayidx389
  %arrayidx391 = getelementptr inbounds i32, ptr %132, i64 %indvars.iv210
  store i32 %add314, ptr %arrayidx391
  %133 = or i64 %indvars.iv203, 3
  %arrayidx394 = getelementptr inbounds ptr, ptr %3, i64 %131
  %134 = load ptr, ptr %arrayidx394
  %arrayidx396 = getelementptr inbounds i32, ptr %134, i64 %indvars.iv210
  store i32 %add346, ptr %arrayidx396
  %indvars.iv.next204 = add nuw nsw i64 %indvars.iv203, 4
  %arrayidx399 = getelementptr inbounds ptr, ptr %3, i64 %133
  %135 = load ptr, ptr %arrayidx399
  %arrayidx401 = getelementptr inbounds i32, ptr %135, i64 %indvars.iv210
  store i32 %add378, ptr %arrayidx401
  %inc403 = add nuw nsw i32 %blky.0189, 1
  %exitcond209.not = icmp eq i32 %inc403, 4
  br i1 %exitcond209.not, label %for.inc405, label %for.cond246.preheader

for.inc405:
  %indvars.iv.next211 = add nuw nsw i64 %indvars.iv210, 1
  %exitcond215.not = icmp eq i64 %indvars.iv.next211, %wide.trip.count214.pre-phi
  br i1 %exitcond215.not, label %for.end407, label %for.body217

for.end407:
  %conv408 = sext i16 %ref to i32
  tail call void @SetupLargerBlocks(i32 signext %list, i32 signext %conv408, i32 signext %mul10)
  %136 = load ptr, ptr @search_setup_done
  %arrayidx410 = getelementptr inbounds ptr, ptr %136, i64 %idxprom
  %137 = load ptr, ptr %arrayidx410
  %arrayidx412 = getelementptr inbounds i32, ptr %137, i64 %idxprom1
  store i32 1, ptr %arrayidx412
  call void @llvm.lifetime.end.p0(i64 512, ptr nonnull %orig_blocks)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %pmv)
  ret void
}

declare void @SetMotionVectorPredictor(ptr nocapture writeonly, ptr nocapture readonly, ptr nocapture readonly, i16 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32)
declare ptr @FastLine16Y_11(ptr, i32 signext, i32 signext, i32 signext, i32 signext)
declare ptr @UMVLine16Y_11(ptr, i32 signext, i32 signext, i32 signext, i32 signext)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
