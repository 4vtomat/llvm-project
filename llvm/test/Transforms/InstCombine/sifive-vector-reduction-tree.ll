; RUN: opt < %s -passes='instcombine<no-verify-fixpoint>' -pass-remarks='instcombine' -pass-remarks-missed='instcombine' -pass-remarks-output=%t -S 2>&1 | FileCheck %s
; RUN: cat %t | FileCheck --check-prefix=CHECK-REMARKS %s

; CHECK-REMARKS: optimized reduction tree using accumulator
; CHECK-REMARKS: optimized reduction tree using accumulator
; CHECK-REMARKS: optimized reduction tree using accumulator
; CHECK-REMARKS: optimized reduction tree using accumulator
; CHECK-REMARKS: cannot optimize reduction tree using accumulator: operands have different type
; CHECK-REMARKS: cannot optimize reduction tree using accumulator: operands have different type
; CHECK-REMARKS: cannot optimize reduction tree using accumulator: operands have different type

define float @tree_sums_v16f32(float %start, <16 x float> %v0, <16 x float> %v1) {
; CHECK-LABEL: define float @tree_sums_v16f32(
; CHECK-SAME: float [[START:%.*]], <16 x float> [[V0:%.*]], <16 x float> [[V1:%.*]]) {
; CHECK-NEXT:    [[COMBINED_REDUCE:%.*]] = fadd fast <16 x float> [[V0]], [[V1]]
; CHECK-NEXT:    [[OP_RDX419_1:%.*]] = call fast float @llvm.vector.reduce.fadd.v16f32(float [[START]], <16 x float> [[COMBINED_REDUCE]])
; CHECK-NEXT:    [[OP_RDX420_1:%.*]] = fadd fast float [[START]], [[START]]
; CHECK-NEXT:    [[OP_RDX422_1:%.*]] = fadd fast float [[OP_RDX419_1]], [[OP_RDX420_1]]
; CHECK-NEXT:    [[OP_RDX423_1:%.*]] = fmul fast float [[START]], 7.000000e+00
; CHECK-NEXT:    [[OP_RDX424_1:%.*]] = fadd fast float [[OP_RDX422_1]], [[OP_RDX423_1]]
; CHECK-NEXT:    ret float [[OP_RDX424_1]]
;
  %op.rdx419 = call fast float @llvm.vector.reduce.fadd.v16f32(float %start, <16 x float> %v0)
  %op.rdx420 = fadd fast float %start, %start
  %op.rdx421 = fadd fast float %start, %start
  %op.rdx422 = fadd fast float %op.rdx419, %op.rdx420
  %op.rdx424 = fadd fast float %op.rdx422, %op.rdx421
  %op.rdx419.1 = call fast float @llvm.vector.reduce.fadd.v16f32(float %start, <16 x float> %v1)
  %op.rdx420.1 = fadd fast float %start, %start
  %op.rdx421.1 = fadd fast float %start, %start
  %op.rdx422.1 = fadd fast float %op.rdx419.1, %op.rdx420.1
  %op.rdx423.1 = fadd fast float %op.rdx421.1, %op.rdx424
  %op.rdx424.1 = fadd fast float %op.rdx422.1, %op.rdx423.1

  ret float %op.rdx424.1
}

define i32 @tree_sums_v16i32(i32 %start, <16 x i32> %v0, <16 x i32> %v1) {
; CHECK-LABEL: define i32 @tree_sums_v16i32(
; CHECK-SAME: i32 [[START:%.*]], <16 x i32> [[V0:%.*]], <16 x i32> [[V1:%.*]]) {
; CHECK-NEXT:    [[COMBINED_REDUCE:%.*]] = add <16 x i32> [[V0]], [[V1]]
; CHECK-NEXT:    [[OP_RDX419_1:%.*]] = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> [[COMBINED_REDUCE]])
; CHECK-NEXT:    [[OP_RDX420_1:%.*]] = shl i32 [[START]], 1
; CHECK-NEXT:    [[OP_RDX422_1:%.*]] = add i32 [[OP_RDX419_1]], [[OP_RDX420_1]]
; CHECK-NEXT:    [[OP_RDX423_1:%.*]] = mul i32 [[START]], 6
; CHECK-NEXT:    [[OP_RDX424_1:%.*]] = add i32 [[OP_RDX422_1]], [[OP_RDX423_1]]
; CHECK-NEXT:    ret i32 [[OP_RDX424_1]]
;
  %op.rdx419 = call  i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %v0)
  %op.rdx420 = add  i32 %start, %start
  %op.rdx421 = add  i32 %start, %start
  %op.rdx422 = add  i32 %op.rdx419, %op.rdx420
  %op.rdx424 = add  i32 %op.rdx422, %op.rdx421
  %op.rdx419.1 = call  i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %v1)
  %op.rdx420.1 = add  i32 %start, %start
  %op.rdx421.1 = add  i32 %start, %start
  %op.rdx422.1 = add  i32 %op.rdx419.1, %op.rdx420.1
  %op.rdx423.1 = add  i32 %op.rdx421.1, %op.rdx424
  %op.rdx424.1 = add  i32 %op.rdx422.1, %op.rdx423.1

  ret i32 %op.rdx424.1
}

define i32 @trivial_tree_sums_v16i32(i32 %start, <16 x i32> %v0, <16 x i32> %v1) {
; CHECK-LABEL: define i32 @trivial_tree_sums_v16i32(
; CHECK-SAME: i32 [[START:%.*]], <16 x i32> [[V0:%.*]], <16 x i32> [[V1:%.*]]) {
; CHECK-NEXT:    [[COMBINED_REDUCE:%.*]] = add <16 x i32> [[V0]], [[V1]]
; CHECK-NEXT:    [[OP_RDX419_1:%.*]] = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> [[COMBINED_REDUCE]])
; CHECK-NEXT:    ret i32 [[OP_RDX419_1]]
;
  %op.rdx419 = call  i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %v0)
  %op.rdx419.1 = call  i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %v1)
  %op.rdx420.1 = add  i32 %op.rdx419, %op.rdx419.1

  ret i32 %op.rdx420.1
}

define i32 @reduce_before_use(<4 x i32> %0, <4 x i32> %1) {
; CHECK-LABEL: define i32 @reduce_before_use(
; CHECK-SAME: <4 x i32> [[TMP0:%.*]], <4 x i32> [[TMP1:%.*]]) {
; CHECK-NEXT:    [[TMP3:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[TMP0]])
; CHECK-NEXT:    [[COMBINED_REDUCE:%.*]] = shl <4 x i32> [[TMP1]], <i32 1, i32 1, i32 1, i32 1>
; CHECK-NEXT:    [[TMP4:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[COMBINED_REDUCE]])
; CHECK-NEXT:    [[TMP5:%.*]] = add i32 [[TMP4]], [[TMP3]]
; CHECK-NEXT:    ret i32 [[TMP5]]
;
  %3 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %0)
  %4 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %1)
  %5 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %1)
  %6 = add i32 %4, %5
  %7 = or i32 %6, 0
  %8 = add i32 %7, %3
  ret i32 %8
}

; Negative tests
define float @negative_tree_sums_v16f32(float %start, <16 x float> %v0, <8 x float> %v1) {
; CHECK-LABEL: define float @negative_tree_sums_v16f32(
; CHECK-SAME: float [[START:%.*]], <16 x float> [[V0:%.*]], <8 x float> [[V1:%.*]]) {
; CHECK-NEXT:    [[OP_RDX419:%.*]] = call fast float @llvm.vector.reduce.fadd.v16f32(float [[START]], <16 x float> [[V0]])
; CHECK-NEXT:    [[OP_RDX420:%.*]] = fadd fast float [[START]], [[START]]
; CHECK-NEXT:    [[OP_RDX421:%.*]] = fadd fast float [[START]], [[START]]
; CHECK-NEXT:    [[OP_RDX422:%.*]] = fadd fast float [[OP_RDX419]], [[OP_RDX420]]
; CHECK-NEXT:    [[OP_RDX424:%.*]] = fadd fast float [[OP_RDX422]], [[OP_RDX421]]
; CHECK-NEXT:    [[OP_RDX419_1:%.*]] = call fast float @llvm.vector.reduce.fadd.v8f32(float [[START]], <8 x float> [[V1]])
; CHECK-NEXT:    [[OP_RDX420_1:%.*]] = fadd fast float [[START]], [[START]]
; CHECK-NEXT:    [[OP_RDX421_1:%.*]] = fadd fast float [[START]], [[START]]
; CHECK-NEXT:    [[OP_RDX422_1:%.*]] = fadd fast float [[OP_RDX419_1]], [[OP_RDX420_1]]
; CHECK-NEXT:    [[OP_RDX423_1:%.*]] = fadd fast float [[OP_RDX421_1]], [[OP_RDX424]]
; CHECK-NEXT:    [[OP_RDX424_1:%.*]] = fadd fast float [[OP_RDX422_1]], [[OP_RDX423_1]]
; CHECK-NEXT:    ret float [[OP_RDX424_1]]
;
  %op.rdx419 = call fast float @llvm.vector.reduce.fadd.v16f32(float %start, <16 x float> %v0)
  %op.rdx420 = fadd fast float %start, %start
  %op.rdx421 = fadd fast float %start, %start
  %op.rdx422 = fadd fast float %op.rdx419, %op.rdx420
  %op.rdx424 = fadd fast float %op.rdx422, %op.rdx421
  %op.rdx419.1 = call fast float @llvm.vector.reduce.fadd.v8f32(float %start, <8 x float> %v1)
  %op.rdx420.1 = fadd fast float %start, %start
  %op.rdx421.1 = fadd fast float %start, %start
  %op.rdx422.1 = fadd fast float %op.rdx419.1, %op.rdx420.1
  %op.rdx423.1 = fadd fast float %op.rdx421.1, %op.rdx424
  %op.rdx424.1 = fadd fast float %op.rdx422.1, %op.rdx423.1

  ret float %op.rdx424.1
}

define i32 @negative_tree_sums_v16i32(i32 %start, <16 x i32> %v0, <8 x i32> %v1) {
; CHECK-LABEL: define i32 @negative_tree_sums_v16i32(
; CHECK-SAME: i32 [[START:%.*]], <16 x i32> [[V0:%.*]], <8 x i32> [[V1:%.*]]) {
; CHECK-NEXT:    [[OP_RDX419:%.*]] = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> [[V0]])
; CHECK-NEXT:    [[OP_RDX420:%.*]] = shl i32 [[START]], 1
; CHECK-NEXT:    [[OP_RDX421:%.*]] = shl i32 [[START]], 1
; CHECK-NEXT:    [[OP_RDX422:%.*]] = add i32 [[OP_RDX419]], [[OP_RDX420]]
; CHECK-NEXT:    [[OP_RDX424:%.*]] = add i32 [[OP_RDX422]], [[OP_RDX421]]
; CHECK-NEXT:    [[OP_RDX419_1:%.*]] = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> [[V1]])
; CHECK-NEXT:    [[OP_RDX420_1:%.*]] = shl i32 [[START]], 1
; CHECK-NEXT:    [[OP_RDX421_1:%.*]] = shl i32 [[START]], 1
; CHECK-NEXT:    [[OP_RDX422_1:%.*]] = add i32 [[OP_RDX419_1]], [[OP_RDX420_1]]
; CHECK-NEXT:    [[OP_RDX423_1:%.*]] = add i32 [[OP_RDX421_1]], [[OP_RDX424]]
; CHECK-NEXT:    [[OP_RDX424_1:%.*]] = add i32 [[OP_RDX422_1]], [[OP_RDX423_1]]
; CHECK-NEXT:    ret i32 [[OP_RDX424_1]]
;
  %op.rdx419 = call  i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %v0)
  %op.rdx420 = add  i32 %start, %start
  %op.rdx421 = add  i32 %start, %start
  %op.rdx422 = add  i32 %op.rdx419, %op.rdx420
  %op.rdx424 = add  i32 %op.rdx422, %op.rdx421
  %op.rdx419.1 = call  i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %v1)
  %op.rdx420.1 = add  i32 %start, %start
  %op.rdx421.1 = add  i32 %start, %start
  %op.rdx422.1 = add  i32 %op.rdx419.1, %op.rdx420.1
  %op.rdx423.1 = add  i32 %op.rdx421.1, %op.rdx424
  %op.rdx424.1 = add  i32 %op.rdx422.1, %op.rdx423.1

  ret i32 %op.rdx424.1
}
