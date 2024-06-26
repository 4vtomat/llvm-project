//===-- RISCVInstCombineIntrinsic.cpp - RISC-V specific InstCombine pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// RISC-V target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCVTargetTransformInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;

#define DEBUG_TYPE "riscvtti"

static cl::opt<bool>
    DisableVectorOpt("riscv-disable-vector-instcombine",
                     cl::desc("Disable InstCombine for vector intrinsics"),
                     cl::init(false), cl::Hidden);

static CallInst *CreateIntrinsic(IntrinsicInst *II, Intrinsic::ID IID,
                                 ArrayRef<Type *> Types,
                                 ArrayRef<Value *> Args) {
  Function *Merge = Intrinsic::getDeclaration(II->getModule(), IID, Types);
  return CallInst::Create(Merge, Args);
}

static Value *getVSplat(Value *Op, Value *VL) {
  if (auto *II = dyn_cast<IntrinsicInst>(Op))
    if ((II->getIntrinsicID() == Intrinsic::riscv_vmv_v_x ||
         II->getIntrinsicID() == Intrinsic::riscv_vfmv_v_f) &&
        isa<UndefValue>(II->getArgOperand(0)) &&
        II->getArgOperand(2) == VL)
      return II->getOperand(1);

  return nullptr;
}

// Look for a splat, ignoring whether the operand is scalar or vector.
static Value *getVSplatOrScalar(Value *Op, Value *VL) {
  Type *Ty = Op->getType();
  if (Ty->isIntegerTy() || Ty->isFloatingPointTy())
    return Op;

  return getVSplat(Op, VL);
}

// Return 0 if Zvl extension is not enabled.
static unsigned getVLMAX(const ScalableVectorType *type,
                         const RISCVSubtarget *ST) {
  if (!ST->hasStdExtZvl())
    return 0;
  // type->getPrimitiveSizeInBits().getKnownMinValue() / RISCV::RVVBitsPerBlock
  // is LMUL. Do not use (type->getPrimitiveSizeInBits().getKnownMinValue() /
  // RISCV::RVVBitsPerBlock), because
  // type->getPrimitiveSizeInBits().getKnownMinValue() may be smaller than
  // RISCV::RVVBitsPerBlock.
  unsigned SEW = type->getScalarType()->getScalarSizeInBits();
  return ((ST->getRealMinVLen() / SEW) *
          type->getPrimitiveSizeInBits().getKnownMinValue()) /
         RISCV::RVVBitsPerBlock;
}

// Try to match
//   (vand (vmerge 0, -1, C), A)
// To
//   (vmerge 0, A, C)
static Instruction *foldVAndWithVMerge(Value *LHS, Value *RHS, Value *VL) {
  auto *II2 = dyn_cast<IntrinsicInst>(LHS);
  if (!II2 || II2->getIntrinsicID() != Intrinsic::riscv_vmerge ||
      !isa<UndefValue>(II2->getArgOperand(0)) || II2->getArgOperand(4) != VL)
    return nullptr;

  Value *FalseVal = getVSplatOrScalar(II2->getArgOperand(1), VL);
  Value *TrueVal = getVSplatOrScalar(II2->getArgOperand(2), VL);
  auto *FalseC = dyn_cast_or_null<ConstantInt>(FalseVal);
  auto *TrueC = dyn_cast_or_null<ConstantInt>(TrueVal);
  if (FalseC && TrueC && FalseC->isZero() && TrueC->isMinusOne()) {
    return CreateIntrinsic(II2, Intrinsic::riscv_vmerge,
                           {II2->getType(), RHS->getType(), VL->getType()},
                           {UndefValue::get(II2->getType()),
                            II2->getArgOperand(1), RHS, II2->getArgOperand(3),
                            VL});
  }

  // FIXME: Handle the (vmerge -1, 0, C) case. Trickier if other AND operand
  // is a scalar since the first operand of vmerge can't be scalar.

  return nullptr;
}

// Try to match
//   (vxor (vmerge 0, (vxor A, B), C), A)
// To
//   (vmerge A, B, C)
static Instruction *foldVXorWithVMergeVXor(Value *LHS, Value *RHS, Value *VL) {
  auto *II2 = dyn_cast<IntrinsicInst>(LHS);
  if (!II2 || II2->getIntrinsicID() != Intrinsic::riscv_vmerge ||
      !isa<UndefValue>(II2->getArgOperand(0)) || II2->getArgOperand(4) != VL)
    return nullptr;

  auto MatchBitSelect = [](Value *Other, Value *Op0, Value *Op1,
                           Value *VL) -> Value * {
    // One operand of the vmerge needs to be 0.
    auto *Zero = dyn_cast_or_null<ConstantInt>(getVSplatOrScalar(Op0, VL));
    if (!Zero || !Zero->isZero())
      return nullptr;

    // If this is a VXOR involving Other, return the other operand.
    auto *II3 = dyn_cast<IntrinsicInst>(Op1);
    if (II3 && II3->getIntrinsicID() == Intrinsic::riscv_vxor &&
        isa<UndefValue>(II3->getArgOperand(0)) && II3->getArgOperand(3) == VL) {
      if (II3->getOperand(1) == Other)
        return II3->getArgOperand(2);
      if (II3->getArgOperand(2) == Other)
        return II3->getArgOperand(1);
    }

    return nullptr;
  };

  Value *FalseVal = II2->getArgOperand(1);
  Value *TrueVal = II2->getArgOperand(2);
  Value *A = RHS;
  Value *C = II2->getArgOperand(3);

  // The VXOR could be either the True or False value of the vmerge, handle
  // both cases. We need to be careful not to put a scalar as the first
  // operand of vmerge.
  if (A->getType()->isVectorTy()) {
    // (vxor (vmerge 0, (vxor A, B), C), A) -> (vmerge A, B, C)
    if (Value *B = MatchBitSelect(A, FalseVal, TrueVal, VL))
      return CreateIntrinsic(II2, Intrinsic::riscv_vmerge,
                             {II2->getType(), B->getType(), VL->getType()},
                             {UndefValue::get(II2->getType()), A, B, C, VL});
  }
  // (vxor (vmerge (vxor A, B), 0, C), A) -> (vmerge B, A, C)
  if (Value *B = MatchBitSelect(A, TrueVal, FalseVal, VL)) {
    if (B->getType()->isVectorTy())
      return CreateIntrinsic(II2, Intrinsic::riscv_vmerge,
                             {II2->getType(), A->getType(), VL->getType()},
                             {UndefValue::get(II2->getType()), B, A, C, VL});
  }

  return nullptr;
}

static Instruction *foldBinaryOp(InstCombiner &IC, IntrinsicInst &II) {
  bool HasFRM = false;
  switch (II.getIntrinsicID()) {
  case Intrinsic::riscv_vfadd:
  case Intrinsic::riscv_vfsub:
  case Intrinsic::riscv_vfrsub:
  case Intrinsic::riscv_vfmul:
    HasFRM = true;
    break;
  }

  if (HasFRM) {
    auto *FRMC = dyn_cast<ConstantInt>(II.getArgOperand(3));
    if (!FRMC || FRMC->getZExtValue() != 7)
      return nullptr;
  }

  Value *VL = II.getArgOperand(3 + HasFRM);
  Value *LHSScalar = getVSplat(II.getArgOperand(1), VL);
  if (!LHSScalar)
    return nullptr;
  Value *RHSScalar = getVSplatOrScalar(II.getArgOperand(2), VL);
  if (!RHSScalar)
    return nullptr;
  Value *Result;
  switch (II.getIntrinsicID()) {
  default: llvm_unreachable("Unexpected intrinsic");
  case Intrinsic::riscv_vadd:
    Result = IC.Builder.CreateAdd(LHSScalar, RHSScalar);
    break;
  case Intrinsic::riscv_vsub:
    Result = IC.Builder.CreateSub(LHSScalar, RHSScalar);
    break;
  case Intrinsic::riscv_vrsub:
    Result = IC.Builder.CreateSub(RHSScalar, LHSScalar);
    break;
  case Intrinsic::riscv_vfadd:
    Result = IC.Builder.CreateFAdd(LHSScalar, RHSScalar);
    break;
  case Intrinsic::riscv_vfsub:
    Result = IC.Builder.CreateFSub(LHSScalar, RHSScalar);
    break;
  case Intrinsic::riscv_vfrsub:
    Result = IC.Builder.CreateFSub(RHSScalar, LHSScalar);
    break;
  case Intrinsic::riscv_vand:
    Result = IC.Builder.CreateAnd(LHSScalar, RHSScalar);
    break;
  case Intrinsic::riscv_vor:
    Result = IC.Builder.CreateOr(LHSScalar, RHSScalar);
    break;
  case Intrinsic::riscv_vxor:
    Result = IC.Builder.CreateXor(LHSScalar, RHSScalar);
    break;
  case Intrinsic::riscv_vsll: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, TypeWidth - 1);
    // If RHSScalar comes from vsll.vx or vsll.vi, its type is i64 for RV64 and
    // i32 for RV32.
    RHSScalar = IC.Builder.CreateZExtOrTrunc(RHSScalar, Ty);
    Result = IC.Builder.CreateShl(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vsrl: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, TypeWidth - 1);
    // If RHSScalar comes from vsrl.vx or vsrl.vi, its type is i64 for RV64 and
    // i32 for RV32.
    RHSScalar = IC.Builder.CreateZExtOrTrunc(RHSScalar, Ty);
    Result = IC.Builder.CreateLShr(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vsra: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, TypeWidth - 1);
    // If RHSScalar comes from vsra.vx or vsra.vi, its type is i64 for RV64 and
    // i32 for RV32.
    RHSScalar = IC.Builder.CreateZExtOrTrunc(RHSScalar, Ty);
    Result = IC.Builder.CreateAShr(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vmul:
    Result = IC.Builder.CreateMul(LHSScalar, RHSScalar);
    break;
  case Intrinsic::riscv_vmulh: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateSExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateSExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateMul(LHSScalar, RHSScalar);
    Result = IC.Builder.CreateAShr(Result, ConstantInt::get(DestTy, TypeWidth));
    Result = IC.Builder.CreateTrunc(Result, Ty);
    break;
  }
  case Intrinsic::riscv_vmulhu: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateZExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateMul(LHSScalar, RHSScalar);
    Result = IC.Builder.CreateLShr(
        Result,
        Constant::getIntegerValue(
            DestTy, APInt(DestTy->getScalarSizeInBits(), TypeWidth, true)));
    Result = IC.Builder.CreateTrunc(Result, Ty);
    break;
  }
  case Intrinsic::riscv_vmulhsu: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateSExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateMul(LHSScalar, RHSScalar);
    Result = IC.Builder.CreateAShr(
        Result,
        Constant::getIntegerValue(
            DestTy, APInt(DestTy->getScalarSizeInBits(), TypeWidth, true)));
    Result = IC.Builder.CreateTrunc(Result, Ty);
    break;
  }
  case Intrinsic::riscv_vfmul:
    Result = IC.Builder.CreateFMul(LHSScalar, RHSScalar);
    break;
  case Intrinsic::riscv_vwmul: {
    unsigned TypeWidth = LHSScalar->getType()->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateSExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateSExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateMul(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwmulu: {
    unsigned TypeWidth = LHSScalar->getType()->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateZExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateMul(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwmulsu: {
    unsigned TypeWidth = LHSScalar->getType()->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateSExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateMul(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwaddu: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateZExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateAdd(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwadd: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateSExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateSExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateAdd(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwsubu: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateZExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateSub(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwsub: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    LHSScalar = IC.Builder.CreateSExt(LHSScalar, DestTy);
    RHSScalar = IC.Builder.CreateSExt(RHSScalar, DestTy);
    Result = IC.Builder.CreateSub(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwaddu_w: {
    Type *Ty = LHSScalar->getType();
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, Ty);
    Result = IC.Builder.CreateAdd(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwadd_w: {
    Type *Ty = LHSScalar->getType();
    RHSScalar = IC.Builder.CreateSExt(RHSScalar, Ty);
    Result = IC.Builder.CreateAdd(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwsubu_w: {
    Type *Ty = LHSScalar->getType();
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, Ty);
    Result = IC.Builder.CreateSub(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vwsub_w: {
    Type *Ty = LHSScalar->getType();
    RHSScalar = IC.Builder.CreateSExt(RHSScalar, Ty);
    Result = IC.Builder.CreateSub(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vnsrl: {
    Type *LHSTy = LHSScalar->getType();
    unsigned TypeWidth = LHSTy->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth / 2);
    RHSScalar = IC.Builder.CreateZExtOrTrunc(RHSScalar, LHSTy);
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, TypeWidth - 1);
    Result = IC.Builder.CreateLShr(LHSScalar, RHSScalar);
    Result = IC.Builder.CreateTrunc(Result, DestTy);
    break;
  }
  case Intrinsic::riscv_vnsra: {
    Type *LHSTy = LHSScalar->getType();
    unsigned TypeWidth = LHSTy->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth / 2);
    RHSScalar = IC.Builder.CreateZExtOrTrunc(RHSScalar, LHSTy);
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, TypeWidth - 1);
    Result = IC.Builder.CreateAShr(LHSScalar, RHSScalar);
    Result = IC.Builder.CreateTrunc(Result, DestTy);
    break;
  }
  case Intrinsic::riscv_vminu: {
    Type *Ty = LHSScalar->getType();
    Result = IC.Builder.CreateIntrinsic(Intrinsic::umin, {Ty},
                                        {LHSScalar, RHSScalar});
    break;
  }
  case Intrinsic::riscv_vmin: {
    Type *Ty = LHSScalar->getType();
    Result = IC.Builder.CreateIntrinsic(Intrinsic::smin, {Ty},
                                        {LHSScalar, RHSScalar});
    break;
  }
  case Intrinsic::riscv_vmaxu: {
    Type *Ty = LHSScalar->getType();
    Result = IC.Builder.CreateIntrinsic(Intrinsic::umax, {Ty},
                                        {LHSScalar, RHSScalar});
    break;
  }
  case Intrinsic::riscv_vmax: {
    Type *Ty = LHSScalar->getType();
    Result = IC.Builder.CreateIntrinsic(Intrinsic::smax, {Ty},
                                        {LHSScalar, RHSScalar});
    break;
  }
  }
  Type *ResultTy = Result->getType();
  Intrinsic::ID IID;
  if (ResultTy->isIntegerTy())
    IID = Intrinsic::riscv_vmv_v_x;
  else if (ResultTy->isFloatingPointTy())
    IID = Intrinsic::riscv_vfmv_v_f;
  else
    return nullptr;

  return CreateIntrinsic(&II, IID, {II.getType(), VL->getType()},
                         {II.getArgOperand(0), Result, VL});
}

static Instruction *foldTernaryOp(InstCombiner &IC, IntrinsicInst &II) {
  bool HasFRM = false;
  switch (II.getIntrinsicID()) {
  case Intrinsic::riscv_vfmacc:
  case Intrinsic::riscv_vfnmacc:
  case Intrinsic::riscv_vfmsac:
  case Intrinsic::riscv_vfnmsac:
  case Intrinsic::riscv_vfmadd:
  case Intrinsic::riscv_vfnmadd:
  case Intrinsic::riscv_vfmsub:
  case Intrinsic::riscv_vfnmsub:
    HasFRM = true;
    break;
  }

  if (HasFRM) {
    auto *FRMC = dyn_cast<ConstantInt>(II.getArgOperand(3));
    if (!FRMC || FRMC->getZExtValue() != 7)
      return nullptr;
  }

  Value *VL = II.getArgOperand(3 + HasFRM);
  Value *VDScalar = getVSplat(II.getArgOperand(0), VL);
  if (!VDScalar)
    return nullptr;
  Value *VS1Scalar = getVSplatOrScalar(II.getArgOperand(1), VL);
  if (!VS1Scalar)
    return nullptr;
  Value *VS2Scalar = getVSplat(II.getArgOperand(2), VL);
  if (!VS2Scalar)
    return nullptr;
  Value *Result;
  switch (II.getIntrinsicID()) {
  case Intrinsic::riscv_vmacc:
    Result = IC.Builder.CreateAdd(VDScalar,
                                  IC.Builder.CreateMul(VS1Scalar, VS2Scalar));
    break;
  case Intrinsic::riscv_vnmsac:
    Result = IC.Builder.CreateSub(VDScalar,
                                  IC.Builder.CreateMul(VS1Scalar, VS2Scalar));
    break;
  case Intrinsic::riscv_vmadd:
    Result = IC.Builder.CreateAdd(VS2Scalar,
                                  IC.Builder.CreateMul(VS1Scalar, VDScalar));
    break;
  case Intrinsic::riscv_vnmsub:
    Result = IC.Builder.CreateSub(VS2Scalar,
                                  IC.Builder.CreateMul(VS1Scalar, VDScalar));
    break;
  case Intrinsic::riscv_vwmaccu: {
    unsigned TypeWidth = VS1Scalar->getType()->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    VS1Scalar = IC.Builder.CreateZExt(VS1Scalar, DestTy);
    VS2Scalar = IC.Builder.CreateZExt(VS2Scalar, DestTy);
    Result = IC.Builder.CreateAdd(VDScalar,
                                  IC.Builder.CreateMul(VS1Scalar, VS2Scalar));
    break;
  }
  case Intrinsic::riscv_vwmacc: {
    unsigned TypeWidth = VS1Scalar->getType()->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    VS1Scalar = IC.Builder.CreateSExt(VS1Scalar, DestTy);
    VS2Scalar = IC.Builder.CreateSExt(VS2Scalar, DestTy);
    Result = IC.Builder.CreateAdd(VDScalar,
                                  IC.Builder.CreateMul(VS1Scalar, VS2Scalar));
    break;
  }
  case Intrinsic::riscv_vwmaccsu: {
    unsigned TypeWidth = VS1Scalar->getType()->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    VS1Scalar = IC.Builder.CreateSExt(VS1Scalar, DestTy);
    VS2Scalar = IC.Builder.CreateZExt(VS2Scalar, DestTy);
    Result = IC.Builder.CreateAdd(VDScalar,
                                  IC.Builder.CreateMul(VS1Scalar, VS2Scalar));
    break;
  }
  case Intrinsic::riscv_vwmaccus: {
    unsigned TypeWidth = VS1Scalar->getType()->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth * 2);
    VS1Scalar = IC.Builder.CreateZExt(VS1Scalar, DestTy);
    VS2Scalar = IC.Builder.CreateSExt(VS2Scalar, DestTy);
    Result = IC.Builder.CreateAdd(VDScalar,
                                  IC.Builder.CreateMul(VS1Scalar, VS2Scalar));
    break;
  }
  case Intrinsic::riscv_vfmacc:
    // vfmacc.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) + vd[i]
    Result = IC.Builder.CreateIntrinsic(Intrinsic::fma, {VDScalar->getType()},
                                        {VS1Scalar, VS2Scalar, VDScalar});
    break;
  case Intrinsic::riscv_vfnmacc:
    // vfnmacc.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) - vd[i]
    Result =
        IC.Builder.CreateIntrinsic(Intrinsic::fma, {VDScalar->getType()},
                                   {IC.Builder.CreateFNeg(VS1Scalar), VS2Scalar,
                                    IC.Builder.CreateFNeg(VDScalar)});
    break;
  case Intrinsic::riscv_vfmsac:
    // vfmsac.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vs2[i]) - vd[i]
    Result = IC.Builder.CreateIntrinsic(
        Intrinsic::fma, {VDScalar->getType()},
        {VS1Scalar, VS2Scalar, IC.Builder.CreateFNeg(VDScalar)});
    break;
  case Intrinsic::riscv_vfnmsac:
    // vfnmsac.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vs2[i]) + vd[i]
    Result = IC.Builder.CreateIntrinsic(
        Intrinsic::fma, {VDScalar->getType()},
        {IC.Builder.CreateFNeg(VS1Scalar), VS2Scalar, VDScalar});
    break;
  case Intrinsic::riscv_vfmadd:
    // vfmadd.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) + vs2[i]
    Result = IC.Builder.CreateIntrinsic(Intrinsic::fma, {VDScalar->getType()},
                                        {VS1Scalar, VDScalar, VS2Scalar});
    break;
  case Intrinsic::riscv_vfnmadd:
    // vfnmadd.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) - vs2[i]
    Result =
        IC.Builder.CreateIntrinsic(Intrinsic::fma, {VDScalar->getType()},
                                   {IC.Builder.CreateFNeg(VS1Scalar), VDScalar,
                                    IC.Builder.CreateFNeg(VS2Scalar)});
    break;
  case Intrinsic::riscv_vfmsub:
    // vfmsub.vv vd, vs1, vs2, vm    # vd[i] = +(vs1[i] * vd[i]) - vs2[i]
    Result = IC.Builder.CreateIntrinsic(
        Intrinsic::fma, {VDScalar->getType()},
        {VS1Scalar, VDScalar, IC.Builder.CreateFNeg(VS2Scalar)});
    break;
  case Intrinsic::riscv_vfnmsub:
    // vfnmsub.vv vd, vs1, vs2, vm   # vd[i] = -(vs1[i] * vd[i]) + vs2[i]
    Result = IC.Builder.CreateIntrinsic(
        Intrinsic::fma, {VDScalar->getType()},
        {IC.Builder.CreateFNeg(VS1Scalar), VDScalar, VS2Scalar});
    break;
  }
  Type *ResultTy = Result->getType();
  Intrinsic::ID IID;
  if (ResultTy->isIntegerTy())
    IID = Intrinsic::riscv_vmv_v_x;
  else if (ResultTy->isFloatingPointTy())
    IID = Intrinsic::riscv_vfmv_v_f;
  else
    return nullptr;

  Value *PassThru = II.getArgOperand(0);
  // FIXME: Only use undef if the policy is agnostic.
  // if (!cast<ConstantInt>(II.getArgOperand(4))->isZero())
  PassThru = UndefValue::get(II.getType());

  return CreateIntrinsic(&II, IID, {II.getType(), VL->getType()},
                         {PassThru, Result, VL});
}

static std::optional<ICmpInst::Predicate> getPredicate(Intrinsic::ID IID) {
  switch (IID) {
  default:
    return std::nullopt;
  case Intrinsic::riscv_vmseq:
    return ICmpInst::ICMP_EQ;
  case Intrinsic::riscv_vmsne:
    return ICmpInst::ICMP_NE;
  case Intrinsic::riscv_vmsltu:
    return ICmpInst::ICMP_ULT;
  case Intrinsic::riscv_vmslt:
    return ICmpInst::ICMP_SLT;
  case Intrinsic::riscv_vmsleu:
    return ICmpInst::ICMP_ULE;
  case Intrinsic::riscv_vmsle:
    return ICmpInst::ICMP_SLE;
  case Intrinsic::riscv_vmsgtu:
    return ICmpInst::ICMP_UGT;
  case Intrinsic::riscv_vmsgt:
    return ICmpInst::ICMP_SGT;
  case Intrinsic::riscv_vmsgeu:
    return ICmpInst::ICMP_UGE;
  case Intrinsic::riscv_vmsge:
    return ICmpInst::ICMP_SGE;
  case Intrinsic::riscv_vmfeq:
    return ICmpInst::FCMP_OEQ;
  case Intrinsic::riscv_vmfne:
    // vmfne writes 1 to the destination element when either operand is NaN,
    // whereas the other comparisons write 0 when either operand is NaN.
    return ICmpInst::FCMP_UNE;
  case Intrinsic::riscv_vmflt:
    return ICmpInst::FCMP_OLT;
  case Intrinsic::riscv_vmfle:
    return ICmpInst::FCMP_OLE;
  case Intrinsic::riscv_vmfgt:
    return ICmpInst::FCMP_OGT;
  case Intrinsic::riscv_vmfge:
    return ICmpInst::FCMP_OGE;
  }
}

// Try to match
//   (vfirst (vmslt (vmv.v.x A), (vmv.v.x B)))
// To
//   (select (icmp slt A, B), 0, -1)
static Instruction *foldVFirstWithCompare(InstCombiner &IC, IntrinsicInst &II) {
  Type *Ty = II.getType();
  Value *Val = II.getArgOperand(0);
  Value *VL = II.getArgOperand(1);

  // If vl == 0, vfirst.m returns -1.
  auto *VLC = dyn_cast<ConstantInt>(VL);
  if (VLC && VLC->isZero()) {
    Value *MinusOne = Constant::getIntegerValue(
        Ty, APInt(Ty->getScalarSizeInBits(), -1, true));
    return IC.replaceInstUsesWith(II, MinusOne);
  }

  // The remaining cases are vl is not zero or we don't know it.
  // If we don't know it, skip the folding.
  if (!isKnownNonZero(VL, {IC.getDataLayout()}))
    return nullptr;

  auto *ValII = dyn_cast<IntrinsicInst>(Val);
  if (!ValII)
    return nullptr;

  std::optional<ICmpInst::Predicate> Pred =
      getPredicate(ValII->getIntrinsicID());
  if (!Pred.has_value())
    return nullptr;
  if (ValII->getArgOperand(2) != VL)
    return nullptr;

  Value *LHSScalar = getVSplat(ValII->getArgOperand(0), VL);
  if (!LHSScalar)
    return nullptr;
  Value *RHSScalar = getVSplatOrScalar(ValII->getArgOperand(1), VL);
  if (!RHSScalar)
    return nullptr;

  Value *CmpV;
  if (LHSScalar->getType()->isIntegerTy())
    CmpV = IC.Builder.CreateICmp(Pred.value(), LHSScalar, RHSScalar);
  else if (LHSScalar->getType()->isFloatingPointTy())
    CmpV = IC.Builder.CreateFCmp(Pred.value(), LHSScalar, RHSScalar);
  else
    return nullptr;

  Value *V = IC.Builder.CreateSelect(
      CmpV, Constant::getNullValue(Ty),
      Constant::getIntegerValue(Ty,
                                APInt(Ty->getScalarSizeInBits(), -1, true)));

  return IC.replaceInstUsesWith(II, V);
}

//   (vmerge (vmslt (vmv.v.x A), (vmv.v.x B)), c, d)
// To
//   (select (icmp slt A, B), d, c)
static Instruction *foldVMergeWithCompare(InstCombiner &IC, IntrinsicInst &II) {
  if (!isa<UndefValue>(II.getArgOperand(0)))
    return nullptr;

  Value *LHS = II.getArgOperand(1);
  Value *RHS = II.getArgOperand(2);
  Value *Mask = II.getArgOperand(3);
  Value *VL = II.getArgOperand(4);

  auto *MaskII = dyn_cast<IntrinsicInst>(Mask);
  if (!MaskII)
    return nullptr;

  std::optional<ICmpInst::Predicate> Pred =
      getPredicate(MaskII->getIntrinsicID());
  if (!Pred.has_value())
    return nullptr;
  if (MaskII->getArgOperand(2) != VL)
    return nullptr;

  Value *MaskLHSScalar = getVSplat(MaskII->getArgOperand(0), VL);
  if (!MaskLHSScalar)
    return nullptr;
  Value *MaskRHSScalar = getVSplatOrScalar(MaskII->getArgOperand(1), VL);
  if (!MaskRHSScalar)
    return nullptr;

  if (RHS->getType()->isIntegerTy())
    RHS = IC.Builder.CreateIntrinsic(
        Intrinsic::riscv_vmv_v_x, {LHS->getType(), VL->getType()},
        {UndefValue::get(LHS->getType()), RHS, VL});
  else if (RHS->getType()->isFloatingPointTy())
    RHS = IC.Builder.CreateIntrinsic(
        Intrinsic::riscv_vfmv_v_f, {LHS->getType(), VL->getType()},
        {UndefValue::get(LHS->getType()), RHS, VL});
  else
    assert(RHS->getType()->isVectorTy() && "vmerge has an unexpected type.");

  Value *CmpV;
  if (MaskLHSScalar->getType()->isIntegerTy())
    CmpV = IC.Builder.CreateICmp(Pred.value(), MaskLHSScalar, MaskRHSScalar);
  else if (MaskLHSScalar->getType()->isFloatingPointTy())
    CmpV = IC.Builder.CreateFCmp(Pred.value(), MaskLHSScalar, MaskRHSScalar);
  else
    llvm_unreachable("Unexpected types for scalar comparison.");

  return SelectInst::Create(CmpV, RHS, LHS);
}

// Fold (vmv.x.s (vmv.v.x X, vl)) -> X
static Instruction *foldVMV_X_S(InstCombiner &IC, IntrinsicInst &II) {
  auto *ValII = dyn_cast<IntrinsicInst>(II.getArgOperand(0));
  if (!ValII || ValII->getIntrinsicID() != Intrinsic::riscv_vmv_v_x ||
      !isa<UndefValue>(ValII->getArgOperand(0)))
    return nullptr;

  // Replace with the scalar input to the vmv.v.x.
  // NOTE: If the VL of the vmv.v.x is zero, the vector value is undefined
  // since it uses a tail agnostic policy. We should still be allowed to fold
  // to the scalar in that case so we don't need to check VL.
  return IC.replaceInstUsesWith(II, ValII->getArgOperand(1));
}

// Fold (vfmv.f.s (vfmv.v.f X, vl)) -> X
static Instruction *foldVMV_F_S(InstCombiner &IC, IntrinsicInst &II) {
  // Look for a splat from scalar.
  auto *ValII = dyn_cast<IntrinsicInst>(II.getArgOperand(0));
  if (!ValII || ValII->getIntrinsicID() != Intrinsic::riscv_vfmv_v_f ||
      !isa<UndefValue>(ValII->getArgOperand(0)))
    return nullptr;

  // Replace with the scalar input to the vfmv.v.f.
  // NOTE: If the VL of the vfmv.v.f is zero, the vector value is undefined
  // since it uses a tail agnostic policy. We should still be allowed to fold
  // to the scalar in that case so we don't need to check VL.
  return IC.replaceInstUsesWith(II, ValII->getArgOperand(1));
}

static Instruction *foldVMSNEWithVPCompare(InstCombiner &IC,
                                           IntrinsicInst &II) {
  using namespace PatternMatch;
  // TODO: Reverse predicate if intrinsics compares with one
  if (!match(II.getOperand(1), m_ZeroInt()))
    return nullptr;

  Value *Pred, *VL, *Mask;
  if (!match(II.getOperand(0), m_OneUse(m_Intrinsic<Intrinsic::vp_zext>(
                                   m_Value(Pred), m_Value(Mask), m_Value(VL)))))
    return nullptr;

  if (!maskIsAllOneOrUndef(Mask) || !Pred->getType()->isIntOrIntVectorTy(1))
    return nullptr;

  if (!match(VL, m_TruncOrSelf(m_Specific(II.getOperand(2)))))
    return nullptr;

  if (auto *VPIntrin = dyn_cast<VPIntrinsic>(Pred))
    if (VPIntrin->getVectorLengthParam() != VL)
      return nullptr;

  return IC.replaceInstUsesWith(II, Pred);
}

// Return true if II is an intrinsic to widen signed elements.
static bool isSignedWcvt(IntrinsicInst *II) {
  return II->getIntrinsicID() == Intrinsic::riscv_vwadd ||
         II->getIntrinsicID() == Intrinsic::riscv_vsext;
}

// Transform (v<bop> (wext a) (wext b)) into (vw<bop> a, b)
static Instruction *foldVwcvtWithVBinaryOp(InstCombiner &IC, IntrinsicInst &II) {
  // Return true if V is a widening conversion intrinsic with VL vector length.
  auto isWcvtWithVL = [](IntrinsicInst *II, Value *VL) {
    if (!II)
      return false;
    switch (II->getIntrinsicID()) {
    default:
      return false;
    case Intrinsic::riscv_vwadd:
    case Intrinsic::riscv_vwaddu: {
      auto *C = dyn_cast<ConstantInt>(II->getArgOperand(2));
      return C && C->isZero() && isa<UndefValue>(II->getArgOperand(0)) && II->getArgOperand(3) == VL;
    }
    case Intrinsic::riscv_vzext:
    case Intrinsic::riscv_vsext:
      // Only use v[sz]ext.vf2 to make up widen operations.
      if (II->getType()->getScalarSizeInBits() !=
          2 * II->getArgOperand(1)->getType()->getScalarSizeInBits())
        return false;
      return isa<UndefValue>(II->getArgOperand(0)) && II->getArgOperand(2) == VL;
    case Intrinsic::riscv_vfwcvt_f_f_v:
      return isa<UndefValue>(II->getArgOperand(0)) && II->getArgOperand(2) == VL;
    }
  };

  bool HasFRM = II.arg_size() == 5;

  auto *Op0 = dyn_cast<IntrinsicInst>(II.getArgOperand(1));
  auto *Op1 = dyn_cast<IntrinsicInst>(II.getArgOperand(2));
  Value *VL = II.getArgOperand(3 + HasFRM);
  if (!isWcvtWithVL(Op0, VL) || !isWcvtWithVL(Op1, VL))
    return nullptr;

  bool Signed, MixSigned = false;
  if (isSignedWcvt(Op0) == isSignedWcvt(Op1)) {
    Signed = isSignedWcvt(Op0);
  } else {
    // vwmulsu is the only binary operation that uses both of signed-widening
    // and unsigned-widening.
    if (II.getIntrinsicID() != Intrinsic::riscv_vmul)
      return nullptr;
    if (isSignedWcvt(Op1))
      std::swap(Op0, Op1);
    MixSigned = true;
  }

  Intrinsic::ID NewOp;
  switch (II.getIntrinsicID()) {
  default:
    llvm_unreachable(
        "Unexpected instruction not having variety to widen arguments");
  case Intrinsic::riscv_vadd:
    NewOp = Signed ? Intrinsic::riscv_vwadd : Intrinsic::riscv_vwaddu;
    break;
  case Intrinsic::riscv_vsub:
    NewOp = Signed ? Intrinsic::riscv_vwsub : Intrinsic::riscv_vwsubu;
    break;
  case Intrinsic::riscv_vfadd:
    NewOp = Intrinsic::riscv_vfwadd;
    break;
  case Intrinsic::riscv_vfsub:
    NewOp = Intrinsic::riscv_vfwsub;
    break;
  case Intrinsic::riscv_vmul:
    NewOp = MixSigned
                ? Intrinsic::riscv_vwmulsu
                : (Signed ? Intrinsic::riscv_vwmul : Intrinsic::riscv_vwmulu);
    break;
  case Intrinsic::riscv_vfmul:
    NewOp = Intrinsic::riscv_vfwmul;
    break;
  }
  if (HasFRM)
    return CreateIntrinsic(&II, NewOp,
                           {II.getType(), Op0->getArgOperand(1)->getType(),
                            Op1->getArgOperand(1)->getType(), VL->getType()},
                           {II.getArgOperand(0), Op0->getArgOperand(1),
                            Op1->getArgOperand(1), II.getArgOperand(3), VL});

  return CreateIntrinsic(&II, NewOp,
                         {II.getType(), Op0->getArgOperand(1)->getType(),
                          Op1->getArgOperand(1)->getType(), VL->getType()},
                         {II.getArgOperand(0), Op0->getArgOperand(1), Op1->getArgOperand(1), VL});
}

static Instruction *foldVBroadcast(InstCombiner &IC, IntrinsicInst &II) {
  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  case Intrinsic::riscv_vaadd:
  case Intrinsic::riscv_vaaddu:
  case Intrinsic::riscv_vsmul:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3),
           II.getArgOperand(4)});
    // These instructions are commutable so check the other operand.
    if (II.getArgOperand(2)->getType()->isVectorTy()) {
      if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(3)))
        return CreateIntrinsic(
            &II, IID,
            {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
            {II.getArgOperand(0), II.getArgOperand(2), V, II.getArgOperand(3),
             II.getArgOperand(4)});
    }
    break;
  case Intrinsic::riscv_vadd:
  case Intrinsic::riscv_vand:
  case Intrinsic::riscv_vmax:
  case Intrinsic::riscv_vmaxu:
  case Intrinsic::riscv_vmin:
  case Intrinsic::riscv_vminu:
  case Intrinsic::riscv_vmul:
  case Intrinsic::riscv_vmulh:
  case Intrinsic::riscv_vmulhu:
  case Intrinsic::riscv_vor:
  case Intrinsic::riscv_vsadd:
  case Intrinsic::riscv_vsaddu:
  case Intrinsic::riscv_vsub:
  case Intrinsic::riscv_vxor:
  case Intrinsic::riscv_vfmax:
  case Intrinsic::riscv_vfmin:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3)});
    // These instructions are commutable so check the other operand.
    if (II.getArgOperand(2)->getType()->isVectorTy()) {
      if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(3))) {
        // Some intrinsics need their opcode changed to commute them.
        switch (IID) {
        default:
          break;
        case Intrinsic::riscv_vsub:  IID = Intrinsic::riscv_vrsub;  break;
        case Intrinsic::riscv_vfdiv: IID = Intrinsic::riscv_vfrdiv; break;
        case Intrinsic::riscv_vfsub: IID = Intrinsic::riscv_vfrsub; break;
        }
        return CreateIntrinsic(
            &II, IID,
            {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
            {II.getArgOperand(0), II.getArgOperand(2), V, II.getArgOperand(3)});
      }
    }
    break;
  case Intrinsic::riscv_vfadd:
  case Intrinsic::riscv_vfdiv:
  case Intrinsic::riscv_vfmul:
  case Intrinsic::riscv_vfsub:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(4)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3),
           II.getArgOperand(4)});
    // These instructions are commutable so check the other operand.
    if (II.getArgOperand(2)->getType()->isVectorTy()) {
      if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(4))) {
        // Some intrinsics need their opcode changed to commute them.
        switch (IID) {
        default:
          break;
        case Intrinsic::riscv_vsub:
          IID = Intrinsic::riscv_vrsub;
          break;
        case Intrinsic::riscv_vfdiv:
          IID = Intrinsic::riscv_vfrdiv;
          break;
        case Intrinsic::riscv_vfsub:
          IID = Intrinsic::riscv_vfrsub;
          break;
        }
        return CreateIntrinsic(
            &II, IID,
            {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
            {II.getArgOperand(0), II.getArgOperand(2), V, II.getArgOperand(3),
             II.getArgOperand(4)});
      }
    }
    break;
  case Intrinsic::riscv_vasub:
  case Intrinsic::riscv_vasubu:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3),
           II.getArgOperand(4)});

    // These instructions are not commutable.
    break;
  case Intrinsic::riscv_vdiv:
  case Intrinsic::riscv_vdivu:
  case Intrinsic::riscv_vmulhsu:
  case Intrinsic::riscv_vrem:
  case Intrinsic::riscv_vremu:
  case Intrinsic::riscv_vssub:
  case Intrinsic::riscv_vssubu:
  case Intrinsic::riscv_vwadd_w:
  case Intrinsic::riscv_vwaddu_w:
  case Intrinsic::riscv_vwsub_w:
  case Intrinsic::riscv_vwsubu_w:
  case Intrinsic::riscv_vfsgnj:
  case Intrinsic::riscv_vfsgnjn:
  case Intrinsic::riscv_vfsgnjx:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3)});
    // These instructions are not commutable.
    break;
  case Intrinsic::riscv_vfwadd_w:
  case Intrinsic::riscv_vfwsub_w:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(4)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3),
           II.getArgOperand(4)});
    // These instructions are not commutable.
    break;
  case Intrinsic::riscv_vssrl:
  case Intrinsic::riscv_vssra:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3))) {
      // Expect II.getArgOperand(2)->getType() is a XLen value type.
      Value *ShiftAmount =
          IC.Builder.CreateZExtOrTrunc(V, II.getArgOperand(3)->getType());
      return CreateIntrinsic(&II, IID,
                             {II.getType(), ShiftAmount->getType(),
                              II.getArgOperand(3)->getType()},
                             {II.getArgOperand(0), II.getArgOperand(1),
                              ShiftAmount, II.getArgOperand(3),
                              II.getArgOperand(4)});
    }
    break;
  case Intrinsic::riscv_vsll:
  case Intrinsic::riscv_vsrl:
  case Intrinsic::riscv_vsra:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3))) {
      // Expect II.getArgOperand(2)->getType() is a XLen value type.
      Value *ShiftAmount =
          IC.Builder.CreateZExtOrTrunc(V, II.getArgOperand(3)->getType());
      return CreateIntrinsic(&II, IID,
                             {II.getType(), ShiftAmount->getType(),
                              II.getArgOperand(3)->getType()},
                             {II.getArgOperand(0), II.getArgOperand(1),
                              ShiftAmount, II.getArgOperand(3)});
    }
    break;
  case Intrinsic::riscv_vwadd:
  case Intrinsic::riscv_vwaddu:
  case Intrinsic::riscv_vwmul:
  case Intrinsic::riscv_vwmulu:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), II.getArgOperand(1)->getType(), V->getType(),
           II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3)});
    // These instructions are commutable so check the other operand.
    if (II.getArgOperand(2)->getType()->isVectorTy())
      if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(3)))
        return CreateIntrinsic(
            &II, IID,
            {II.getType(), II.getArgOperand(2)->getType(), V->getType(),
             II.getArgOperand(3)->getType()},
            {II.getArgOperand(0), II.getArgOperand(2), V, II.getArgOperand(3)});
    break;
  case Intrinsic::riscv_vfwadd:
  case Intrinsic::riscv_vfwmul:
    // These instructions are commutable so check the other operand.
    if (II.getArgOperand(2)->getType()->isVectorTy())
      if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(4)))
        return CreateIntrinsic(&II, IID,
                               {II.getType(), II.getArgOperand(2)->getType(),
                                V->getType(), II.getArgOperand(3)->getType()},
                               {II.getArgOperand(0), II.getArgOperand(2), V,
                                II.getArgOperand(3), II.getArgOperand(4)});
    [[fallthrough]];
  case Intrinsic::riscv_vfwsub:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(4)))
      return CreateIntrinsic(&II, IID,
                             {II.getType(), II.getArgOperand(1)->getType(),
                              V->getType(), II.getArgOperand(3)->getType()},
                             {II.getArgOperand(0), II.getArgOperand(1), V,
                              II.getArgOperand(3), II.getArgOperand(4)});
    break;
  case Intrinsic::riscv_vwmulsu:
  case Intrinsic::riscv_vwsub:
  case Intrinsic::riscv_vwsubu:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), II.getArgOperand(1)->getType(), V->getType(),
           II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3)});
    // These instructions are not commutable.
    break;
  case Intrinsic::riscv_vnsrl:
  case Intrinsic::riscv_vnsra:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3))) {
      // Expect II.getArgOperand(2)->getType() is a XLen value type.
      Value *ShiftAmount =
          IC.Builder.CreateZExtOrTrunc(V, II.getArgOperand(3)->getType());
      return CreateIntrinsic(&II, IID,
                             {II.getType(), II.getArgOperand(1)->getType(),
                              ShiftAmount->getType(),
                              II.getArgOperand(3)->getType()},
                             {II.getArgOperand(0), II.getArgOperand(1),
                              ShiftAmount, II.getArgOperand(3)});
    }
    break;
  case Intrinsic::riscv_vmacc:
  case Intrinsic::riscv_vnmsac:
    if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), V, II.getArgOperand(2), II.getArgOperand(3),
           II.getArgOperand(4)});
    // These instructions are commutable so check the other multiply operand.
    if (II.getArgOperand(1)->getType()->isVectorTy())
      if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3)))
        return CreateIntrinsic(
            &II, IID,
            {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
            {II.getArgOperand(0), V, II.getArgOperand(1), II.getArgOperand(3),
             II.getArgOperand(4)});
    // FIXME: If these were tail agnostic there would be more commuting
    // options for these intrinsics.
    break;
  case Intrinsic::riscv_vfmacc:
  case Intrinsic::riscv_vfnmacc:
  case Intrinsic::riscv_vfmsac:
  case Intrinsic::riscv_vfnmsac:
    if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(4)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), V, II.getArgOperand(2), II.getArgOperand(3),
           II.getArgOperand(4), II.getArgOperand(5)});
    // These instructions are commutable so check the other multiply operand.
    if (II.getArgOperand(1)->getType()->isVectorTy())
      if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(4)))
        return CreateIntrinsic(
            &II, IID,
            {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
            {II.getArgOperand(0), V, II.getArgOperand(1), II.getArgOperand(3),
             II.getArgOperand(4), II.getArgOperand(5)});
    // FIXME: If these were tail agnostic there would be more commuting
    // options for these intrinsics.
    break;
  case Intrinsic::riscv_vmadd:
  case Intrinsic::riscv_vnmsub:
    if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), V, II.getArgOperand(2), II.getArgOperand(3),
           II.getArgOperand(4)});
    // FIXME: If these were tail agnostic there would be more commuting
    // options for these intrinsics.
    break;
  case Intrinsic::riscv_vfmadd:
  case Intrinsic::riscv_vfnmadd:
  case Intrinsic::riscv_vfmsub:
  case Intrinsic::riscv_vfnmsub:
    if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(4)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), V, II.getArgOperand(2), II.getArgOperand(3),
           II.getArgOperand(4), II.getArgOperand(5)});
    // FIXME: If these were tail agnostic there would be more commuting
    // options for these intrinsics.
    break;
  case Intrinsic::riscv_vwmacc:
  case Intrinsic::riscv_vwmaccu:
  case Intrinsic::riscv_vwmaccsu:
    if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(3)))
      return CreateIntrinsic(&II, IID,
                             {II.getType(), V->getType(),
                              II.getArgOperand(2)->getType(),
                              II.getArgOperand(3)->getType()},
                             {II.getArgOperand(0), V, II.getArgOperand(2),
                              II.getArgOperand(3), II.getArgOperand(4)});
    // These instructions are commutable so check the other multiply operand.
    if (II.getArgOperand(1)->getType()->isVectorTy()) {
      // Commute the opcode for vwmaccsu.
      if (IID == Intrinsic::riscv_vwmaccsu)
        IID = Intrinsic::riscv_vwmaccus;
      if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3))) {
        return CreateIntrinsic(&II, IID,
                               {II.getType(), V->getType(),
                                II.getArgOperand(1)->getType(),
                                II.getArgOperand(3)->getType()},
                               {II.getArgOperand(0), V, II.getArgOperand(1),
                                II.getArgOperand(3), II.getArgOperand(4)});
      }
    }
    break;
  case Intrinsic::riscv_vfwmacc:
  case Intrinsic::riscv_vfwnmacc:
  case Intrinsic::riscv_vfwmsac:
  case Intrinsic::riscv_vfwnmsac:
    if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(4)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(2)->getType(),
           II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), V, II.getArgOperand(2), II.getArgOperand(3),
           II.getArgOperand(4), II.getArgOperand(5)});
    // These instructions are commutable so check the other multiply operand.
    if (II.getArgOperand(1)->getType()->isVectorTy()) {
      // Commute the opcode for vwmaccsu.
      if (IID == Intrinsic::riscv_vwmaccsu)
        IID = Intrinsic::riscv_vwmaccus;
      if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(4))) {
        return CreateIntrinsic(
            &II, IID,
            {II.getType(), V->getType(), II.getArgOperand(1)->getType(),
             II.getArgOperand(3)->getType()},
            {II.getArgOperand(0), V, II.getArgOperand(1), II.getArgOperand(3),
             II.getArgOperand(4), II.getArgOperand(5)});
      }
    }
    break;
  case Intrinsic::riscv_vmsne:
    if (Instruction *R = foldVMSNEWithVPCompare(IC, II))
      return R;
    [[fallthrough]];
  case Intrinsic::riscv_vmseq:
  case Intrinsic::riscv_vmslt:
  case Intrinsic::riscv_vmsltu:
  case Intrinsic::riscv_vmsle:
  case Intrinsic::riscv_vmsleu:
  case Intrinsic::riscv_vmsgt:
  case Intrinsic::riscv_vmsgtu:
  case Intrinsic::riscv_vmsge:
  case Intrinsic::riscv_vmsgeu:
  case Intrinsic::riscv_vmfeq:
  case Intrinsic::riscv_vmfne:
  case Intrinsic::riscv_vmflt:
  case Intrinsic::riscv_vmfle:
  case Intrinsic::riscv_vmfgt:
  case Intrinsic::riscv_vmfge:
    if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(2)))
      return CreateIntrinsic(&II, IID,
                             {II.getArgOperand(0)->getType(), V->getType(),
                              II.getArgOperand(2)->getType()},
                             {II.getArgOperand(0), V, II.getArgOperand(2)});
    // These instructions are commutable so check the other operand.
    if (II.getArgOperand(1)->getType()->isVectorTy()) {
      if (Value *V = getVSplat(II.getArgOperand(0), II.getArgOperand(2))) {
        // Most of these intrinsics need their opcode changed to commute them.
        switch (IID) {
        default:
          break;
        case Intrinsic::riscv_vmslt:  IID = Intrinsic::riscv_vmsgt;  break;
        case Intrinsic::riscv_vmsltu: IID = Intrinsic::riscv_vmsgtu; break;
        case Intrinsic::riscv_vmsle:  IID = Intrinsic::riscv_vmsge;  break;
        case Intrinsic::riscv_vmsleu: IID = Intrinsic::riscv_vmsgeu; break;
        case Intrinsic::riscv_vmsgt:  IID = Intrinsic::riscv_vmslt;  break;
        case Intrinsic::riscv_vmsgtu: IID = Intrinsic::riscv_vmsltu; break;
        case Intrinsic::riscv_vmsge:  IID = Intrinsic::riscv_vmsle;  break;
        case Intrinsic::riscv_vmsgeu: IID = Intrinsic::riscv_vmsleu; break;
        case Intrinsic::riscv_vmflt:  IID = Intrinsic::riscv_vmfgt;  break;
        case Intrinsic::riscv_vmfle:  IID = Intrinsic::riscv_vmfge;  break;
        case Intrinsic::riscv_vmfgt:  IID = Intrinsic::riscv_vmflt;  break;
        case Intrinsic::riscv_vmfge:  IID = Intrinsic::riscv_vmfle;  break;
        }
        return CreateIntrinsic(&II, IID,
                               {II.getArgOperand(1)->getType(), V->getType(),
                                II.getArgOperand(2)->getType()},
                               {II.getArgOperand(1), V, II.getArgOperand(2)});
      }
    }
    break;
  case Intrinsic::riscv_vmerge:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(4))) {
      // For floating point we need to change the intrinsic.
      if (V->getType()->isFloatingPointTy())
        IID = Intrinsic::riscv_vfmerge;
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(4)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3),
           II.getArgOperand(4)});
    }
  }

  return nullptr;
}

// Fold
//   (vmv.x.s (vrgather.vx (vle X), Y, 1)) -> load (X + Y)
//   (vfmv.f.s (vrgather.vx (vle X), Y, 1)) -> load (X + Y)
//   (vmv.x.s (vle X)) -> load (X)
//   (vfmv.f.s (vle X)) -> load (X)
static Instruction *foldVmvVRgatherVle(InstCombiner &IC, IntrinsicInst &II,
                                       const RISCVSubtarget *ST) {
  if (II.getIntrinsicID() != Intrinsic::riscv_vmv_x_s &&
      II.getIntrinsicID() != Intrinsic::riscv_vfmv_f_s)
    return nullptr;
  if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(0))) {
    uint64_t Offset;
    IntrinsicInst *Vle;
    if (II2->getIntrinsicID() == Intrinsic::riscv_vrgather_vx &&
        isa<UndefValue>(II2->getArgOperand(0)) &&
        isa<ConstantInt>(II2->getArgOperand(2)) &&
        isa<ConstantInt>(II2->getArgOperand(3)) &&
        cast<ConstantInt>(II2->getArgOperand(3))->getZExtValue() == 1) {
      Offset = cast<ConstantInt>(II2->getArgOperand(2))->getZExtValue();
      if (getVLMAX(cast<ScalableVectorType>(II2->getType()), ST) <= Offset)
        return nullptr;
      if (auto *II3 = dyn_cast<IntrinsicInst>(II2->getArgOperand(1)))
        Vle = II3;
      else
        return nullptr;
    } else {
      Offset = 0;
      Vle = II2;
    }
    if (Vle->getIntrinsicID() == Intrinsic::riscv_vle &&
        isa<UndefValue>(Vle->getArgOperand(0)) &&
        isa<ConstantInt>(Vle->getArgOperand(2)) &&
        !cast<ConstantInt>(Vle->getArgOperand(2))->isZero()) {
      PointerType *SrcPtrTy = Vle->getType()->getScalarType()->getPointerTo();
      IRBuilderBase::InsertPointGuard Guard(IC.Builder);
      IC.Builder.SetInsertPoint(Vle);
      Value *SrcPtr =
          IC.Builder.CreatePointerCast(Vle->getArgOperand(1), SrcPtrTy);
      SrcPtr = IC.Builder.CreateGEP(Vle->getType()->getScalarType(), SrcPtr,
                                    IC.Builder.getIntN(ST->getXLen(), Offset));
      return IC.replaceInstUsesWith(
          II, IC.Builder.CreateLoad(Vle->getType()->getScalarType(), SrcPtr));
    }
  }
  return nullptr;
}
std::optional<Instruction *>
RISCVTTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  if (DisableVectorOpt)
    return std::nullopt;

  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  default:
    break;
  case Intrinsic::riscv_vand: {
    if (!isa<UndefValue>(II.getArgOperand(0)))
      break;
    Value *LHS = II.getArgOperand(1);
    Value *RHS = II.getArgOperand(2);
    Value *VL = II.getArgOperand(3);
    if (Instruction *V = foldVAndWithVMerge(LHS, RHS, VL))
      return V;
    // And is commutable, try the other order.
    if (Instruction *V = foldVAndWithVMerge(RHS, LHS, VL))
      return V;
    // If two sources are both scalar, fold it to scalar operation + vmv.v.x.
    if (Instruction *V = foldBinaryOp(IC, II))
      return V;
    break;
  }
  case Intrinsic::riscv_vxor: {
    if (!isa<UndefValue>(II.getArgOperand(0)))
      break;
    Value *LHS = II.getArgOperand(1);
    Value *RHS = II.getArgOperand(2);
    Value *VL = II.getArgOperand(3);

    if (Instruction *V = foldVXorWithVMergeVXor(LHS, RHS, VL))
      return V;
    // Xor is commutable, try the other order.
    if (Instruction *V = foldVXorWithVMergeVXor(RHS, LHS, VL))
      return V;
    // If two sources are both scalar, fold it to scalar operation + vmv.v.x.
    if (Instruction *V = foldBinaryOp(IC, II))
      return V;

    break;
  }
  case Intrinsic::riscv_vfcvt_f_xu_v:
    // Try to convert to vwfcvt_f_xu_v.
    if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(1))) {
      // Look for the vwcvtu.x.x.v idiom with the same VL.
      if (II2->getIntrinsicID() == Intrinsic::riscv_vwaddu &&
          isa<UndefValue>(II2->getArgOperand(0)) &&
          II2->getArgOperand(3) == II.getArgOperand(2) &&
          isa<ConstantInt>(II2->getArgOperand(2)) &&
          cast<ConstantInt>(II2->getArgOperand(2))->isZero())
        return CreateIntrinsic(
            &II, Intrinsic::riscv_vfwcvt_f_xu_v,
            {II.getType(), II2->getArgOperand(1)->getType(),
             II.getArgOperand(2)->getType()},
            {II.getArgOperand(0), II2->getArgOperand(1), II.getArgOperand(2)});
    }
    break;
  case Intrinsic::riscv_vfcvt_f_x_v:
    // Try to convert to vwfcvt_f_x_v.
    if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(1))) {
      // Look for the vwcvt.x.x.v idiom with the same VL.
      if (II2->getIntrinsicID() == Intrinsic::riscv_vwadd &&
          isa<UndefValue>(II2->getArgOperand(0)) &&
          II2->getArgOperand(3) == II.getArgOperand(2) &&
          isa<ConstantInt>(II2->getArgOperand(2)) &&
          cast<ConstantInt>(II2->getArgOperand(2))->isZero())
        return CreateIntrinsic(
            &II, Intrinsic::riscv_vfwcvt_f_x_v,
            {II.getType(), II2->getArgOperand(1)->getType(),
             II.getArgOperand(2)->getType()},
            {II.getArgOperand(0), II2->getArgOperand(1), II.getArgOperand(2)});
    }
    break;
  case Intrinsic::riscv_vadd:
    if (isa<UndefValue>(II.getArgOperand(0))) {
      for (int i = 0; i != 2; ++i) {
        if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(1 + i))) {
          if (II2->getIntrinsicID() == Intrinsic::riscv_vmul &&
              isa<UndefValue>(II2->getArgOperand(0)) &&
              II.getArgOperand(3) == II2->getArgOperand(3) &&
              // cannot combine vmul + vadd.vx to vmacc
              II.getArgOperand(1 + (1 - i))->getType()->isVectorTy()) {
            // Add a tail agnostic policy
            Value *Policy = ConstantInt::get(II.getArgOperand(3)->getType(), 1);
            return CreateIntrinsic(
                &II, Intrinsic::riscv_vmacc,
                {II.getType(), II2->getArgOperand(2)->getType(),
                 II.getArgOperand(3)->getType()},
                {II.getArgOperand(1 + (1 - i)), II2->getArgOperand(2),
                 II2->getArgOperand(1), II.getArgOperand(3), Policy});
          }
        }
      }
    }

    if (Instruction *V = foldVwcvtWithVBinaryOp(IC, II))
      return V;

    if (Instruction *V = foldBinaryOp(IC, II))
      return V;
    break;
  case Intrinsic::riscv_vmul:
    if (Instruction *V = foldVwcvtWithVBinaryOp(IC, II))
      return V;
    if (Instruction *V = foldBinaryOp(IC, II))
      return V;
    break;
  case Intrinsic::riscv_vfmul:
    if (Instruction *V = foldVwcvtWithVBinaryOp(IC, II))
      return V;
    if (Instruction *V = foldBinaryOp(IC, II))
      return V;
    break;
  case Intrinsic::riscv_vor: {
    if (isa<UndefValue>(II.getArgOperand(0))) {
      // combine (vor (vsll (vwaddu a, 0), HalfSewOfVor), (vwaddu, c, 0)) to
      //         (vwmaccu (vwaddu a, c), (1 << HalfSewOfVor) - 1, a)
      unsigned HalfSewOfVor = II.getType()->getScalarSizeInBits() / 2;
      Value *VL = II.getArgOperand(3);
      for (int i = 0; i != 2; ++i) {
        if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(1 + i))) {
          if (auto *II3 =
                  dyn_cast<IntrinsicInst>(II.getArgOperand(1 + (1 - i)))) {
            if (II2->getIntrinsicID() == Intrinsic::riscv_vsll &&
                isa<UndefValue>(II2->getArgOperand(0)) &&
                isa<ConstantInt>(II2->getArgOperand(2)) &&
                cast<ConstantInt>(II2->getArgOperand(2))->getZExtValue() ==
                    HalfSewOfVor &&
                VL == II2->getArgOperand(3) &&
                II3->getIntrinsicID() == Intrinsic::riscv_vwaddu &&
                isa<UndefValue>(II3->getArgOperand(0)) &&
                isa<ConstantInt>(II3->getArgOperand(2)) &&
                cast<ConstantInt>(II3->getArgOperand(2))->isZero() &&
                VL == II3->getArgOperand(3)) {
              if (auto *II4 = dyn_cast<IntrinsicInst>(II2->getArgOperand(1))) {
                if (II4->getIntrinsicID() == Intrinsic::riscv_vwaddu &&
                    isa<UndefValue>(II4->getArgOperand(0)) &&
                    isa<ConstantInt>(II4->getArgOperand(2)) &&
                    cast<ConstantInt>(II4->getArgOperand(2))->isZero() &&
                    VL == II4->getArgOperand(3)) {
                  Value *Vwaddu = IC.Builder.CreateIntrinsic(
                      Intrinsic::riscv_vwaddu,
                      {II.getType(), II4->getArgOperand(1)->getType(),
                       II3->getArgOperand(1)->getType(), VL->getType()},
                      {UndefValue::get(II.getType()), II4->getArgOperand(1),
                       II3->getArgOperand(1), VL});
                  ConstantInt *Multiplier = ConstantInt::get(
                      Type::getIntNTy(II.getContext(), HalfSewOfVor),
                      (1 << HalfSewOfVor) - 1);
                  Value *Policy =
                      ConstantInt::get(II.getArgOperand(3)->getType(), 1);
                  return CreateIntrinsic(
                      &II, Intrinsic::riscv_vwmaccu,
                      {Vwaddu->getType(), Multiplier->getType(),
                       II4->getArgOperand(1)->getType(), VL->getType()},
                      {Vwaddu, Multiplier, II4->getArgOperand(1), VL, Policy});
                }
              }
            }
          }
        }
      }
    }
    if (Instruction *V = foldBinaryOp(IC, II))
      return V;
    break;
  }
  case Intrinsic::riscv_vsub:
  case Intrinsic::riscv_vfadd:
  case Intrinsic::riscv_vfsub:
    if (Instruction *V = foldVwcvtWithVBinaryOp(IC, II))
      return V;
    if (Instruction *V = foldBinaryOp(IC, II))
      return V;
    break;
  case Intrinsic::riscv_vrsub:
  case Intrinsic::riscv_vfrsub:
  case Intrinsic::riscv_vsll:
  case Intrinsic::riscv_vsrl:
  case Intrinsic::riscv_vsra:
  case Intrinsic::riscv_vmulh:
  case Intrinsic::riscv_vmulhu:
  case Intrinsic::riscv_vmulhsu:
  case Intrinsic::riscv_vwmul:
  case Intrinsic::riscv_vwmulu:
  case Intrinsic::riscv_vwmulsu:
  case Intrinsic::riscv_vwaddu:
  case Intrinsic::riscv_vwadd:
  case Intrinsic::riscv_vwsubu:
  case Intrinsic::riscv_vwsub:
  case Intrinsic::riscv_vwaddu_w:
  case Intrinsic::riscv_vwadd_w:
  case Intrinsic::riscv_vwsubu_w:
  case Intrinsic::riscv_vwsub_w:
  case Intrinsic::riscv_vnsrl:
  case Intrinsic::riscv_vnsra:
  case Intrinsic::riscv_vminu:
  case Intrinsic::riscv_vmin:
  case Intrinsic::riscv_vmaxu:
  case Intrinsic::riscv_vmax:
    // TODO: Add more intrinsics here.
    if (Instruction *V = foldBinaryOp(IC, II))
      return V;
    break;
  case Intrinsic::riscv_vmacc:
    // combine (vmacc a, (vwcvt b), (vwcvt c))
    // to      (vwmacc a, b, c)
    if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(1)))
      if (auto *II3 = dyn_cast<IntrinsicInst>(II.getArgOperand(2))) {
        Intrinsic::ID Vwcvt[] = {Intrinsic::riscv_vwadd,
                                 Intrinsic::riscv_vwaddu};
        Intrinsic::ID Vwmacc[] = {Intrinsic::riscv_vwmacc,
                                  Intrinsic::riscv_vwmaccu};
        Value *VL = II.getArgOperand(3);
        for (int i = 0; i != 2; ++i)
          if (II2->getIntrinsicID() == Vwcvt[i] &&
              II3->getIntrinsicID() == Vwcvt[i] &&
              isa<UndefValue>(II2->getArgOperand(0)) &&
              isa<UndefValue>(II3->getArgOperand(0)) &&
              isa<ConstantInt>(II2->getArgOperand(2)) &&
              cast<ConstantInt>(II2->getArgOperand(2))->isZero() &&
              isa<ConstantInt>(II3->getArgOperand(2)) &&
              cast<ConstantInt>(II3->getArgOperand(2))->isZero() &&
              VL == II2->getArgOperand(3) && VL == II3->getArgOperand(3))
            return CreateIntrinsic(
                &II, Vwmacc[i],
                {II.getType(), II2->getArgOperand(1)->getType(),
                 II3->getArgOperand(1)->getType(), VL->getType()},
                {II.getArgOperand(0), II2->getArgOperand(1),
                 II3->getArgOperand(1), VL, II.getArgOperand(4)});
      }
    if (Instruction *V = foldTernaryOp(IC, II))
      return V;
    break;
  case Intrinsic::riscv_vnmsac:
  case Intrinsic::riscv_vmadd:
  case Intrinsic::riscv_vnmsub:
  case Intrinsic::riscv_vwmaccu:
  case Intrinsic::riscv_vwmacc:
  case Intrinsic::riscv_vwmaccsu:
  case Intrinsic::riscv_vwmaccus:
  case Intrinsic::riscv_vfmacc:
  case Intrinsic::riscv_vfnmacc:
  case Intrinsic::riscv_vfmsac:
  case Intrinsic::riscv_vfnmsac:
  case Intrinsic::riscv_vfmadd:
  case Intrinsic::riscv_vfnmadd:
  case Intrinsic::riscv_vfmsub:
  case Intrinsic::riscv_vfnmsub:
    if (Instruction *V = foldTernaryOp(IC, II))
      return V;
    break;
  case Intrinsic::riscv_vfirst:
    if (Instruction *V = foldVFirstWithCompare(IC, II))
      return V;
    break;
  case Intrinsic::riscv_vmerge:
  case Intrinsic::riscv_vfmerge:
    if (Instruction *V = foldVMergeWithCompare(IC, II))
      return V;
    break;
  case Intrinsic::riscv_vmv_x_s:
    if (Instruction *V = foldVMV_X_S(IC, II))
      return V;
    if (Instruction *V = foldVmvVRgatherVle(IC, II, ST))
      return V;
    break;
  case Intrinsic::riscv_vfmv_f_s:
    if (Instruction *V = foldVMV_F_S(IC, II))
      return V;
    if (Instruction *V = foldVmvVRgatherVle(IC, II, ST))
      return V;
    break;
  case Intrinsic::riscv_vslideup:
    // combine (vslideup a, (vslidedown undef, a, b), b) to a
    if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(1))) {
      if (II2->getIntrinsicID() == Intrinsic::riscv_vslidedown &&
          isa<UndefValue>(II2->getArgOperand(0)) &&
          // same source
          II.getArgOperand(0) == II2->getArgOperand(1) &&
          // same offset
          II.getArgOperand(2) == II2->getArgOperand(2) &&
          isa<ConstantInt>(II.getArgOperand(2)) &&
          isa<ConstantInt>(II.getArgOperand(3)) &&
          isa<ConstantInt>(II2->getArgOperand(3))) {
        uint64_t Offset =
            cast<ConstantInt>(II.getArgOperand(2))->getZExtValue();
        uint64_t SlideupVL =
            cast<ConstantInt>(II.getArgOperand(3))->getZExtValue();
        uint64_t SlidedownVL =
            cast<ConstantInt>(II2->getArgOperand(3))->getZExtValue();
        if ((SlideupVL <= (Offset + SlidedownVL)) &&
            // If (Offset + SlidedownVL) is greater than VLMAX, the output of
            // slidedown will contain 0, but 0 does not belong to
            // II.getArgOperand(0).
            ((Offset + SlidedownVL) <=
             getVLMAX(cast<ScalableVectorType>(II.getType()), ST)))
          return IC.replaceInstUsesWith(II, II.getArgOperand(0));
      }
    }
    break;
  }

  // Try to fold broadcasts.
  if (Instruction *V = foldVBroadcast(IC, II))
    return V;

  switch (IID) {
  case Intrinsic::riscv_vsll:
  case Intrinsic::riscv_vsrl:
  case Intrinsic::riscv_vsra:
  case Intrinsic::riscv_vnsra:
  case Intrinsic::riscv_vnsrl:
  case Intrinsic::riscv_vssra:
  case Intrinsic::riscv_vssrl: {
    Type *ShAmtTy = II.getArgOperand(2)->getType();
    // If shift amount is a scalar, we can use SimplifyDemandedBits on it. Only
    // the lower log2(SEW) bits are needed. Where SEW is the scalar size of
    // the source vector.
    if (ShAmtTy->isIntegerTy()) {
      unsigned TypeWidth =
          II.getArgOperand(1)->getType()->getScalarSizeInBits();
      unsigned BitWidth = ShAmtTy->getIntegerBitWidth();
      KnownBits ShAmtKnown(BitWidth);
      APInt DemandedBits = APInt::getLowBitsSet(BitWidth, Log2_32(TypeWidth));
      if (IC.SimplifyDemandedBits(&II, 2, DemandedBits, ShAmtKnown))
        return &II;
    }
    break;
  }
  }

  return std::nullopt;
}
