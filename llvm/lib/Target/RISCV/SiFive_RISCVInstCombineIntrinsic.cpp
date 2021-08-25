//===-- RISCVInstCombineIntrinsic.cpp - RISCV specific InstCombine pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// RISCV target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "RISCVTargetTransformInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;

#define DEBUG_TYPE "riscvtti"

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
  Value *VL = II.getArgOperand(3);
  Value *LHSScalar = getVSplat(II.getArgOperand(1), VL);
  if (!LHSScalar)
    return nullptr;
  Value *RHSScalar = getVSplatOrScalar(II.getArgOperand(2), VL);
  if (!RHSScalar)
    return nullptr;
  Value *Result;
  switch (II.getIntrinsicID()) {
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
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, (1 << Log2_32(TypeWidth)) - 1);
    // If RHSScalar comes from vsll.vx or vsll.vi, its type is i64 for RV64 and
    // i32 for RV32.
    Type *RTy = RHSScalar->getType();
    if (TypeWidth < RTy->getScalarSizeInBits())
      RHSScalar = IC.Builder.CreateTrunc(RHSScalar, Ty);
    Result = IC.Builder.CreateShl(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vsrl: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, (1 << Log2_32(TypeWidth)) - 1);
    // If RHSScalar comes from vsrl.vx or vsrl.vi, its type is i64 for RV64 and
    // i32 for RV32.
    Type *RTy = RHSScalar->getType();
    if (TypeWidth < RTy->getScalarSizeInBits())
      RHSScalar = IC.Builder.CreateTrunc(RHSScalar, Ty);
    Result = IC.Builder.CreateLShr(LHSScalar, RHSScalar);
    break;
  }
  case Intrinsic::riscv_vsra: {
    Type *Ty = LHSScalar->getType();
    unsigned TypeWidth = Ty->getScalarSizeInBits();
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, (1 << Log2_32(TypeWidth)) - 1);
    // If RHSScalar comes from vsra.vx or vsra.vi, its type is i64 for RV64 and
    // i32 for RV32.
    Type *RTy = RHSScalar->getType();
    if (TypeWidth < RTy->getScalarSizeInBits())
      RHSScalar = IC.Builder.CreateTrunc(RHSScalar, Ty);
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
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, LHSTy);
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, (1 << Log2_32(TypeWidth)) - 1);
    Result = IC.Builder.CreateLShr(LHSScalar, RHSScalar);
    Result = IC.Builder.CreateTrunc(Result, DestTy);
    break;
  }
  case Intrinsic::riscv_vnsra: {
    Type *LHSTy = LHSScalar->getType();
    unsigned TypeWidth = LHSTy->getScalarSizeInBits();
    Type *DestTy = IC.Builder.getIntNTy(TypeWidth / 2);
    RHSScalar = IC.Builder.CreateZExt(RHSScalar, LHSTy);
    RHSScalar = IC.Builder.CreateAnd(RHSScalar, (1 << Log2_32(TypeWidth)) - 1);
    Result = IC.Builder.CreateAShr(LHSScalar, RHSScalar);
    Result = IC.Builder.CreateTrunc(Result, DestTy);
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

static Optional<ICmpInst::Predicate> getPredicate(Intrinsic::ID IID) {
  switch (IID) {
  default:
    return None;
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
  if (!isKnownNonZero(VL, IC.getDataLayout()))
    return nullptr;

  auto *ValII = dyn_cast<IntrinsicInst>(Val);
  if (!ValII)
    return nullptr;

  Optional<ICmpInst::Predicate> Pred = getPredicate(ValII->getIntrinsicID());
  if (!Pred.hasValue())
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
    CmpV = IC.Builder.CreateICmp(Pred.getValue(), LHSScalar, RHSScalar);
  else if (LHSScalar->getType()->isFloatingPointTy())
    CmpV = IC.Builder.CreateFCmp(Pred.getValue(), LHSScalar, RHSScalar);
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
//   (select (icmp slt A, B), c, d)
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

  Optional<ICmpInst::Predicate> Pred = getPredicate(MaskII->getIntrinsicID());
  if (!Pred.hasValue())
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
    CmpV = IC.Builder.CreateICmp(Pred.getValue(), MaskLHSScalar, MaskRHSScalar);
  else if (MaskLHSScalar->getType()->isFloatingPointTy())
    CmpV = IC.Builder.CreateFCmp(Pred.getValue(), MaskLHSScalar, MaskRHSScalar);
  else
    llvm_unreachable("Unexpected types for scalar comparison.");

  Value *V = IC.Builder.CreateSelect(CmpV, LHS, RHS);

  return IC.replaceInstUsesWith(II, V);
}

/// This function handles following case
///
///     A  ->  B    cast to fixed
///     PHI
///     B  ->  A    cast from fixed
///
/// All the related PHI nodes can be replaced by new PHI nodes with type A.
/// The uses of \p II can be changed to the new PHI node corresponding to \p PN.
/// NOTE: This is based on optimizeBitCastFromPhi with the load/store handling
/// removed.
static Instruction *optimizeVCastFromFixedPhi(IntrinsicInst &II, PHINode *PN,
                                              InstCombiner &IC) {
  Value *Src = II.getArgOperand(0);
  Type *SrcTy = Src->getType(); // Type B
  Type *DestTy = II.getType();  // Type A

  SmallVector<PHINode *, 4> PhiWorklist;
  SmallSetVector<PHINode *, 4> OldPhiNodes;

  // Find all of the A->B casts and PHI nodes.
  // We need to inspect all related PHI nodes, but PHIs can be cyclic, so
  // OldPhiNodes is used to track all known PHI nodes, before adding a new
  // PHI to PhiWorklist, it is checked against and added to OldPhiNodes first.
  PhiWorklist.push_back(PN);
  OldPhiNodes.insert(PN);
  while (!PhiWorklist.empty()) {
    auto *OldPN = PhiWorklist.pop_back_val();
    for (Value *IncValue : OldPN->incoming_values()) {
      if (auto *PNode = dyn_cast<PHINode>(IncValue)) {
        if (OldPhiNodes.insert(PNode))
          PhiWorklist.push_back(PNode);
        continue;
      }

      auto *VCastTo = dyn_cast<IntrinsicInst>(IncValue);
      // We can't handle other instructions.
      if (!VCastTo ||
          VCastTo->getIntrinsicID() != Intrinsic::riscv_vcast_to_fixed)
        return nullptr;

      // Verify it's a A->B cast.
      Type *TyA = VCastTo->getArgOperand(0)->getType();
      Type *TyB = VCastTo->getType();
      if (TyA != DestTy || TyB != SrcTy)
        return nullptr;
    }
  }

  // Check that each user of each old PHI node is something that we can
  // rewrite, so that all of the old PHI nodes can be cleaned up afterwards.
  for (auto *OldPN : OldPhiNodes) {
    for (User *V : OldPN->users()) {
      if (auto *VCastFrom = dyn_cast<IntrinsicInst>(V)) {
        if (VCastFrom->getIntrinsicID() != Intrinsic::riscv_vcast_from_fixed)
          return nullptr;
        // Verify it's a B->A cast.
        Type *TyB = VCastFrom->getArgOperand(0)->getType();
        Type *TyA = VCastFrom->getType();
        if (TyA != DestTy || TyB != SrcTy)
          return nullptr;
      } else if (auto *PHI = dyn_cast<PHINode>(V)) {
        // As long as the user is another old PHI node, then even if we don't
        // rewrite it, the PHI web we're considering won't have any users
        // outside itself, so it'll be dead.
        if (OldPhiNodes.count(PHI) == 0)
          return nullptr;
      } else {
        return nullptr;
      }
    }
  }

  // For each old PHI node, create a corresponding new PHI node with a type A.
  SmallDenseMap<PHINode *, PHINode *> NewPNodes;
  for (auto *OldPN : OldPhiNodes) {
    IC.Builder.SetInsertPoint(OldPN);
    PHINode *NewPN = IC.Builder.CreatePHI(DestTy, OldPN->getNumOperands());
    NewPNodes[OldPN] = NewPN;
  }

  // Fill in the operands of new PHI nodes.
  for (auto *OldPN : OldPhiNodes) {
    PHINode *NewPN = NewPNodes[OldPN];
    for (unsigned j = 0, e = OldPN->getNumOperands(); j != e; ++j) {
      Value *V = OldPN->getOperand(j);
      Value *NewV = nullptr;
      if (auto *VCastTo = dyn_cast<IntrinsicInst>(V)) {
        assert(VCastTo->getIntrinsicID() == Intrinsic::riscv_vcast_to_fixed &&
               "Unexpected intrinsic");
        NewV = VCastTo->getArgOperand(0);
      } else if (auto *PrevPN = dyn_cast<PHINode>(V)) {
        NewV = NewPNodes[PrevPN];
      }
      assert(NewV);
      NewPN->addIncoming(NewV, OldPN->getIncomingBlock(j));
    }
  }

  // Traverse all accumulated PHI nodes and process its users,
  // which are vcast_from intrinsics. Without this processing
  // NewPHI nodes could be replicated and could lead to extra
  // moves generated after DeSSA.

  // Replace users of BitCast B->A with NewPHI. These will help
  // later to get rid off a closure formed by OldPHI nodes.
  Instruction *RetVal = nullptr;
  for (auto *OldPN : OldPhiNodes) {
    PHINode *NewPN = NewPNodes[OldPN];
    for (User *V : make_early_inc_range(OldPN->users())) {
      if (auto *VCastTo = dyn_cast<IntrinsicInst>(V)) {
        Type *TyB = VCastTo->getArgOperand(0)->getType();
        Type *TyA = VCastTo->getType();
        assert(TyA == DestTy && TyB == SrcTy);
        (void)TyA;
        (void)TyB;
        Instruction *I = IC.replaceInstUsesWith(*VCastTo, NewPN);
        if (VCastTo == &II)
          RetVal = I;
      } else if (auto *PHI = dyn_cast<PHINode>(V)) {
        assert(OldPhiNodes.contains(PHI));
        (void)PHI;
      } else {
        llvm_unreachable("all uses should be handled");
      }
    }
  }

  return RetVal;
}

static Instruction *foldVBroadcast(InstCombiner &IC, IntrinsicInst &II) {
  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  case Intrinsic::riscv_vaadd:
  case Intrinsic::riscv_vaaddu:
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
  case Intrinsic::riscv_vsmul:
  case Intrinsic::riscv_vsub:
  case Intrinsic::riscv_vxor:
  case Intrinsic::riscv_vfadd:
  case Intrinsic::riscv_vfdiv:
  case Intrinsic::riscv_vfmax:
  case Intrinsic::riscv_vfmin:
  case Intrinsic::riscv_vfmul:
  case Intrinsic::riscv_vfsub:
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
  case Intrinsic::riscv_vasub:
  case Intrinsic::riscv_vasubu:
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
  case Intrinsic::riscv_vfwadd_w:
  case Intrinsic::riscv_vfwsub_w:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3)});
    // These instructions are not commutable.
    break;
  case Intrinsic::riscv_vwadd:
  case Intrinsic::riscv_vwaddu:
  case Intrinsic::riscv_vwmul:
  case Intrinsic::riscv_vwmulu:
  case Intrinsic::riscv_vfwadd:
  case Intrinsic::riscv_vfwmul:
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
  case Intrinsic::riscv_vwmulsu:
  case Intrinsic::riscv_vwsub:
  case Intrinsic::riscv_vwsubu:
  case Intrinsic::riscv_vfwsub:
    if (Value *V = getVSplat(II.getArgOperand(2), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), II.getArgOperand(1)->getType(), V->getType(),
           II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), II.getArgOperand(1), V, II.getArgOperand(3)});
    // These instructions are not commutable.
    break;
  case Intrinsic::riscv_vmacc:
  case Intrinsic::riscv_vnmsac:
  case Intrinsic::riscv_vfmacc:
  case Intrinsic::riscv_vfnmacc:
  case Intrinsic::riscv_vfmsac:
  case Intrinsic::riscv_vfnmsac:
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
  case Intrinsic::riscv_vmadd:
  case Intrinsic::riscv_vnmsub:
  case Intrinsic::riscv_vfmadd:
  case Intrinsic::riscv_vfnmadd:
  case Intrinsic::riscv_vfmsub:
  case Intrinsic::riscv_vfnmsub:
    if (Value *V = getVSplat(II.getArgOperand(1), II.getArgOperand(3)))
      return CreateIntrinsic(
          &II, IID,
          {II.getType(), V->getType(), II.getArgOperand(3)->getType()},
          {II.getArgOperand(0), V, II.getArgOperand(2), II.getArgOperand(3),
           II.getArgOperand(4)});
    // FIXME: If these were tail agnostic there would be more commuting
    // options for these intrinsics.
    break;
  case Intrinsic::riscv_vwmacc:
  case Intrinsic::riscv_vwmaccu:
  case Intrinsic::riscv_vwmaccsu:
  case Intrinsic::riscv_vfwmacc:
  case Intrinsic::riscv_vfwnmacc:
  case Intrinsic::riscv_vfwmsac:
  case Intrinsic::riscv_vfwnmsac:
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
  case Intrinsic::riscv_vmseq:
  case Intrinsic::riscv_vmsne:
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

Optional<Instruction *>
RISCVTTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  default:
    break;
  case Intrinsic::riscv_vcast_from_fixed:
    if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(0)))
      if (II2->getIntrinsicID() == Intrinsic::riscv_vcast_to_fixed &&
          II.getType() == II2->getArgOperand(0)->getType())
        return IC.replaceInstUsesWith(II, II2->getArgOperand(0));

    if (auto *PN = dyn_cast<PHINode>(II.getArgOperand(0)))
      if (auto *I = optimizeVCastFromFixedPhi(II, PN, IC))
        return I;

    break;
  case Intrinsic::riscv_vcast_to_fixed:
    if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(0)))
      if (II2->getIntrinsicID() == Intrinsic::riscv_vcast_from_fixed &&
          II.getType() == II2->getArgOperand(0)->getType())
        return IC.replaceInstUsesWith(II, II2->getArgOperand(0));
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
  case Intrinsic::riscv_vadd:
  case Intrinsic::riscv_vsub:
  case Intrinsic::riscv_vrsub:
  case Intrinsic::riscv_vfadd:
  case Intrinsic::riscv_vfsub:
  case Intrinsic::riscv_vfrsub:
  case Intrinsic::riscv_vor:
  case Intrinsic::riscv_vsll:
  case Intrinsic::riscv_vsrl:
  case Intrinsic::riscv_vsra:
  case Intrinsic::riscv_vmul:
  case Intrinsic::riscv_vmulh:
  case Intrinsic::riscv_vmulhu:
  case Intrinsic::riscv_vmulhsu:
  case Intrinsic::riscv_vfmul:
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
  case Intrinsic::riscv_vnsra: {
    // TODO: Add more intrinsics here.
    if (Instruction *V = foldBinaryOp(IC, II))
      return V;
    break;
  }
  case Intrinsic::riscv_vfirst: {
    if (Instruction *V = foldVFirstWithCompare(IC, II))
      return V;
    break;
  }
  case Intrinsic::riscv_vmerge:
  case Intrinsic::riscv_vfmerge: {
    if (Instruction *V = foldVMergeWithCompare(IC, II))
      return V;
    break;
  }
  }

  // Try to fold broadcasts.
  if (Instruction *V = foldVBroadcast(IC, II))
    return V;

  return None;
}
