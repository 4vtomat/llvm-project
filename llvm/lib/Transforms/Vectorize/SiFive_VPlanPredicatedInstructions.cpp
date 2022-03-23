//===- SiFive_VPlanPredicatedInstructions.cpp - Vectorizer Plan -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Supports emission of the predicated versions of the vector instructions.
///
//===----------------------------------------------------------------------===//

#include "SiFive_VPlanPredicatedInstructions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/VectorBuilder.h"

using namespace llvm;
void llvm::widenPredicatedInstruction(Instruction *Op, VPValue *Def,
                                      VPUser &User, VPTransformState &State,
                                      VPValue *BlockInMask, VPValue *EVL,
                                      unsigned Part) {
  IRBuilderBase &BuilderIR = State.Builder;
  VectorBuilder Builder(BuilderIR);
  auto &&MaskValue = [&](unsigned Part, ElementCount EC) -> Value * {
    // The outermost mask can be lowered as an all ones mask when using EVL.
    if (auto *VPI = dyn_cast<VPInstruction>(BlockInMask))
      if (VPI && VPI->getOpcode() == VPInstruction::ActiveLaneMask)
        return BuilderIR.getTrueVector(EC);
    return State.get(BlockInMask, Part);
  };

  unsigned Opcode =
      Op ? Op->getOpcode() : cast<VPInstruction>(User).getOpcode();

  auto &&CreateCast = [&](CastInst *CI) {
    Value *SrcVal = State.get(User.getOperand(0), Part);
    auto *SrcTy = cast<VectorType>(SrcVal->getType());
    auto *DestTy = VectorType::get(CI->getType(), SrcTy->getElementCount());
    Builder.setMask(MaskValue(Part, DestTy->getElementCount()));
    Builder.setEVL(State.get(EVL, Part));
    Value *V = Builder.createVectorInstruction(CI->getOpcode(), DestTy,
                                               {SrcVal}, "vp.cast");
    State.set(Def, V, Part);
  };

  switch (Opcode) {
  case VPInstruction::Not: {
    assert(!Op && "Expected with no-op only.");
    Value *A = State.get(User.getOperand(0), Part);
    auto *PredTy = cast<VectorType>(A->getType());
    Value *MaskArg = BuilderIR.getTrueVector(State.VF);
    Value *EVLArg = State.get(State.EVL, Part);
    Builder.setMask(MaskArg).setEVL(EVLArg);
    Value *V = Builder.createVectorInstruction(Instruction::Xor, PredTy,
                                               {A, MaskArg}, "pred.not");
    State.set(Def, V, Part);
    return;
  }
  case Instruction::Select: {
    assert(!Op && "Expected with no-op only.");
    Value *Cond = State.get(User.getOperand(0), Part);
    Value *Op1 = State.get(User.getOperand(1), Part);
    Value *Op2 = State.get(User.getOperand(2), Part);
    Value *EVLArg = State.get(EVL, Part);
    // Emit vp.merge intrinsic to keep same tail policy in entire loop.
    // Otherwise, this will lead to switching it to/from tail-agnostic, which is
    // not performant and may break optimizations in backend. Keep name of the
    // value as "vp.op.select" for debugging purposes
    Value *V = BuilderIR.CreateIntrinsic(Intrinsic::vp_merge, {Op1->getType()},
                                         {Cond, Op1, Op2, EVLArg}, nullptr,
                                         "vp.op.select");
    State.set(Def, V, Part);
    return;
  }
  case VPInstruction::ICmpULE: {
    assert(!Op && "Expected with no-op only.");
    Value *IV = State.get(User.getOperand(0), Part);
    Value *TC = State.get(User.getOperand(1), Part);
    Value *PredArg = BuilderIR.getInt8(CmpInst::ICMP_ULE);
    Value *MaskArg = BuilderIR.getTrueVector(State.VF);
    Value *EVLArg = State.get(State.EVL, Part);
    Builder.setMask(MaskArg).setEVL(EVLArg);
    Value *V =
        Builder.createVectorInstruction(Instruction::ICmp, IV->getType(),
                                        {IV, TC, PredArg}, "pred.active.lane");
    State.set(Def, V, Part);
    return;
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    //===------------------ compare instructions --------------------------===//
    // Widen compares. Generate vector compares.
    bool FCmp = (Opcode == Instruction::FCmp);
    auto *Cmp = cast<CmpInst>(Op);
    Value *A = State.get(User.getOperand(0), Part);
    Value *B = State.get(User.getOperand(1), Part);
    Value *C = nullptr;

    VectorType *OpTy = cast<VectorType>(A->getType());
    Value *MaskArg = MaskValue(Part, OpTy->getElementCount());
    Builder.setMask(MaskArg);
    Value *EVLArg = State.get(EVL, Part);
    Builder.setEVL(EVLArg);
    Value *PredArg = BuilderIR.getInt8(Cmp->getPredicate());

    if (FCmp) {
      IRBuilder<>::FastMathFlagGuard FMFG(BuilderIR);
      BuilderIR.setFastMathFlags(Cmp->getFastMathFlags());
      C = Builder.createVectorInstruction(Opcode, OpTy, {A, B, PredArg},
                                          "vp.op.fcmp");
    } else {
      C = Builder.createVectorInstruction(Opcode, OpTy, {A, B, PredArg},
                                          "vp.op.icmp");
    }
    // The result of vp_icmp or vp_fcmp may contain lanes that are undef due
    // to the mask. We don't need undef boolean values.
    // Convert undef lanes to false by inserting a vp_merge.
    Value *AllFalse = BuilderIR.getFalseVector(OpTy->getElementCount());
    // FIXME: this currently is expanded to:
    //    vsetvli zero, a2, e8, mf4, ta, mu
    //    vmand.mm        v15, v15, v0
    //    vmandn.mm       v16, v14, v0
    //    vmor.mm v0, v15, v16
    //    vsetvli zero, zero, e64, m2, ta, mu
    Value *V = BuilderIR.CreateIntrinsic(Intrinsic::vp_merge, {C->getType()},
                                         {MaskArg, C, AllFalse, EVLArg},
                                         nullptr, "vp.op.merge");

    State.set(Def, V, Part);
    return;
  }
  case Instruction::SExt:
  case Instruction::ZExt:
  case Instruction::Trunc: {
    //===------------------ Int-to-Int cast instructions ------------------===//
    auto *CI = cast<CastInst>(Op);

    assert(isa<IntegerType>(CI->getType()) && "Invalid destination Int type.");
    IntegerType *DestElemTy = cast<IntegerType>(CI->getType());
    IntegerType *SrcElemTy = cast<IntegerType>(CI->getOperand(0)->getType());
    if (Opcode == Instruction::Trunc)
      assert(DestElemTy->getBitWidth() < SrcElemTy->getBitWidth() &&
             "Cannot truncate to a larger size.");
    else
      assert(DestElemTy->getBitWidth() > SrcElemTy->getBitWidth() &&
             "Cannot extend to a smaller size.");
    return CreateCast(CI);
  }
  case Instruction::FPExt:
  case Instruction::FPTrunc: {
    //===------------------ Float-to-Float cast instructions --------------===//
    auto *CI = cast<CastInst>(Op);
    Type *DestElemTy = CI->getType();
    Type *SrcElemTy = CI->getOperand(0)->getType();
    assert(DestElemTy->isFloatingPointTy() && SrcElemTy->isFloatingPointTy() &&
           "Invalid destination/source type for float extension.");
    if (Opcode == Instruction::FPTrunc)
      assert(DestElemTy->getTypeID() < SrcElemTy->getTypeID() &&
             "Cannot extend to a larger size.");
    else
      assert(DestElemTy->getTypeID() > SrcElemTy->getTypeID() &&
             "Cannot extend to a smaller size.");
    return CreateCast(CI);
  }
  case Instruction::FPToUI:
  case Instruction::FPToSI: {
    //===------------------ Float-to-Int cast instructions ----------------===//
    auto *CI = cast<CastInst>(Op);
    Type *DestElemTy = CI->getType();
    Type *SrcElemTy = CI->getOperand(0)->getType();
    assert(DestElemTy->isIntegerTy() && SrcElemTy->isFloatingPointTy() &&
           "Invalid destination/source type for float to int cast.");
    return CreateCast(CI);
  }
  case Instruction::UIToFP:
  case Instruction::SIToFP: {
    //===------------------ Int-to-Float cast instructions ----------------===//
    auto *CI = cast<CastInst>(Op);
    Type *DestElemTy = CI->getType();
    Type *SrcElemTy = CI->getOperand(0)->getType();
    assert(SrcElemTy->isIntegerTy() && DestElemTy->isFloatingPointTy() &&
           "Invalid destination/source type for float to int cast.");
    return CreateCast(CI);
  }
  case Instruction::IntToPtr:
  case Instruction::PtrToInt: {
    //===------------------ Int-Ptr cast instructions ---------------------===//
    auto *CI = cast<CastInst>(Op);
    Type *DestElemTy = CI->getType();
    Type *SrcElemTy = CI->getOperand(0)->getType();
    if (Opcode == Instruction::IntToPtr)
      assert(SrcElemTy->isIntegerTy() && DestElemTy->isPointerTy() &&
             "Invalid destination/source type for int to ptr cast.");
    else
      assert(DestElemTy->isIntegerTy() && SrcElemTy->isPointerTy() &&
             "Invalid destination/source type for ptr to int cast.");
    return CreateCast(CI);
  }
  default:
    break;
  }

  //===------------------- Other Binary and Unary Ops ---------------------===//
  if (Instruction::isBinaryOp(Opcode) || Instruction::isUnaryOp(Opcode)) {
    assert(((Instruction::isBinaryOp(Opcode) &&
             (!Op || Op->getNumOperands() == 2)) ||
            (Instruction::isUnaryOp(Opcode) &&
             (!Op || Op->getNumOperands() == 1))) &&
           "Invalid number of operands.");

    // Just widen unops and binops.

    SmallVector<Value *, 4> Ops;
    for (unsigned I = 0, E = Instruction::isBinaryOp(Opcode) ? 2 : 1; I < E;
         ++I) {
      VPValue *VPOp = User.getOperand(I);
      Ops.push_back(State.get(VPOp, Part));
    }

    VectorType *OpTy = cast<VectorType>(Ops[0]->getType());
    // FIXME: This is a hack because we are not being honest here.
    Value *MaskArg;
    if (Op)
      MaskArg = MaskValue(Part, OpTy->getElementCount());
    else
      MaskArg = BuilderIR.getTrueVector(OpTy->getElementCount());
    Builder.setMask(MaskArg);
    Builder.setEVL(State.get(EVL, Part));

    Value *V = Builder.createVectorInstruction(Opcode, OpTy, Ops, "vp.op");

    if (Op)
      if (auto *VecOp = dyn_cast<Instruction>(V))
        VecOp->copyIRFlags(Op);

    // Use this vector value for all users of the original instruction.
    State.set(Def, V, Part);
    return;
  }
  llvm_unreachable("Unexpected opcode.");
}
