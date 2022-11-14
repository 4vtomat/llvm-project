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
#include "VPlan.h"
#include "VPlanValue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/VectorBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#define DEBUG_TYPE "loop-vectorize"

using namespace llvm;

/// Return true if instruction can stay unmasked regardless to a mask on its
/// basic block.
/// For now only assume that all integer operations, except for division and
/// remain, can be unmasked.
/// On RISC-V integer division and remain don't raise exception, so they
/// technically could be unmasked, but generating unmasked instruction may
/// violate LLVM's principles
/// TODO: Whether instruction should be masked or unmasked has to be decided
/// during VPlan construction by looking at the target and exceptions that
/// are enabled.
static bool canUnaryOrBinaryOpBeUnmasked(const unsigned Opcode, Type *ElementType) {
  assert((Instruction::isUnaryOp(Opcode) || Instruction::isBinaryOp(Opcode)) &&
         "Unary or Binary operation is expected.");
  if (!ElementType->isIntegerTy() && !ElementType->isFloatingPointTy()) {
    return false;
  }
  switch (Opcode) {
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
    return false;
  }
  return true;
}

static void widenSelectInstruction(VPTransformState &State,
                                   const unsigned VPOpCode, VPValue *Def,
                                   VPUser &User, const unsigned Part,
                                   StringRef Name) {
  VPValue *EVL = State.Plan->getEVL();
  Value *Cond = State.get(User.getOperand(0), Part);
  Value *Op1 = State.get(User.getOperand(1), Part);
  Value *Op2 = State.get(User.getOperand(2), Part);
  Value *EVLArg = State.get(EVL, Part);
  Value *V = State.Builder.CreateIntrinsic(VPOpCode, {Op1->getType()},
                                           {Cond, Op1, Op2, EVLArg}, nullptr,
                                           "vp.op.select");
  State.set(Def, V, Part);
}

namespace llvm {
void widenPredicatedInstruction(Instruction *Op, VPValue *Def, VPUser &User,
                                VPTransformState &State, VPValue *BlockInMask,
                                unsigned Part) {
  VPValue *EVL = State.Plan->getEVL();
  IRBuilderBase &BuilderIR = State.Builder;
  VectorBuilder Builder(BuilderIR);
  auto &&MaskValue = [&](unsigned Part, ElementCount EC) -> Value * {
    if (!BlockInMask)
      return BuilderIR.getTrueVector(State.VF);
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
    // TODO: Whether instruction should be masked or unmasked has to be decided
    // during VPlan construction by looking at the target and exceptions that
    // are enabled.
    // Since LV is targeting RVV, use all-true mask for conversions.
    Builder.setMask(BuilderIR.getTrueVector(SrcTy->getElementCount()));
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
    Value *EVLArg = State.get(EVL, Part);
    Builder.setMask(MaskArg).setEVL(EVLArg);
    Value *V = Builder.createVectorInstruction(Instruction::Xor, PredTy,
                                               {A, MaskArg}, "pred.not");
    State.set(Def, V, Part);
    return;
  }
  case Instruction::Select: {
    assert((!Op || isa<VPWidenSelectRecipe>(Def->getDef())) &&
           "Expected with no-op only or VPWidenSelectRecipe.");
    widenSelectInstruction(State, Intrinsic::vp_select, Def, User, Part,
                           "vp.op.select");
    return;
  }
  case VPInstruction::ICmpULE: {
    assert(!Op && "Expected with no-op only.");
    Value *IV = State.get(User.getOperand(0), Part);
    Value *TC = State.get(User.getOperand(1), Part);
    StringRef PredicateStr = CmpInst::getPredicateName(CmpInst::ICMP_ULE);
    auto *PredicateMDS = MDString::get(IV->getContext(), PredicateStr);
    Value *PredArg = MetadataAsValue::get(IV->getContext(), PredicateMDS);

    Value *MaskArg = BuilderIR.getTrueVector(State.VF);
    Value *EVLArg = State.get(EVL, Part);
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

    StringRef PredicateStr = CmpInst::getPredicateName(Cmp->getPredicate());
    auto *PredicateMDS = MDString::get(Cmp->getContext(), PredicateStr);
    Value *PredArg = MetadataAsValue::get(Cmp->getContext(), PredicateMDS);

    if (FCmp) {
      IRBuilder<>::FastMathFlagGuard FMFG(BuilderIR);
      BuilderIR.setFastMathFlags(Cmp->getFastMathFlags());
      C = Builder.createVectorInstruction(Opcode, OpTy, {A, B, PredArg},
                                          "vp.op.fcmp");
    } else {
      C = Builder.createVectorInstruction(Opcode, OpTy, {A, B, PredArg},
                                          "vp.op.icmp");
    }

    State.set(Def, C, Part);
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
    Value *MaskArg;
    if (Op && !canUnaryOrBinaryOpBeUnmasked(Opcode, OpTy->getElementType()))
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

void widenPredicatedCall(CallInst &CI, VPValue *Def, VPUser &ArgOperands,
                         VPTransformState &State, Intrinsic::ID VPID,
                         unsigned Part) {
  IRBuilderBase &Builder = State.Builder;
  SmallVector<Type *, 2> TysForDecl = {CI.getType()};
  SmallVector<Value *, 4> Args;
  for (auto &I : enumerate(ArgOperands.operands())) {
    Value *Arg = State.get(I.value(), Part);
    if (isVectorIntrinsicWithOverloadTypeAtArg(VPID, I.index()))
      TysForDecl.push_back(Arg->getType());
    Args.push_back(Arg);
  }

  Args.push_back(Builder.getTrueVector(State.VF));
  Args.push_back(State.get(State.Plan->getEVL(), Part));
  auto *DestTy = VectorType::get(CI.getType(), State.VF);
  CallInst *V = Builder.CreateIntrinsic(VPID, DestTy, Args, nullptr, "vp.op");
  if (isa<FPMathOperator>(V))
    V->copyFastMathFlags(&CI);
  State.set(Def, V, Part);
}

void VPSelectInstruction::execute(VPTransformState &State) {
  if (!State.Plan->getEVL()) {
    // For non RVV VLA vectorization, reuse existing mechanism to generate the
    // vector code
    VPInstruction::execute(State);
    return;
  }

  assert(!State.Instance && "VPInstruction executing an Instance");
  IRBuilderBase::FastMathFlagGuard FMFGuard(State.Builder);
  State.Builder.setFastMathFlags(getFastMathFlags());

  unsigned VPOpCode = Intrinsic::vp_select;
  StringRef Name = "vp.op.select";
  if (hasTailUndisturbedPolicy()) {
    VPOpCode = Intrinsic::vp_merge;
    Name = "vp.op.merge";
  }

  for (unsigned Part = 0; Part < State.UF; ++Part)
    widenSelectInstruction(State, VPOpCode, this, *this, Part, Name);

}

/// Build and return either `vp.gather`/`vp.scatter` or
/// `vp.strided_load`/`vp.strided_store` if previous analysis indicated it's
/// possible to be used
Instruction *
widenPredicatedMemoryInstruction(VPWidenMemoryInstructionRecipe &VPWMIR,
                                 VPTransformState &State, unsigned Part,
                                 ArrayRef<Value *> BlockInMaskParts) {
  assert(Part == 0 && "Cannot support Part > 0 for RVV VLA vectorization");
  Value *EVLPart = State.get(State.Plan->getEVL(), Part);
  assert(EVLPart && "EVL must be set prior to generation of vp-intrinsics");

  VPValue *VPAddr = VPWMIR.getAddr();
  Value *VectorGep = State.get(VPAddr, Part);
  auto *PtrsTy = cast<VectorType>(VectorGep->getType());
  ElementCount NumElts = PtrsTy->getElementCount();
  auto &Builder = State.Builder;

  auto MaskValue = [&](unsigned Part, ElementCount EC) -> Value * {
    // The outermost mask can be lowered as an all ones mask when using
    // EVL.
    VPValue *Mask = VPWMIR.getMask();
    auto *IMask = dyn_cast_or_null<VPInstruction>(Mask);
    if (!Mask || (IMask && IMask->getOpcode() == VPInstruction::ICmpULE))
      return Builder.getTrueVector(EC);

    return BlockInMaskParts[Part];
  };
  Value *BlockInMaskPart = MaskValue(Part, NumElts);

  if (VPWMIR.isStore()) {
    VPValue *StoredValue = VPWMIR.getStoredValue();
    Value *StoredVal = State.get(StoredValue, Part);

    if (VPWMIR.isStrided()) {
      Value *Ptr = State.get(VPAddr, VPIteration(0, 0));
      const SCEV *SCEVStride = VPWMIR.getStride();
      auto &DL = State.CFG.PrevBB->getModule()->getDataLayout();
      SCEVExpander Exp(*(State.SE), DL, "stride");
      Instruction *InsertPoint = &*State.Builder.GetInsertPoint();
      assert(Exp.isSafeToExpandAt(SCEVStride, InsertPoint) &&
             "It's not safe to expand that SCEV in the vector loop. That was "
             "not caught by isSafeStrideAccessInfo.");
      Value *Stride =
          Exp.expandCodeFor(SCEVStride, SCEVStride->getType(), InsertPoint);
      LLVM_DEBUG(llvm::dbgs()
                 << "Generating strided store for addr = " << *VPAddr
                 << " with a stride = " << *Stride << '\n');
      auto *PtrTy = cast<PointerType>(PtrsTy->getElementType());
      Value *Operands[] = {StoredVal, Ptr, Stride, BlockInMaskPart, EVLPart};
      return Builder.CreateIntrinsic(
          Intrinsic::experimental_vp_strided_store,
          {StoredVal->getType(), PtrTy, Stride->getType()}, Operands);
    }
    auto *DataTy = cast<VectorType>(StoredVal->getType());
    LLVM_DEBUG(llvm::dbgs() << "Indexed store for " << *VPAddr << "\n");
    Value *Operands[] = {StoredVal, VectorGep, BlockInMaskPart, EVLPart};
    return Builder.CreateIntrinsic(Intrinsic::vp_scatter, {DataTy, PtrsTy},
                                   Operands);
  } else {
    auto *DataTy = VectorType::get(VPWMIR.getElementType(), State.VF);
    if (VPWMIR.isStrided()) {
      Value *Ptr = State.get(VPAddr, VPIteration(0, 0));
      const SCEV *SCEVStride = VPWMIR.getStride();
      auto &DL = State.CFG.PrevBB->getModule()->getDataLayout();
      SCEVExpander Exp(*(State.SE), DL, "stride");
      Instruction *InsertPoint = &*State.Builder.GetInsertPoint();
      assert(Exp.isSafeToExpandAt(SCEVStride, InsertPoint) &&
             "It's not safe to expand that SCEV in the vector loop. That was "
             "not caught by isSafeStrideAccessInfo.");
      Value *Stride =
          Exp.expandCodeFor(SCEVStride, SCEVStride->getType(), InsertPoint);
      auto *PtrTy = cast<PointerType>(PtrsTy->getElementType());
      LLVM_DEBUG(llvm::dbgs()
                 << "Generating strided load for addr = " << *VPAddr
                 << " with a stride = " << *Stride << '\n');
      Value *Operands[] = {Ptr, Stride, BlockInMaskPart, EVLPart};
      return Builder.CreateIntrinsic(Intrinsic::experimental_vp_strided_load,
                                      {DataTy, PtrTy, Stride->getType()},
                                      Operands, nullptr, "vp.strided.load");
    }
    LLVM_DEBUG(llvm::dbgs() << "Indexed load for " << VPAddr << "\n");
    Value *Operands[] = {VectorGep, BlockInMaskPart, EVLPart};
    return Builder.CreateIntrinsic(Intrinsic::vp_gather, {DataTy, PtrsTy},
                                   Operands, nullptr, "vp.gather");
  }
}

} // namespace llvm
