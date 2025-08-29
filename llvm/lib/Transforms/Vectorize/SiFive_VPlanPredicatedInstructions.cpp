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
#include "VPlanHelpers.h"
#include "VPlanUtils.h"
#include "VPlanValue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsRISCV.h"
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

static Value *widenSelectInstruction(VPTransformState &State,
                                     const unsigned VPOpCode, VPValue *Def,
                                     VPUser &User, const Twine &Name) {
  VPValue *EVL = State.EVL;
  Value *Cond = State.get(User.getOperand(0));
  Value *Op1 = State.get(User.getOperand(1));
  Value *Op2 = State.get(User.getOperand(2));
  Value *EVLArg = State.get(EVL, /*NeedsScalar=*/true);
  return State.Builder.CreateIntrinsic(VPOpCode, {Op1->getType()},
                                       {Cond, Op1, Op2, EVLArg}, nullptr, Name);
}

/// Construct vector popcount of the vector \p V
static Value *createVectorPopcount(IRBuilderBase &Builder, Value *V,
                                   Value *EVL) {
  ElementCount EC = cast<VectorType>(V->getType())->getElementCount();
  Value *Operands[] = {V, Builder.getTrueVector(EC), EVL};

  return Builder.CreateIntrinsic(Intrinsic::experimental_vp_popcount,
                                 {V->getType()}, Operands);
}

/// Compress \p VectorToCompress using RVV vcompress intrinsic
static Value *compressVector(IRBuilderBase &Builder, Value *Mask,
                             Value *VectorToCompress, Value *EVL) {
  assert(Mask != nullptr && "Compress mask must be provided");
  assert(EVL != nullptr && "EVL for RVV-intrinsic must be provided");
  Type *VTy = VectorToCompress->getType();
  Value *Operands[] = {VectorToCompress, Mask, EVL};
  CallInst *Compress = Builder.CreateIntrinsic(
      Intrinsic::experimental_vp_compress, {VTy}, Operands);
  return Compress;
}

static Value *expandVector(IRBuilderBase &Builder, Value *Mask,
                           Value *VectorToExpand, Value *EVL) {
  assert(Mask != nullptr && "Expand mask must be provided");
  assert(EVL != nullptr && "EVL for RVV-intrinsic must be provided");
  Type *VTy = VectorToExpand->getType();
  Value *Operands[] = {VectorToExpand, Mask, EVL};
  CallInst *Expand = Builder.CreateIntrinsic(Intrinsic::experimental_vp_expand,
                                             {VTy}, Operands);
  return Expand;
}

namespace llvm {
Value *widenPredicatedInstruction(Instruction *Op, VPValue *Def, VPUser &User,
                                  VPTransformState &State,
                                  VPValue *BlockInMask) {
  VPValue *EVL = State.EVL;
  IRBuilderBase &BuilderIR = State.Builder;
  auto &&MaskValue = [&](ElementCount EC) -> Value * {
    if (!BlockInMask)
      return BuilderIR.getTrueVector(State.VF);
    // The outermost mask can be lowered as an all ones mask when using EVL.
    if (auto *VPI = dyn_cast<VPInstruction>(BlockInMask))
      if (VPI && VPI->getOpcode() == VPInstruction::ActiveLaneMask)
        return BuilderIR.getTrueVector(EC);
    return State.get(BlockInMask);
  };

  unsigned Opcode =
      Op ? Op->getOpcode() : cast<VPInstruction>(User).getOpcode();

  switch (Opcode) {
  case VPInstruction::Not: {
    assert(!Op && "Expected with no-op only.");
    Value *A = State.get(User.getOperand(0));
    auto *PredTy = cast<VectorType>(A->getType());
    Value *MaskArg = BuilderIR.getTrueVector(State.VF);
    Value *EVLArg = State.get(EVL, /*NeedsScalar=*/true);
    return BuilderIR.CreateIntrinsic(
      MaskArg->getType(), Intrinsic::vp_xor,
        {A, MaskArg, MaskArg, EVLArg}, nullptr, "pred.not");
  }
  case Instruction::Select: {
    assert((!Op || isa<VPWidenSelectRecipe>(Def->getDefiningRecipe())) &&
           "Expected with no-op only or VPWidenSelectRecipe.");
    return widenSelectInstruction(State, Intrinsic::vp_select, Def, User,
                                  "vp.op.select");
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    //===------------------ compare instructions --------------------------===//
    // Widen compares. Generate vector compares.
    bool FCmp = (Opcode == Instruction::FCmp);
    Value *A = State.get(User.getOperand(0));
    Value *B = State.get(User.getOperand(1));

    assert((Op || cast<VPInstruction>(Def)) && "Invalid recipe");
    CmpInst::Predicate Pred = cast<VPRecipeWithIRFlags>(Def)->getPredicate();
    VectorType *OpTy = cast<VectorType>(A->getType());
    Value *MaskArg = MaskValue(OpTy->getElementCount());
    Value *EVLArg = State.get(EVL, /*NeedScalar=*/true);

    StringRef PredicateStr = CmpInst::getPredicateName(Pred);
    auto *PredicateMDS = MDString::get(A->getContext(), PredicateStr);
    Value *PredArg = MetadataAsValue::get(A->getContext(), PredicateMDS);

    if (FCmp) {
      IRBuilder<>::FastMathFlagGuard FMFG(BuilderIR);
      auto *V = BuilderIR.CreateIntrinsic(
        MaskArg->getType(), Intrinsic::vp_fcmp,
          {A, B, PredArg, MaskArg, EVLArg}, nullptr, "vp.op.fcmp");
      if (auto *VPF = dyn_cast<VPRecipeWithIRFlags>(Def))
        VPF->applyFlags(cast<Instruction>(*V));

      return V;
    }
    
    return BuilderIR.CreateIntrinsic(MaskArg->getType(), Intrinsic::vp_icmp,
                                     {A, B, PredArg, MaskArg, EVLArg}, nullptr,
                                     "vp.op.icmp");
  }
  case Instruction::SExt:
  case Instruction::ZExt:
  case Instruction::Trunc:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt: {
    assert(isa<VPWidenCastRecipe>(Def) &&
           "VPWidenCastRecipe is expected in CreateCast lambda");
    Value *SrcVal = State.get(User.getOperand(0));
    auto *SrcTy = cast<VectorType>(SrcVal->getType());
    auto *VPWC = cast<VPWidenCastRecipe>(Def);
    Type *DestTy = VPWC->getResultType();
    auto *DestVecTy = VectorType::get(DestTy, SrcTy->getElementCount());
    // TODO: Whether instruction should be masked or unmasked has to be decided
    // during VPlan construction by looking at the target and exceptions that
    // are enabled.
    // Since LV is targeting RVV, use all-true mask for conversions.
    Value *MaskArg = BuilderIR.getTrueVector(SrcTy->getElementCount());
    Value *EVLArg = State.get(EVL, /*NeedsScalar=*/true);
    auto VPID = VPIntrinsic::getForOpcode(VPWC->getOpcode());
    return BuilderIR.CreateIntrinsic(DestVecTy, VPID, {SrcVal, MaskArg, EVLArg},
                                     nullptr, "vp.cast");
  }
  default:
    break;
  }

  //===------------------- Other Binary and Unary Ops ---------------------===//
  if (Instruction::isBinaryOp(Opcode) || Instruction::isUnaryOp(Opcode)) {
    // Just widen unops and binops.

    SmallVector<Value *, 4> Ops;
    for (unsigned I = 0, E = Instruction::isBinaryOp(Opcode) ? 2 : 1; I < E;
         ++I) {
      VPValue *VPOp = User.getOperand(I);
      Ops.push_back(State.get(VPOp));
    }

    VectorType *OpTy = cast<VectorType>(Ops[0]->getType());
    Value *MaskArg = nullptr;
    if (Op && !canUnaryOrBinaryOpBeUnmasked(Opcode, OpTy->getElementType()))
      MaskArg = MaskValue(OpTy->getElementCount());
    Value *V =
        widenPredicatedArithmeticOp(State, Opcode, Ops, MaskArg, "vp.op");

    if (auto *VPF = dyn_cast<VPRecipeWithIRFlags>(Def))
      if (auto *VecOp = dyn_cast<Instruction>(V);
          VecOp && isa<FPMathOperator>(V))
        VPF->applyFlags(*VecOp);

    return V;
  }
  llvm_unreachable("Unexpected opcode.");
}

void widenPredicatedIntrinsic(CallInst *CI, VPValue *Def,
                              VPTransformState &State, Intrinsic::ID VPID,
                              const TargetTransformInfo *TTI) {
  IRBuilderBase &Builder = State.Builder;
  auto *Recipe = cast<VPWidenIntrinsicRecipe>(Def);
  SmallVector<Type *, 2> TysForDecl;
  // Add return type if intrinsic is overloaded on it.
  if (isVectorIntrinsicWithOverloadTypeAtArg(VPID, -1, TTI))
    TysForDecl.push_back(VectorType::get(
        Recipe->getResultType()->getScalarType(), State.VF));
  SmallVector<Value *, 4> Args;
  for (auto I : enumerate(Recipe->operands())) {
    Value *Arg;
    if (!isVectorIntrinsicWithScalarOpAtArg(VPID, I.index(), TTI))
      Arg = State.get(I.value());
    else
      Arg = State.get(I.value(), VPLane(0));
    if (isVectorIntrinsicWithOverloadTypeAtArg(VPID, I.index(), TTI))
      TysForDecl.push_back(Arg->getType());
    Args.push_back(Arg);
  }

  Args.push_back(Builder.getTrueVector(State.VF));
  Args.push_back(State.get(State.EVL, /*NeedsScalar=*/true));
  CallInst *V =
      Builder.CreateIntrinsic(VPID, TysForDecl, Args, nullptr, "vp.op");
  if (auto *VPF = dyn_cast<VPRecipeWithIRFlags>(Def);
      VPF && isa<FPMathOperator>(V))
    VPF->applyFlags(*V);
  State.set(Def, V);
}

void VPSelectInstruction::execute(VPTransformState &State) {
  if (!State.EVL) {
    // For non RVV VLA vectorization, reuse existing mechanism to generate the
    // vector code
    VPInstruction::execute(State);
    return;
  }

  assert(!State.Lane && "VPInstruction executing a lane");

  unsigned VPOpCode = Intrinsic::vp_select;
  StringRef Name = "vp.op.select";
  if (hasTailUndisturbedPolicy()) {
    VPOpCode = Intrinsic::vp_merge;
    Name = "vp.op.merge";
  }

  Value *V = widenSelectInstruction(State, VPOpCode, this, *this, Name);
  State.set(this, V);
}

/// Generate following sequence to update scalar monotonic variable:
///   %0 = vp.popcount(%mask, %rvl)
///   %1 = mul %0, %step          // where %step is a step of monotonic
///   %monotonic.update = add %monotonic, %1
void VPMonotonicUpdateInstruction::execute(VPTransformState &State) {
  auto &Builder = State.Builder;
  Value *V = State.get(getIncomingValue(), /*NeedsScalar=*/true);
  Value *Step = State.get(getStepValue(), /*NeedsScalar=*/true);
  Value *Mask = State.get(getMask());
  Value *EVL = State.get(State.EVL, /*NeedsScalar=*/true);
  Value *Vpop = createVectorPopcount(Builder, Mask, EVL);
  Vpop = Builder.CreateZExtOrTrunc(Vpop, Step->getType());

  Value *Mult = Builder.CreateMul(Vpop, Step);
  const auto *OrigUpdateOp = cast<BinaryOperator>(MD.getUpdateOp());

  Value *NewV = Builder.CreateBinOp(OrigUpdateOp->getOpcode(), V, Mult,
                                    "monotonic.update");
  State.set(this, NewV, /*IsScalar=*/true);
}

/// Generate phi for the monotonic:
///   %monotonic = phi [%monotonic.update, %vector.latch]
void VPMonotonicHeaderPHIRecipe::execute(VPTransformState &State) {
  IRBuilder<>::InsertPointGuard Guard(State.Builder);
  State.Builder.SetInsertPoint(State.CFG.PrevBB->getFirstNonPHIIt());

  Value *StartV = State.get(getStartValue(), /*NeedsScalar=*/true);
  auto *Phi = State.Builder.CreatePHI(StartV->getType(), 2, "monotonic.phi");
  BasicBlock *PreheaderBB =
      State.CFG.VPBB2IRBB.at(getParent()->getCFGPredecessor(0));
  Phi->addIncoming(StartV, PreheaderBB);

  State.set(this, Phi, /*IsScalar*/ true);
}

/// Build and return either `vp.gather`/`vp.scatter` or
/// `vp.strided_load`/`vp.strided_store` if previous analysis indicated it's
/// possible to be used
Instruction *widenPredicatedMemoryInstruction(VPWidenMemoryRecipe &VPWMIR,
                                              VPTransformState &State,
                                              Value *BlockInMaskPart) {
  Value *EVLPart = State.get(State.EVL, /*NeedsScalar=*/true);
  assert(EVLPart && "EVL must be set prior to generation of vp-intrinsics");

  VPValue *VPAddr = VPWMIR.getAddr();
  ElementCount NumElts = State.VF;
  auto &Builder = State.Builder;
  const Align Alignment = getLoadStoreAlignment(&VPWMIR.getIngredient());

  VPValue *Mask = VPWMIR.getMask();

  if (!Mask)
    BlockInMaskPart = Builder.getTrueVector(NumElts);

  if (auto *VPWMSIR = dyn_cast<VPWidenStoreEVLRecipe>(&VPWMIR)) {
    VPValue *StoredValue = VPWMSIR->getStoredValue();
    Value *StoredVal = State.get(StoredValue);
    if (VPWMIR.isMonotonic())
      StoredVal = compressVector(Builder, BlockInMaskPart, StoredVal, EVLPart);

    if (VPWMIR.isStrided()) {
      Value *Ptr = State.get(VPAddr, VPLane(0));
      auto *PtrTy = cast<PointerType>(Ptr->getType());
      CallInst *VS = nullptr;
      if (!VPWMIR.isConsecutive()) {
        Value *Stride = State.get(VPWMIR.getStride(), VPLane(0));
        LLVM_DEBUG(llvm::dbgs() << "Generating strided store for addr = ";
                   const auto *Instr = VPAddr->getDefiningRecipe();
                   VPSlotTracker SlotTracker((Instr && Instr->getParent())
                                                 ? Instr->getParent()->getPlan()
                                                 : nullptr);
                   Instr->print(dbgs(), Twine(), SlotTracker);
                   dbgs() << " with a stride = " << *Stride << '\n');
        Value *Operands[] = {StoredVal, Ptr, Stride, BlockInMaskPart, EVLPart};
        VS = Builder.CreateIntrinsic(
            Intrinsic::experimental_vp_strided_store,
            {StoredVal->getType(), PtrTy, Stride->getType()}, Operands);
      } else {
        if (VPWMIR.isMonotonic()) {
          EVLPart = createVectorPopcount(Builder, BlockInMaskPart, EVLPart);
          BlockInMaskPart = Builder.getTrueVector(NumElts);
        }
        Value *Operands[] = {StoredVal, Ptr, BlockInMaskPart, EVLPart};
        VS = Builder.CreateIntrinsic(Intrinsic::vp_store,
                                     {StoredVal->getType(), PtrTy}, Operands);
      }

      VS->addParamAttr(
          1, Attribute::getWithAlignment(VS->getContext(), Alignment));
      return VS;
    }
    auto *DataTy = cast<VectorType>(StoredVal->getType());
    LLVM_DEBUG(llvm::dbgs()
               << "Indexed store for " << VPAddr->getDefiningRecipe() << "\n");
    Value *VectorGep = State.get(VPAddr);
    Value *Operands[] = {StoredVal, VectorGep, BlockInMaskPart, EVLPart};
    auto *PtrsTy = cast<VectorType>(VectorGep->getType());
    CallInst *VS = Builder.CreateIntrinsic(Intrinsic::vp_scatter,
                                           {DataTy, PtrsTy}, Operands);
    VS->addParamAttr(1,
                     Attribute::getWithAlignment(VS->getContext(), Alignment));
    return VS;
  }
  // Handle loads: strided (consecutive or non-consecutive), indexed
  auto *DataTy = VectorType::get(VPWMIR.getElementType(), State.VF);
  CallInst *VL = nullptr;
  if (VPWMIR.isStrided()) {
    Value *Ptr = State.get(VPAddr, VPLane(0));
    auto *PtrTy = cast<PointerType>(Ptr->getType());
    if (!VPWMIR.isConsecutive()) {
      Value *Stride = State.get(VPWMIR.getStride(), VPLane(0));
      LLVM_DEBUG(llvm::dbgs()
                 << "Generating strided load for addr = " << VPAddr->getDefiningRecipe()
                 << " with a stride = " << *Stride << '\n');
      Value *Operands[] = {Ptr, Stride, BlockInMaskPart, EVLPart};
      VL = Builder.CreateIntrinsic(Intrinsic::experimental_vp_strided_load,
                                   {DataTy, PtrTy, Stride->getType()}, Operands,
                                   nullptr, "vp.strided.load");
    } else {
      Value *Operands[] = {Ptr, BlockInMaskPart, EVLPart};
      // Expand load requires to load only necessary number of elements, which is
      // determined by vcpop. The load needs to be done without a mask.
      if (VPWMIR.isMonotonic()) {
        Operands[1] = Builder.getTrueVector(NumElts);
        Operands[2] = createVectorPopcount(Builder, BlockInMaskPart, EVLPart);
      }

      VL = Builder.CreateIntrinsic(Intrinsic::vp_load, {DataTy, PtrTy}, Operands);
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Indexed load for " << VPAddr << "\n");
    Value *VectorGep = State.get(VPAddr);
    Value *Operands[] = {VectorGep, BlockInMaskPart, EVLPart};
    auto *PtrsTy = cast<VectorType>(VectorGep->getType());
    VL = Builder.CreateIntrinsic(Intrinsic::vp_gather, {DataTy, PtrsTy},
                                 Operands, nullptr, "vp.gather");
  }
  VL->addParamAttr(0,
                   Attribute::getWithAlignment(VL->getContext(), Alignment));

  // For expand load expand the data according to the mask
  if (VPWMIR.isMonotonic())
    return cast<Instruction>(
        expandVector(Builder, BlockInMaskPart, VL, EVLPart));
  return VL;
}

Instruction *widenPredicatedArithmeticOp(VPTransformState &State,
                                         unsigned Opcode, ArrayRef<Value *> Ops,
                                         Value *Mask, const Twine &Name) {
  assert(((Instruction::isBinaryOp(Opcode) && (Ops.size() == 2)) ||
          (Instruction::isUnaryOp(Opcode) && (Ops.size() == 1))) &&
         "Invalid number of operands.");
  VPValue *EVL = State.EVL;
  Value *EVLPart =
      State.EVL ? State.get(EVL, /*NeedsScalar=*/true) : State.EVLPlaceholder;
  EVLPart =
      State.Builder.CreateZExtOrTrunc(EVLPart, State.Builder.getInt32Ty());
  assert(EVLPart && "EVL was not created");
  if (!Mask)
    Mask = State.Builder.getTrueVector(State.VF);

  auto VPID = VPIntrinsic::getForOpcode(Opcode);
  SmallVector<Value *> VPOps;
  VPOps.append(Ops.begin(), Ops.end());
  VPOps.push_back(Mask);
  VPOps.push_back(EVLPart);

  return cast<Instruction>(State.Builder.CreateIntrinsic(
      Ops[0]->getType(), VPID, VPOps, nullptr, Name));
}
} // namespace llvm
