//===- VPlanRecipes.cpp - Implementations for VPlan recipes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains implementations for different VPlan recipes.
///
//===----------------------------------------------------------------------===//

#include "VPlan.h"
#include "VPlanAnalysis.h"
#if SIFIVE_CUSTOMIZATION
#include "SiFive_VPlanPredicatedInstructions.h"
#endif // SIFIVE_CUSTOMIZATION
#include "VPlanValue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsRISCV.h" // SIFIVE
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h" // SIFIVE
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <cassert>

using namespace llvm;

using VectorParts = SmallVector<Value *, 2>;

namespace llvm {
extern cl::opt<bool> EnableVPlanNativePath;
}

#define LV_NAME "loop-vectorize"
#define DEBUG_TYPE LV_NAME

#if SIFIVE_CUSTOMIZATION
extern cl::opt<uint64_t> LoopVectorizerVLUpperBound;
#endif

static Value *GetSetVL(VPTransformState &State, Value *EVL) {
#if SIFIVE_CUSTOMIZATION
  if (!EVL) {
    // Set EVL to all ones to get vlmax
    Type *I64Type = State.Builder.getInt64Ty();
    EVL = ConstantInt::get(I64Type, APInt::getAllOnes(64));
    // TODO: with MaxSafeDist analysis available for uncountable loop, support
    // it here for correctness"
    Value *VFArg = State.Builder.getInt32(State.VF.getKnownMinValue());
    return State.Builder.CreateIntrinsic(
        State.Builder.getInt32Ty(), Intrinsic::experimental_get_vector_length,
        {EVL, VFArg, State.Builder.getTrue()});
  }
#endif // SIFIVE_CUSTOMIZATION
  assert(EVL->getType()->isIntegerTy() &&
         "Requested vector length should be an integer.");

  // TODO: Add support for MaxSafeDist for correct loop emission.
  Value *VFArg = State.Builder.getInt32(State.VF.getKnownMinValue());

  assert(State.MaxSafeNumElems != 0 &&
         "Max safe number of elements that can be vectorized cannot be 0.");
  // Clamp RVL for finite dependence distances and user provided upper bounds
  if (State.MaxSafeNumElems != VPTransformState::UnknownNumSafeElems ||
      LoopVectorizerVLUpperBound) {
    uint64_t MaxRVL;
    if (State.MaxSafeNumElems != VPTransformState::UnknownNumSafeElems &&
        LoopVectorizerVLUpperBound)
      MaxRVL = std::min(State.MaxSafeNumElems, LoopVectorizerVLUpperBound.getValue());
    else
      MaxRVL = State.MaxSafeNumElems != VPTransformState::UnknownNumSafeElems
                   ? State.MaxSafeNumElems
                   : LoopVectorizerVLUpperBound;

    Constant *RVLUpperBound = ConstantInt::get(EVL->getType(), MaxRVL);
    EVL = State.Builder.CreateBinaryIntrinsic(Intrinsic::umin, EVL,
                                              RVLUpperBound);
  }

  Value *GVL = State.Builder.CreateIntrinsic(
      State.Builder.getInt32Ty(), Intrinsic::experimental_get_vector_length,
      {EVL, VFArg, State.Builder.getTrue()});
  return GVL;
}

bool VPRecipeBase::mayWriteToMemory() const {
  switch (getVPDefID()) {
  case VPInterleaveSC:
    return cast<VPInterleaveRecipe>(this)->getNumStoreOperands() > 0;
  case VPWidenMemoryInstructionSC: {
    return cast<VPWidenMemoryInstructionRecipe>(this)->isStore();
  }
  case VPReplicateSC:
  case VPWidenCallSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayWriteToMemory();
  case VPBranchOnMaskSC:
  case VPScalarIVStepsSC:
  case VPPredInstPHISC:
    return false;
  case VPBlendSC:
  case VPReductionSC:
  case VPWidenCanonicalIVSC:
  case VPWidenCastSC:
  case VPWidenGEPSC:
  case VPWidenIntOrFpInductionSC:
  case VPWidenPHISC:
  case VPWidenSC:
  case VPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getVPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayWriteToMemory()) &&
           "underlying instruction may write to memory");
    return false;
  }
  default:
    return true;
  }
}

bool VPRecipeBase::mayReadFromMemory() const {
  switch (getVPDefID()) {
  case VPWidenMemoryInstructionSC: {
    return !cast<VPWidenMemoryInstructionRecipe>(this)->isStore();
  }
  case VPReplicateSC:
  case VPWidenCallSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayReadFromMemory();
  case VPBranchOnMaskSC:
  case VPScalarIVStepsSC:
  case VPPredInstPHISC:
    return false;
  case VPBlendSC:
  case VPReductionSC:
  case VPWidenCanonicalIVSC:
  case VPWidenCastSC:
  case VPWidenGEPSC:
  case VPWidenIntOrFpInductionSC:
  case VPWidenPHISC:
  case VPWidenSC:
  case VPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getVPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayReadFromMemory()) &&
           "underlying instruction may read from memory");
    return false;
  }
  default:
    return true;
  }
}

bool VPRecipeBase::mayHaveSideEffects() const {
  switch (getVPDefID()) {
  case VPDerivedIVSC:
  case VPPredInstPHISC:
  case VPScalarCastSC:
    return false;
  case VPInstructionSC:
    switch (cast<VPInstruction>(this)->getOpcode()) {
    case Instruction::Or:
    case Instruction::ICmp:
    case Instruction::Select:
    case VPInstruction::Not:
    case VPInstruction::CalculateTripCountMinusVF:
    case VPInstruction::CanonicalIVIncrementForPart:
      return false;
    default:
      return true;
    }
  case VPWidenCallSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayHaveSideEffects();
#if SIFIVE_CUSTOMIZATION
  case VPVectorPointerSC:
#endif // SIFIVE_CUSTOMIZATION
  case VPBlendSC:
  case VPReductionSC:
  case VPScalarIVStepsSC:
  case VPWidenCanonicalIVSC:
  case VPWidenCastSC:
  case VPWidenGEPSC:
  case VPWidenIntOrFpInductionSC:
  case VPWidenPHISC:
  case VPWidenPointerInductionSC:
  case VPWidenSC:
  case VPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getVPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayHaveSideEffects()) &&
           "underlying instruction has side-effects");
    return false;
  }
  case VPInterleaveSC:
    return mayWriteToMemory();
  case VPWidenMemoryInstructionSC:
    assert(cast<VPWidenMemoryInstructionRecipe>(this)
                   ->getIngredient()
                   .mayHaveSideEffects() == mayWriteToMemory() &&
           "mayHaveSideffects result for ingredient differs from this "
           "implementation");
    return mayWriteToMemory();
  case VPReplicateSC: {
    auto *R = cast<VPReplicateRecipe>(this);
    return R->getUnderlyingInstr()->mayHaveSideEffects();
  }
  default:
    return true;
  }
}

void VPLiveOut::fixPhi(VPlan &Plan, VPTransformState &State) {
  auto Lane = VPLane::getLastLaneForVF(State.VF);
  VPValue *ExitValue = getOperand(0);
  if (vputils::isUniformAfterVectorization(ExitValue))
    Lane = VPLane::getFirstLane();
  VPBasicBlock *MiddleVPBB =
      cast<VPBasicBlock>(Plan.getVectorLoopRegion()->getSingleSuccessor());
  assert(MiddleVPBB->getNumSuccessors() == 0 &&
         "the middle block must not have any successors");
  BasicBlock *MiddleBB = State.CFG.VPBB2IRBB[MiddleVPBB];
  Phi->addIncoming(State.get(ExitValue, VPIteration(State.UF - 1, Lane)),
                   MiddleBB);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPLiveOut::print(raw_ostream &O, VPSlotTracker &SlotTracker) const {
  O << "Live-out ";
  getPhi()->printAsOperand(O);
  O << " = ";
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << "\n";
}
#endif

void VPRecipeBase::insertBefore(VPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  Parent = InsertPos->getParent();
  Parent->getRecipeList().insert(InsertPos->getIterator(), this);
}

void VPRecipeBase::insertBefore(VPBasicBlock &BB,
                                iplist<VPRecipeBase>::iterator I) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(I == BB.end() || I->getParent() == &BB);
  Parent = &BB;
  BB.getRecipeList().insert(I, this);
}

void VPRecipeBase::insertAfter(VPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  Parent = InsertPos->getParent();
  Parent->getRecipeList().insertAfter(InsertPos->getIterator(), this);
}

void VPRecipeBase::removeFromParent() {
  assert(getParent() && "Recipe not in any VPBasicBlock");
  getParent()->getRecipeList().remove(getIterator());
  Parent = nullptr;
}

iplist<VPRecipeBase>::iterator VPRecipeBase::eraseFromParent() {
  assert(getParent() && "Recipe not in any VPBasicBlock");
  return getParent()->getRecipeList().erase(getIterator());
}

void VPRecipeBase::moveAfter(VPRecipeBase *InsertPos) {
  removeFromParent();
  insertAfter(InsertPos);
}

void VPRecipeBase::moveBefore(VPBasicBlock &BB,
                              iplist<VPRecipeBase>::iterator I) {
  removeFromParent();
  insertBefore(BB, I);
}

FastMathFlags VPRecipeWithIRFlags::getFastMathFlags() const {
  assert(OpType == OperationType::FPMathOp &&
         "recipe doesn't have fast math flags");
  FastMathFlags Res;
  Res.setAllowReassoc(FMFs.AllowReassoc);
  Res.setNoNaNs(FMFs.NoNaNs);
  Res.setNoInfs(FMFs.NoInfs);
  Res.setNoSignedZeros(FMFs.NoSignedZeros);
  Res.setAllowReciprocal(FMFs.AllowReciprocal);
  Res.setAllowContract(FMFs.AllowContract);
  Res.setApproxFunc(FMFs.ApproxFunc);
  return Res;
}

VPInstruction::VPInstruction(unsigned Opcode, CmpInst::Predicate Pred,
                             VPValue *A, VPValue *B, DebugLoc DL,
                             const Twine &Name)
    : VPRecipeWithIRFlags(VPDef::VPInstructionSC, ArrayRef<VPValue *>({A, B}),
                          Pred, DL),
      Opcode(Opcode), Name(Name.str()) {
  assert(Opcode == Instruction::ICmp &&
         "only ICmp predicates supported at the moment");
}

VPInstruction::VPInstruction(unsigned Opcode,
                             std::initializer_list<VPValue *> Operands,
                             FastMathFlags FMFs, DebugLoc DL, const Twine &Name)
    : VPRecipeWithIRFlags(VPDef::VPInstructionSC, Operands, FMFs, DL),
      Opcode(Opcode), Name(Name.str()) {
  // Make sure the VPInstruction is a floating-point operation.
  assert(isFPMathOp() && "this op can't take fast-math flags");
}

Value *VPInstruction::generateInstruction(VPTransformState &State,
                                          unsigned Part) {
  IRBuilderBase &Builder = State.Builder;
  Builder.SetCurrentDebugLocation(getDebugLoc());

  if (Instruction::isBinaryOp(getOpcode())) {
    if (Part != 0 && vputils::onlyFirstPartUsed(this))
      return State.get(this, 0);

    Value *A = State.get(getOperand(0), Part);
    Value *B = State.get(getOperand(1), Part);
#if SIFIVE_CUSTOMIZATION
    if (State.Plan->useVLAVectorizer() && A->getType()->isVectorTy())
      return llvm::widenPredicatedInstruction(nullptr, this, *this, State,
                                              nullptr, Part);
    // FIXME: Remove with a proper representation of VFxUF in a VPlan. Currently
    // VFxUF is replaced with RVL if it's available. It can happen that after
    // such replacement types of operands do not match. In this case we have to
    // convert RVL operand to the proper type
    if (A->getType() != B->getType()) {
      if (getOperand(0) == State.RVL)
        A = Builder.CreateZExtOrTrunc(A, B->getType());
      else if (getOperand(1) == State.RVL)
        B = Builder.CreateZExtOrTrunc(B, A->getType());
    }
#endif // SIFIVE_CUSTOMIZATION
    auto *Res =
        Builder.CreateBinOp((Instruction::BinaryOps)getOpcode(), A, B, Name);
    if (auto *I = dyn_cast<Instruction>(Res))
      setFlags(I);
    return Res;
  }

  switch (getOpcode()) {
  case VPInstruction::Not: {
    Value *A = State.get(getOperand(0), Part);
#if SIFIVE_CUSTOMIZATION
    if (State.Plan->useVLAVectorizer() && A->getType()->isVectorTy())
      return llvm::widenPredicatedInstruction(nullptr, this, *this, State,
                                              nullptr, Part);
#endif // SIFIVE_CUSTOMIZATION
    return Builder.CreateNot(A, Name);
  }
  case Instruction::ICmp: {
    Value *A = State.get(getOperand(0), Part);
#if SIFIVE_CUSTOMIZATION
    if (State.Plan->useVLAVectorizer() && A->getType()->isVectorTy())
      return llvm::widenPredicatedInstruction(nullptr, this, *this, State,
                                              nullptr, Part);
#endif // SIFIVE_CUSTOMIZATION
    Value *B = State.get(getOperand(1), Part);
    return Builder.CreateCmp(getPredicate(), A, B, Name);
  }
  case Instruction::Select: {
    Value *Cond = State.get(getOperand(0), Part);
    Value *Op1 = State.get(getOperand(1), Part);
    Value *Op2 = State.get(getOperand(2), Part);
#if SIFIVE_CUSTOMIZATION
    if (State.Plan->useVLAVectorizer() && Cond->getType()->isVectorTy())
      return llvm::widenPredicatedInstruction(nullptr, this, *this, State,
                                              nullptr, Part);

#endif // SIFIVE_CUSTOMIZATION
    return Builder.CreateSelect(Cond, Op1, Op2, Name);
  }
  case VPInstruction::ActiveLaneMask: {
    // Get first lane of vector induction variable.
    Value *VIVElem0 = State.get(getOperand(0), VPIteration(Part, 0));
    // Get the original loop tripcount.
    Value *ScalarTC = State.get(getOperand(1), VPIteration(Part, 0));

    auto *Int1Ty = Type::getInt1Ty(Builder.getContext());
    auto *PredTy = VectorType::get(Int1Ty, State.VF);
    return Builder.CreateIntrinsic(Intrinsic::get_active_lane_mask,
                                   {PredTy, ScalarTC->getType()},
                                   {VIVElem0, ScalarTC}, nullptr, Name);
  }
  case VPInstruction::FirstOrderRecurrenceSplice: {
    // Generate code to combine the previous and current values in vector v3.
    //
    //   vector.ph:
    //     v_init = vector(..., ..., ..., a[-1])
    //     br vector.body
    //
    //   vector.body
    //     i = phi [0, vector.ph], [i+4, vector.body]
    //     v1 = phi [v_init, vector.ph], [v2, vector.body]
    //     v2 = a[i, i+1, i+2, i+3];
    //     v3 = vector(v1(3), v2(0, 1, 2))

    // For the first part, use the recurrence phi (v1), otherwise v2.
    auto *V1 = State.get(getOperand(0), 0);
    Value *PartMinus1 = Part == 0 ? V1 : State.get(getOperand(1), Part - 1);
    if (!PartMinus1->getType()->isVectorTy())
      return PartMinus1;
#if SIFIVE_CUSTOMIZATION
    if (State.Plan->useVLAVectorizer()) {
      Value *V2 = State.get(getOperand(1), Part);
      Value *PrevRVL = State.get(State.Plan->getPrevRVL(), Part);
      Value *RVL = State.get(State.RVL, Part);

      auto *IdxTy = Builder.getInt32Ty();
      Value *Shift = ConstantInt::get(IdxTy, -1);
      Value *Mask = Builder.getTrueVector(State.VF);

      return Builder.CreateIntrinsic(
          Intrinsic::experimental_vp_splice, {PartMinus1->getType()},
          {PartMinus1, V2, Shift, Mask, PrevRVL, RVL}, nullptr);
    }
#endif // SIFIVE_CUSTOMIZATION
    Value *V2 = State.get(getOperand(1), Part);
    return Builder.CreateVectorSplice(PartMinus1, V2, -1, Name);
  }
  case VPInstruction::CalculateTripCountMinusVF: {
    Value *ScalarTC = State.get(getOperand(0), {0, 0});
    Value *Step =
        createStepForVF(Builder, ScalarTC->getType(), State.VF, State.UF);
    Value *Sub = Builder.CreateSub(ScalarTC, Step);
    Value *Cmp = Builder.CreateICmp(CmpInst::Predicate::ICMP_UGT, ScalarTC, Step);
    Value *Zero = ConstantInt::get(ScalarTC->getType(), 0);
    return Builder.CreateSelect(Cmp, Sub, Zero);
  }
#if SIFIVE_CUSTOMIZATION
  case VPInstruction::ExplicitVectorLength: {
    // TODO: Restructure this code with an explicit remainder loop, vsetvli can
    // be outside of the main loop.
    assert(Part == 0 && "No unrolling expected for predicated vectorization.");
    Value *EVL = nullptr;
    if (!State.Plan->isUncountable()) {
      assert(getNumOperands() != 0 &&
             "Countable loop vectorization must use EVL");
      // Compute VTC - IV as the EVL(requested vector length).
      Value *Index = State.get(getOperand(0), 0);
      Value *TripCount = State.get(getOperand(1), VPIteration(0, 0));
      EVL = State.Builder.CreateSub(TripCount, Index);
    }
    // Set VLMAX if EVL is nullptr
    Value *SetVL = GetSetVL(State, EVL);
    State.RVL = this;
    return SetVL;
  }
  case VPInstruction::ExplicitVectorLengthIVIncrement: {
    assert(Part == 0 && "Expected unroll factor 1 for VP vectorization.");
    Value *Phi = State.get(getOperand(0), 0);
    Value *EVL = State.get(getOperand(1), 0);
    assert(EVL->getType()->getScalarSizeInBits() <=
               Phi->getType()->getScalarSizeInBits() &&
           "EVL type must be smaller than Phi type.");
    EVL = Builder.CreateIntCast(EVL, Phi->getType(), /*isSigned=*/false);
    return Builder.CreateAdd(Phi, EVL, Name, hasNoUnsignedWrap(),
                             hasNoSignedWrap());
  }
#endif // SIFIVE_CUSTOMIZATION
  case VPInstruction::CanonicalIVIncrementForPart: {
    auto *IV = State.get(getOperand(0), VPIteration(0, 0));
    if (Part == 0)
      return IV;

    // The canonical IV is incremented by the vectorization factor (num of SIMD
    // elements) times the unroll part.
    Value *Step = createStepForVF(Builder, IV->getType(), State.VF, Part);
    return Builder.CreateAdd(IV, Step, Name, hasNoUnsignedWrap(),
                             hasNoSignedWrap());
  }
  case VPInstruction::BranchOnCond: {
    if (Part != 0)
      return nullptr;

    Value *Cond = State.get(getOperand(0), VPIteration(Part, 0));
    VPRegionBlock *ParentRegion = getParent()->getParent();
    VPBasicBlock *Header = ParentRegion->getEntryBasicBlock();

    // Replace the temporary unreachable terminator with a new conditional
    // branch, hooking it up to backward destination for exiting blocks now and
    // to forward destination(s) later when they are created.
    BranchInst *CondBr =
        Builder.CreateCondBr(Cond, Builder.GetInsertBlock(), nullptr);

    if (getParent()->isExiting())
      CondBr->setSuccessor(1, State.CFG.VPBB2IRBB[Header]);

    CondBr->setSuccessor(0, nullptr);
    Builder.GetInsertBlock()->getTerminator()->eraseFromParent();
    return CondBr;
  }
  case VPInstruction::BranchOnCount: {
    if (Part != 0)
      return nullptr;
    // First create the compare.
    Value *IV = State.get(getOperand(0), Part);
    Value *TC = State.get(getOperand(1), Part);
    Value *Cond = Builder.CreateICmpEQ(IV, TC);

    // Now create the branch.
    auto *Plan = getParent()->getPlan();
    VPRegionBlock *TopRegion = Plan->getVectorLoopRegion();
    VPBasicBlock *Header = TopRegion->getEntry()->getEntryBasicBlock();

    // Replace the temporary unreachable terminator with a new conditional
    // branch, hooking it up to backward destination (the header) now and to the
    // forward destination (the exit/middle block) later when it is created.
    // Note that CreateCondBr expects a valid BB as first argument, so we need
    // to set it to nullptr later.
    BranchInst *CondBr = Builder.CreateCondBr(Cond, Builder.GetInsertBlock(),
                                              State.CFG.VPBB2IRBB[Header]);
    CondBr->setSuccessor(0, nullptr);
    Builder.GetInsertBlock()->getTerminator()->eraseFromParent();
    return CondBr;
  }
#if SIFIVE_CUSTOMIZATION
  case VPInstruction::CSAInitMask: {
    if (Part == 0) {
      Value *InitMask = State.get(getOperand(0), 0);
      State.set(this, InitMask, Part);
      return InitMask;
    }
    Value *V = State.get(this, Part - 1);
    State.set(this, V, Part);
    return V;
  }
  case VPInstruction::CSAInitData: {
    if (Part == 0) {
      Type *ElemTyp = getOperand(0)->getUnderlyingValue()->getType();
      Value *InitData = PoisonValue::get(VectorType::get(ElemTyp, State.VF));
      State.set(this, InitData, Part);
      return InitData;
    }
    Value *V = State.get(this, Part - 1);
    State.set(this, V, Part);
    return V;
  }
  case VPInstruction::CSAMaskPhi: {
    if (Part == 0) {
      IRBuilder<>::InsertPointGuard Guard(State.Builder);
      State.Builder.SetInsertPoint(State.CFG.PrevBB->getFirstNonPHI());
      BasicBlock *PreheaderBB = State.CFG.getPreheaderBBFor(this);
      Value *InitMask = State.get(getOperand(0), Part);
      PHINode *MaskPhi =
          State.Builder.CreatePHI(InitMask->getType(), 2, "csa.mask.phi");
      MaskPhi->addIncoming(InitMask, PreheaderBB);
      State.set(this, MaskPhi, Part);
      return MaskPhi;
    }
    Value *V =State.get(this, Part - 1);
    State.set(this, V, Part);
    return V;
  }
  case VPInstruction::CSAMaskSel: {
    if (!State.EnableRISCVCSA) {
      Value *WidenedCond = State.get(getOperand(0), Part);
      Value *MaskPhi = State.get(getOperand(1), Part);
      Value *AnyActive = State.get(getOperand(4), Part);
      // If not the first Part, use the mask from the previous unrolled Part
      Value *OldMask = Part == 0 ? MaskPhi : State.get(this, Part - 1);
      Value *MaskSel = State.Builder.CreateSelect(AnyActive, WidenedCond,
                                                  OldMask, "csa.mask.sel");
      // MaskPhi wants to use the most recently updated mask. That's the one
      // that corresponds to the last Part.
      if (Part == State.UF - 1)
        cast<PHINode>(MaskPhi)->addIncoming(MaskSel, State.CFG.PrevBB);
      State.set(this, MaskSel, Part);
      return MaskSel;
    }

    // NewMask can be calculated as (vmsbf(NewMask) & OldMask) | NewMask
    Value *WidenedCond = State.get(getOperand(0), Part);
    Value *AllTrue = State.get(getOperand(2), Part);
    Value *AllFalse = State.get(getOperand(3), Part);
    Value *RVL =
        State.Plan->useVLAVectorizer()
            ? State.get(State.RVL, Part)
            : getRuntimeVF(Builder, State.Builder.getInt32Ty(), State.VF);

    Value *UndistCond = State.Builder.CreateIntrinsic(
        WidenedCond->getType(), Intrinsic::vp_merge,
        {AllTrue, WidenedCond, AllFalse, RVL});

    Value *InitRVL =
        State.Plan->useVLAVectorizer()
            ? State.get(State.Plan->getInitRVL(), Part)
            : getRuntimeVF(Builder, State.Builder.getInt32Ty(), State.VF);

    Value *InitRVL64 =
        State.Builder.CreateZExtOrTrunc(InitRVL, State.Builder.getInt64Ty());
    Value *SBF = State.Builder.CreateIntrinsic(WidenedCond->getType(),
                                               Intrinsic::riscv_vmsbf,
                                               {UndistCond, InitRVL64});
    Value *MaskPhi = State.get(getOperand(1), Part);
    Value *OldMask = Part == 0 ? MaskPhi : State.get(this, Part - 1);
    Value *InitRVL32 =
        State.Builder.CreateZExtOrTrunc(InitRVL, State.Builder.getInt32Ty());
    Value *VAnd =
        State.Builder.CreateIntrinsic(WidenedCond->getType(), Intrinsic::vp_and,
                                      {SBF, OldMask, AllTrue, InitRVL32});
    Value *NewMask =
        State.Builder.CreateIntrinsic(WidenedCond->getType(), Intrinsic::vp_or,
                                      {VAnd, UndistCond, AllTrue, InitRVL32});

    // MaskPhi wants to use the most recently updated mask. That's the one
    // that corresponds to the last Part.
    if (Part == State.UF - 1)
      cast<PHINode>(MaskPhi)->addIncoming(NewMask, State.CFG.PrevBB);

    State.set(this, NewMask, Part);
    return NewMask;
  }
  case VPInstruction::CSAAnyActive: {
    Value *WidenedCond = State.get(getOperand(0), Part);
    Value *AllOnesMask = State.get(getOperand(1), Part);

    Value *RVL =
        State.Plan->useVLAVectorizer()
            ? State.get(State.RVL, Part)
            : getRuntimeVF(Builder, State.Builder.getInt32Ty(), State.VF);

    Value *StartValue =
        ConstantInt::get(WidenedCond->getType()->getScalarType(), 0);
    Value *AnyActive = State.Builder.CreateIntrinsic(
        WidenedCond->getType()->getScalarType(), Intrinsic::vp_reduce_or,
        {StartValue, WidenedCond, AllOnesMask, RVL}, nullptr,
        "csa.cond.anyactive");
    State.set(this, AnyActive, Part);
    return AnyActive;
  }
  case VPInstruction::CSAVLPhi: {
    IRBuilder<>::InsertPointGuard Guard(State.Builder);
    State.Builder.SetInsertPoint(State.CFG.PrevBB->getFirstNonPHI());
    BasicBlock *PreheaderBB = State.CFG.getPreheaderBBFor(this);

    // InitVL can be anything since it won't be used if no mask was active
    Value *InitVL = ConstantInt::get(State.Builder.getInt32Ty(), 0);
    PHINode *VLPhi =
        State.Builder.CreatePHI(InitVL->getType(), 2, "csa.vl.phi");
    VLPhi->addIncoming(InitVL, PreheaderBB);
    State.set(this, VLPhi, Part);
    return VLPhi;
  }
  case VPInstruction::CSAVLSel: {
    Value *AnyActive = State.get(getOperand(0), Part);
    Value *VLPhi = State.get(getOperand(1), Part);
    Value *RVL =
        State.Plan->useVLAVectorizer()
            ? State.get(State.RVL, Part)
            : getRuntimeVF(Builder, State.Builder.getInt32Ty(), State.VF);

    Value *VLSel =
        State.Builder.CreateSelect(AnyActive, RVL, VLPhi, "csa.vl.sel");
    cast<PHINode>(VLPhi)->addIncoming(VLSel, State.CFG.PrevBB);
    State.set(this, VLSel, Part);
    return VLSel;
  }
  case VPInstruction::ExitingCond: {
    VPValue *VPVectorCond = getOperand(0);
    assert(VPVectorCond && "Mask cannot be null for vfirst");
    // Create vfirst
    Value *Mask = State.get(VPVectorCond, Part);
    Value *RVL = State.get(State.RVL, 0);
    assert(RVL && "VL is null for uncountable loops");
    Value *VFirstI = Builder.CreateIntrinsic(
        Intrinsic::vp_first, {Mask->getType()},
        {Mask,
         Builder.getTrueVector(
             cast<VectorType>(Mask->getType())->getElementCount()),
         RVL});
    State.setVFirst(VFirstI);

    // Create cmp
    Value *Cond = Builder.CreateICmp(ICmpInst::ICMP_SGE, VFirstI,
                                     ConstantInt::get(VFirstI->getType(), 0));
    State.set(this, Cond, Part);
    return Cond;
  }
  // TODO: This case can be removed when support for Call instruction is added
  // to VPlan in upstream. For now it helps catch any use of VPInstruction for
  // Call opcode that is now supported by the new VPCallInstruction recipe.
  case Instruction::Call:
    llvm_unreachable("This opcode is handled by the VPCallInstruction recipe");
#endif // SIFIVE_CUSTOMIZATION
  case VPInstruction::ComputeReductionResult: {
    if (Part != 0)
      return State.get(this, 0);

    // FIXME: The cross-recipe dependency on VPReductionPHIRecipe is temporary
    // and will be removed by breaking up the recipe further.
    auto *PhiR = cast<VPReductionPHIRecipe>(getOperand(0));
    auto *OrigPhi = cast<PHINode>(PhiR->getUnderlyingValue());
    // Get its reduction variable descriptor.
    const RecurrenceDescriptor &RdxDesc = PhiR->getRecurrenceDescriptor();

    RecurKind RK = RdxDesc.getRecurrenceKind();

    State.setDebugLocFrom(getDebugLoc());

    VPValue *LoopExitingDef = getOperand(1);
    Type *PhiTy = OrigPhi->getType();
    VectorParts RdxParts(State.UF);
    for (unsigned Part = 0; Part < State.UF; ++Part)
      RdxParts[Part] = State.get(LoopExitingDef, Part);

    // If the vector reduction can be performed in a smaller type, we truncate
    // then extend the loop exit value to enable InstCombine to evaluate the
    // entire expression in the smaller type.
    // TODO: Handle this in truncateToMinBW.
    if (State.VF.isVector() && PhiTy != RdxDesc.getRecurrenceType()) {
      Type *RdxVecTy = VectorType::get(RdxDesc.getRecurrenceType(), State.VF);
      for (unsigned Part = 0; Part < State.UF; ++Part)
        RdxParts[Part] = Builder.CreateTrunc(RdxParts[Part], RdxVecTy);
    }
    // Reduce all of the unrolled parts into a single vector.
    Value *ReducedPartRdx = RdxParts[0];
    unsigned Op = RecurrenceDescriptor::getOpcode(RK);

    if (PhiR->isOrdered()) {
      ReducedPartRdx = RdxParts[State.UF - 1];
    } else {
      // Floating-point operations should have some FMF to enable the reduction.
      IRBuilderBase::FastMathFlagGuard FMFG(Builder);
      Builder.setFastMathFlags(RdxDesc.getFastMathFlags());
      for (unsigned Part = 1; Part < State.UF; ++Part) {
        Value *RdxPart = RdxParts[Part];
        if (Op != Instruction::ICmp && Op != Instruction::FCmp)
          ReducedPartRdx = Builder.CreateBinOp(
              (Instruction::BinaryOps)Op, RdxPart, ReducedPartRdx, "bin.rdx");
        else if (RecurrenceDescriptor::isAnyOfRecurrenceKind(RK)) {
          TrackingVH<Value> ReductionStartValue =
              RdxDesc.getRecurrenceStartValue();
          ReducedPartRdx = createAnyOfOp(Builder, ReductionStartValue, RK,
                                         ReducedPartRdx, RdxPart);
#if SIFIVE_CUSTOMIZATION
        }  else if (RecurrenceDescriptor::isFindLastIVRecurrenceKind(RK)) {
          ReducedPartRdx = createFindLastIVOp(Builder, ReducedPartRdx, RdxPart);
#endif // SIFIVE_CUSTOMIZATION
        } else
          ReducedPartRdx = createMinMaxOp(Builder, RK, ReducedPartRdx, RdxPart);
      }
    }

    // Create the reduction after the loop. Note that inloop reductions create
    // the target reduction in the loop using a Reduction recipe.
    if (State.VF.isVector() && !PhiR->isInLoop()) {
#if SIFIVE_CUSTOMIZATION
      if (State.Plan->useVLAVectorizer()) {
        Value *InitRVL = State.get(State.Plan->getInitRVL(), 0);
        assert(InitRVL &&
               "InitRVL must be initialized in emitIterationCountCheck when "
               "using VP intrinsic to generate unordered reduction");
        ReducedPartRdx = createTargetReduction(Builder, RdxDesc, ReducedPartRdx,
                                               InitRVL, OrigPhi);
      } else {
        ReducedPartRdx =
            createTargetReduction(Builder, RdxDesc, ReducedPartRdx, OrigPhi);
      }
      // Adjust the final scalar result after the loop if the target prefers
      // that.
      // FIXME: Handle situation that the start value and identity are equal.
      if (PhiR->postFixStartValue()) {
        IRBuilderBase::FastMathFlagGuard FMFG(Builder);
        Builder.setFastMathFlags(RdxDesc.getFastMathFlags());
        Value *StartV = PhiR->getStartValue()->getLiveInIRValue();
        // Truncate start value if the reduction is performed in a smaller type.
        if (PhiTy != RdxDesc.getRecurrenceType())
          StartV = Builder.CreateTrunc(StartV, RdxDesc.getRecurrenceType());
        ReducedPartRdx = Builder.CreateBinOp((Instruction::BinaryOps)Op, StartV,
                                             ReducedPartRdx);
      }
#else
      ReducedPartRdx =
          createTargetReduction(Builder, RdxDesc, ReducedPartRdx, OrigPhi);
#endif // SIFIVE_CUSTOMIZATION
      // If the reduction can be performed in a smaller type, we need to extend
      // the reduction to the wider type before we branch to the original loop.
      if (PhiTy != RdxDesc.getRecurrenceType())
        ReducedPartRdx = RdxDesc.isSigned()
                             ? Builder.CreateSExt(ReducedPartRdx, PhiTy)
                             : Builder.CreateZExt(ReducedPartRdx, PhiTy);
    }
#if SIFIVE_CUSTOMIZATION
    if (RecurrenceDescriptor::isFindLastIVRecurrenceKind(RK))
      ReducedPartRdx =
          createSentinelValueHandling(Builder, RdxDesc, ReducedPartRdx);
#endif // SIFIVE_CUSTOMIZATION

    // If there were stores of the reduction value to a uniform memory address
    // inside the loop, create the final store here.
    if (StoreInst *SI = RdxDesc.IntermediateStore) {
      auto *NewSI = Builder.CreateAlignedStore(
          ReducedPartRdx, SI->getPointerOperand(), SI->getAlign());
      propagateMetadata(NewSI, SI);
    }

    return ReducedPartRdx;
  }
  default:
    llvm_unreachable("Unsupported opcode for instruction");
  }
}

#if !defined(NDEBUG)
bool VPInstruction::isFPMathOp() const {
  // Inspired by FPMathOperator::classof. Notable differences are that we don't
  // support Call, PHI and Select opcodes here yet.
  return Opcode == Instruction::FAdd || Opcode == Instruction::FMul ||
         Opcode == Instruction::FNeg || Opcode == Instruction::FSub ||
         Opcode == Instruction::FDiv || Opcode == Instruction::FRem ||
#if SIFIVE_CUSTOMIZATION
         Opcode == Instruction::Select ||
#endif // SIFIVE_CUSTOMIZATION
         Opcode == Instruction::FCmp || Opcode == Instruction::Select;
}
#endif

void VPInstruction::execute(VPTransformState &State) {
  assert(!State.Instance && "VPInstruction executing an Instance");
  IRBuilderBase::FastMathFlagGuard FMFGuard(State.Builder);
  assert((hasFastMathFlags() == isFPMathOp() ||
          getOpcode() == Instruction::Select) &&
         "Recipe not a FPMathOp but has fast-math flags?");
  if (hasFastMathFlags())
    State.Builder.setFastMathFlags(getFastMathFlags());
  for (unsigned Part = 0; Part < State.UF; ++Part) {
    Value *GeneratedValue = generateInstruction(State, Part);
    if (!hasResult())
      continue;
    assert(GeneratedValue && "generateInstruction must produce a value");
    State.set(this, GeneratedValue, Part);
  }
}
bool VPInstruction::onlyFirstLaneUsed(const VPValue *Op) const {
  assert(is_contained(operands(), Op) && "Op must be an operand of the recipe");
  if (Instruction::isBinaryOp(getOpcode()))
    return vputils::onlyFirstLaneUsed(this);

  switch (getOpcode()) {
  default:
    return false;
  case Instruction::ICmp:
    // TODO: Cover additional opcodes.
    return vputils::onlyFirstLaneUsed(this);
  case VPInstruction::ActiveLaneMask:
#if SIFIVE_CUSTOMIZATION
  case VPInstruction::ExplicitVectorLength:
  case VPInstruction::ExplicitVectorLengthIVIncrement:
#endif // SIFIVE_CUSTOMIZATION
  case VPInstruction::CalculateTripCountMinusVF:
  case VPInstruction::CanonicalIVIncrementForPart:
  case VPInstruction::BranchOnCount:
#if SIFIVE_CUSTOMIZATION
  case VPInstruction::ExitingCond:
#endif
    // TODO: Cover additional operands.
    return getOperand(0) == Op;
  };
  llvm_unreachable("switch should return");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPInstruction::dump() const {
  VPSlotTracker SlotTracker(getParent()->getPlan());
  print(dbgs(), "", SlotTracker);
}

void VPInstruction::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";

  if (hasResult()) {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  switch (getOpcode()) {
  case VPInstruction::Not:
    O << "not";
    break;
  case VPInstruction::SLPLoad:
    O << "combined load";
    break;
  case VPInstruction::SLPStore:
    O << "combined store";
    break;
  case VPInstruction::ActiveLaneMask:
    O << "active lane mask";
    break;
  case VPInstruction::FirstOrderRecurrenceSplice:
    O << "first-order splice";
    break;
  case VPInstruction::BranchOnCond:
    O << "branch-on-cond";
    break;
  case VPInstruction::CalculateTripCountMinusVF:
    O << "TC > VF ? TC - VF : 0";
    break;
  case VPInstruction::CanonicalIVIncrementForPart:
    O << "VF * Part +";
    break;
  case VPInstruction::BranchOnCount:
    O << "branch-on-count";
    break;
  case VPInstruction::ComputeReductionResult:
    O << "compute-reduction-result";
    break;
#if SIFIVE_CUSTOMIZATION
  case VPInstruction::ExplicitVectorLength:
    O << "EXPLICIT-VECTOR-LENGTH";
    break;
  case VPInstruction::ExplicitVectorLengthIVIncrement:
    O << "EXPLICIT-VECTOR-LENGTH +";
    break;
  case VPInstruction::CSAInitMask:
    O << "csa-init-mask";
    break;
  case VPInstruction::CSAInitData:
    O << "csa-init-data";
    break;
  case VPInstruction::CSAMaskPhi:
    O << "csa-mask-phi";
    break;
  case VPInstruction::CSAMaskSel:
    O << "csa-mask-sel";
    break;
  case VPInstruction::CSAVLPhi:
    O << "csa-vl-phi";
    break;
  case VPInstruction::CSAVLSel:
    O << "csa-vl-sel";
    break;
  case VPInstruction::CSAAnyActive:
    O << "csa-anyactive";
    break;
  case VPInstruction::ExitingCond:
    O << "exiting-cond";
    break;
#endif // SIFIVE_CUSTOMIZATION
  default:
    O << Instruction::getOpcodeName(getOpcode());
  }

  printFlags(O);
  printOperands(O, SlotTracker);

  if (auto DL = getDebugLoc()) {
    O << ", !dbg ";
    DL.print(O);
  }
}
#endif

void VPWidenCallRecipe::execute(VPTransformState &State) {
  assert(State.VF.isVector() && "not widening");
  auto &CI = *cast<CallInst>(getUnderlyingInstr());
  assert(!isa<DbgInfoIntrinsic>(CI) &&
         "DbgInfoIntrinsic should have been dropped during VPlan construction");
  State.setDebugLocFrom(getDebugLoc());

#if SIFIVE_CUSTOMIZATION
  if (State.Plan->useVLAVectorizer()) {
    // Skip if CI doesn't have vp form.
    if (Intrinsic::ID VPID = VPIntrinsic::getVPIntrinsicID(VectorIntrinsicID);
        VPIntrinsic::isVPIntrinsic(VPID)) {
      for (unsigned Part = 0; Part < State.UF; ++Part) {
        llvm::widenPredicatedCall(CI, this, *this, State, VPID, Part);
        Value *V = State.get(this, Part);
        State.addMetadata(V, &CI);
      }
      return;
    }
  }
#endif // SIFIVE_CUSTOMIZATION

  bool UseIntrinsic = VectorIntrinsicID != Intrinsic::not_intrinsic;
  FunctionType *VFTy = nullptr;
  if (Variant)
    VFTy = Variant->getFunctionType();
  for (unsigned Part = 0; Part < State.UF; ++Part) {
    SmallVector<Type *, 2> TysForDecl;
    // Add return type if intrinsic is overloaded on it.
    if (UseIntrinsic &&
        isVectorIntrinsicWithOverloadTypeAtArg(VectorIntrinsicID, -1))
      TysForDecl.push_back(
          VectorType::get(CI.getType()->getScalarType(), State.VF));
    SmallVector<Value *, 4> Args;
    for (const auto &I : enumerate(operands())) {
      // Some intrinsics have a scalar argument - don't replace it with a
      // vector.
      Value *Arg;
      if (UseIntrinsic &&
          isVectorIntrinsicWithScalarOpAtArg(VectorIntrinsicID, I.index()))
        Arg = State.get(I.value(), VPIteration(0, 0));
      // Some vectorized function variants may also take a scalar argument,
      // e.g. linear parameters for pointers. This needs to be the scalar value
      // from the start of the respective part when interleaving.
      else if (VFTy && !VFTy->getParamType(I.index())->isVectorTy())
        Arg = State.get(I.value(), VPIteration(Part, 0));
      else
        Arg = State.get(I.value(), Part);
      if (UseIntrinsic &&
          isVectorIntrinsicWithOverloadTypeAtArg(VectorIntrinsicID, I.index()))
        TysForDecl.push_back(Arg->getType());
      Args.push_back(Arg);
    }

    Function *VectorF;
    if (UseIntrinsic) {
      // Use vector version of the intrinsic.
      Module *M = State.Builder.GetInsertBlock()->getModule();
      VectorF = Intrinsic::getDeclaration(M, VectorIntrinsicID, TysForDecl);
      assert(VectorF && "Can't retrieve vector intrinsic.");
    } else {
#ifndef NDEBUG
      assert(Variant != nullptr && "Can't create vector function.");
#endif
      VectorF = Variant;
#if SIFIVE_CUSTOMIZATION
      // Add VL as an explicit final argument to SiFive NF Library functions
      if (VectorF->getName().starts_with(SiFiveNFLibraryPrefix) &&
          State.Plan->useVLAVectorizer()) {
        Value *RVL = State.get(State.RVL, Part);
        Args.push_back(RVL);
      }
#endif // SIFIVE_CUSTOMIZATION
    }

    SmallVector<OperandBundleDef, 1> OpBundles;
    CI.getOperandBundlesAsDefs(OpBundles);
    CallInst *V = State.Builder.CreateCall(VectorF, Args, OpBundles);

    if (isa<FPMathOperator>(V))
      V->copyFastMathFlags(&CI);

    State.set(this, V, Part);
    State.addMetadata(V, &CI);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenCallRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-CALL ";

  auto *CI = cast<CallInst>(getUnderlyingInstr());
  if (CI->getType()->isVoidTy())
    O << "void ";
  else {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  O << "call @" << CI->getCalledFunction()->getName() << "(";
  printOperands(O, SlotTracker);
  O << ")";

  if (VectorIntrinsicID)
    O << " (using vector intrinsic)";
  else {
    O << " (using library function";
    if (Variant->hasName())
      O << ": " << Variant->getName();
    O << ")";
  }
}

void VPWidenSelectRecipe::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-SELECT ";
  printAsOperand(O, SlotTracker);
  O << " = select ";
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(1)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(2)->printAsOperand(O, SlotTracker);
  O << (isInvariantCond() ? " (condition is loop invariant)" : "");
}
#endif

#if SIFIVE_CUSTOMIZATION
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPSelectInstruction::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = select ";
  printFlags(O);
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << ' ';
  getOperand(1)->printAsOperand(O, SlotTracker);
  O << ' ';
  getOperand(2)->printAsOperand(O, SlotTracker);
  if (TP != TailPolicy::Unknown) {
    O << " tail policy = ";
    O << (TP == TailPolicy::Agnostic ? "agnostic" : "undisturbed");
  }

  if (auto DL = getDebugLoc()) {
    O << ", !dbg ";
    DL.print(O);
  }
}

void VPMonotonicUpdateInstruction::print(raw_ostream &O, const Twine &Indent, VPSlotTracker &SlotTracker) const {
  O << Indent << "monotonic-update ";
  printAsOperand(O, SlotTracker);
  O << " = ";
  O << MD.getUpdateOp()->getOpcodeName();
  O << ' ';
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(1)->printAsOperand(O, SlotTracker);
  O << " @";
  getMask()->printAsOperand(O, SlotTracker);

  if (auto DL = getDebugLoc()) {
    O << ", !dbg ";
    DL.print(O);
  }
}

void VPMonotonicHeaderPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                       VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = monotonic-phi ";
  printOperands(O, SlotTracker);
}
#endif
#endif // SIFIVE_CUSTOMIZATION

void VPWidenSelectRecipe::execute(VPTransformState &State) {
  State.setDebugLocFrom(getDebugLoc());

  // The condition can be loop invariant but still defined inside the
  // loop. This means that we can't just use the original 'cond' value.
  // We have to take the 'vectorized' value and pick the first lane.
  // Instcombine will make this a no-op.
  auto *InvarCond =
      isInvariantCond() ? State.get(getCond(), VPIteration(0, 0)) : nullptr;

  for (unsigned Part = 0; Part < State.UF; ++Part) {
    Value *Cond = InvarCond ? InvarCond : State.get(getCond(), Part);
    Value *Op0 = State.get(getOperand(1), Part);
    Value *Op1 = State.get(getOperand(2), Part);
#if SIFIVE_CUSTOMIZATION
    Value *Sel;
    if (State.Plan->useVLAVectorizer() && Cond->getType()->isVectorTy()) {
      Value *RVLArg = State.get(State.RVL, Part);
      Sel = State.Builder.CreateIntrinsic(Intrinsic::vp_select, {Op0->getType()},
                                          {Cond, Op0, Op1, RVLArg}, nullptr,
                                          "vp.widen.select");
    } else {
      Sel = State.Builder.CreateSelect(Cond, Op0, Op1);
    }
#else
    Value *Sel = State.Builder.CreateSelect(Cond, Op0, Op1);
#endif // SIFIVE_CUSTOMIZATION
    State.set(this, Sel, Part);
    State.addMetadata(Sel, dyn_cast_or_null<Instruction>(getUnderlyingValue()));
  }
}

VPRecipeWithIRFlags::FastMathFlagsTy::FastMathFlagsTy(
    const FastMathFlags &FMF) {
  AllowReassoc = FMF.allowReassoc();
  NoNaNs = FMF.noNaNs();
  NoInfs = FMF.noInfs();
  NoSignedZeros = FMF.noSignedZeros();
  AllowReciprocal = FMF.allowReciprocal();
  AllowContract = FMF.allowContract();
  ApproxFunc = FMF.approxFunc();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPRecipeWithIRFlags::printFlags(raw_ostream &O) const {
  switch (OpType) {
  case OperationType::Cmp:
    O << " " << CmpInst::getPredicateName(getPredicate());
    break;
  case OperationType::DisjointOp:
    if (DisjointFlags.IsDisjoint)
      O << " disjoint";
    break;
  case OperationType::PossiblyExactOp:
    if (ExactFlags.IsExact)
      O << " exact";
    break;
  case OperationType::OverflowingBinOp:
    if (WrapFlags.HasNUW)
      O << " nuw";
    if (WrapFlags.HasNSW)
      O << " nsw";
    break;
  case OperationType::FPMathOp:
    getFastMathFlags().print(O);
    break;
  case OperationType::GEPOp:
    if (GEPFlags.IsInBounds)
      O << " inbounds";
    break;
  case OperationType::NonNegOp:
    if (NonNegFlags.NonNeg)
      O << " nneg";
    break;
  case OperationType::Other:
    break;
  }
  if (getNumOperands() > 0)
    O << " ";
}
#endif

void VPWidenRecipe::execute(VPTransformState &State) {
  State.setDebugLocFrom(getDebugLoc());
#if SIFIVE_CUSTOMIZATION
  auto *I = cast_or_null<Instruction>(getUnderlyingValue());
  if (I && State.Plan->useVLAVectorizer() &&
      State.get(getOperand(0), 0)->getType()->isVectorTy() &&
      !isa<BitCastInst>(I) && !isa<FreezeInst>(I)) {
    // Bitcasts are not supported.
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Value *V = llvm::widenPredicatedInstruction(I, this, *this, State,
                                                  nullptr, Part);
      State.set(this, V, Part);
      //Value *V = State.get(this, Part);
      State.addMetadata(V, I);
    }
    return;
  }
#endif // SIFIVE_CUSTOMIZATION
  auto &Builder = State.Builder;
  switch (Opcode) {
  case Instruction::Call:
  case Instruction::Br:
  case Instruction::PHI:
  case Instruction::GetElementPtr:
  case Instruction::Select:
    llvm_unreachable("This instruction is handled by a different recipe.");
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::FNeg:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    // Just widen unops and binops.
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      SmallVector<Value *, 2> Ops;
      for (VPValue *VPOp : operands())
        Ops.push_back(State.get(VPOp, Part));

      Value *V = Builder.CreateNAryOp(Opcode, Ops);

      if (auto *VecOp = dyn_cast<Instruction>(V))
        setFlags(VecOp);

      // Use this vector value for all users of the original instruction.
      State.set(this, V, Part);
      State.addMetadata(V, dyn_cast_or_null<Instruction>(getUnderlyingValue()));
    }

    break;
  }
  case Instruction::Freeze: {
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Value *Op = State.get(getOperand(0), Part);

      Value *Freeze = Builder.CreateFreeze(Op);
      State.set(this, Freeze, Part);
    }
    break;
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    // Widen compares. Generate vector compares.
    bool FCmp = Opcode == Instruction::FCmp;
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Value *A = State.get(getOperand(0), Part);
      Value *B = State.get(getOperand(1), Part);
      Value *C = nullptr;
      if (FCmp) {
        // Propagate fast math flags.
        IRBuilder<>::FastMathFlagGuard FMFG(Builder);
        if (auto *I = dyn_cast_or_null<Instruction>(getUnderlyingValue()))
          Builder.setFastMathFlags(I->getFastMathFlags());
        C = Builder.CreateFCmp(getPredicate(), A, B);
      } else {
        C = Builder.CreateICmp(getPredicate(), A, B);
      }
      State.set(this, C, Part);
      State.addMetadata(C, dyn_cast_or_null<Instruction>(getUnderlyingValue()));
    }

    break;
  }
  default:
    // This instruction is not vectorized by simple widening.
    LLVM_DEBUG(dbgs() << "LV: Found an unhandled opcode : "
                      << Instruction::getOpcodeName(Opcode));
    llvm_unreachable("Unhandled instruction!");
  } // end of switch.

#if !defined(NDEBUG)
  // Verify that VPlan type inference results agree with the type of the
  // generated values.
  for (unsigned Part = 0; Part < State.UF; ++Part) {
    assert(VectorType::get(State.TypeAnalysis.inferScalarType(this),
                           State.VF) == State.get(this, Part)->getType() &&
           "inferred type and type from generated instructions do not match");
  }
#endif
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenRecipe::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN ";
  printAsOperand(O, SlotTracker);
  O << " = " << Instruction::getOpcodeName(Opcode);
  printFlags(O);
  printOperands(O, SlotTracker);
}
#endif

void VPWidenCastRecipe::execute(VPTransformState &State) {
  State.setDebugLocFrom(getDebugLoc());
#if SIFIVE_CUSTOMIZATION
  auto *I = cast_or_null<Instruction>(getUnderlyingValue());
  if (I && State.Plan->useVLAVectorizer() &&
      State.get(getOperand(0), 0)->getType()->isVectorTy() &&
      !isa<BitCastInst>(I) && !isa<FreezeInst>(I)) {
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Value *V = llvm::widenPredicatedInstruction(I, this, *this, State,
                                                  nullptr, Part);
      State.set(this, V, Part);
      // Value *V = State.get(this, Part);
      State.addMetadata(V, I);
    }
    return;
  }
#endif // SIFIVE_CUSTOMIZATION
  auto &Builder = State.Builder;
  /// Vectorize casts.
  assert(State.VF.isVector() && "Not vectorizing?");
  Type *DestTy = VectorType::get(getResultType(), State.VF);
  VPValue *Op = getOperand(0);
  for (unsigned Part = 0; Part < State.UF; ++Part) {
    if (Part > 0 && Op->isLiveIn()) {
      // FIXME: Remove once explicit unrolling is implemented using VPlan.
      State.set(this, State.get(this, 0), Part);
      continue;
    }
    Value *A = State.get(Op, Part);
    Value *Cast = Builder.CreateCast(Instruction::CastOps(Opcode), A, DestTy);
    State.set(this, Cast, Part);
    State.addMetadata(Cast, cast_or_null<Instruction>(getUnderlyingValue()));
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenCastRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-CAST ";
  printAsOperand(O, SlotTracker);
  O << " = " << Instruction::getOpcodeName(Opcode) << " ";
  printFlags(O);
  printOperands(O, SlotTracker);
  O << " to " << *getResultType();
}
#endif

/// This function adds
/// (StartIdx * Step, (StartIdx + 1) * Step, (StartIdx + 2) * Step, ...)
/// to each vector element of Val. The sequence starts at StartIndex.
/// \p Opcode is relevant for FP induction variable.
static Value *getStepVector(Value *Val, Value *StartIdx, Value *Step,
                            Instruction::BinaryOps BinOp, ElementCount VF,
                            IRBuilderBase &Builder) {
  assert(VF.isVector() && "only vector VFs are supported");

  // Create and check the types.
  auto *ValVTy = cast<VectorType>(Val->getType());
  ElementCount VLen = ValVTy->getElementCount();

  Type *STy = Val->getType()->getScalarType();
  assert((STy->isIntegerTy() || STy->isFloatingPointTy()) &&
         "Induction Step must be an integer or FP");
  assert(Step->getType() == STy && "Step has wrong type");

  SmallVector<Constant *, 8> Indices;

  // Create a vector of consecutive numbers from zero to VF.
  VectorType *InitVecValVTy = ValVTy;
  if (STy->isFloatingPointTy()) {
    Type *InitVecValSTy =
        IntegerType::get(STy->getContext(), STy->getScalarSizeInBits());
    InitVecValVTy = VectorType::get(InitVecValSTy, VLen);
  }
  Value *InitVec = Builder.CreateStepVector(InitVecValVTy);

  // Splat the StartIdx
  Value *StartIdxSplat = Builder.CreateVectorSplat(VLen, StartIdx);

  if (STy->isIntegerTy()) {
    InitVec = Builder.CreateAdd(InitVec, StartIdxSplat);
    Step = Builder.CreateVectorSplat(VLen, Step);
    assert(Step->getType() == Val->getType() && "Invalid step vec");
    // FIXME: The newly created binary instructions should contain nsw/nuw
    // flags, which can be found from the original scalar operations.
    Step = Builder.CreateMul(InitVec, Step);
    return Builder.CreateAdd(Val, Step, "induction");
  }

  // Floating point induction.
  assert((BinOp == Instruction::FAdd || BinOp == Instruction::FSub) &&
         "Binary Opcode should be specified for FP induction");
  InitVec = Builder.CreateUIToFP(InitVec, ValVTy);
  InitVec = Builder.CreateFAdd(InitVec, StartIdxSplat);

  Step = Builder.CreateVectorSplat(VLen, Step);
  Value *MulOp = Builder.CreateFMul(InitVec, Step);
  return Builder.CreateBinOp(BinOp, Val, MulOp, "induction");
}

/// A helper function that returns an integer or floating-point constant with
/// value C.
static Constant *getSignedIntOrFpConstant(Type *Ty, int64_t C) {
  return Ty->isIntegerTy() ? ConstantInt::getSigned(Ty, C)
                           : ConstantFP::get(Ty, C);
}

static Value *getRuntimeVFAsFloat(IRBuilderBase &B, Type *FTy,
                                  ElementCount VF) {
  assert(FTy->isFloatingPointTy() && "Expected floating point type!");
  Type *IntTy = IntegerType::get(FTy->getContext(), FTy->getScalarSizeInBits());
  Value *RuntimeVF = getRuntimeVF(B, IntTy, VF);
  return B.CreateUIToFP(RuntimeVF, FTy);
}

void VPWidenIntOrFpInductionRecipe::execute(VPTransformState &State) {
  assert(!State.Instance && "Int or FP induction being replicated.");

  Value *Start = getStartValue()->getLiveInIRValue();
  const InductionDescriptor &ID = getInductionDescriptor();
  TruncInst *Trunc = getTruncInst();
  IRBuilderBase &Builder = State.Builder;
  assert(IV->getType() == ID.getStartValue()->getType() && "Types must match");
  assert(State.VF.isVector() && "must have vector VF");

  // The value from the original loop to which we are mapping the new induction
  // variable.
  Instruction *EntryVal = Trunc ? cast<Instruction>(Trunc) : IV;

  // Fast-math-flags propagate from the original induction instruction.
  IRBuilder<>::FastMathFlagGuard FMFG(Builder);
  if (ID.getInductionBinOp() && isa<FPMathOperator>(ID.getInductionBinOp()))
    Builder.setFastMathFlags(ID.getInductionBinOp()->getFastMathFlags());

  // Now do the actual transformations, and start with fetching the step value.
  Value *Step = State.get(getStepValue(), VPIteration(0, 0));

  assert((isa<PHINode>(EntryVal) || isa<TruncInst>(EntryVal)) &&
         "Expected either an induction phi-node or a truncate of it!");

  // Construct the initial value of the vector IV in the vector loop preheader
  auto CurrIP = Builder.saveIP();
  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  Builder.SetInsertPoint(VectorPH->getTerminator());
  if (isa<TruncInst>(EntryVal)) {
    assert(Start->getType()->isIntegerTy() &&
           "Truncation requires an integer type");
    auto *TruncType = cast<IntegerType>(EntryVal->getType());
    Step = Builder.CreateTrunc(Step, TruncType);
    Start = Builder.CreateCast(Instruction::Trunc, Start, TruncType);
  }

  Value *Zero = getSignedIntOrFpConstant(Start->getType(), 0);
  Value *SplatStart = Builder.CreateVectorSplat(State.VF, Start);
  Value *SteppedStart = getStepVector(
      SplatStart, Zero, Step, ID.getInductionOpcode(), State.VF, State.Builder);

  // We create vector phi nodes for both integer and floating-point induction
  // variables. Here, we determine the kind of arithmetic we will perform.
  Instruction::BinaryOps AddOp;
  Instruction::BinaryOps MulOp;
  if (Step->getType()->isIntegerTy()) {
    AddOp = Instruction::Add;
    MulOp = Instruction::Mul;
  } else {
    AddOp = ID.getInductionOpcode();
    MulOp = Instruction::FMul;
  }

  // Multiply the vectorization factor by the step using integer or
  // floating-point arithmetic as appropriate.
  Type *StepType = Step->getType();
  Value *RuntimeVF;
  if (Step->getType()->isFloatingPointTy())
    RuntimeVF = getRuntimeVFAsFloat(Builder, StepType, State.VF);
  else
    RuntimeVF = getRuntimeVF(Builder, StepType, State.VF);
  Value *Mul = Builder.CreateBinOp(MulOp, Step, RuntimeVF);

  // Create a vector splat to use in the induction update.
  //
  // FIXME: If the step is non-constant, we create the vector splat with
  //        IRBuilder. IRBuilder can constant-fold the multiply, but it doesn't
  //        handle a constant vector splat.
  Value *SplatVF = isa<Constant>(Mul)
                       ? ConstantVector::getSplat(State.VF, cast<Constant>(Mul))
                       : Builder.CreateVectorSplat(State.VF, Mul);
  Builder.restoreIP(CurrIP);

  // We may need to add the step a number of times, depending on the unroll
  // factor. The last of those goes into the PHI.
  PHINode *VecInd = PHINode::Create(SteppedStart->getType(), 2, "vec.ind");
  VecInd->insertBefore(State.CFG.PrevBB->getFirstInsertionPt());
  VecInd->setDebugLoc(EntryVal->getDebugLoc());
  Instruction *LastInduction = VecInd;
  for (unsigned Part = 0; Part < State.UF; ++Part) {
    State.set(this, LastInduction, Part);

    if (isa<TruncInst>(EntryVal))
      State.addMetadata(LastInduction, EntryVal);

#if SIFIVE_CUSTOMIZATION
    if (State.Plan->useVLAVectorizer()) {
      // FIXME: Remove this code with a proper representation of pointer induction
      // in a VPlan.
      assert(!State.RVL &&
             "Runtime VL is available, but code was not updated to use it.");
      Value *RVLPart = nullptr;
      Value *RVLPartCast = nullptr;
      if (!State.RVLPlaceholder) {
        Type *I32Ty = Builder.getInt32Ty();
        State.RVLPlaceholder = RVLPart = State.Builder.CreateLoad(
            I32Ty, UndefValue::get(I32Ty->getPointerTo()));
      } else {
        RVLPart = State.RVLPlaceholder;
      }
      RVLPartCast = StepType->isIntegerTy()
                        ? Builder.CreateSExtOrTrunc(RVLPart, StepType)
                        : Builder.CreateUIToFP(RVLPart, StepType);
      Value *Mul = Builder.CreateBinOp(MulOp, Step, RVLPartCast);
      SplatVF = Builder.CreateVectorSplat(State.VF, Mul);
      LastInduction = widenPredicatedArithmeticOp(
          State, AddOp, {LastInduction, SplatVF}, Part,
          /*Mask=*/nullptr, "step.add");
    } else
#endif // SIFIVE_CUSTOMIZATION
    LastInduction = cast<Instruction>(
        Builder.CreateBinOp(AddOp, LastInduction, SplatVF, "step.add"));
    LastInduction->setDebugLoc(EntryVal->getDebugLoc());
  }

  LastInduction->setName("vec.ind.next");
  VecInd->addIncoming(SteppedStart, VectorPH);
  // Add induction update using an incorrect block temporarily. The phi node
  // will be fixed after VPlan execution. Note that at this point the latch
  // block cannot be used, as it does not exist yet.
  // TODO: Model increment value in VPlan, by turning the recipe into a
  // multi-def and a subclass of VPHeaderPHIRecipe.
  VecInd->addIncoming(LastInduction, VectorPH);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenIntOrFpInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-INDUCTION";
  if (getTruncInst()) {
    O << "\\l\"";
    O << " +\n" << Indent << "\"  " << VPlanIngredient(IV) << "\\l\"";
    O << " +\n" << Indent << "\"  ";
    getVPValue(0)->printAsOperand(O, SlotTracker);
  } else
    O << " " << VPlanIngredient(IV);

  O << ", ";
  getStepValue()->printAsOperand(O, SlotTracker);
}
#endif

bool VPWidenIntOrFpInductionRecipe::isCanonical() const {
  // The step may be defined by a recipe in the preheader (e.g. if it requires
  // SCEV expansion), but for the canonical induction the step is required to be
  // 1, which is represented as live-in.
  if (getStepValue()->getDefiningRecipe())
    return false;
  auto *StepC = dyn_cast<ConstantInt>(getStepValue()->getLiveInIRValue());
  auto *StartC = dyn_cast<ConstantInt>(getStartValue()->getLiveInIRValue());
  return StartC && StartC->isZero() && StepC && StepC->isOne();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPDerivedIVRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << Indent << "= DERIVED-IV ";
  getStartValue()->printAsOperand(O, SlotTracker);
  O << " + ";
  getCanonicalIV()->printAsOperand(O, SlotTracker);
  O << " * ";
  getStepValue()->printAsOperand(O, SlotTracker);
}
#endif

void VPScalarIVStepsRecipe::execute(VPTransformState &State) {
  // Fast-math-flags propagate from the original induction instruction.
  IRBuilder<>::FastMathFlagGuard FMFG(State.Builder);
  if (hasFastMathFlags())
    State.Builder.setFastMathFlags(getFastMathFlags());

  /// Compute scalar induction steps. \p ScalarIV is the scalar induction
  /// variable on which to base the steps, \p Step is the size of the step.

  Value *BaseIV = State.get(getOperand(0), VPIteration(0, 0));
  Value *Step = State.get(getStepValue(), VPIteration(0, 0));
  IRBuilderBase &Builder = State.Builder;

  // Ensure step has the same type as that of scalar IV.
  Type *BaseIVTy = BaseIV->getType()->getScalarType();
  assert(BaseIVTy == Step->getType() && "Types of BaseIV and Step must match!");

  // We build scalar steps for both integer and floating-point induction
  // variables. Here, we determine the kind of arithmetic we will perform.
  Instruction::BinaryOps AddOp;
  Instruction::BinaryOps MulOp;
  if (BaseIVTy->isIntegerTy()) {
    AddOp = Instruction::Add;
    MulOp = Instruction::Mul;
  } else {
    AddOp = InductionOpcode;
    MulOp = Instruction::FMul;
  }

  // Determine the number of scalars we need to generate for each unroll
  // iteration.
  bool FirstLaneOnly = vputils::onlyFirstLaneUsed(this);
  // Compute the scalar steps and save the results in State.
  Type *IntStepTy =
      IntegerType::get(BaseIVTy->getContext(), BaseIVTy->getScalarSizeInBits());
  Type *VecIVTy = nullptr;
  Value *UnitStepVec = nullptr, *SplatStep = nullptr, *SplatIV = nullptr;
  if (!FirstLaneOnly && State.VF.isScalable()) {
    VecIVTy = VectorType::get(BaseIVTy, State.VF);
    UnitStepVec =
        Builder.CreateStepVector(VectorType::get(IntStepTy, State.VF));
    SplatStep = Builder.CreateVectorSplat(State.VF, Step);
    SplatIV = Builder.CreateVectorSplat(State.VF, BaseIV);
  }

  unsigned StartPart = 0;
  unsigned EndPart = State.UF;
  unsigned StartLane = 0;
  unsigned EndLane = FirstLaneOnly ? 1 : State.VF.getKnownMinValue();
  if (State.Instance) {
    StartPart = State.Instance->Part;
    EndPart = StartPart + 1;
    StartLane = State.Instance->Lane.getKnownLane();
    EndLane = StartLane + 1;
  }
  for (unsigned Part = StartPart; Part < EndPart; ++Part) {
    Value *StartIdx0 = createStepForVF(Builder, IntStepTy, State.VF, Part);

    if (!FirstLaneOnly && State.VF.isScalable()) {
      auto *SplatStartIdx = Builder.CreateVectorSplat(State.VF, StartIdx0);
      auto *InitVec = Builder.CreateAdd(SplatStartIdx, UnitStepVec);
      if (BaseIVTy->isFloatingPointTy())
        InitVec = Builder.CreateSIToFP(InitVec, VecIVTy);
      auto *Mul = Builder.CreateBinOp(MulOp, InitVec, SplatStep);
      auto *Add = Builder.CreateBinOp(AddOp, SplatIV, Mul);
      State.set(this, Add, Part);
      // It's useful to record the lane values too for the known minimum number
      // of elements so we do those below. This improves the code quality when
      // trying to extract the first element, for example.
    }

    if (BaseIVTy->isFloatingPointTy())
      StartIdx0 = Builder.CreateSIToFP(StartIdx0, BaseIVTy);

    for (unsigned Lane = StartLane; Lane < EndLane; ++Lane) {
      Value *StartIdx = Builder.CreateBinOp(
          AddOp, StartIdx0, getSignedIntOrFpConstant(BaseIVTy, Lane));
      // The step returned by `createStepForVF` is a runtime-evaluated value
      // when VF is scalable. Otherwise, it should be folded into a Constant.
      assert((State.VF.isScalable() || isa<Constant>(StartIdx)) &&
             "Expected StartIdx to be folded to a constant when VF is not "
             "scalable");
      auto *Mul = Builder.CreateBinOp(MulOp, StartIdx, Step);
      auto *Add = Builder.CreateBinOp(AddOp, BaseIV, Mul);
      State.set(this, Add, VPIteration(Part, Lane));
    }
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPScalarIVStepsRecipe::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << " = SCALAR-STEPS ";
  printOperands(O, SlotTracker);
}
#endif

void VPWidenGEPRecipe::execute(VPTransformState &State) {
  assert(State.VF.isVector() && "not widening");
  auto *GEP = cast<GetElementPtrInst>(getUnderlyingInstr());
  // Construct a vector GEP by widening the operands of the scalar GEP as
  // necessary. We mark the vector GEP 'inbounds' if appropriate. A GEP
  // results in a vector of pointers when at least one operand of the GEP
  // is vector-typed. Thus, to keep the representation compact, we only use
  // vector-typed operands for loop-varying values.

  if (areAllOperandsInvariant()) {
    // If we are vectorizing, but the GEP has only loop-invariant operands,
    // the GEP we build (by only using vector-typed operands for
    // loop-varying values) would be a scalar pointer. Thus, to ensure we
    // produce a vector of pointers, we need to either arbitrarily pick an
    // operand to broadcast, or broadcast a clone of the original GEP.
    // Here, we broadcast a clone of the original.
    //
    // TODO: If at some point we decide to scalarize instructions having
    //       loop-invariant operands, this special case will no longer be
    //       required. We would add the scalarization decision to
    //       collectLoopScalars() and teach getVectorValue() to broadcast
    //       the lane-zero scalar value.
    SmallVector<Value *> Ops;
    for (unsigned I = 0, E = getNumOperands(); I != E; I++)
      Ops.push_back(State.get(getOperand(I), VPIteration(0, 0)));

    auto *NewGEP =
        State.Builder.CreateGEP(GEP->getSourceElementType(), Ops[0],
                                ArrayRef(Ops).drop_front(), "", isInBounds());
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Value *EntryPart = State.Builder.CreateVectorSplat(State.VF, NewGEP);
      State.set(this, EntryPart, Part);
      State.addMetadata(EntryPart, GEP);
    }
  } else {
    // If the GEP has at least one loop-varying operand, we are sure to
    // produce a vector of pointers. But if we are only unrolling, we want
    // to produce a scalar GEP for each unroll part. Thus, the GEP we
    // produce with the code below will be scalar (if VF == 1) or vector
    // (otherwise). Note that for the unroll-only case, we still maintain
    // values in the vector mapping with initVector, as we do for other
    // instructions.
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      // The pointer operand of the new GEP. If it's loop-invariant, we
      // won't broadcast it.
      auto *Ptr = isPointerLoopInvariant()
                      ? State.get(getOperand(0), VPIteration(0, 0))
                      : State.get(getOperand(0), Part);

      // Collect all the indices for the new GEP. If any index is
      // loop-invariant, we won't broadcast it.
      SmallVector<Value *, 4> Indices;
      for (unsigned I = 1, E = getNumOperands(); I < E; I++) {
        VPValue *Operand = getOperand(I);
        if (isIndexLoopInvariant(I - 1))
          Indices.push_back(State.get(Operand, VPIteration(0, 0)));
        else
          Indices.push_back(State.get(Operand, Part));
      }

      // Create the new GEP. Note that this GEP may be a scalar if VF == 1,
      // but it should be a vector, otherwise.
      auto *NewGEP = State.Builder.CreateGEP(GEP->getSourceElementType(), Ptr,
                                             Indices, "", isInBounds());
      assert((State.VF.isScalar() || NewGEP->getType()->isVectorTy()) &&
             "NewGEP is not a pointer vector");
      State.set(this, NewGEP, Part);
      State.addMetadata(NewGEP, GEP);
    }
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenGEPRecipe::print(raw_ostream &O, const Twine &Indent,
                             VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-GEP ";
  O << (isPointerLoopInvariant() ? "Inv" : "Var");
  for (size_t I = 0; I < getNumOperands() - 1; ++I)
    O << "[" << (isIndexLoopInvariant(I) ? "Inv" : "Var") << "]";

  O << " ";
  printAsOperand(O, SlotTracker);
  O << " = getelementptr";
  printFlags(O);
  printOperands(O, SlotTracker);
}
#endif

void VPVectorPointerRecipe ::execute(VPTransformState &State) {
  auto &Builder = State.Builder;
  State.setDebugLocFrom(getDebugLoc());
  for (unsigned Part = 0; Part < State.UF; ++Part) {
    // Calculate the pointer for the specific unroll-part.
    Value *PartPtr = nullptr;
    // Use i32 for the gep index type when the value is constant,
    // or query DataLayout for a more suitable index type otherwise.
    const DataLayout &DL =
        Builder.GetInsertBlock()->getModule()->getDataLayout();
    Type *IndexTy = State.VF.isScalable() && (IsReverse || Part > 0)
                        ? DL.getIndexType(IndexedTy->getPointerTo())
                        : Builder.getInt32Ty();
    Value *Ptr = State.get(getOperand(0), VPIteration(0, 0));
    bool InBounds = isInBounds();
    if (IsReverse) {
      // If the address is consecutive but reversed, then the
      // wide store needs to start at the last vector element.
      // RunTimeVF =  VScale * VF.getKnownMinValue()
      // For fixed-width VScale is 1, then RunTimeVF = VF.getKnownMinValue()
#if SIFIVE_CUSTOMIZATION
      Value *RunTimeVF;
      if (State.Plan->useVLAVectorizer()) {
        VPValue *RVL = State.RVL;
        // If RVL is not nullptr, then RVL must be a valid value set during plan
        // creation and must be used to correctly reverse the address
        RunTimeVF = State.get(RVL, Part);
        if (RunTimeVF->getType() != IndexTy)
          RunTimeVF = Builder.CreateZExtOrTrunc(RunTimeVF, IndexTy);
      } else {
        RunTimeVF = getRuntimeVF(Builder, IndexTy, State.VF);
      }
#else
      Value *RunTimeVF = getRuntimeVF(Builder, IndexTy, State.VF);
#endif // SIFIVE_CUSTOMIZATION
      // NumElt = -Part * RunTimeVF
      Value *NumElt = Builder.CreateMul(
          ConstantInt::get(IndexTy, -(int64_t)Part), RunTimeVF);
      // LastLane = 1 - RunTimeVF
      Value *LastLane =
          Builder.CreateSub(ConstantInt::get(IndexTy, 1), RunTimeVF);
      PartPtr = Builder.CreateGEP(IndexedTy, Ptr, NumElt, "", InBounds);
      PartPtr = Builder.CreateGEP(IndexedTy, PartPtr, LastLane, "", InBounds);
    } else {
      Value *Increment = createStepForVF(Builder, IndexTy, State.VF, Part);
      PartPtr = Builder.CreateGEP(IndexedTy, Ptr, Increment, "", InBounds);
    }

    State.set(this, PartPtr, Part);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPVectorPointerRecipe::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << " = vector-pointer ";
  if (IsReverse)
    O << "(reverse) ";

  printOperands(O, SlotTracker);
}
#endif

void VPBlendRecipe::execute(VPTransformState &State) {
  State.setDebugLocFrom(getDebugLoc());
  // We know that all PHIs in non-header blocks are converted into
  // selects, so we don't have to worry about the insertion order and we
  // can just use the builder.
  // At this point we generate the predication tree. There may be
  // duplications since this is a simple recursive scan, but future
  // optimizations will clean it up.

  unsigned NumIncoming = getNumIncomingValues();

  // Generate a sequence of selects of the form:
  // SELECT(Mask3, In3,
  //        SELECT(Mask2, In2,
  //               SELECT(Mask1, In1,
  //                      In0)))
  // Note that Mask0 is never used: lanes for which no path reaches this phi and
  // are essentially undef are taken from In0.
 VectorParts Entry(State.UF);
  for (unsigned In = 0; In < NumIncoming; ++In) {
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      // We might have single edge PHIs (blocks) - use an identity
      // 'select' for the first PHI operand.
      Value *In0 = State.get(getIncomingValue(In), Part);
      if (In == 0)
        Entry[Part] = In0; // Initialize with the first incoming value.
      else {
        // Select between the current value and the previous incoming edge
        // based on the incoming mask.
        Value *Cond = State.get(getMask(In), Part);
#if SIFIVE_CUSTOMIZATION
        if (State.Plan->useVLAVectorizer() && Cond->getType()->isVectorTy()) {
          Value *RVLArg = State.get(State.RVL, Part);
          Entry[Part] = State.Builder.CreateIntrinsic(
              Intrinsic::vp_select, {In0->getType()},
              {Cond, In0, Entry[Part], RVLArg}, nullptr, "predphi");
        } else
#endif // SIFIVE_CUSTOMIZATION
        Entry[Part] =
            State.Builder.CreateSelect(Cond, In0, Entry[Part], "predphi");
      }
    }
  }
  for (unsigned Part = 0; Part < State.UF; ++Part)
    State.set(this, Entry[Part], Part);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPBlendRecipe::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "BLEND ";
  printAsOperand(O, SlotTracker);
  O << " =";
  if (getNumIncomingValues() == 1) {
    // Not a User of any mask: not really blending, this is a
    // single-predecessor phi.
    O << " ";
    getIncomingValue(0)->printAsOperand(O, SlotTracker);
  } else {
    for (unsigned I = 0, E = getNumIncomingValues(); I < E; ++I) {
      O << " ";
      getIncomingValue(I)->printAsOperand(O, SlotTracker);
      O << "/";
      getMask(I)->printAsOperand(O, SlotTracker);
    }
  }
}

void VPReductionRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "REDUCE ";
  printAsOperand(O, SlotTracker);
  O << " = ";
  getChainOp()->printAsOperand(O, SlotTracker);
  O << " +";
  if (isa<FPMathOperator>(getUnderlyingInstr()))
    O << getUnderlyingInstr()->getFastMathFlags();
  O << " reduce." << Instruction::getOpcodeName(RdxDesc.getOpcode()) << " (";
  getVecOp()->printAsOperand(O, SlotTracker);
  if (getCondOp()) {
    O << ", ";
    getCondOp()->printAsOperand(O, SlotTracker);
  }
  O << ")";
  if (RdxDesc.IntermediateStore)
    O << " (with final reduction value stored in invariant address sank "
         "outside of loop)";
}
#endif

bool VPReplicateRecipe::shouldPack() const {
  // Find if the recipe is used by a widened recipe via an intervening
  // VPPredInstPHIRecipe. In this case, also pack the scalar values in a vector.
  return any_of(users(), [](const VPUser *U) {
    if (auto *PredR = dyn_cast<VPPredInstPHIRecipe>(U))
      return any_of(PredR->users(), [PredR](const VPUser *U) {
        return !U->usesScalars(PredR);
      });
    return false;
  });
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPReplicateRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << (IsUniform ? "CLONE " : "REPLICATE ");

  if (!getUnderlyingInstr()->getType()->isVoidTy()) {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }
  if (auto *CB = dyn_cast<CallBase>(getUnderlyingInstr())) {
    O << "call";
    printFlags(O);
    O << "@" << CB->getCalledFunction()->getName() << "(";
    interleaveComma(make_range(op_begin(), op_begin() + (getNumOperands() - 1)),
                    O, [&O, &SlotTracker](VPValue *Op) {
                      Op->printAsOperand(O, SlotTracker);
                    });
    O << ")";
  } else {
    O << Instruction::getOpcodeName(getUnderlyingInstr()->getOpcode());
    printFlags(O);
    printOperands(O, SlotTracker);
  }

  if (shouldPack())
    O << " (S->V)";
}
#endif

/// Checks if \p C is uniform across all VFs and UFs. It is considered as such
/// if it is either defined outside the vector region or its operand is known to
/// be uniform across all VFs and UFs (e.g. VPDerivedIV or VPCanonicalIVPHI).
/// TODO: Uniformity should be associated with a VPValue and there should be a
/// generic way to check.
static bool isUniformAcrossVFsAndUFs(VPScalarCastRecipe *C) {
  return C->isDefinedOutsideVectorRegions() ||
         isa<VPDerivedIVRecipe>(C->getOperand(0)) ||
         isa<VPCanonicalIVPHIRecipe>(C->getOperand(0));
}

Value *VPScalarCastRecipe ::generate(VPTransformState &State, unsigned Part) {
  assert(vputils::onlyFirstLaneUsed(this) &&
         "Codegen only implemented for first lane.");
  switch (Opcode) {
  case Instruction::SExt:
  case Instruction::ZExt:
  case Instruction::Trunc: {
    // Note: SExt/ZExt not used yet.
    Value *Op = State.get(getOperand(0), VPIteration(Part, 0));
    return State.Builder.CreateCast(Instruction::CastOps(Opcode), Op, ResultTy);
  }
  default:
    llvm_unreachable("opcode not implemented yet");
  }
}

void VPScalarCastRecipe ::execute(VPTransformState &State) {
  bool IsUniformAcrossVFsAndUFs = isUniformAcrossVFsAndUFs(this);
  for (unsigned Part = 0; Part != State.UF; ++Part) {
    Value *Res;
    // Only generate a single instance, if the recipe is uniform across UFs and
    // VFs.
    if (Part > 0 && IsUniformAcrossVFsAndUFs)
      Res = State.get(this, VPIteration(0, 0));
    else
      Res = generate(State, Part);
    State.set(this, Res, VPIteration(Part, 0));
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPScalarCastRecipe ::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << Indent << "SCALAR-CAST ";
  printAsOperand(O, SlotTracker);
  O << " = " << Instruction::getOpcodeName(Opcode) << " ";
  printOperands(O, SlotTracker);
  O << " to " << *ResultTy;
}
#endif

#if SIFIVE_CUSTOMIZATION
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPCSAHeaderPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                 VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = csa-data-phi ";
  printOperands(O, SlotTracker);

}
#endif

void VPCSAHeaderPHIRecipe::execute(VPTransformState &State) {
  // PrevBB is this BB
  IRBuilder<>::InsertPointGuard Guard(State.Builder);
  State.Builder.SetInsertPoint(State.CFG.PrevBB->getFirstNonPHI());

  Value *InitData = State.get(getVPInitData(), 0);
  PHINode *DataPhi =
      State.Builder.CreatePHI(InitData->getType(), 2, "csa.data.phi");
  BasicBlock *PreheaderBB = State.CFG.getPreheaderBBFor(this);
  DataPhi->addIncoming(InitData, PreheaderBB);

  // Use the same DataPhi for all Parts
  for (unsigned Part = 0; Part < State.UF; ++Part)
    State.set(this, DataPhi, Part);
}

InstructionCost VPCSAHeaderPHIRecipe::overhead(ElementCount VF,
                                               VPCostContext &Ctx) const {
  if (VF.isScalar())
    return 0;

  InstructionCost C = 0;
  auto *VectorTy =
      VectorType::get(getUnderlyingValue()->getType(), VF);
  auto *MaskTy =
      VectorType::get(IntegerType::getInt1Ty(VectorTy->getContext()), VF);
  auto *Int32VecTy =
      VectorType::get(IntegerType::getInt32Ty(VectorTy->getContext()), VF);

  constexpr TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  // TODO: When we move to VPlan based CM, the costs of recipes in PH and exit
  // should be added as overhead to the vector loop automatically. When that 
  // happens, the generation of overhead for those recipes in this function can
  // be removed.

  // All True/False Mask
  C += Ctx.TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, MaskTy);
  C += Ctx.TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, MaskTy);

  // CSAInitMask
  C += Ctx.TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VectorTy);
  // CSAInitData
  C += Ctx.TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VectorTy);

  // CSAExtractScalar
  // StepVector
  ArrayRef<Value *> Args;
  IntrinsicCostAttributes CostAttrs(Intrinsic::experimental_stepvector,
                                    Int32VecTy, Args);
  C += Ctx.TTI->getIntrinsicInstrCost(CostAttrs, CostKind);
  // NegOneSplat
  C += Ctx.TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, Int32VecTy);
  // LastIdx
  C += Ctx.TTI->getMinMaxReductionCost(Intrinsic::smax, Int32VecTy,
                                       FastMathFlags(), CostKind);
  // ExtractFromVec
  C += Ctx.TTI->getArithmeticInstrCost(Instruction::ExtractElement, VectorTy,
                                       CostKind);
  // LastIdxGeZero
  C += Ctx.TTI->getArithmeticInstrCost(Instruction::ICmp, Int32VecTy, CostKind);
  // ChooseFromVecOrInit
  C += Ctx.TTI->getArithmeticInstrCost(Instruction::Select,
                                       VectorTy->getScalarType(), CostKind);
  return C * Ctx.TTI->getCSAOverheadFactor();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPCSADataUpdateRecipe::print(raw_ostream &O, const Twine &Indent,
                                 VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = csa-data-update ";
  printOperands(O, SlotTracker);
}
#endif

void VPCSADataUpdateRecipe::execute(VPTransformState &State) {
  if (!State.EnableRISCVCSA) {
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Value *AnyActive = State.get(getVPAnyActive(), Part);
      Value *DataUpdate = getVPDataPhi() == getVPTrue()
                              ? State.get(getVPFalse(), Part)
                              : State.get(getVPTrue(), Part);
      PHINode *DataPhi = cast<PHINode>(State.get(getVPDataPhi(), Part));
      // If not the first Part, use the mask from the previous unrolled Part
      Value *OldData = Part == 0 ? DataPhi : State.get(this, Part - 1);
      Value *DataSel = State.Builder.CreateSelect(AnyActive, DataUpdate,
                                                  OldData, "csa.data.sel");

      if (Part == State.UF - 1)
        DataPhi->addIncoming(DataSel, State.CFG.PrevBB);
      State.set(this, DataSel, Part);
    }
    return;
  }

  for (unsigned Part = 0; Part < State.UF; ++Part) {
    // We can't use the NewMask to update the data. We must use the condition
    // vector since it is possible that condition vector is all false but
    // lanes from a prior iteration on 0..RVL are active in NewMask.
    Value *Cond = State.get(getVPCondToUse(), Part);
    Value *DataPhi = State.get(getVPDataPhi(), Part);
    Value *UndistData = getVPDataPhi() == getVPTrue()
                            ? State.get(getVPFalse(), Part)
                            : State.get(getVPTrue(), Part);
    Value *RVL =
        State.Plan->useVLAVectorizer()
            ? State.get(State.RVL, Part)
            : getRuntimeVF(State.Builder, State.Builder.getInt32Ty(), State.VF);
    Value *RVL32 =
        State.Builder.CreateZExtOrTrunc(RVL, State.Builder.getInt32Ty());

    Value *OldData = Part == 0 ? DataPhi : State.get(this, Part - 1);
    Value *NewData = State.Builder.CreateIntrinsic(
        DataPhi->getType(), Intrinsic::vp_merge,
        {Cond, UndistData, OldData, RVL32});
    if (Part == State.UF - 1)
      cast<PHINode>(DataPhi)->addIncoming(NewData, State.CFG.PrevBB);
    State.set(this, NewData, Part);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPCSAExtractScalarRecipe::print(raw_ostream &O, const Twine &Indent,
                                 VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = CSA-EXTRACT-SCALAR ";
  printOperands(O, SlotTracker);
}
#endif

void VPCSAExtractScalarRecipe::execute(VPTransformState &State) {
  IRBuilder<>::InsertPointGuard Guard(State.Builder);
  State.Builder.SetInsertPoint(State.CFG.ExitBB->getFirstNonPHI());

  unsigned LastPart = State.UF - 1;
  Value *MaskSel = State.get(getVPMaskSel(), LastPart);
  Value *DataSel = State.get(getVPDataSel(), LastPart);
  Value *InitRVL =
      State.Plan->useVLAVectorizer()
          ? State.get(State.Plan->getInitRVL(), 0)
          : getRuntimeVF(State.Builder, State.Builder.getInt32Ty(), State.VF);
  Value *InitRVL32 =
      State.Builder.CreateZExtOrTrunc(InitRVL, State.Builder.getInt32Ty());

  Value *VLToUse =
      State.EnableRISCVCSA ? InitRVL32 : State.get(getVPCSAVLSel(), LastPart);
  Value *InitScalar = getVPInitScalar()->getLiveInIRValue();

  Value *IndexVec = State.Builder.CreateStepVector(
      VectorType::get(State.Builder.getInt32Ty(), State.VF), "csa.step");
  Value *NegOne = ConstantInt::get(IndexVec->getType()->getScalarType(), -1);
  Value *LastIdx = State.Builder.CreateIntrinsic(
      NegOne->getType(), Intrinsic::vp_reduce_smax,
      {NegOne, IndexVec, MaskSel, VLToUse});
  Value *ExtractFromVec =
      State.Builder.CreateExtractElement(DataSel, LastIdx, "csa.extract");
  Value *Zero = ConstantInt::get(LastIdx->getType(), 0);
  Value *LastIdxGEZero = State.Builder.CreateICmpSGE(LastIdx, Zero);
  Value *ChooseFromVecOrInit =
      State.Builder.CreateSelect(LastIdxGEZero, ExtractFromVec, InitScalar);
  State.set(this, ChooseFromVecOrInit, 0);
}

InstructionCost VPMonotonicHeaderPHIRecipe::overhead(ElementCount VF,
                                                     VPCostContext &Ctx) const {
  return 0;
}
#endif // SIFIVE_CUSTOMIZATION

void VPBranchOnMaskRecipe::execute(VPTransformState &State) {
  assert(State.Instance && "Branch on Mask works only on single instance.");

  unsigned Part = State.Instance->Part;
  unsigned Lane = State.Instance->Lane.getKnownLane();

  Value *ConditionBit = nullptr;
  VPValue *BlockInMask = getMask();
  if (BlockInMask) {
    ConditionBit = State.get(BlockInMask, Part);
    if (ConditionBit->getType()->isVectorTy())
      ConditionBit = State.Builder.CreateExtractElement(
          ConditionBit, State.Builder.getInt32(Lane));
  } else // Block in mask is all-one.
    ConditionBit = State.Builder.getTrue();

  // Replace the temporary unreachable terminator with a new conditional branch,
  // whose two destinations will be set later when they are created.
  auto *CurrentTerminator = State.CFG.PrevBB->getTerminator();
  assert(isa<UnreachableInst>(CurrentTerminator) &&
         "Expected to replace unreachable terminator with conditional branch.");
  auto *CondBr = BranchInst::Create(State.CFG.PrevBB, nullptr, ConditionBit);
  CondBr->setSuccessor(0, nullptr);
  ReplaceInstWithInst(CurrentTerminator, CondBr);
}

void VPPredInstPHIRecipe::execute(VPTransformState &State) {
  assert(State.Instance && "Predicated instruction PHI works per instance.");
  Instruction *ScalarPredInst =
      cast<Instruction>(State.get(getOperand(0), *State.Instance));
  BasicBlock *PredicatedBB = ScalarPredInst->getParent();
  BasicBlock *PredicatingBB = PredicatedBB->getSinglePredecessor();
  assert(PredicatingBB && "Predicated block has no single predecessor.");
  assert(isa<VPReplicateRecipe>(getOperand(0)) &&
         "operand must be VPReplicateRecipe");

  // By current pack/unpack logic we need to generate only a single phi node: if
  // a vector value for the predicated instruction exists at this point it means
  // the instruction has vector users only, and a phi for the vector value is
  // needed. In this case the recipe of the predicated instruction is marked to
  // also do that packing, thereby "hoisting" the insert-element sequence.
  // Otherwise, a phi node for the scalar value is needed.
  unsigned Part = State.Instance->Part;
  if (State.hasVectorValue(getOperand(0), Part)) {
    Value *VectorValue = State.get(getOperand(0), Part);
    InsertElementInst *IEI = cast<InsertElementInst>(VectorValue);
    PHINode *VPhi = State.Builder.CreatePHI(IEI->getType(), 2);
    VPhi->addIncoming(IEI->getOperand(0), PredicatingBB); // Unmodified vector.
    VPhi->addIncoming(IEI, PredicatedBB); // New vector with inserted element.
    if (State.hasVectorValue(this, Part))
      State.reset(this, VPhi, Part);
    else
      State.set(this, VPhi, Part);
    // NOTE: Currently we need to update the value of the operand, so the next
    // predicated iteration inserts its generated value in the correct vector.
    State.reset(getOperand(0), VPhi, Part);
  } else {
    Type *PredInstType = getOperand(0)->getUnderlyingValue()->getType();
    PHINode *Phi = State.Builder.CreatePHI(PredInstType, 2);
    Phi->addIncoming(PoisonValue::get(ScalarPredInst->getType()),
                     PredicatingBB);
    Phi->addIncoming(ScalarPredInst, PredicatedBB);
    if (State.hasScalarValue(this, *State.Instance))
      State.reset(this, Phi, *State.Instance);
    else
      State.set(this, Phi, *State.Instance);
    // NOTE: Currently we need to update the value of the operand, so the next
    // predicated iteration inserts its generated value in the correct vector.
    State.reset(getOperand(0), Phi, *State.Instance);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPPredInstPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << Indent << "PHI-PREDICATED-INSTRUCTION ";
  printAsOperand(O, SlotTracker);
  O << " = ";
  printOperands(O, SlotTracker);
}

void VPWidenMemoryInstructionRecipe::print(raw_ostream &O, const Twine &Indent,
                                           VPSlotTracker &SlotTracker) const {
#if SIFIVE_CUSTOMIZATION
  if (this->Speculative)
    O << Indent << "WIDEN-SPECULATIVE-MEMORY-INSTRUCTION ";
  else
    O << Indent << "WIDEN ";
#else
  O << Indent << "WIDEN ";
#endif

  if (!isStore()) {
#if SIFIVE_CUSTOMIZATION
    if (this->Speculative) {
      interleaveComma(definedValues(), O, [&O, &SlotTracker](VPValue *Op) {
        Op->printAsOperand(O, SlotTracker);
      });
    } else
#endif // SIFIVE_CUSTOMIZATION
    getVPSingleValue()->printAsOperand(O, SlotTracker);
    O << " = ";
  }
#if SIFIVE_CUSTOMIZATION
  O << Instruction::getOpcodeName(getIngredient().getOpcode()) << " ";
#else
  O << Instruction::getOpcodeName(Ingredient.getOpcode()) << " ";
#endif // SIFIVE_CUSTOMIZATION

  printOperands(O, SlotTracker);
}
#endif

void VPCanonicalIVPHIRecipe::execute(VPTransformState &State) {
  Value *Start = getStartValue()->getLiveInIRValue();
  PHINode *EntryPart = PHINode::Create(Start->getType(), 2, "index");
  EntryPart->insertBefore(State.CFG.PrevBB->getFirstInsertionPt());

  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  EntryPart->addIncoming(Start, VectorPH);
  EntryPart->setDebugLoc(getDebugLoc());
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part)
    State.set(this, EntryPart, Part);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPCanonicalIVPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                   VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = CANONICAL-INDUCTION ";
  printOperands(O, SlotTracker);
}
#endif

bool VPCanonicalIVPHIRecipe::isCanonical(
    InductionDescriptor::InductionKind Kind, VPValue *Start,
    VPValue *Step) const {
  // Must be an integer induction.
  if (Kind != InductionDescriptor::IK_IntInduction)
    return false;
  // Start must match the start value of this canonical induction.
  if (Start != getStartValue())
    return false;

  // If the step is defined by a recipe, it is not a ConstantInt.
  if (Step->getDefiningRecipe())
    return false;

  ConstantInt *StepC = dyn_cast<ConstantInt>(Step->getLiveInIRValue());
  return StepC && StepC->isOne();
}

bool VPWidenPointerInductionRecipe::onlyScalarsGenerated(bool IsScalable) {
  return IsScalarAfterVectorization &&
         (!IsScalable || vputils::onlyFirstLaneUsed(this));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenPointerInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = WIDEN-POINTER-INDUCTION ";
  getStartValue()->printAsOperand(O, SlotTracker);
  O << ", " << *IndDesc.getStep();
}
#endif

void VPExpandSCEVRecipe::execute(VPTransformState &State) {
  assert(!State.Instance && "cannot be used in per-lane");
  const DataLayout &DL = State.CFG.PrevBB->getModule()->getDataLayout();
  SCEVExpander Exp(SE, DL, "induction");

  Value *Res = Exp.expandCodeFor(Expr, Expr->getType(),
                                 &*State.Builder.GetInsertPoint());
  assert(!State.ExpandedSCEVs.contains(Expr) &&
         "Same SCEV expanded multiple times");
  State.ExpandedSCEVs[Expr] = Res;
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part)
    State.set(this, Res, {Part, 0});
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPExpandSCEVRecipe::print(raw_ostream &O, const Twine &Indent,
                               VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  getVPSingleValue()->printAsOperand(O, SlotTracker);
  O << " = EXPAND SCEV " << *Expr;
}
#endif

void VPWidenCanonicalIVRecipe::execute(VPTransformState &State) {
  Value *CanonicalIV = State.get(getOperand(0), 0);
  Type *STy = CanonicalIV->getType();
  IRBuilder<> Builder(State.CFG.PrevBB->getTerminator());
  ElementCount VF = State.VF;
  Value *VStart = VF.isScalar()
                      ? CanonicalIV
                      : Builder.CreateVectorSplat(VF, CanonicalIV, "broadcast");
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part) {
    Value *VStep = createStepForVF(Builder, STy, VF, Part);
    if (VF.isVector()) {
      VStep = Builder.CreateVectorSplat(VF, VStep);
      VStep =
          Builder.CreateAdd(VStep, Builder.CreateStepVector(VStep->getType()));
    }
    Value *CanonicalVectorIV = Builder.CreateAdd(VStart, VStep, "vec.iv");
    State.set(this, CanonicalVectorIV, Part);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenCanonicalIVRecipe::print(raw_ostream &O, const Twine &Indent,
                                     VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = WIDEN-CANONICAL-INDUCTION ";
  printOperands(O, SlotTracker);
}
#endif

void VPFirstOrderRecurrencePHIRecipe::execute(VPTransformState &State) {
  auto &Builder = State.Builder;
  // Create a vector from the initial value.
  auto *VectorInit = getStartValue()->getLiveInIRValue();

  Type *VecTy = State.VF.isScalar()
                    ? VectorInit->getType()
                    : VectorType::get(VectorInit->getType(), State.VF);

  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  if (State.VF.isVector()) {
    auto *IdxTy = Builder.getInt32Ty();
    auto *One = ConstantInt::get(IdxTy, 1);
    IRBuilder<>::InsertPointGuard Guard(Builder);
    Builder.SetInsertPoint(VectorPH->getTerminator());

#if SIFIVE_CUSTOMIZATION
    Value *RuntimeVF = nullptr;
    if (State.Plan->useVLAVectorizer()) {
      assert(State.Plan->getInitRVL() &&
             "InitRVL must be constructed to correctly handle "
             "VPFirstOrderRecurrencePHIRecipe");
      Value *InitRVL = State.get(State.Plan->getInitRVL(), 0);
      RuntimeVF = State.Builder.CreateTrunc(InitRVL, IdxTy);
    } else {
      RuntimeVF = getRuntimeVF(Builder, IdxTy, State.VF);
    }
#endif // SIFIVE_CUSTOMIZATION
    auto *LastIdx = Builder.CreateSub(RuntimeVF, One);
    VectorInit = Builder.CreateInsertElement(
        PoisonValue::get(VecTy), VectorInit, LastIdx, "vector.recur.init");
  }

  // Create a phi node for the new recurrence.
  PHINode *EntryPart = PHINode::Create(VecTy, 2, "vector.recur");
  EntryPart->insertBefore(State.CFG.PrevBB->getFirstInsertionPt());
  EntryPart->addIncoming(VectorInit, VectorPH);
  State.set(this, EntryPart, 0);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPFirstOrderRecurrencePHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                            VPSlotTracker &SlotTracker) const {
  O << Indent << "FIRST-ORDER-RECURRENCE-PHI ";
  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

void VPReductionPHIRecipe::execute(VPTransformState &State) {
  auto &Builder = State.Builder;

  // Reductions do not have to start at zero. They can start with
  // any loop invariant values.
  VPValue *StartVPV = getStartValue();
  Value *StartV = StartVPV->getLiveInIRValue();

  // In order to support recurrences we need to be able to vectorize Phi nodes.
  // Phi nodes have cycles, so we need to vectorize them in two stages. This is
  // stage #1: We create a new vector PHI node with no incoming edges. We'll use
  // this value when we vectorize all of the instructions that use the PHI.
  bool ScalarPHI = State.VF.isScalar() || IsInLoop;
  Type *VecTy = ScalarPHI ? StartV->getType()
                          : VectorType::get(StartV->getType(), State.VF);

  BasicBlock *HeaderBB = State.CFG.PrevBB;
  assert(State.CurrentVectorLoop->getHeader() == HeaderBB &&
         "recipe must be in the vector loop header");
  unsigned LastPartForNewPhi = isOrdered() ? 1 : State.UF;
  for (unsigned Part = 0; Part < LastPartForNewPhi; ++Part) {
    Instruction *EntryPart = PHINode::Create(VecTy, 2, "vec.phi");
    EntryPart->insertBefore(HeaderBB->getFirstInsertionPt());
    State.set(this, EntryPart, Part);
  }

  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);

#if SIFIVE_CUSTOMIZATION
  bool PostSV = postFixStartValue();
#endif // SIFIVE_CUSTOMIZATION

  Value *Iden = nullptr;
  RecurKind RK = RdxDesc.getRecurrenceKind();
  if (RecurrenceDescriptor::isMinMaxRecurrenceKind(RK) ||
      RecurrenceDescriptor::isAnyOfRecurrenceKind(RK)) {
    // MinMax and AnyOf reductions have the start value as their identity.
    if (ScalarPHI) {
      Iden = StartV;
    } else {
      IRBuilderBase::InsertPointGuard IPBuilder(Builder);
      Builder.SetInsertPoint(VectorPH->getTerminator());
      StartV = Iden =
          Builder.CreateVectorSplat(State.VF, StartV, "minmax.ident");
    }
#if SIFIVE_CUSTOMIZATION
  } else if (RecurrenceDescriptor::isFindLastIVRecurrenceKind(RK)) {
    // [I|F]FindLastIV will use a sentinel value as the identity to initialize
    // the reduction phi. In the middle block, createSentinelValueHandling will
    // generate checks to verify if the reduction result is the sentinel value.
    // If the result is the sentinel value, it will be corrected back to the
    // start value.
    // TODO: The sentinel value is not always necessary. When the start value is
    // a constant, and smaller than the start value of the induction variable,
    // the start value can be directly used to initialize the reduction phi.
    StartV = Iden = RdxDesc.getRecurrenceIdentity(RK, VecTy->getScalarType(),
                                                  RdxDesc.getFastMathFlags());
    if (!ScalarPHI) {
      IRBuilderBase::InsertPointGuard IPBuilder(Builder);
      Builder.SetInsertPoint(VectorPH->getTerminator());
      StartV = Iden = Builder.CreateVectorSplat(State.VF, Iden);
    }
#endif // SIFIVE_CUSTOMIZATION
  } else {
    Iden = RdxDesc.getRecurrenceIdentity(RK, VecTy->getScalarType(),
                                         RdxDesc.getFastMathFlags());

    if (!ScalarPHI) {
      Iden = Builder.CreateVectorSplat(State.VF, Iden);
#if SIFIVE_CUSTOMIZATION
      if (PostSV) {
        StartV = Iden;
      } else {
#endif // SIFIVE_CUSTOMIZATION
        IRBuilderBase::InsertPointGuard IPBuilder(Builder);
        Builder.SetInsertPoint(VectorPH->getTerminator());
        Constant *Zero = Builder.getInt32(0);
        StartV = Builder.CreateInsertElement(Iden, StartV, Zero);
#if SIFIVE_CUSTOMIZATION
      }
#endif // SIFIVE_CUSTOMIZATION
    }
  }

  for (unsigned Part = 0; Part < LastPartForNewPhi; ++Part) {
    Value *EntryPart = State.get(this, Part);
    // Make sure to add the reduction start value only to the
    // first unroll part.
    Value *StartVal = (Part == 0) ? StartV : Iden;
    cast<PHINode>(EntryPart)->addIncoming(StartVal, VectorPH);
  }
}

#if SIFIVE_CUSTOMIZATION
InstructionCost VPReductionPHIRecipe::overhead(ElementCount VF,
                                               VPCostContext &Ctx) const {
  // There is no overhead when VF is scalar or the reduction is in-loop.
  if (VF.isScalar() || IsInLoop)
    return 0;

  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  RecurKind RdxKind = RdxDesc.getRecurrenceKind();
  Type *ElementTy = RdxDesc.getRecurrenceType();
  auto *VectorTy = cast<VectorType>(ToVectorTy(ElementTy, VF));
  // TODO: Add broadcast cost for all recurrence kinds
  switch (RdxKind) {
  case RecurKind::Add:
  case RecurKind::Mul:
  case RecurKind::Or:
  case RecurKind::And:
  case RecurKind::Xor:
  case RecurKind::FAdd:
  case RecurKind::FMul:
  case RecurKind::FMulAdd:
    return Ctx.TTI->getArithmeticReductionCost(
        RdxDesc.getOpcode(), VectorTy, RdxDesc.getFastMathFlags(), CostKind);
  case RecurKind::SMin:
  case RecurKind::SMax:
  case RecurKind::UMin:
  case RecurKind::UMax:
  case RecurKind::FMin:
  case RecurKind::FMax:
  case RecurKind::FMinimum:
  case RecurKind::FMaximum: {
    Intrinsic::ID Id = getMinMaxReductionIntrinsicOp(RdxKind);
    return Ctx.TTI->getMinMaxReductionCost(
        Id, VectorTy, RdxDesc.getFastMathFlags(), CostKind);
  }
  case RecurKind::IAnyOf:
  case RecurKind::FAnyOf: {
    // The cost references the instructions created in
    // llvm::createAnyOfTargetReduction
    auto *VecCondTy = cast<VectorType>(CmpInst::makeCmpResultType(VectorTy));
    InstructionCost O =
        Ctx.TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VectorTy);
    O += Ctx.TTI->getCmpSelInstrCost(Instruction::ICmp, VectorTy, VecCondTy,
                                     CmpInst::ICMP_NE, CostKind);
    O += Ctx.TTI->getArithmeticReductionCost(
        Instruction::Or, VecCondTy, RdxDesc.getFastMathFlags(), CostKind);
    O += Ctx.TTI->getCmpSelInstrCost(Instruction::Select, ElementTy,
                                     CmpInst::makeCmpResultType(ElementTy),
                                     CmpInst::BAD_ICMP_PREDICATE, CostKind);
    return O;
  }
  case RecurKind::IFindLastIV:
  case RecurKind::FFindLastIV: {
    // Emit reduce.smax to get the last induction value
    InstructionCost O = Ctx.TTI->getMinMaxReductionCost(
        Intrinsic::smax, VectorTy, FastMathFlags(), CostKind);
    // Sentinel value handling
    O += Ctx.TTI->getCmpSelInstrCost(Instruction::ICmp, ElementTy, nullptr,
                                     CmpInst::ICMP_NE, CostKind);
    O += Ctx.TTI->getCmpSelInstrCost(Instruction::Select, ElementTy,
                                     CmpInst::makeCmpResultType(ElementTy),
                                     CmpInst::BAD_ICMP_PREDICATE, CostKind);
    return O;
  }
  case RecurKind::None:
    llvm_unreachable("Unexpected reduction kind.");
  }
  return InstructionCost::getInvalid();
}
#endif // SIFIVE_CUSTOMIZATION

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPReductionPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                 VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-REDUCTION-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

void VPWidenPHIRecipe::execute(VPTransformState &State) {
  assert(EnableVPlanNativePath &&
         "Non-native vplans are not expected to have VPWidenPHIRecipes.");

  Value *Op0 = State.get(getOperand(0), 0);
  Type *VecTy = Op0->getType();
  Value *VecPhi = State.Builder.CreatePHI(VecTy, 2, "vec.phi");
  State.set(this, VecPhi, 0);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                             VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-PHI ";

  auto *OriginalPhi = cast<PHINode>(getUnderlyingValue());
  // Unless all incoming values are modeled in VPlan  print the original PHI
  // directly.
  // TODO: Remove once all VPWidenPHIRecipe instances keep all relevant incoming
  // values as VPValues.
  if (getNumOperands() != OriginalPhi->getNumOperands()) {
    O << VPlanIngredient(OriginalPhi);
    return;
  }

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

// TODO: It would be good to use the existing VPWidenPHIRecipe instead and
// remove VPActiveLaneMaskPHIRecipe.
void VPActiveLaneMaskPHIRecipe::execute(VPTransformState &State) {
  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part) {
    Value *StartMask = State.get(getOperand(0), Part);
    PHINode *EntryPart =
        State.Builder.CreatePHI(StartMask->getType(), 2, "active.lane.mask");
    EntryPart->addIncoming(StartMask, VectorPH);
    EntryPart->setDebugLoc(getDebugLoc());
    State.set(this, EntryPart, Part);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPActiveLaneMaskPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                      VPSlotTracker &SlotTracker) const {
  O << Indent << "ACTIVE-LANE-MASK-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

#if SIFIVE_CUSTOMIZATION
void VPEVLBasedIVPHIRecipe::execute(VPTransformState &State) {
  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  assert(State.UF == 1 && "Expected unroll factor 1 for VP vectorization.");
  // FIXME: Initial VL must be explicitly represented in VPlan, but as a
  // temporary solution emit initial computation of VL here
  Value *Start = nullptr;
  if (getOperand(0) == State.Plan->getInitRVL()) {
    IRBuilder<>::InsertPointGuard Guard(State.Builder);
    BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
    State.Builder.SetInsertPoint(VectorPH->getTerminator());
    Start = GetSetVL(State, State.get(&State.Plan->getVectorTripCount(), 0));
    State.set(State.Plan->getInitRVL(), Start, 0);
  } else {
    Start = State.get(getOperand(0), VPIteration(0, 0));
  }
  PHINode *EntryPart =
      State.Builder.CreatePHI(Start->getType(), 2, "evl.based.iv");
  EntryPart->addIncoming(Start, VectorPH);
  EntryPart->setDebugLoc(getDebugLoc());
  State.set(this, EntryPart, 0);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPEVLBasedIVPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent << "EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif
#endif // SIFIVE_CUSTOMIZATION
