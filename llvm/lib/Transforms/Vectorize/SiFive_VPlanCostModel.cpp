//===- SiFive_VPlanCostModel.h - Vectorizer Cost Model --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// VPlan-based cost model
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"

#include "SiFive_VPlanCostModel.h"
#include "VPlan.h"
#include "VPlanCFG.h"
#include "VPlanValue.h"

using namespace llvm;

#define DEBUG_TYPE "vplan-cost-model"

#if SIFIVE_CUSTOMIZATION
static cl::opt<bool> SiFiveEstimateRegesterPressure(
    "sifive-vplan-cost-model-estimate-regpressure", cl::init(true), cl::Hidden,
    cl::desc(
        "Control whether cost model should estimate register pressure or not"));
#endif // SIFIVE_CUSTOMIZATION

static ElementCount getElementCount(const std::pair<unsigned, bool> LMUL,
                                    const unsigned SEW) {
  unsigned KnownMinValue = LMUL.second
                               ? RISCV::RVVBitsPerBlock / (LMUL.first * SEW)
                               : RISCV::RVVBitsPerBlock * LMUL.first / SEW;
  return ElementCount::get(KnownMinValue, true);
}

static ElementCount getElementCount(const RVVPair &RVVP) {
  return getElementCount(RVVP.getLMUL(), RVVP.getSEW());
}

static Type *getRecipeType(const VPRecipeBase *VPR) {
  if (!VPR->hasUnderlyingInstr())
    return nullptr;
  return VPR->getUnderlyingInstr()->getType();
}

namespace llvm {
ElementCount RVVPair::getElementCount(const int LMULExp, const unsigned SEW) {
  return ::getElementCount(getIntFromLMULKind((LMULKind)LMULExp), SEW);
}

RVVPair RVVPair::getWithType(Type *Ty, const RVVPair &RVVP) {
  if (Ty->isVoidTy() || Ty->isAggregateType() || RVVP.getType() == Ty)
    return RVVP;
  return get(Ty, ::getElementCount(RVVP), RVVP.DL);
}

// FIXME: Unify with `RvvHintAttr` in Clang
RVVPair RVVPair::getWithExponent(Type *Ty, const int LMULExp,
                                 const DataLayout &DL) {
  if (LMULExp < -3 || LMULExp > 3)
    return RVVPair(Ty, LMULKind::Unsupported, DL);
  return RVVPair(Ty, (RVVPair::LMULKind)LMULExp, DL);
}

// FIXME: Unify with `RvvHintAttr` in Clang
std::pair<unsigned, bool> RVVPair::getIntFromLMULKind(LMULKind Kind) {
  int LMULExp = (int)Kind;
  assert(LMULExp >= -3 && LMULExp <= 3 &&
         "LMUL exponent is not a valid or not supported.");
  return {1u << std::abs(LMULExp), LMULExp < 0};
}

// FIXME: Unify with `RvvHintAttr` in Clang
StringRef RVVPair::getStringFromLMULKind(LMULKind Kind) const {
  switch (Kind) {
  case LMULKind::Mf8:
    return "mf8";
  case LMULKind::Mf4:
    return "mf4";
  case LMULKind::Mf2:
    return "mf2";
  case LMULKind::M1:
    return "m1";
  case LMULKind::M2:
    return "m2";
  case LMULKind::M4:
    return "m4";
  case LMULKind::M8:
    return "m8";
  case LMULKind::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("Unsupported LMUL Kind");
}

Type *VPlanCostModel::getVectorType(Type *Ty, const RVVPair &RVVP) {
  assert(!isa_and_nonnull<VectorType>(Ty) &&
         "Cannot convert non-scalar type to VectorType for a given (LMUL, SEW) "
         "pair");
  if (!RVVP)
    return nullptr;

  if (Ty->isVoidTy() || Ty->isMetadataTy())
    return Ty;
  return ScalableVectorType::get(Ty, getElementCount(RVVP).getKnownMinValue());
}

InstructionCost VPlanCostModel::getCost(const RVVPair &RVL) {
  if (!RVL) {
    LLVM_DEBUG(dbgs() << "VPlanCM: unsupported Runtime VL = " << RVL << '\n');
    return InstructionCost::getInvalid();
  }
  InstructionCost VectorIterCost = 0;
  for (const VPBlockBase *Block : vp_depth_first_deep(Plan.getEntry()))
    VectorIterCost += getCost(Block, RVL);

  if (!VectorIterCost.isValid())
    return VectorIterCost;

  // FIXME: Move estimate of entire loop from `isMoreProfitable` here such that
  // the function will return estimation of the entire VPlan.
  LLVM_DEBUG(dbgs() << "VPlanCM: " << RVL
                    << " vec_iter_cost = " << VectorIterCost << '\n');
  return VectorIterCost;
}

InstructionCost VPlanCostModel::getCost(const VPBlockBase *Block,
                                        const RVVPair &RVL) {
  return TypeSwitch<const VPBlockBase *, InstructionCost>(Block)
      .Case<VPBasicBlock>([&](const VPBasicBlock *BBlock) {
        InstructionCost Cost = 0;
        for (const VPRecipeBase &Recipe : *BBlock)
          Cost += getCost(&Recipe, RVL);
        return Cost;
      })
      .Default([&](const VPBlockBase *BBlock) -> InstructionCost { return 0; });
}

InstructionCost VPlanCostModel::getCost(const VPRecipeBase *Recipe,
                                        const RVVPair &RVL) {
  InstructionCost Cost =
      TypeSwitch<const VPRecipeBase *, InstructionCost>(Recipe)
          .Case<VPWidenMemoryInstructionRecipe>(
              [&](const VPWidenMemoryInstructionRecipe *VPWMIR) {
                return getMemoryOpCost(VPWMIR, RVL);
              })
          .Case<VPInterleaveRecipe>([&](const VPInterleaveRecipe *VPI) {
            return getInterleavedMemoryOpCost(VPI, RVL);
          })
          .Case<VPWidenIntOrFpInductionRecipe>(
              [&](const VPWidenIntOrFpInductionRecipe *IVR) -> InstructionCost {
                Value *Start = IVR->getStartValue()->getLiveInIRValue();
                const TruncInst *Trunc = IVR->getTruncInst();
                Type *VectorTy = Trunc ? getVectorType(Trunc->getType(), RVL)
                                       : getVectorType(Start->getType(), RVL);
                Instruction::BinaryOps AddOp;
                const InductionDescriptor &ID = IVR->getInductionDescriptor();
                if (Start->getType()->isIntegerTy())
                  AddOp = Instruction::Add;
                else
                  AddOp = ID.getInductionOpcode();
                return TTI.getArithmeticInstrCost(AddOp, VectorTy, CostKind);
              })
          .Case<VPCanonicalIVPHIRecipe, VPScalarIVStepsRecipe,
                VPReductionPHIRecipe, VPWidenPointerInductionRecipe>(
              [&](const VPRecipeBase *IVR) -> InstructionCost { return 1; })
          .Case<VPReductionRecipe>([&](const VPReductionRecipe *VPR) {
            return getReductionCost(VPR, RVL);
          })
          .Case<VPReplicateRecipe>([&](const VPReplicateRecipe *VPR) {
            return getReplicateOpCost(VPR, RVL);
          })
          .Case<VPInstruction>(
              [&](const VPInstruction *VPI) -> InstructionCost {
                return getInstructionCost(VPI, RVL);
              })
          .Default([&](const VPRecipeBase *R) -> InstructionCost {
            // FIXME: The code here should not access underlying instruction and
            // should completely rely on information that can be taken directly
            // from VPRecipe or VPInstruction.
            if (!Recipe->hasUnderlyingInstr())
              return 0;
            const Instruction *I = Recipe->getUnderlyingInstr();
            ElementCount VF = getElementCount(RVL);
            unsigned Opcode = I->getOpcode();
            switch (Opcode) {
            case Instruction::GetElementPtr:
              // We mark this instruction as zero-cost because the cost of GEPs
              // in vectorized code depends on whether the corresponding memory
              // instruction is scalarized or not. Therefore, we handle GEPs
              // with the memory instruction cost.
              return 0;
            case Instruction::Br:
              return 0;
            case Instruction::PHI:
              return 0;
            case Instruction::UDiv:
            case Instruction::SDiv:
            case Instruction::URem:
            case Instruction::SRem: {
              Type *VectorTy = getVectorType(I->getType(), RVL);
              return TTI.getArithmeticInstrCost(Opcode, VectorTy, CostKind);
            }
            case Instruction::Add:
            case Instruction::FAdd:
            case Instruction::Sub:
            case Instruction::FSub:
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
              Type *VectorTy = getVectorType(I->getType(), RVL);
              // Certain instructions can be cheaper to vectorize if they have a
              // constant second vector operand. One example of this are shifts
              // on x86.
              Value *Op2 = I->getOperand(1);
              auto Op2Info = TTI.getOperandInfo(Op2);
              if (Op2Info.Kind == TargetTransformInfo::OK_AnyValue &&
                  Legal.isUniform(Op2, VF))
                Op2Info.Kind = TargetTransformInfo::OK_UniformValue;

              SmallVector<const Value *, 4> Operands(I->operand_values());
              const unsigned RegID =
                  TTI.getRegisterClassForType(true /*vector*/, VectorTy);
              const unsigned NumUsedRegs = TTI.getRegUsageForType(VectorTy);
              addRegisterUsage(Recipe->getVPSingleValue(), RegID, NumUsedRegs);
              InstructionCost Cost = getRegisterPressureCost(RegID, VectorTy);
              return Cost + TTI.getArithmeticInstrCost(
                                I->getOpcode(), VectorTy, CostKind,
                                {TargetTransformInfo::OK_AnyValue,
                                 TargetTransformInfo::OP_None},
                                Op2Info, Operands, I);
            }
            case Instruction::FNeg: {
              Type *VectorTy = getVectorType(I->getType(), RVL);
              const unsigned RegID =
                  TTI.getRegisterClassForType(true /*vector*/, VectorTy);
              const unsigned NumUsedRegs = TTI.getRegUsageForType(VectorTy);
              addRegisterUsage(Recipe->getVPSingleValue(), RegID, NumUsedRegs);
              InstructionCost Cost = getRegisterPressureCost(RegID, VectorTy);
              return Cost + TTI.getArithmeticInstrCost(
                                I->getOpcode(), VectorTy, CostKind,
                                {TargetTransformInfo::OK_AnyValue,
                                 TargetTransformInfo::OP_None},
                                {TargetTransformInfo::OK_AnyValue,
                                 TargetTransformInfo::OP_None},
                                I->getOperand(0), I);
            }
            case Instruction::Select: {
              Type *VectorTy = getVectorType(I->getType(), RVL);
              const SelectInst *SI = cast<SelectInst>(I);
              Type *CondTy = SI->getCondition()->getType();
              CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
              if (auto *Cmp = dyn_cast<CmpInst>(SI->getCondition()))
                Pred = Cmp->getPredicate();
              const unsigned RegID =
                  TTI.getRegisterClassForType(true /*vector*/, VectorTy);
              const unsigned NumUsedRegs = TTI.getRegUsageForType(VectorTy);
              addRegisterUsage(Recipe->getVPSingleValue(), RegID, NumUsedRegs);
              InstructionCost Cost = getRegisterPressureCost(RegID, VectorTy);
              return Cost + TTI.getCmpSelInstrCost(I->getOpcode(), VectorTy,
                                                   CondTy, Pred, CostKind, I);
            }
            case Instruction::ICmp:
            case Instruction::FCmp: {
              Type *VectorTy = getVectorType(I->getType(), RVL);
              const unsigned RegID =
                  TTI.getRegisterClassForType(true /*vector*/, VectorTy);
              const unsigned NumUsedRegs = TTI.getRegUsageForType(VectorTy);
              addRegisterUsage(Recipe->getVPSingleValue(), RegID, NumUsedRegs);
              InstructionCost Cost = getRegisterPressureCost(RegID, VectorTy);
              return Cost + TTI.getCmpSelInstrCost(
                                I->getOpcode(), VectorTy, nullptr,
                                cast<CmpInst>(I)->getPredicate(), CostKind, I);
            }
            case Instruction::BitCast:
              if (I->getType()->isPointerTy())
                return 0;
              [[fallthrough]];
            case Instruction::ZExt:
            case Instruction::SExt:
            case Instruction::FPToUI:
            case Instruction::FPToSI:
            case Instruction::FPExt:
            case Instruction::PtrToInt:
            case Instruction::IntToPtr:
            case Instruction::SIToFP:
            case Instruction::UIToFP:
            case Instruction::Trunc:
            case Instruction::FPTrunc: {
              Type *VectorTy = getVectorType(I->getType(), RVL);
              // Computes the CastContextHint from a Load/Store instruction.

              unsigned Opcode = I->getOpcode();
              TTI::CastContextHint CCH = TTI::CastContextHint::None;
              Type *SrcScalarTy = I->getOperand(0)->getType();
              Type *SrcVecTy = VectorTy->isVectorTy()
                                   ? ToVectorTy(SrcScalarTy, VF)
                                   : SrcScalarTy;

              const unsigned RegID =
                  TTI.getRegisterClassForType(true /*vector*/, VectorTy);
              const unsigned NumUsedRegs = TTI.getRegUsageForType(VectorTy);
              addRegisterUsage(Recipe->getVPSingleValue(), RegID, NumUsedRegs);
              InstructionCost Cost = getRegisterPressureCost(RegID, VectorTy);
              return Cost + TTI.getCastInstrCost(Opcode, VectorTy, SrcVecTy,
                                                 CCH, CostKind, I);
            }
            case Instruction::Call: {
              const CallInst *CI = cast<CallInst>(I);
              InstructionCost CallCost = getVectorCallCost(CI, RVL);
              if (getVectorIntrinsicIDForCall(CI, &TLI)) {
                InstructionCost IntrinsicCost = getVectorIntrinsicCost(CI, RVL);
                return std::min(CallCost, IntrinsicCost);
              }
              return CallCost;
            }
            case Instruction::ExtractValue:
              return TTI.getInstructionCost(I, CostKind);
            case Instruction::Load:
            case Instruction::Store:
              return getMemoryOpCost(
                  I, getLoadStoreType(const_cast<Instruction *>(I)),
                  /*IsConsecutive=*/true,
                  /*IsMasked=*/false, /*IsReverse=*/false,
                  /*Speculative=*/false);
            default: {
              Type *VectorTy = getVectorType(I->getType(), RVL);
              const unsigned RegID =
                  TTI.getRegisterClassForType(true /*vector*/, VectorTy);
              InstructionCost Cost = 0;
              for (const VPValue *VPV : Recipe->definedValues()) {
                const unsigned NumUsedRegs = TTI.getRegUsageForType(VectorTy);
                addRegisterUsage(VPV, RegID, NumUsedRegs);
                Cost += getRegisterPressureCost(RegID, VectorTy);
              }
              // This opcode is unknown. Assume that it is the same as 'mul'.
              return Cost + TTI.getArithmeticInstrCost(Instruction::Mul,
                                                       VectorTy, CostKind);
            }
            } // end of switch.
          });

  VisitedRecipes.insert(Recipe);
  // Traverse operands of the recipe and if operand is no longer used, free
  // registers it occupied.
  for (const VPValue *VPV : Recipe->operands()) {
    if (llvm::all_of(VPV->users(), [&](const VPUser *VPU) -> bool {
          if (auto *VPRU = dyn_cast<VPRecipeBase>(VPU))
            return VisitedRecipes.count(VPRU);
          return false;
        })) {
      auto RegUsageIt = RegistersUsage.LiveRecipes.find(VPV);
      if (RegUsageIt == RegistersUsage.LiveRecipes.end())
        continue;

      for (auto &RegisterUsage : RegUsageIt->second) {
        // Free registers of the VPValue as they're no longer used
        unsigned &RegUsed = RegistersUsage.LiveRegister[RegisterUsage.first];
        assert(RegisterUsage.second <= RegUsed &&
               "Invalid number of registers in use");
        RegUsed -= RegisterUsage.second;
        // zero used number of registers by VPValue to avoid "double free"
        RegisterUsage.second = 0;
      }
    }
  }

  RVVPair VPRecipeRVL = getRecipeType(Recipe)
                            ? RVVPair::getWithType(getRecipeType(Recipe), RVL)
                            : RVL;
  LLVM_DEBUG(dbgs() << "VPlanCM: cost " << Cost << " for RVL " << VPRecipeRVL
                    << " for VPInstruction: ";
             VPSlotTracker SlotTracker((Recipe->getParent())
                                           ? Recipe->getParent()->getPlan()
                                           : nullptr);
             Recipe->print(dbgs(), Twine(), SlotTracker); dbgs() << '\n');
  LLVM_DEBUG(dbgs() << "VPlanCM: Current registers usage"
                    << (SiFiveEstimateRegesterPressure ? "" : "(ignored by CM)")
                    << ':');
  LLVM_DEBUG(for (const auto RegUsage
                  : RegistersUsage.LiveRegister) {
    dbgs() << '\t' << TTI.getRegisterClassName(RegUsage.first) << " = "
           << RegUsage.second;
  });
  LLVM_DEBUG(dbgs() << '\n');
  return Cost;
}

InstructionCost VPlanCostModel::getVectorCallCost(const CallInst *CI,
                                                  const RVVPair &RVL) const {
  Function *F = CI->getCalledFunction();
  Type *ScalarRetTy = CI->getType();
  SmallVector<Type *, 4> Tys, ScalarTys;
  for (auto &ArgOp : CI->args())
    ScalarTys.push_back(ArgOp->getType());

  // Estimate cost of scalarized vector call. The source operands are assumed
  // to be vectors, so we need to extract individual elements from there,
  // execute VF scalar calls, and then gather the result into the vector return
  // value.
  InstructionCost ScalarCallCost =
      TTI.getCallInstrCost(F, ScalarRetTy, ScalarTys, CostKind);

  // Compute corresponding vector type for return value and arguments.
  // FIXME: Some types can stay scalar, so this code should not blindly convert
  // all types to vector type
  Type *RetTy = getVectorType(ScalarRetTy, RVL);
  for (Type *ScalarTy : ScalarTys)
    Tys.push_back(getVectorType(ScalarTy, RVL));

  // FIXME: The compute cost to execute the loop (we're using scalable vectors)
  // which will extract and insert elements
  InstructionCost ScalarizationCost = InstructionCost::getInvalid();
  // FIXME: This should use getEstimatedVLFor for better estimation of the
  // scalarization
  ElementCount VF = getElementCount(RVL);
  InstructionCost Cost =
      ScalarCallCost * VF.getKnownMinValue() + ScalarizationCost;

  // If we can't emit a vector call for this function, then the currently found
  // cost is the cost we need to return.
  InstructionCost MaskCost = 0;
  VFShape Shape = VFShape::get(*CI, VF, false /*HasGlobalPred*/);
  Function *VecFunc =
      VFDatabase(*const_cast<CallInst *>(CI)).getVectorizedFunction(Shape);
  // If we want an unmasked vector function but can't find one matching the VF,
  // maybe we can find vector function that does use a mask and synthesize
  // an all-true mask.
  if (!VecFunc) {
    Shape = VFShape::get(*CI, VF, /*HasGlobalPred=*/true);
    VecFunc =
        VFDatabase(*const_cast<CallInst *>(CI)).getVectorizedFunction(Shape);
    // If we found one, add in the cost of creating a mask
    if (VecFunc)
      MaskCost = TTI.getShuffleCost(
          TargetTransformInfo::SK_Broadcast,
          VectorType::get(
              IntegerType::getInt1Ty(VecFunc->getFunctionType()->getContext()),
              VF));
  }

  if (CI->isNoBuiltin() || !VecFunc)
    return Cost;

  // If the corresponding vector cost is cheaper, return its cost.
  InstructionCost VectorCallCost =
      TTI.getCallInstrCost(nullptr, RetTy, Tys, CostKind) + MaskCost;
  if (VectorCallCost < Cost) {
    Cost = VectorCallCost;
  }
  return Cost;
}

InstructionCost
VPlanCostModel::getVectorIntrinsicCost(const CallInst *CI,
                                       const RVVPair &RVL) const {
  Intrinsic::ID ID =
      getVectorIntrinsicIDForCall(CI, &TLI, Legal.useVLAVectorizer());
  assert(ID && "Expected intrinsic call!");
  Type *RetTy = getVectorType(CI->getType(), RVL);
  FastMathFlags FMF;
  if (auto *FPMO = dyn_cast<FPMathOperator>(CI))
    FMF = FPMO->getFastMathFlags();

  SmallVector<const Value *> Arguments(CI->args());
  FunctionType *FTy = CI->getCalledFunction()->getFunctionType();
  SmallVector<Type *> ParamTys;
  std::transform(FTy->param_begin(), FTy->param_end(),
                 std::back_inserter(ParamTys),
                 [&](Type *Ty) { return getVectorType(Ty, RVL); });

  IntrinsicCostAttributes CostAttrs(ID, RetTy, Arguments, ParamTys, FMF,
                                    dyn_cast<IntrinsicInst>(CI));
  return TTI.getIntrinsicInstrCost(CostAttrs, CostKind);
}

InstructionCost VPlanCostModel::getMemoryOpCost(const Instruction *I, Type *Ty,
                                                bool IsConsecutive,
                                                bool IsMasked, bool IsReverse,
                                                bool IsSpeculative) const {
  const Align Alignment = getLoadStoreAlignment(const_cast<Instruction *>(I));
  const Value *Ptr = getLoadStorePointerOperand(I);
  unsigned AS = getLoadStoreAddressSpace(const_cast<Instruction *>(I));
  if (IsConsecutive) {
    InstructionCost Cost = 0;
    if (!IsSpeculative || IsMasked) {
      Cost += TTI.getMaskedMemoryOpCost(I->getOpcode(), Ty, Alignment, AS,
                                        CostKind);
    } else {
      TTI::OperandValueInfo OpInfo = TTI::getOperandInfo(I->getOperand(0));
      Cost += TTI.getMemoryOpCost(I->getOpcode(), Ty, Alignment, AS, CostKind,
                                  OpInfo, I);
    }
    if (IsReverse)
      Cost +=
          TTI.getShuffleCost(TargetTransformInfo::SK_Reverse,
                             cast<VectorType>(Ty), std::nullopt, CostKind, 0);
    return Cost;
  }
  // FIXME: There should be a special code for strided memory access
  return TTI.getAddressComputationCost(Ty) +
         TTI.getGatherScatterOpCost(I->getOpcode(), Ty, Ptr, IsMasked,
                                    Alignment, CostKind, I);
}

InstructionCost
VPlanCostModel::getMemoryOpCost(const VPWidenMemoryInstructionRecipe *VPWMIR,
                                const RVVPair &RVL) {
  const Instruction *I = &VPWMIR->getIngredient();
  Type *ValTy = VPWMIR->getElementType();
  const bool IsMasked = VPWMIR->getMask() != nullptr;
  auto *VectorTy = cast<VectorType>(getVectorType(ValTy, RVL));

  InstructionCost Cost = 0;
  if (!VPWMIR->isStore()) {
    const unsigned RegID =
        TTI.getRegisterClassForType(true /*vector*/, VectorTy);
    const unsigned NumUsedRegs = TTI.getRegUsageForType(VectorTy);
    addRegisterUsage(VPWMIR->getVPSingleValue(), RegID, NumUsedRegs);
    Cost = getRegisterPressureCost(RegID, VectorTy);
  }
  return Cost + getMemoryOpCost(I, VectorTy, VPWMIR->isConsecutive(), IsMasked,
                                VPWMIR->isReverse(), VPWMIR->isSpeculative());
}

InstructionCost VPlanCostModel::getInstructionCost(const VPInstruction *VPI,
                                                   const RVVPair &RVL) const {
  switch (VPI->getOpcode()) {
  case Instruction::FMul: {
    const Value *UV = VPI->getOperand(0)->getUnderlyingValue();
    if (!UV)
      return 0;
    Type *VectorTy = getVectorType(UV->getType(), RVL);
    return TTI.getArithmeticInstrCost(Instruction::FMul, VectorTy, CostKind);
  }
  case Instruction::Select:
    // VPSelectInstruction is generated to emit TU policy. Currently it has no
    // overhead in HW
    return 0;
  case VPInstruction::FirstOrderRecurrenceSplice: {
    auto *V = VPI->getOperand(0)->getUnderlyingValue();
    auto *VectorTy = getVectorType(V->getType(), RVL);
    return TTI.getShuffleCost(TargetTransformInfo::SK_Splice,
                              cast<VectorType>(VectorTy), std::nullopt,
                              CostKind, /*Index*/ -1);
  }
  case VPInstruction::CanonicalIVIncrement:
  case VPInstruction::BranchOnCount:
    return 1;
  default:
    return 0;
  }
}

InstructionCost
VPlanCostModel::getInterleavedMemoryOpCost(const VPInterleaveRecipe *VPI,
                                           const RVVPair &RVL) {
  const InterleaveGroup<Instruction> *Group = VPI->getInterleaveGroup();
  const unsigned InterleaveFactor = Group->getFactor();
  const Instruction *I = Group->getMember(0);
  unsigned AS = getLoadStoreAddressSpace(const_cast<Instruction *>(I));
  Type *ValTy = getLoadStoreType(const_cast<Instruction *>(I));
  const bool IsMasked = VPI->getMask() != nullptr;
  auto *VectorTy = cast<VectorType>(getVectorType(ValTy, RVL));
  ElementCount VF = getElementCount(RVL);
  auto *WideVecTy = VectorType::get(ValTy, VF * InterleaveFactor);

  if (!TTI.isLegalVectorInterleave(VectorType::get(ValTy, VF), InterleaveFactor,
                                   I->getModule()->getDataLayout())) {
    LLVM_DEBUG(dbgs() << "InterleaveGroup = "; VPSlotTracker SlotTracker(
                   (VPI->getParent()) ? VPI->getParent()->getPlan() : nullptr);
               VPI->print(dbgs(), Twine(), SlotTracker);
               dbgs() << " is illegal for " << RVL << '\n');
    assert(0 && "InterleaveGroup is illegal for a given RVL");
    // Even though such candidates should be filtered out before VPlan is
    // constructed, make sure we won't select this candidate for vectorization
    return InstructionCost::getInvalid();
  }

  assert(VF.isScalable() && "Cost model for Interleaved Memory Access is only "
                            "implemented for scalable vectors");
  const unsigned RegID = TTI.getRegisterClassForType(true /*vector*/, VectorTy);

  InstructionCost Cost = TTI.getInterleavedMemoryOpCost(
      I->getOpcode(), WideVecTy, InterleaveFactor, /*Indices=*/{},
      Group->getAlign(), AS, CostKind, IsMasked, /*UseMaskForGaps=*/false);

  for (const VPValue *VPV : VPI->definedValues()) {
    const unsigned NumUsedRegs = TTI.getRegUsageForType(VectorTy);
    addRegisterUsage(VPV, RegID, NumUsedRegs);
    Cost += getRegisterPressureCost(RegID, VectorTy);
  }

  if (Group->isReverse()) {
    Cost += Group->getNumMembers() *
            TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, VectorTy,
                               std::nullopt, CostKind, 0);
  }
  return Cost;
}

InstructionCost VPlanCostModel::getReductionCost(const VPReductionRecipe *VPR,
                                                 const RVVPair &RVL) const {
  const RecurrenceDescriptor &RdxDesc = VPR->getRecurrenceDescriptor();

  RecurKind RdxKind = RdxDesc.getRecurrenceKind();
  Type *ElementTy = RdxDesc.getRecurrenceType();
  auto *VectorTy = cast<VectorType>(getVectorType(ElementTy, RVL));
  switch (RdxKind) {
  case RecurKind::Add:
  case RecurKind::Mul:
  case RecurKind::Or:
  case RecurKind::And:
  case RecurKind::Xor:
  case RecurKind::FAdd:
  case RecurKind::FMul:
  case RecurKind::FMulAdd:
    return TTI.getArithmeticReductionCost(
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
    return TTI.getMinMaxReductionCost(Id, VectorTy, RdxDesc.getFastMathFlags(),
                                      CostKind);
  }
  default:
    assert(0 && "Expected arithmetic or min/max reduction");
  }
  return InstructionCost::getInvalid();
}

InstructionCost VPlanCostModel::getReplicateOpCost(const VPReplicateRecipe *VPR,
                                                   const RVVPair &RVL) const {
  if (VPR->isUniform())
    return 1;
  const Instruction *I = VPR->getUnderlyingInstr();
  if (isa<AllocaInst>(I)) {
    ElementCount VF = getElementCount(RVL);
    // We cannot easily widen alloca to a scalable alloca, as
    // the result would need to be a vector of pointers.
    if (VF.isScalable())
      return InstructionCost::getInvalid();
    Type *VectorTy = getVectorType(I->getType(), RVL);
    // This opcode is unknown. Assume that it is the same as 'mul'.
    return TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy, CostKind);
  }

  assert(0 &&
         "non-uniform replicate recipe is not yet supported by VLA vectorizer");
  // FIXME:This estimation is not correct. It should return VLMAX
  return getElementCount(RVL).getKnownMinValue();
}

void VPlanCostModel::addRegisterUsage(const VPValue *VPV, const unsigned RegID,
                                      const unsigned NumRegs) {
  LLVM_DEBUG(dbgs() << "VPlanCM: " << *VPV << " will use " << NumRegs << ' '
                    << TTI.getRegisterClassName(RegID) << " registers\n");
  RegistersUsage.LiveRecipes[VPV][RegID] += NumRegs;
  RegistersUsage.LiveRegister[RegID] += NumRegs;
}

InstructionCost VPlanCostModel::getRegisterPressureCost(const unsigned RegID,
                                                        Type *Ty) const {
  if (!SiFiveEstimateRegesterPressure)
    return 0;

  const unsigned Count = RegistersUsage.LiveRegister.find(RegID)->second;
  const unsigned MaxNumRegisters = TTI.getNumberOfRegisters(RegID);
  if (Count > MaxNumRegisters) {
    // Need to spill excessive registers, which will require to store them
    // into memory
    InstructionCost Cost =
        (Count - MaxNumRegisters) *
        TTI.getMemoryOpCost(Instruction::Store, Ty, Align(64), 0, CostKind);
    Cost += (Count - MaxNumRegisters) *
            TTI.getMemoryOpCost(Instruction::Load, Ty, Align(64), 0, CostKind);
    LLVM_DEBUG(dbgs() << "VPlanCM: Spill and Reload of "
                      << (Count - MaxNumRegisters) << " registers is required"
                      << ". Cost increased by " << Cost << '\n');
    return Cost;
  }
  return 0;
}

} // namespace llvm
