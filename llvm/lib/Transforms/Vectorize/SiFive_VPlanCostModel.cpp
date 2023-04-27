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

static ElementCount getElementCount(const RVVPair &RVVP) {
  std::pair<unsigned, bool> LMUL = RVVP.getLMUL();
  unsigned KnownMinValue =
      LMUL.second ? RISCV::RVVBitsPerBlock / (LMUL.first * RVVP.getSEW())
                  : RISCV::RVVBitsPerBlock * LMUL.first / RVVP.getSEW();
  return ElementCount::get(KnownMinValue, true);
}

namespace llvm {
// FIXME: Unify with `RvvHintAttr` in Clang
RVVPair RVVPair::getWithExponent(Type *Ty, const unsigned LMULExp,
                                 const DataLayout &DL) {
  assert(LMULExp != 4 && LMULExp <= 7 &&
         "LMUL exponent is not a valid or not supported.");
  return RVVPair(Ty, (RVVPair::LMULKind)LMULExp, DL);
}

// FIXME: Unify with `RvvHintAttr` in Clang
std::pair<unsigned, bool> RVVPair::getIntFromLMULKind(LMULKind Kind) const {
  unsigned LMULExp = (unsigned)Kind;
  assert(LMULExp != 4 && LMULExp <= 7 &&
         "LMUL exponent is not a valid or not supported.");
  return {LMULExp < 4 ? 1 << LMULExp : 1 << (8 - LMULExp),
          (LMULExp & 0b100) != 0};
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
  }
  llvm_unreachable("Unsupported LMUL Kind");
}

Type *VPlanCostModel::getVectorType(Type *Ty, const RVVPair &RVVP) {
  assert(!isa_and_nonnull<VectorType>(Ty) &&
         "Cannot convert non-scalar type to VectorType for a given (LMUL, SEW) "
         "pair");
  if (Ty->isVoidTy() || Ty->isMetadataTy())
    return Ty;
  return ScalableVectorType::get(Ty, getElementCount(RVVP).getKnownMinValue());
}

InstructionCost VPlanCostModel::getCost(const RVVPair &RVL) const {
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
                                        const RVVPair &RVL) const {
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
                                        const RVVPair &RVL) const {
  InstructionCost Cost =
      TypeSwitch<const VPRecipeBase *, InstructionCost>(Recipe)
          .Case<VPWidenMemoryInstructionRecipe>(
              [&](const VPWidenMemoryInstructionRecipe *VPWMIR) {
                return getMemoryOpCost(VPWMIR, RVL);
              })
          .Case<VPCanonicalIVPHIRecipe, VPScalarIVStepsRecipe,
                VPWidenIntOrFpInductionRecipe, VPReductionPHIRecipe,
                VPWidenPointerInductionRecipe>(
              [&](const VPRecipeBase *IVR) -> InstructionCost { return 1; })
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
                  Legal.isUniform(Op2))
                Op2Info.Kind = TargetTransformInfo::OK_UniformValue;

              SmallVector<const Value *, 4> Operands(I->operand_values());
              return TTI.getArithmeticInstrCost(
                  I->getOpcode(), VectorTy, CostKind,
                  {TargetTransformInfo::OK_AnyValue,
                   TargetTransformInfo::OP_None},
                  Op2Info, Operands, I);
            }
            case Instruction::FNeg: {
              Type *VectorTy = getVectorType(I->getType(), RVL);
              return TTI.getArithmeticInstrCost(
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
              return TTI.getCmpSelInstrCost(I->getOpcode(), VectorTy, CondTy,
                                             Pred, CostKind, I);
            }
            case Instruction::ICmp:
            case Instruction::FCmp: {
              Type *VectorTy = getVectorType(I->getType(), RVL);
              return TTI.getCmpSelInstrCost(I->getOpcode(), VectorTy, nullptr,
                                             cast<CmpInst>(I)->getPredicate(),
                                             CostKind, I);
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

              return TTI.getCastInstrCost(Opcode, VectorTy, SrcVecTy, CCH,
                                           CostKind, I);
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
                  /*IsMasked=*/false, /*IsReverse*/false,
                  /*Speculative=*/false);
            case Instruction::Alloca:
              // We cannot easily widen alloca to a scalable alloca, as
              // the result would need to be a vector of pointers.
              if (VF.isScalable())
                return InstructionCost::getInvalid();
              [[fallthrough]];
            default: {
              Type *VectorTy = getVectorType(I->getType(), RVL);
              // This opcode is unknown. Assume that it is the same as 'mul'.
              return TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy,
                                                 CostKind);
            }
            } // end of switch.
          });

  LLVM_DEBUG(dbgs() << "VPlanCM: cost " << Cost << " for RVL " << RVL
                    << " for VPInstruction: ";
             VPSlotTracker SlotTracker((Recipe->getParent())
                                           ? Recipe->getParent()->getPlan()
                                           : nullptr);
             Recipe->print(dbgs(), Twine(), SlotTracker); dbgs() << '\n');
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

InstructionCost VPlanCostModel::getVectorIntrinsicCost(const CallInst *CI,
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
                                const RVVPair &RVL) const {
  const Instruction *I = &VPWMIR->getIngredient();
  Type *ValTy = VPWMIR->getElementType();
  const bool IsMasked = VPWMIR->getMask() != nullptr;
  Type *VectorTy = getVectorType(ValTy, RVL);

  return getMemoryOpCost(I, VectorTy, VPWMIR->isConsecutive(), IsMasked,
                         VPWMIR->isReverse(), VPWMIR->isSpeculative());
}

InstructionCost VPlanCostModel::getInstructionCost(const VPInstruction *VPI,
                                                   const RVVPair &RVL) const {
  switch (VPI->getOpcode()) {
    case Instruction::Select:
      // VPSelectInstruction is generated to emit TU policy. Currently it has no
      // overhead in HW
      return 0;
    case VPInstruction::CanonicalIVIncrement:
    case VPInstruction::CanonicalIVIncrementNUW:
    case VPInstruction::BranchOnCount:
      return 1;
    default:
      return 0;
  }
}
} // namespace llvm
