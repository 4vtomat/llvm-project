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
                                     VPUser &User, const unsigned Part,
                                     const Twine &Name) {
  VPValue *RVL = State.RVL;
  Value *Cond = State.get(User.getOperand(0), Part);
  Value *Op1 = State.get(User.getOperand(1), Part);
  Value *Op2 = State.get(User.getOperand(2), Part);
  Value *RVLArg = State.get(RVL, Part, /*NeedsScalar=*/true);
  return State.Builder.CreateIntrinsic(VPOpCode, {Op1->getType()},
                                       {Cond, Op1, Op2, RVLArg}, nullptr, Name);
}

/// Construct vector popcount of the vector \p V
static Value *createVectorPopcount(IRBuilderBase &Builder, Value *V,
                                   Value *RVL) {
  ElementCount EC = cast<VectorType>(V->getType())->getElementCount();
  Value *Operands[] = {Builder.getTrueVector(EC), V, RVL};

  return Builder.CreateIntrinsic(Intrinsic::experimental_vp_popcount,
                                 {V->getType()}, Operands);
}

/// Compress \p VectorToCompress using RVV vcompress intrinsic
static Value *compressVector(IRBuilderBase &Builder, Value *Mask,
                             Value *VectorToCompress, Value *RVL) {
  assert(Mask != nullptr && "Compress mask must be provided");
  assert(RVL != nullptr && "RVL for RVV-intrinsic must be provided");
  Type *VTy = VectorToCompress->getType();
  Value *Operands[] = {VectorToCompress, Mask, RVL};
  CallInst *Compress = Builder.CreateIntrinsic(
      Intrinsic::experimental_vp_compress, {VTy}, Operands);
  return Compress;
}

namespace llvm {
Value *widenPredicatedInstruction(Instruction *Op, VPValue *Def, VPUser &User,
                                  VPTransformState &State, VPValue *BlockInMask,
                                  unsigned Part) {
  VPValue *RVL = State.RVL;
  IRBuilderBase &BuilderIR = State.Builder;
  VectorBuilder Builder(BuilderIR);
  auto &&MaskValue = [&](unsigned Part, ElementCount EC) -> Value * {
    if (!BlockInMask)
      return BuilderIR.getTrueVector(State.VF);
    // The outermost mask can be lowered as an all ones mask when using RVL.
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
    Builder.setEVL(State.get(RVL, Part, /*NeedsScalar=*/true));
    return Builder.createVectorInstruction(CI->getOpcode(), DestTy, {SrcVal},
                                           "vp.cast");
  };

  switch (Opcode) {
  case VPInstruction::Not: {
    assert(!Op && "Expected with no-op only.");
    Value *A = State.get(User.getOperand(0), Part);
    auto *PredTy = cast<VectorType>(A->getType());
    Value *MaskArg = BuilderIR.getTrueVector(State.VF);
    Value *RVLArg = State.get(RVL, Part, /*NeedsScalar=*/true);
    Builder.setMask(MaskArg).setEVL(RVLArg);
    return Builder.createVectorInstruction(Instruction::Xor, PredTy,
                                           {A, MaskArg}, "pred.not");
  }
  case Instruction::Select: {
    assert((!Op || isa<VPWidenSelectRecipe>(Def->getDefiningRecipe())) &&
           "Expected with no-op only or VPWidenSelectRecipe.");
    return widenSelectInstruction(State, Intrinsic::vp_select, Def, User, Part,
                                  "vp.op.select");
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    //===------------------ compare instructions --------------------------===//
    // Widen compares. Generate vector compares.
    bool FCmp = (Opcode == Instruction::FCmp);
    auto *Cmp = cast<CmpInst>(Op);
    Value *A = State.get(User.getOperand(0), Part);
    Value *B = State.get(User.getOperand(1), Part);

    VectorType *OpTy = cast<VectorType>(A->getType());
    Value *MaskArg = MaskValue(Part, OpTy->getElementCount());
    Builder.setMask(MaskArg);
    Value *RVLArg = State.get(RVL, Part, /*NeedsScalar=*/true);
    Builder.setEVL(RVLArg);

    StringRef PredicateStr = CmpInst::getPredicateName(Cmp->getPredicate());
    auto *PredicateMDS = MDString::get(Cmp->getContext(), PredicateStr);
    Value *PredArg = MetadataAsValue::get(Cmp->getContext(), PredicateMDS);

    if (FCmp) {
      IRBuilder<>::FastMathFlagGuard FMFG(BuilderIR);
      BuilderIR.setFastMathFlags(Cmp->getFastMathFlags());
      return Builder.createVectorInstruction(Opcode, OpTy, {A, B, PredArg},
                                          "vp.op.fcmp");
    }
    return Builder.createVectorInstruction(Opcode, OpTy, {A, B, PredArg},
                                           "vp.op.icmp");
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
    // Just widen unops and binops.

    SmallVector<Value *, 4> Ops;
    for (unsigned I = 0, E = Instruction::isBinaryOp(Opcode) ? 2 : 1; I < E;
         ++I) {
      VPValue *VPOp = User.getOperand(I);
      Ops.push_back(State.get(VPOp, Part));
    }

    VectorType *OpTy = cast<VectorType>(Ops[0]->getType());
    Value *MaskArg = nullptr;
    if (Op && !canUnaryOrBinaryOpBeUnmasked(Opcode, OpTy->getElementType()))
      MaskArg = MaskValue(Part, OpTy->getElementCount());
    Value *V =
        widenPredicatedArithmeticOp(State, Opcode, Ops, Part, MaskArg, "vp.op");

    if (Op)
      if (auto *VecOp = dyn_cast<Instruction>(V))
        VecOp->copyIRFlags(Op);

    return V;
  }
  llvm_unreachable("Unexpected opcode.");
}

void widenPredicatedCall(CallInst &CI, VPValue *Def, VPUser &ArgOperands,
                         VPTransformState &State, Intrinsic::ID VPID,
                         unsigned Part) {
  IRBuilderBase &Builder = State.Builder;
  auto *DestTy = VectorType::get(CI.getType(), State.VF);
  SmallVector<Type *, 2> TysForDecl = {DestTy};
  SmallVector<Value *, 4> Args;
  for (auto I : enumerate(ArgOperands.operands())) {
    Value *Arg;
    if (!isVectorIntrinsicWithScalarOpAtArg(VPID, I.index()))
      Arg = State.get(I.value(), Part);
    else
      Arg = State.get(I.value(), VPIteration(0, 0));
    if (isVectorIntrinsicWithOverloadTypeAtArg(VPID, I.index()))
      TysForDecl.push_back(Arg->getType());
    Args.push_back(Arg);
  }

  Args.push_back(Builder.getTrueVector(State.VF));
  Args.push_back(State.get(State.RVL, Part, /*NeedsScalar=*/ true));
  CallInst *V =
      Builder.CreateIntrinsic(VPID, TysForDecl, Args, nullptr, "vp.op");
  if (isa<FPMathOperator>(V))
    V->copyFastMathFlags(&CI);
  State.set(Def, V, Part);
}

void VPSelectInstruction::execute(VPTransformState &State) {
  if (!State.RVL) {
    // For non RVV VLA vectorization, reuse existing mechanism to generate the
    // vector code
    VPInstruction::execute(State);
    return;
  }

  assert(!State.Instance && "VPInstruction executing an Instance");

  unsigned VPOpCode = Intrinsic::vp_select;
  StringRef Name = "vp.op.select";
  if (hasTailUndisturbedPolicy()) {
    VPOpCode = Intrinsic::vp_merge;
    Name = "vp.op.merge";
  }

  for (unsigned Part = 0; Part < State.UF; ++Part) {
    Value *V = widenSelectInstruction(State, VPOpCode, this, *this, Part, Name);
    State.set(this, V, Part);
  }
}

/// Generate following sequence to update scalar monotonic variable:
///   %0 = vp.popcount(%mask, %rvl)
///   %1 = mul %0, %step          // where %step is a step of monotonic
///   %monotonic.update = add %monotonic, %1
void VPMonotonicUpdateInstruction::execute(VPTransformState &State) {
  assert(State.UF == 1 && "Unrolling is not supported");
  auto &Builder = State.Builder;
  Value *V = State.get(getIncomingValue(), 0, /*NeedsScalar=*/true);
  Value *Step = State.get(getStepValue(), 0, /*NeedsScalar=*/true);
  Value *Mask = State.get(getMask(), 0);
  Value *RVL = State.get(State.RVL, 0, /*NeedsScalar=*/true);
  Value *Vpop = createVectorPopcount(Builder, Mask, RVL);
  Vpop = Builder.CreateZExtOrTrunc(Vpop, Step->getType());

  Value *Mult = Builder.CreateMul(Vpop, Step);
  const auto *OrigUpdateOp = cast<BinaryOperator>(MD.getUpdateOp());

  Value *NewV = Builder.CreateBinOp(OrigUpdateOp->getOpcode(), V, Mult,
                                    "monotonic.update");
  State.set(this, NewV, 0, /*IsScalar=*/true);
}

/// Generate phi for the monotonic:
///   %monotonic = phi [%monotonic.update, %vector.latch]
void VPMonotonicHeaderPHIRecipe::execute(VPTransformState &State) {
  IRBuilder<>::InsertPointGuard Guard(State.Builder);
  State.Builder.SetInsertPoint(State.CFG.PrevBB->getFirstNonPHI());

  Value *StartV = State.get(getStartValue(), 0, /*NeedsScalar=*/true);
  auto *Phi = State.Builder.CreatePHI(StartV->getType(), 2, "monotonic.phi");
  BasicBlock *PreheaderBB = State.CFG.getPreheaderBBFor(this);
  Phi->addIncoming(StartV, PreheaderBB);

  // Use the same Phi for all Parts
  for (unsigned Part = 0; Part < State.UF; ++Part)
    State.set(this, Phi, Part, /*IsScalar*/ true);
}

/// Build and return either `vp.gather`/`vp.scatter` or
/// `vp.strided_load`/`vp.strided_store` if previous analysis indicated it's
/// possible to be used
Instruction *
widenPredicatedMemoryInstruction(VPWidenMemoryInstructionRecipe &VPWMIR,
                                 VPTransformState &State, unsigned Part,
                                 ArrayRef<Value *> BlockInMaskParts) {
  assert(Part == 0 && "Cannot support Part > 0 for RVV VLA vectorization");
  Value *RVLPart = State.get(State.RVL, Part, /*NeedsScalar=*/true);
  assert(RVLPart && "RVL must be set prior to generation of vp-intrinsics");

  VPValue *VPAddr = VPWMIR.getAddr();
  ElementCount NumElts = State.VF;
  auto &Builder = State.Builder;
  const Align Alignment = getLoadStoreAlignment(&VPWMIR.getIngredient());

  auto MaskValue = [&](unsigned Part, ElementCount EC) -> Value * {
    // The outermost mask can be lowered as an all ones mask when using
    // RVL.
    VPValue *Mask = VPWMIR.getMask();
    if (!Mask)
      return Builder.getTrueVector(EC);

    return BlockInMaskParts[Part];
  };
  Value *BlockInMaskPart = MaskValue(Part, NumElts);

  if (VPWMIR.isStore()) {
    VPValue *StoredValue = VPWMIR.getStoredValue();
    Value *StoredVal = State.get(StoredValue, Part);
    if (VPWMIR.isMonotonic())
      StoredVal = compressVector(Builder, BlockInMaskPart, StoredVal, RVLPart);

    if (VPWMIR.isStrided()) {
      Value *Ptr = State.get(VPAddr, VPIteration(0, 0));
      const SCEV *SCEVStride = VPWMIR.getStrideInBytes();
      auto *PtrTy = cast<PointerType>(Ptr->getType());
      CallInst *VS = nullptr;
      if (!VPWMIR.isConsecutive()) {
        auto &DL = State.CFG.PrevBB->getModule()->getDataLayout();
        SCEVExpander Exp(*State.SE, DL, "stride");
        Instruction *InsertPoint = &*State.Builder.GetInsertPoint();
        assert(Exp.isSafeToExpandAt(SCEVStride, InsertPoint) &&
               "It's not safe to expand that SCEV in the vector loop. That was "
               "not caught by isSafeStrideAccessInfo.");
        Value *Stride = Exp.expandCodeFor(
            SCEVStride, SCEVStride->getType(), InsertPoint);
        LLVM_DEBUG(llvm::dbgs()
                   << "Generating strided store for addr = " << *VPAddr
                   << " with a stride = " << *Stride << '\n');
        Value *Operands[] = {StoredVal, Ptr, Stride, BlockInMaskPart, RVLPart};
        VS = Builder.CreateIntrinsic(
            Intrinsic::experimental_vp_strided_store,
            {StoredVal->getType(), PtrTy, Stride->getType()}, Operands);
      } else {
        if (VPWMIR.isMonotonic()) {
          RVLPart = createVectorPopcount(Builder, BlockInMaskPart, RVLPart);
          BlockInMaskPart = Builder.getTrueVector(NumElts);
        }
        Value *Operands[] = {StoredVal, Ptr, BlockInMaskPart, RVLPart};
        VS = Builder.CreateIntrinsic(Intrinsic::vp_store,
                                     {StoredVal->getType(), PtrTy}, Operands);
      }

      VS->addParamAttr(
          1, Attribute::getWithAlignment(VS->getContext(), Alignment));
      return VS;
    }
    auto *DataTy = cast<VectorType>(StoredVal->getType());
    LLVM_DEBUG(llvm::dbgs() << "Indexed store for " << *VPAddr << "\n");
    Value *VectorGep = State.get(VPAddr, Part);
    Value *Operands[] = {StoredVal, VectorGep, BlockInMaskPart, RVLPart};
    auto *PtrsTy = cast<VectorType>(VectorGep->getType());
    CallInst *VS = Builder.CreateIntrinsic(Intrinsic::vp_scatter,
                                           {DataTy, PtrsTy}, Operands);
    VS->addParamAttr(1,
                     Attribute::getWithAlignment(VS->getContext(), Alignment));
    return VS;
  }
  auto *DataTy = VectorType::get(VPWMIR.getElementType(), State.VF);
  if (VPWMIR.isStrided()) {
    Value *Ptr = State.get(VPAddr, VPIteration(0, 0));
    const SCEV *SCEVStride = VPWMIR.getStrideInBytes();
    auto &DL = State.CFG.PrevBB->getModule()->getDataLayout();
    SCEVExpander Exp(*(State.SE), DL, "stride");
    Instruction *InsertPoint = &*State.Builder.GetInsertPoint();
    assert(Exp.isSafeToExpandAt(SCEVStride, InsertPoint) &&
           "It's not safe to expand that SCEV in the vector loop. That was "
           "not caught by isSafeStrideAccessInfo.");
    Value *Stride = Exp.expandCodeFor(
        SCEVStride, SCEVStride->getType(), InsertPoint);
    auto *PtrTy = cast<PointerType>(Ptr->getType());
    LLVM_DEBUG(llvm::dbgs()
               << "Generating strided load for addr = " << *VPAddr
               << " with a stride = " << *Stride << '\n');
    Value *Operands[] = {Ptr, Stride, BlockInMaskPart, RVLPart};
    CallInst *VL =
        Builder.CreateIntrinsic(Intrinsic::experimental_vp_strided_load,
                                {DataTy, PtrTy, Stride->getType()}, Operands,
                                nullptr, "vp.strided.load");
    VL->addParamAttr(
        0, Attribute::getWithAlignment(VL->getContext(), Alignment));
    return VL;
  }
  LLVM_DEBUG(llvm::dbgs() << "Indexed load for " << VPAddr << "\n");
  Value *VectorGep = State.get(VPAddr, Part);
  Value *Operands[] = {VectorGep, BlockInMaskPart, RVLPart};
  auto *PtrsTy = cast<VectorType>(VectorGep->getType());
  CallInst *VL = Builder.CreateIntrinsic(
      Intrinsic::vp_gather, {DataTy, PtrsTy}, Operands, nullptr, "vp.gather");
  VL->addParamAttr(0,
                   Attribute::getWithAlignment(VL->getContext(), Alignment));
  return VL;
}

Instruction *widenPredicatedArithmeticOp(VPTransformState &State,
                                         unsigned Opcode, ArrayRef<Value *> Ops,
                                         unsigned Part, Value *Mask,
                                         const Twine &Name) {
  assert(((Instruction::isBinaryOp(Opcode) && (Ops.size() == 2)) ||
          (Instruction::isUnaryOp(Opcode) && (Ops.size() == 1))) &&
         "Invalid number of operands.");
  VectorBuilder VBuilder(State.Builder);
  VPValue *RVL = State.RVL;
  Value *RVLPart = State.RVL ? State.get(RVL, Part, /*NeedsScalar=*/true)
                             : State.RVLPlaceholder;
  RVLPart =
      State.Builder.CreateZExtOrTrunc(RVLPart, State.Builder.getInt32Ty());
  assert(RVLPart && "RVL was not created");
  if (!Mask)
    Mask = State.Builder.getTrueVector(State.VF);
  VBuilder.setMask(Mask).setEVL(RVLPart);
  return cast<Instruction>(
      VBuilder.createVectorInstruction(Opcode, Ops[0]->getType(), Ops, Name));
}
} // namespace llvm
