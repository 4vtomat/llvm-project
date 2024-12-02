//===----- RISCVCodeGenPrepare.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a RISC-V specific version of CodeGenPrepare.
// It munges the code in the input function to better prepare it for
// SelectionDAG-based code generation. This works around limitations in it's
// basic-block-at-a-time approach.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-codegenprepare"
#define PASS_NAME "RISC-V CodeGenPrepare"

namespace {

class RISCVCodeGenPrepare : public FunctionPass,
                            public InstVisitor<RISCVCodeGenPrepare, bool> {
  const DataLayout *DL;
  const DominatorTree *DT;
  const RISCVSubtarget *ST;

public:
  static char ID;

  RISCVCodeGenPrepare() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<DominatorTreeWrapperPass>(); // SIFIVE
  }

  bool visitInstruction(Instruction &I) { return false; }
  bool visitAnd(BinaryOperator &BO);
  bool visitIntrinsicInst(IntrinsicInst &I);
  bool optimizeReduction(IntrinsicInst &I); // SIFIVE
  bool expandVPStrideLoad(IntrinsicInst &I);
};

} // end anonymous namespace

// Try to optimize (i64 (and (zext/sext (i32 X), C1))) if C1 has bit 31 set,
// but bits 63:32 are zero. If we know that bit 31 of X is 0, we can fill
// the upper 32 bits with ones.
bool RISCVCodeGenPrepare::visitAnd(BinaryOperator &BO) {
  if (!ST->is64Bit())
    return false;

  if (!BO.getType()->isIntegerTy(64))
    return false;

  using namespace PatternMatch;

  // Left hand side should be a zext nneg.
  Value *LHSSrc;
  if (!match(BO.getOperand(0), m_NNegZExt(m_Value(LHSSrc))))
    return false;

  if (!LHSSrc->getType()->isIntegerTy(32))
    return false;

  // Right hand side should be a constant.
  Value *RHS = BO.getOperand(1);

  auto *CI = dyn_cast<ConstantInt>(RHS);
  if (!CI)
    return false;
  uint64_t C = CI->getZExtValue();

  // Look for constants that fit in 32 bits but not simm12, and can be made
  // into simm12 by sign extending bit 31. This will allow use of ANDI.
  // TODO: Is worth making simm32?
  if (!isUInt<32>(C) || isInt<12>(C) || !isInt<12>(SignExtend64<32>(C)))
    return false;

  // Sign extend the constant and replace the And operand.
  C = SignExtend64<32>(C);
  BO.setOperand(1, ConstantInt::get(RHS->getType(), C));

  return true;
}

// LLVM vector reduction intrinsics return a scalar result, but on RISC-V vector
// reduction instructions write the result in the first element of a vector
// register. So when a reduction in a loop uses a scalar phi, we end up with
// unnecessary scalar moves:
//
// loop:
// vfmv.s.f v10, fa0
// vfredosum.vs v8, v8, v10
// vfmv.f.s fa0, v8
//
// This mainly affects ordered fadd reductions, since other types of reduction
// typically use element-wise vectorisation in the loop body. This tries to
// vectorize any scalar phis that feed into a fadd reduction:
//
// loop:
// %phi = phi <float> [ ..., %entry ], [ %acc, %loop ]
// %acc = call float @llvm.vector.reduce.fadd.nxv2f32(float %phi,
//                                                    <vscale x 2 x float> %vec)
//
// ->
//
// loop:
// %phi = phi <vscale x 2 x float> [ ..., %entry ], [ %acc.vec, %loop ]
// %phi.scalar = extractelement <vscale x 2 x float> %phi, i64 0
// %acc = call float @llvm.vector.reduce.fadd.nxv2f32(float %x,
//                                                    <vscale x 2 x float> %vec)
// %acc.vec = insertelement <vscale x 2 x float> poison, float %acc.next, i64 0
//
// Which eliminates the scalar -> vector -> scalar crossing during instruction
// selection.
bool RISCVCodeGenPrepare::visitIntrinsicInst(IntrinsicInst &I) {
#if SIFIVE_CUSTOMIZATION
  if (optimizeReduction(I))
    return true;
#endif // SIFIVE_CUSTOMIZATION

  if (expandVPStrideLoad(I))
    return true;

  if (I.getIntrinsicID() != Intrinsic::vector_reduce_fadd)
    return false;

  auto *PHI = dyn_cast<PHINode>(I.getOperand(0));
  if (!PHI || !PHI->hasOneUse() ||
      !llvm::is_contained(PHI->incoming_values(), &I))
    return false;

  Type *VecTy = I.getOperand(1)->getType();
  IRBuilder<> Builder(PHI);
  auto *VecPHI = Builder.CreatePHI(VecTy, PHI->getNumIncomingValues());

  for (auto *BB : PHI->blocks()) {
    Builder.SetInsertPoint(BB->getTerminator());
    Value *InsertElt = Builder.CreateInsertElement(
        VecTy, PHI->getIncomingValueForBlock(BB), (uint64_t)0);
    VecPHI->addIncoming(InsertElt, BB);
  }

  Builder.SetInsertPoint(&I);
  I.setOperand(0, Builder.CreateExtractElement(VecPHI, (uint64_t)0));

  PHI->eraseFromParent();

  return true;
}

#if SIFIVE_CUSTOMIZATION
// Manually expand unordered reductions with non-zero VL to get better regalloc
// and vsetvli than SelectionDAG can manage. We need to know the VL is non-zero
// so we don't need a passthru for the vredusum. We can also use the VL for
// vfmv.s.f if we know if it is non-zero. SelectionDAG will conservatively use
// a value of 1 to make sure it is non-zero.
bool RISCVCodeGenPrepare::optimizeReduction(IntrinsicInst &II) {
  Intrinsic::ID IID = II.getIntrinsicID();

  bool Changed = false;
  // Canonicalize the start value of vp.reduce.fadd from -0.0 to 0.0
  // if nsz flag is present.
  if (IID == Intrinsic::vp_reduce_fadd &&
      II.getFastMathFlags().noSignedZeros()) {
    using namespace PatternMatch;
    if (match(II.getOperand(0), m_NegZeroFP())) {
      II.setOperand(0, ConstantFP::getZero(II.getOperand(0)->getType()));
      Changed = true;
    }
  }

  // RISCV intrinsic ID.
  Intrinsic::ID RVIID;
  switch (IID) {
  // Floating point reductions.
  case Intrinsic::vp_reduce_fadd:
    RVIID = Intrinsic::riscv_vfredusum;
    break;
  case Intrinsic::vp_reduce_fmin:
    RVIID = Intrinsic::riscv_vfredmin;
    break;
  case Intrinsic::vp_reduce_fmax:
    RVIID = Intrinsic::riscv_vfredmax;
    break;
  // Integer reductions.
  case Intrinsic::vp_reduce_add:
    RVIID = Intrinsic::riscv_vredsum;
    break;
  case Intrinsic::vp_reduce_smax:
    RVIID = Intrinsic::riscv_vredmax;
    break;
  case Intrinsic::vp_reduce_umax:
    RVIID = Intrinsic::riscv_vredmaxu;
    break;
  case Intrinsic::vp_reduce_smin:
    RVIID = Intrinsic::riscv_vredmin;
    break;
  case Intrinsic::vp_reduce_umin:
    RVIID = Intrinsic::riscv_vredminu;
    break;
  case Intrinsic::vp_reduce_and:
    RVIID = Intrinsic::riscv_vredand;
    break;
  case Intrinsic::vp_reduce_or:
    RVIID = Intrinsic::riscv_vredor;
    break;
  case Intrinsic::vp_reduce_xor:
    RVIID = Intrinsic::riscv_vredxor;
    break;
  default:
    return Changed;
  }

  // Must be unordered for fadd reduction.
  if (IID == Intrinsic::vp_reduce_fadd && !II.getFastMathFlags().allowReassoc())
    return Changed;

  Value *Vec = II.getArgOperand(1);

  // FIXME: Only handle scalable vectors for now.
  auto *VecTy = dyn_cast<ScalableVectorType>(Vec->getType());
  if (!VecTy)
    return Changed;

  // TODO: Handle masks that aren't all ones?
  auto *ConstMask = dyn_cast<Constant>(II.getArgOperand(2));
  if (!ConstMask || !ConstMask->isAllOnesValue())
    return Changed;

  // Get the LMUL1 type and ensure that we didn't exceed LMUL=8.
  // FIXME: Support larger LMUL.
  Type *ScalarTy = VecTy->getElementType();
  unsigned SizeInBits = ScalarTy->getPrimitiveSizeInBits();
  const auto &TLI = *ST->getTargetLowering();
  if (!TLI.isTypeLegal(EVT::getEVT(VecTy)) ||
      // We only need Zvfhmin to make half a legal type, but Zvfhmin lacks
      // the operations we need here.
      (ScalarTy->isHalfTy() && !ST->hasVInstructionsF16()) ||
      // Boolean vector (i.e. mask) is a legal type, but it's not valid here.
      SizeInBits == 1)
    return Changed;
  ScalableVectorType *LMul1Ty =
      ScalableVectorType::get(ScalarTy, RISCV::RVVBitsPerBlock / SizeInBits);

  // Try to prove the VL is non-zero.
  Value *VL = II.getArgOperand(3);
  if (!isKnownNonZero(VL, {*DL, DT, nullptr, &II}))
    return Changed;

  // Found non-zero VL, let's rewrite.
  Value *Scalar = II.getArgOperand(0);

  IRBuilder<> Builder(&II);

  // Extend VL from i32 to XLen if needed.
  if (ST->is64Bit())
    VL = Builder.CreateZExt(VL, Builder.getInt64Ty());

  // If Scalar is already a neutral value, simply use the source vector as
  // the start value to avoid creating a new vfmv.s.f / vmv.s.x.
  bool IsScalarNeutral = false;
  if (const auto *C = dyn_cast<Constant>(Scalar))
    switch (IID) {
    case Intrinsic::vp_reduce_umax:
    case Intrinsic::vp_reduce_or:
      IsScalarNeutral = C->isZeroValue();
      break;
    case Intrinsic::vp_reduce_and:
    case Intrinsic::vp_reduce_umin:
      IsScalarNeutral = C->isAllOnesValue();
      break;
    case Intrinsic::vp_reduce_fmax: {
      FastMathFlags FMF = II.getFastMathFlags();
      const auto &APF = cast<ConstantFP>(C)->getValueAPF();
      if (FMF.noNaNs())
        IsScalarNeutral = FMF.noInfs() ? (APF.isLargest() && APF.isNegative())
                                       : APF.isNegInfinity();
      else
        // -QNaN
        IsScalarNeutral = APF.isNaN() && !APF.isSignaling() && APF.isNegative();
      break;
    }
    case Intrinsic::vp_reduce_smax:
      IsScalarNeutral = C->isMinSignedValue();
      break;
    case Intrinsic::vp_reduce_fmin: {
      FastMathFlags FMF = II.getFastMathFlags();
      const auto &APF = cast<ConstantFP>(C)->getValueAPF();
      if (FMF.noNaNs())
        IsScalarNeutral = FMF.noInfs() ? (APF.isLargest() && !APF.isNegative())
                                       : APF.isPosInfinity();
      else
        // +QNaN
        IsScalarNeutral =
            APF.isNaN() && !APF.isSignaling() && !APF.isNegative();
      break;
    }
    case Intrinsic::vp_reduce_smin:
      IsScalarNeutral = cast<ConstantInt>(C)->isMaxValue(/*IsSigned=*/true);
      break;
    }

  Value *ScalarInVec;
  if (IsScalarNeutral) {
    ScalarInVec = Vec;
  } else {
    // Move scalar into vector using vfmv.s.f / vmv.s.x. We need use VecTy to
    // get the correct LMUL in the vsetvli even though vfmv.s.f / vmv.s.x don't
    // care about LMUL.
    ScalarInVec = Builder.CreateIntrinsic(
        ScalarTy->isFloatingPointTy() ? Intrinsic::riscv_vfmv_s_f
                                      : Intrinsic::riscv_vmv_s_x,
        {VecTy, VL->getType()}, {PoisonValue::get(VecTy), Scalar, VL});
  }

  // Convert to LMUL1 to match what vfredusum wants.
  if (ElementCount::isKnownLT(LMul1Ty->getElementCount(),
                              VecTy->getElementCount()))
    ScalarInVec =
        Builder.CreateExtractVector(LMul1Ty, ScalarInVec, Builder.getInt64(0));
  else if (ElementCount::isKnownGT(LMul1Ty->getElementCount(),
                                   VecTy->getElementCount()))
    ScalarInVec = Builder.CreateInsertVector(LMul1Ty, PoisonValue::get(LMul1Ty),
                                             ScalarInVec, Builder.getInt64(0));

  // Do the reduction.
  Value *Reduce;
  // We only care about rounding mode for vp_reduce_fadd.
  if (IID == Intrinsic::vp_reduce_fadd)
    // The 7 here is dynamic rounding mode.
    Reduce =
        Builder.CreateIntrinsic(RVIID, {LMul1Ty, VecTy, VL->getType()},
                                {PoisonValue::get(LMul1Ty), Vec, ScalarInVec,
                                 ConstantInt::get(VL->getType(), 7), VL});
  else
    Reduce = Builder.CreateIntrinsic(
        RVIID, {LMul1Ty, VecTy, VL->getType()},
        {PoisonValue::get(LMul1Ty), Vec, ScalarInVec, VL});

  // Extract the scalar result to match the original intrinsic result type.
  Value *Res = Builder.CreateExtractElement(Reduce, (uint64_t)0);
  Res->takeName(&II);
  II.replaceAllUsesWith(Res);
  II.eraseFromParent();
  return true;
}
#endif // SIFIVE_CUSTOMIZATION

// Always expand zero strided loads so we match more .vx splat patterns, even if
// we have +optimized-zero-stride-loads. RISCVDAGToDAGISel::Select will convert
// it back to a strided load if it's optimized.
bool RISCVCodeGenPrepare::expandVPStrideLoad(IntrinsicInst &II) {
  Value *BasePtr, *VL;

  using namespace PatternMatch;
  if (!match(&II, m_Intrinsic<Intrinsic::experimental_vp_strided_load>(
                      m_Value(BasePtr), m_Zero(), m_AllOnes(), m_Value(VL))))
    return false;

  // If SEW>XLEN then a splat will get lowered as a zero strided load anyway, so
  // avoid expanding here.
  if (II.getType()->getScalarSizeInBits() > ST->getXLen())
    return false;

  if (!isKnownNonZero(VL, {*DL, DT, nullptr, &II}))
    return false;

  auto *VTy = cast<VectorType>(II.getType());

  IRBuilder<> Builder(&II);
  Type *STy = VTy->getElementType();
  Value *Val = Builder.CreateLoad(STy, BasePtr);
  Value *Res = Builder.CreateIntrinsic(Intrinsic::experimental_vp_splat, {VTy},
                                       {Val, II.getOperand(2), VL});

  II.replaceAllUsesWith(Res);
  II.eraseFromParent();
  return true;
}

bool RISCVCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<RISCVTargetMachine>();
  ST = &TM.getSubtarget<RISCVSubtarget>(F);

  DL = &F.getDataLayout();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree(); // SIFIVE

  bool MadeChange = false;
  for (auto &BB : F)
    for (Instruction &I : llvm::make_early_inc_range(BB))
      MadeChange |= visit(I);

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(RISCVCodeGenPrepare, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(RISCVCodeGenPrepare, DEBUG_TYPE, PASS_NAME, false, false)

char RISCVCodeGenPrepare::ID = 0;

FunctionPass *llvm::createRISCVCodeGenPreparePass() {
  return new RISCVCodeGenPrepare();
}
