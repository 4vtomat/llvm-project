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
#include "llvm/IR/Dominators.h" // SIFIVE
#include "llvm/IR/IRBuilder.h"  // SIFIVE
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsRISCV.h" // SIFIVE
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-codegenprepare"
#define PASS_NAME "RISC-V CodeGenPrepare"

STATISTIC(NumZExtToSExt, "Number of SExt instructions converted to ZExt");

namespace {

class RISCVCodeGenPrepare : public FunctionPass,
                            public InstVisitor<RISCVCodeGenPrepare, bool> {
  const DataLayout *DL;
  const RISCVSubtarget *ST;
  const DominatorTree *DT; // SIFIVE

public:
  static char ID;

  RISCVCodeGenPrepare() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<DominatorTreeWrapperPass>(); // SIFIVE
  }

  bool visitInstruction(Instruction &I) { return false; }
  bool visitZExtInst(ZExtInst &I);
  bool visitAnd(BinaryOperator &BO);
  bool visitIntrinsicInst(IntrinsicInst &II); // SIFIVE
};

} // end anonymous namespace

bool RISCVCodeGenPrepare::visitZExtInst(ZExtInst &ZExt) {
  if (!ST->is64Bit())
    return false;

  Value *Src = ZExt.getOperand(0);

  // We only care about ZExt from i32 to i64.
  if (!ZExt.getType()->isIntegerTy(64) || !Src->getType()->isIntegerTy(32))
    return false;

  // Look for an opportunity to replace (i64 (zext (i32 X))) with a sext if we
  // can determine that the sign bit of X is zero via a dominating condition.
  // This often occurs with widened induction variables.
  if (isImpliedByDomCondition(ICmpInst::ICMP_SGE, Src,
                              Constant::getNullValue(Src->getType()), &ZExt,
                              *DL).value_or(false)) {
    auto *SExt = new SExtInst(Src, ZExt.getType(), "", &ZExt);
    SExt->takeName(&ZExt);
    SExt->setDebugLoc(ZExt.getDebugLoc());

    ZExt.replaceAllUsesWith(SExt);
    ZExt.eraseFromParent();
    ++NumZExtToSExt;
    return true;
  }

  // Convert (zext (abs(i32 X, i1 1))) -> (sext (abs(i32 X, i1 1))). If abs of
  // INT_MIN is poison, the sign bit is zero.
  using namespace PatternMatch;
  if (match(Src, m_Intrinsic<Intrinsic::abs>(m_Value(), m_One()))) {
    auto *SExt = new SExtInst(Src, ZExt.getType(), "", &ZExt);
    SExt->takeName(&ZExt);
    SExt->setDebugLoc(ZExt.getDebugLoc());

    ZExt.replaceAllUsesWith(SExt);
    ZExt.eraseFromParent();
    ++NumZExtToSExt;
    return true;
  }

  return false;
}

// Try to optimize (i64 (and (zext/sext (i32 X), C1))) if C1 has bit 31 set,
// but bits 63:32 are zero. If we can prove that bit 31 of X is 0, we can fill
// the upper 32 bits with ones. A separate transform will turn (zext X) into
// (sext X) for the same condition.
bool RISCVCodeGenPrepare::visitAnd(BinaryOperator &BO) {
  if (!ST->is64Bit())
    return false;

  if (!BO.getType()->isIntegerTy(64))
    return false;

  // Left hand side should be sext or zext.
  Instruction *LHS = dyn_cast<Instruction>(BO.getOperand(0));
  if (!LHS || (!isa<SExtInst>(LHS) && !isa<ZExtInst>(LHS)))
    return false;

  Value *LHSSrc = LHS->getOperand(0);
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

  // If we can determine the sign bit of the input is 0, we can replace the
  // And mask constant.
  if (!isImpliedByDomCondition(ICmpInst::ICMP_SGE, LHSSrc,
                               Constant::getNullValue(LHSSrc->getType()),
                               LHS, *DL).value_or(false))
    return false;

  // Sign extend the constant and replace the And operand.
  C = SignExtend64<32>(C);
  BO.setOperand(1, ConstantInt::get(LHS->getType(), C));

  return true;
}

#if SIFIVE_CUSTOMIZATION
// Manually expand unordered reductions with non-zero VL to get better regalloc
// and vsetvli than SelectionDAG can manage. We need to know the VL is non-zero
// so we don't need a passthru for the vredusum. We can also use the VL for
// vfmv.s.f if we know if it is non-zero. SelectionDAG will conservatively use
// a value of 1 to make sure it is non-zero.
bool RISCVCodeGenPrepare::visitIntrinsicInst(IntrinsicInst &II) {
  Intrinsic::ID IID = II.getIntrinsicID();
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
    return false;
  }

  // Must be unordered for fadd reduction.
  if (IID == Intrinsic::vp_reduce_fadd && !II.getFastMathFlags().allowReassoc())
    return false;

  Value *Vec = II.getArgOperand(1);

  // FIXME: Only handle scalable vectors for now.
  auto *VecTy = dyn_cast<ScalableVectorType>(Vec->getType());
  if (!VecTy)
    return false;

  // TODO: Handle masks that aren't all ones?
  auto *ConstMask = dyn_cast<Constant>(II.getArgOperand(2));
  if (!ConstMask || !ConstMask->isAllOnesValue())
    return false;

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
    return false;
  ScalableVectorType *LMul1Ty =
      ScalableVectorType::get(ScalarTy, RISCV::RVVBitsPerBlock / SizeInBits);

  // Try to prove the VL is non-zero.
  Value *VL = II.getArgOperand(3);
  if (!isKnownNonZero(VL, *DL, 0, nullptr, &II, DT))
    return false;

  // Found non-zero VL, let's rewrite.
  Value *Scalar = II.getArgOperand(0);

  IRBuilder<> Builder(&II);

  // Extend VL from i32 to XLen if needed.
  if (ST->is64Bit())
    VL = Builder.CreateZExt(VL, Builder.getInt64Ty());

  // Move scalar into vector using vfmv.s.f / vmv.s.x. We need use VecTy to get
  // the correct LMUL in the vsetvli even though vfmv.s.f / vmv.s.x don't care
  // about LMUL.
  Value *ScalarInVec = Builder.CreateIntrinsic(
      ScalarTy->isFloatingPointTy() ? Intrinsic::riscv_vfmv_s_f
                                    : Intrinsic::riscv_vmv_s_x,
      {VecTy, VL->getType()}, {PoisonValue::get(VecTy), Scalar, VL});

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

bool RISCVCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<RISCVTargetMachine>();
  ST = &TM.getSubtarget<RISCVSubtarget>(F);

  DL = &F.getParent()->getDataLayout();

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
