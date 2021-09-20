//===----- SiFive_RISCVCodeGenPrepare.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a RISCV specific version of CodeGenPrepare.
// It munges the code in the input function to better prepare it for
// SelectionDAG-based code generation. This works around limitations in it's
// basic-block-at-a-time approach.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "riscv-codegenprepare"
#define PASS_NAME "RISCV CodeGenPrepare"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

class RISCVCodeGenPrepare : public FunctionPass {
  const DataLayout *DL;
  const RISCVSubtarget *ST;

public:
  static char ID;

  RISCVCodeGenPrepare() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override { return PASS_NAME; }

  bool runOnFunction(Function &F) override;

private:
  bool optimizeZExt(ZExtInst *I);
  bool optimizeZExtWUses(ZExtInst *I);
  bool optimizeBinaryOperator(BinaryOperator *BO);
  bool optimizeICmp(ICmpInst *ICmp);
};

} // end anonymous namespace

// If the result of a zext.w is used by a GEP in another basic block, duplicate
// the zext to enable add.uw or shXadd.uw.
bool RISCVCodeGenPrepare::optimizeZExtWUses(ZExtInst *I) {
  if (!ST->hasStdExtZba())
    return false;

  BasicBlock *DefBB = I->getParent();

  Value *Src = I->getOperand(0);

  // Needs to be a zext.w.
  if (!Src->getType()->isIntegerTy(32) ||
      !I->getType()->isIntegerTy(64))
    return false;

  // Make sure all users are GEPs or left shifts by constant.
  // NOTE: This isn't strictly necessary, but it ensures we don't extend the
  // live range of Src without removing all non-local users of I.
  bool HasNonLocalUser = false;
  for (auto *U : I->users()) {
    auto *UserI = cast<Instruction>(U);

    if (!isa<GetElementPtrInst>(UserI) &&
        !(UserI->getOpcode() == Instruction::Shl &&
          isa<ConstantInt>(UserI->getOperand(1))))
      return false;

    if (UserI->getParent() != DefBB)
      HasNonLocalUser = true;
  }

  // If all users are local we don't need to do anything.
  if (!HasNonLocalUser)
    return false;

  DenseMap<BasicBlock *, ZExtInst *> InsertedZExts;

  bool MadeChange = false;
  for (auto UI = I->user_begin(), E = I->user_end(); UI != E; ) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    BasicBlock *UserBB = User->getParent();

    // If this user is in the same block as the zext, don't change the zext.
    if (UserBB == DefBB)
      continue;

    // If we have already inserted a zext into this block, use it.
    ZExtInst *&InsertedZExt = InsertedZExts[UserBB];

    if (!InsertedZExt) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      assert(InsertPt != UserBB->end());
      InsertedZExt = new ZExtInst(Src, I->getType(), "", &*InsertPt);
      // Propagate the debug info.
      InsertedZExt->setDebugLoc(I->getDebugLoc());
    }

    // Replace a use of the zext with a use of the new zext.
    TheUse = InsertedZExt;
    MadeChange = true;
  }

  // If the original zext has become dead, remove it.
  if (I->use_empty()) {
    I->eraseFromParent();
    MadeChange = true;
  }

  return MadeChange;
}

bool RISCVCodeGenPrepare::optimizeZExt(ZExtInst *ZExt) {
  if (!ST->is64Bit())
    return false;

  Value *Src = ZExt->getOperand(0);

  // We only care about ZExt from i32 to i64.
  if (!ZExt->getType()->isIntegerTy(64) || !Src->getType()->isIntegerTy(32))
    return false;

  // Look for an opportunity to replace (i64 (zext (i32 X))) with a sext if we
  // can determine that bit 31 of X is zero via a dominating condition. This
  // often occurs with widened induction variables.
  const DataLayout &DL = ZExt->getModule()->getDataLayout();
  if (isImpliedByDomCondition(ICmpInst::ICMP_SGE, Src,
                              Constant::getNullValue(Src->getType()), ZExt,
                              DL)) {
    IRBuilder<> Builder(ZExt);
    Value *SExt = Builder.CreateSExt(Src, ZExt->getType());
    SExt->takeName(ZExt);

    ZExt->replaceAllUsesWith(SExt);
    ZExt->eraseFromParent();
    return true;
  }

  return optimizeZExtWUses(ZExt);
}

// Try to optimize (i64 and (zext/sext (i32 X), C1)) if C1 has bit 31 is one,
// but bits 63:32 are zero. If we can prove that bit 31 of X is 0, we can fill
// the upper 32 bits with ones. A separate transform will turn (zext X) into
// (sext X) for the same condition.
bool RISCVCodeGenPrepare::optimizeBinaryOperator(BinaryOperator *BO) {
  if (!ST->is64Bit())
    return false;

  if (BO->getOpcode() != Instruction::And)
    return false;

  if (!BO->getType()->isIntegerTy(64))
    return false;

  // Left hand side should be sext or zext.
  Instruction *LHS = dyn_cast<Instruction>(BO->getOperand(0));
  if (!LHS || (LHS->getOpcode() != Instruction::SExt &&
               LHS->getOpcode() != Instruction::ZExt))
    return false;

  Value *LHSSrc = LHS->getOperand(0);
  if (!LHSSrc->getType()->isIntegerTy(32))
    return false;

  // Right hand side should be a constant.
  Value *RHS = BO->getOperand(1);

  auto *CI = dyn_cast<ConstantInt>(RHS);
  // Handle the case where constant hoisting may have hidden the constant.
  if (!CI && isa<BitCastInst>(RHS))
    CI = dyn_cast<ConstantInt>(cast<BitCastInst>(RHS)->getOperand(0));
  if (!CI)
    return false;
  uint64_t C = CI->getZExtValue();

  // Look for constants that fit in 32 bits but not simm12, and can be made
  // into simm12 by sign extending bit 31.
  if (!isUInt<32>(C) || isInt<12>(C) || !isInt<12>(SignExtend64(C, 32)))
    return false;

  // If we can determine the sign bit of the input is 0, we can replace the
  // And mask constant.
  const DataLayout &DL = BO->getModule()->getDataLayout();
  if (!isImpliedByDomCondition(ICmpInst::ICMP_SGE, LHSSrc,
                               Constant::getNullValue(LHSSrc->getType()), LHS,
                               DL))
    return false;

  // Sign extend the constant and create a new And.
  C = SignExtend64(C, 32);
  IRBuilder<> Builder(BO);
  Value *NewBO = Builder.CreateAnd(LHS, ConstantInt::get(LHS->getType(), C));
  NewBO->takeName(BO);

  // Remove the old And.
  BO->replaceAllUsesWith(NewBO);
  BO->eraseFromParent();

  // Erase any bitcasts of constants we made dead.
  if (auto *RHSI = dyn_cast<Instruction>(RHS))
    if (RHSI->use_empty())
      RHSI->eraseFromParent();

  return true;
}

bool RISCVCodeGenPrepare::optimizeICmp(ICmpInst *ICmp) {
  if (ST->hasStdExtZbb())
    return false;

  auto *BO = dyn_cast<BinaryOperator>(ICmp->getOperand(0));
  if (!BO)
    return false;

  // Fold (icmp sgt (A + 1), Op1) -> (icmp sge A, Op1) if the add won't wrap.
  // InstCombine normally does this, but it is disabled if the add is part of a
  // min pattern. Without Zbb, the min will be turned into control flow so it
  // is better to separate the add from the cmp so we can sink it.
  if (ICmp->getPredicate() == ICmpInst::ICMP_SGT &&
      BO->getOpcode() == Instruction::Add && BO->hasNoSignedWrap() &&
      match(BO->getOperand(1), m_One())) {
    IRBuilder<> Builder(ICmp);
    Value *NewICmp = Builder.CreateICmpSGE(BO->getOperand(0), ICmp->getOperand(1));
    NewICmp->takeName(ICmp);
    ICmp->replaceAllUsesWith(NewICmp);
    ICmp->eraseFromParent();
    return true;
  }

  return false;
}

bool RISCVCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  auto &TM = TPC->getTM<RISCVTargetMachine>();
  ST = TM.getSubtargetImpl(F);

  DL = &F.getParent()->getDataLayout();

  bool MadeChange = false;
  for (auto &BB : F) {
    for (auto II = BB.begin(), IE = BB.end(); II != IE; ) {
      Instruction *I = &*II++;
      if (auto *ZExt = dyn_cast<ZExtInst>(I))
        MadeChange |= optimizeZExt(ZExt);
      else if (auto *BO = dyn_cast<BinaryOperator>(I))
        MadeChange |= optimizeBinaryOperator(BO);
      else if (auto *ICmp = dyn_cast<ICmpInst>(I))
        MadeChange |= optimizeICmp(ICmp);
    }
  }

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(RISCVCodeGenPrepare, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_END(RISCVCodeGenPrepare, DEBUG_TYPE, PASS_NAME, false, false)

char RISCVCodeGenPrepare::ID = 0;

FunctionPass *llvm::createRISCVCodeGenPreparePass() {
  return new RISCVCodeGenPrepare();
}
