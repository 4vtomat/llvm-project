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
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Pass.h"
#include "RISCVTargetMachine.h"

#define DEBUG_TYPE "riscv-codegenprepare"
#define PASS_NAME "RISCV CodeGenPrepare"

using namespace llvm;

namespace {

class RISCVCodeGenPrepare : public FunctionPass {
public:
  static char ID;

  RISCVCodeGenPrepare() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override { return PASS_NAME; }

  bool runOnFunction(Function &F) override;

private:
};

} // end anonymous namespace

// If the result of a zext.w is used by a GEP in another basic block, duplicate
// the zext to enable add.uw or shXadd.uw.
static bool optimizeZExtWUses(Instruction *I) {
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

bool RISCVCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  auto &TM = TPC->getTM<RISCVTargetMachine>();
  const RISCVSubtarget *ST = TM.getSubtargetImpl(F);

  // TODO: Our only optimizations are for RV64 with Zba.
  if (!ST->is64Bit() || !ST->hasStdExtZba())
    return false;

  bool MadeChange = false;
  for (auto &BB : F) {
    for (auto II = BB.begin(), IE = BB.end(); II != IE; ) {
      Instruction *I = &*II++;
      if (isa<ZExtInst>(I))
        MadeChange |= optimizeZExtWUses(I);
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
