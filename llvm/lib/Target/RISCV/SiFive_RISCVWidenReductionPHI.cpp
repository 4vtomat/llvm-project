//===--------- SiFive_RISCVWidenReductionPHI.cpp  -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass custom widens reduction phi.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-widen-reduction-phi"

// To widen reduction recipe without changing order, vectorizer widen loops to
// like,
// body:
//   %sum.phi = phi float [ %start, %pheader ], [ %sum, %body ]
//   %sum = call float @llvm.vp.reduce.fadd(%sum.phi, %vec, %m, %evl)
// This looks perfect in IR. But it is suboptimal for RVV, since RVV uses scalar
// vector for the start value of reduction. Hence we need vmv.s.f/vmv.f.s
// before/after vfredosum.
// This function tries to solve the issue by widening the phi-node like,
// preheader:
//   %sumvec.init = insertelement <VTy> poison, float %start, i32 0
// body:
//   %sumvec.phi = phi <VTy> [ %sumvec.init, %pheader ], [ %sumvec, %body ]
//   %start = extractelement <VTy> %sumvec, i32 0
//   %sum = call float @llvm.vp.reduce.fadd(%sum.phi, %vec, %m, %evl)
//   %sumvec = insertelement <VTy> poison, float %sum, i32 0

namespace {

class RISCVWidenReductionPHI : public LoopPass {
  Loop *CurLoop = nullptr;

public:
  static char ID; // Pass identification, replacement for typeid

  RISCVWidenReductionPHI() : LoopPass(ID) {}

  bool runOnLoop(Loop *L, LPPassManager &) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override {
    return "RISC-V widen reduction phis";
  }

private:
  bool isFaddReductionPhi(const PHINode *PN);
  void widenReductionPHI(PHINode *PN);
};

} // end anonymous namespace

char RISCVWidenReductionPHI::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVWidenReductionPHI, DEBUG_TYPE,
                      "RISC-V reduction lowering pass", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(RISCVWidenReductionPHI, DEBUG_TYPE,
                    "RISC-V reduction lowering pass", false, false)

Pass *llvm::createRISCVWidenReductionPHIPass() {
  return new RISCVWidenReductionPHI();
}

bool RISCVWidenReductionPHI::isFaddReductionPhi(const PHINode *PN) {
  if (PN->getNumIncomingValues() != 2)
    return false;

  if (!PN->hasOneUse())
    return false;

  Value *V = PN->getIncomingValueForBlock(CurLoop->getLoopLatch());
  auto *II = dyn_cast<IntrinsicInst>(V);
  if (!II || II->getIntrinsicID() != Intrinsic::vp_reduce_fadd)
    return false;

  // Only support UF=1 case. It's enough for our VLA implement now. But if we
  // need to upstream it, we need support unrolled loop.
  return II->getOperand(0) == PN;
}

void RISCVWidenReductionPHI::widenReductionPHI(PHINode *PN) {
  BasicBlock *PreHeader = CurLoop->getLoopPreheader();
  BasicBlock *Header = CurLoop->getHeader();
  BasicBlock *Latch = CurLoop->getLoopLatch();

  auto *Reduction = cast<IntrinsicInst>(PN->getIncomingValueForBlock(Latch));
  assert(Reduction->getIntrinsicID() == Intrinsic::vp_reduce_fadd);

  IRBuilder<> IB(PreHeader->getTerminator());
  IB.SetInsertPoint(PreHeader->getTerminator());
  Value *StartV = PN->getIncomingValueForBlock(PreHeader);
  Type *VecTy = Reduction->getOperand(1)->getType();
  Value *StartVec = IB.CreateInsertElement(VecTy, StartV, (uint64_t)0);

  IB.SetInsertPoint(&Header->front());
  PHINode *NewPHI = IB.CreatePHI(VecTy, 2);
  NewPHI->addIncoming(StartVec, PreHeader);

  IB.SetInsertPoint(Reduction);
  Value *NewStartV = IB.CreateExtractElement(NewPHI, (uint64_t)0);
  Reduction->setOperand(0, NewStartV);

  IB.SetInsertPoint(Reduction->getParent()->getTerminator());
  Value *SumVec = IB.CreateInsertElement(VecTy, Reduction, (uint64_t)0);

  NewPHI->addIncoming(SumVec, Latch);

  // Erase original phi.
  PN->eraseFromParent();
}

bool RISCVWidenReductionPHI::runOnLoop(Loop *L, LPPassManager &) {
  if (skipLoop(L))
    return false;

  if (!L->isLoopSimplifyForm())
    return false;

  CurLoop = L;

  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<RISCVTargetMachine>();
  Function &F = *L->getHeader()->getParent();
  const auto *ST = &TM.getSubtarget<RISCVSubtarget>(F);
  if (!ST->hasVInstructions())
    return false;

  SmallVector<PHINode *, 4> Targets;

  BasicBlock *Header = L->getHeader();
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(&*I);
    if (isFaddReductionPhi(PN))
      Targets.push_back(&*PN);
  }

  if (Targets.empty())
    return false;

  for (PHINode *PN : Targets)
    widenReductionPHI(PN);

  return true;
}
