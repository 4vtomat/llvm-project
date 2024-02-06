//===--- SiFive_ExpandVPReduction.cpp - Expand VPReduction intrinsics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR expansion for vp reductions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "llvm/IR/IntrinsicsRISCV.h"

#define DEBUG_TYPE "expand-vp-reduce"

using namespace llvm;

static void expandReductionMul(IntrinsicInst *II) {
  assert(II->getIntrinsicID() == Intrinsic::vp_reduce_mul);

  LLVMContext &C = II->getContext();
  Value *Start = II->getArgOperand(0);
  Value *Vec = II->getArgOperand(1);
  Value *VL = II->getArgOperand(3);
  Type *VecTy = Vec->getType();
  Type *MaskTy = VecTy->getWithNewBitWidth(1);
  Type *ScalarTy = II->getType();
  Type *i32Ty = Type::getInt32Ty(C);
  Type *XLenTy = II->getModule()->getDataLayout().getLargestLegalIntType(C);
  Value *True = ConstantInt::get(MaskTy, 1);

  BasicBlock *StartBB = II->getParent();
  IRBuilder<> IB(II);
  Value *VLNotZero = IB.CreateICmpNE(VL, ConstantInt::get(i32Ty, 0));
  Instruction *Term = SplitBlockAndInsertIfThen(VLNotZero, II, false);
  BasicBlock *ReductionGuardBB = Term->getParent();

  // Generate guard for reduction to deny VL = 1.
  IB.SetInsertPoint(Term);
  Value *Cond = IB.CreateICmpNE(VL, ConstantInt::get(i32Ty, 1));
  Term = SplitBlockAndInsertIfThen(Cond, Term, false);
  BasicBlock *PaddingOneGuardBB = Term->getParent();

  // Generate guard for padding one to deny power-of-2 VL.
  IB.SetInsertPoint(Term);
  Value *VLPop = IB.CreateUnaryIntrinsic(Intrinsic::ctpop, VL);
  if (auto *CVL = dyn_cast<ConstantInt>(VL))
    VLPop = ConstantInt::get(i32Ty, CVL->getValue().popcount());
  Value *IsNonPowerOf2 = IB.CreateICmpNE(VLPop, ConstantInt::get(i32Ty, 1));
  Term = SplitBlockAndInsertIfThen(IsNonPowerOf2, Term, false);
  BasicBlock *PadOneBB = Term->getParent();

  // NewVL = NextPowerOf2(VL)
  // NewVec[i] = Vec[i], if 0 <= i < VL.
  //           = 1, if VL <= i < NewVL.
  //           = undef, otherwise.
  IB.SetInsertPoint(Term);
  Value *VLCTLZ =
      IB.CreateIntrinsic(i32Ty, Intrinsic::ctlz, {VL, IB.getFalse()});
  Value *Offset = IB.CreateSub(ConstantInt::get(i32Ty, 32), VLCTLZ);
  Value *NewVL = IB.CreateShl(ConstantInt::get(i32Ty, 1), Offset);
  // Could we implement it by llvm IR?
  Value *Ones =
      IB.CreateIntrinsic(VecTy, Intrinsic::riscv_vmv_v_x,
                         {UndefValue::get(VecTy), ConstantInt::get(ScalarTy, 1),
                          IB.CreateZExt(NewVL, XLenTy)});
  Value *NewVec =
      IB.CreateIntrinsic(VecTy, Intrinsic::vp_merge, {True, Vec, Ones, VL});

  auto GetElseBB = [](BasicBlock *BB) {
    return cast<BranchInst>(BB->getTerminator())->getSuccessor(1);
  };

  // Loop to calculate reduce-mul and its result is a scalar vector.
  BasicBlock *PreHeader = GetElseBB(PaddingOneGuardBB);
  BasicBlock *PostLoopBB = GetElseBB(ReductionGuardBB);
  BasicBlock *LoopBody = BasicBlock::Create(PreHeader->getContext(), "loop",
                                            PreHeader->getParent(), PostLoopBB);

  IB.SetInsertPoint(PreHeader->getTerminator());
  PHINode *StartVec = IB.CreatePHI(VecTy, 2);
  StartVec->addIncoming(Vec, PaddingOneGuardBB);
  StartVec->addIncoming(NewVec, PadOneBB);
  PHINode *StartVL = IB.CreatePHI(i32Ty, 2);
  StartVL->addIncoming(VL, PaddingOneGuardBB);
  StartVL->addIncoming(NewVL, PadOneBB);
  IB.CreateBr(LoopBody);
  PreHeader->getTerminator()->eraseFromParent();

  // Generate loop body.
  IB.SetInsertPoint(LoopBody);
  PHINode *VecPhi = IB.CreatePHI(VecTy, 2, "vec");
  VecPhi->addIncoming(StartVec, PreHeader);
  PHINode *VLPhi = IB.CreatePHI(i32Ty, 2, "vl");
  VLPhi->addIncoming(StartVL, PreHeader);

  Value *HalfVL = IB.CreateLShr(VLPhi, ConstantInt::get(i32Ty, 1));
  VLPhi->addIncoming(HalfVL, LoopBody);

  Value *XLenHalfVL = IB.CreateZExt(HalfVL, XLenTy);
  // Could we implement it by llvm IR?
  Value *UpperVec =
      IB.CreateIntrinsic(VecTy, Intrinsic::riscv_vslidedown,
                         {UndefValue::get(VecTy), VecPhi, XLenHalfVL,
                          XLenHalfVL, ConstantInt::get(XLenTy, 1)});
  Value *NextVec = IB.CreateIntrinsic(VecTy, Intrinsic::vp_mul,
                                      {VecPhi, UpperVec, True, HalfVL});
  VecPhi->addIncoming(NextVec, LoopBody);

  IB.CreateCondBr(IB.CreateICmpEQ(HalfVL, ConstantInt::get(i32Ty, 1)),
                  PostLoopBB, LoopBody);

  IB.SetInsertPoint(&PostLoopBB->front());
  PHINode *ScalarVec = IB.CreatePHI(VecTy, 2);
  ScalarVec->addIncoming(Vec, ReductionGuardBB);
  ScalarVec->addIncoming(NextVec, LoopBody);
  Value *Red =
      IB.CreateExtractElement(ScalarVec, ConstantInt::get(ScalarTy, 0));
  Value *RedMulStart = IB.CreateMul(Red, Start);

  // Result
  BasicBlock *ResBB = GetElseBB(StartBB);
  IB.SetInsertPoint(&ResBB->front());
  PHINode *Res = IB.CreatePHI(ScalarTy, 2);
  Res->addIncoming(RedMulStart, PostLoopBB);
  Res->addIncoming(Start, StartBB);
  II->replaceAllUsesWith(Res);
  II->eraseFromParent();
}

static bool runImpl(Function &F) {
  SmallVector<IntrinsicInst *, 4> Replaces;
  for (auto &I : instructions(F)) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      // TODO: Also support vp.reduce.fmul.
      case Intrinsic::vp_reduce_mul: {
        Value *Mask = II->getArgOperand(2);
        auto *SC = dyn_cast_or_null<ConstantInt>(getSplatValue(Mask));
        // Only support true mask now.
        if (SC && SC->isAllOnesValue())
          Replaces.push_back(II);
        break;
      }
      }
    }
  }

  if (Replaces.empty())
    return false;

  for (IntrinsicInst *II : Replaces)
    expandReductionMul(II);

  return true;
}

namespace {
class ExpandVPReductionLegacyPass : public FunctionPass {
public:
  static char ID;

  ExpandVPReductionLegacyPass() : FunctionPass(ID) {
    initializeExpandVPReductionLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override { return runImpl(F); }
};
} // namespace

char ExpandVPReductionLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(ExpandVPReductionLegacyPass, "expand-vp-reduce",
                      "Expand vp.reduce functions", false, false)
INITIALIZE_PASS_END(ExpandVPReductionLegacyPass, "expand-vp-reduce",
                    "Expand vp.reduce functions", false, false)

FunctionPass *llvm::createExpandVPReductionPass() {
  return new ExpandVPReductionLegacyPass();
}
