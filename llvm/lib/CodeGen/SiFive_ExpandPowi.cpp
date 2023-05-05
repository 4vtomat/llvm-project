//===--- SiFive_ExpandPowi.cpp - Expand Powi intrinsics -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR expansion for powi/vp.powi. The expansion is based on
// compiler-rt/__powidf2.c.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "expand-powi"

using namespace llvm;

// The expansion is based on the c code of compiler-rt/__powidf2.c,
// const int recip = b < 0;
// double r = 1;
// while (1) {
//   if (b & 1)
//     r *= a;
//   b /= 2;
//   if (b == 0)
//     break;
//   a *= a;
// }
// return recip ? 1 / r : r;
//
// Expansion of llvm.powi still uses vp intrinsics here. It regards llvm.powi as
// llvm.vp.powi with true mask and maximum vl.
static void expandPowi(IntrinsicInst *II) {
  LLVMContext &C = II->getContext();
  Value *OrigBase = II->getOperand(0);
  Value *OrigExp = II->getOperand(1);
  VectorType *BaseTy = cast<VectorType>(OrigBase->getType());
  Type *ExpTy = OrigExp->getType();
  Type *CondTy = BaseTy->getWithNewType(Type::getInt1Ty(C));
  Value *True = ConstantInt::get(CondTy, 1);
  Value *Mask, *EVL;
  if (II->getIntrinsicID() == Intrinsic::vp_powi) {
    Mask = II->getOperand(2);
    EVL = II->getOperand(3);
  } else {
    assert(II->getIntrinsicID() == Intrinsic::powi);
    Mask = True;
    IRBuilder<> Builder(II);
    EVL = Builder.CreateElementCount(Type::getInt32Ty(C),
                                     BaseTy->getElementCount());
  }

  BasicBlock *PreLoopBB = II->getParent();
  BasicBlock *PostLoopBB = PreLoopBB->splitBasicBlock(II, "powi-post-loop");
  BasicBlock *LoopBody =
      BasicBlock::Create(PreLoopBB->getContext(), "powi-expansion-loop",
                         PreLoopBB->getParent(), PostLoopBB);

  IRBuilder<> Builder(PreLoopBB->getTerminator());
  Builder.CreateBr(LoopBody);
  PreLoopBB->getTerminator()->eraseFromParent();

  Builder.SetInsertPoint(LoopBody);
  // Create phi of base.
  PHINode *Base = Builder.CreatePHI(BaseTy, 2, "base");
  Base->addIncoming(OrigBase, PreLoopBB);
  // Create phi of exponent.
  PHINode *Exp = Builder.CreatePHI(ExpTy, 2, "exp");
  Exp->addIncoming(OrigExp, PreLoopBB);
  // Create phi of res.
  PHINode *Res = Builder.CreatePHI(BaseTy, 2, "res");
  Res->addIncoming(ConstantFP::get(BaseTy, 1.), PreLoopBB);
  // Res *= Base if Exp is odd.
  Value *Tmp = Builder.CreateIntrinsic(BaseTy, Intrinsic::vp_fmul,
                                       {Res, Base, True, EVL});
  Value *And1 = Builder.CreateAnd(Exp, ConstantInt::get(ExpTy, 1));
  Value *IsOdd = Builder.CreateICmpNE(And1, ConstantInt::get(ExpTy, 0));
  Value *NewRes = Builder.CreateSelect(IsOdd, Tmp, Res);
  Res->addIncoming(NewRes, LoopBody);
  // Update Exp.
  Value *NewExp = Builder.CreateLShr(Exp, ConstantInt::get(ExpTy, 1));
  Exp->addIncoming(NewExp, LoopBody);
  // Update Base.
  Value *NewBase = Builder.CreateIntrinsic(BaseTy, Intrinsic::vp_fmul,
                                           {Base, Base, True, EVL});
  Base->addIncoming(NewBase, LoopBody);
  // Check whether NewExp is zero.
  Builder.CreateCondBr(Builder.CreateICmpEQ(NewExp, ConstantInt::get(ExpTy, 0)),
                       PostLoopBB, LoopBody);

  Builder.SetInsertPoint(&PostLoopBB->front());
  // Use reciprocal if power is negative.
  Value *Recip =
      Builder.CreateIntrinsic(BaseTy, Intrinsic::vp_fdiv,
                              {ConstantFP::get(BaseTy, 1.), NewRes, Mask, EVL});
  Value *IsNegative =
      Builder.CreateICmpSLT(OrigExp, ConstantInt::get(ExpTy, 0));
  Value *Powi = Builder.CreateSelect(IsNegative, Recip, NewRes);
  II->replaceAllUsesWith(Powi);
  II->eraseFromParent();
}

static bool runImpl(Function &F) {
  SmallVector<IntrinsicInst *, 4> Replace;
  for (auto &I : instructions(F)) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      // TODO: Add cost model to select small fixed vectors llvm.powi.
      if (II->getIntrinsicID() == Intrinsic::vp_powi ||
          (II->getIntrinsicID() == Intrinsic::powi &&
           isa<ScalableVectorType>(II->getType())))
        Replace.push_back(II);
    }
  }

  if (Replace.empty())
    return false;

  for (IntrinsicInst *II : Replace)
    expandPowi(II);

  return true;
}

namespace {
class ExpandPowiLegacyPass : public FunctionPass {
public:
  static char ID;

  ExpandPowiLegacyPass() : FunctionPass(ID) {
    initializeExpandPowiLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override { return runImpl(F); }
};
} // namespace

char ExpandPowiLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(ExpandPowiLegacyPass, "expand-powi",
                      "Expand powi functions", false, false)
INITIALIZE_PASS_END(ExpandPowiLegacyPass, "expand-powi",
                    "Expand powi functions", false, false)

FunctionPass *llvm::createExpandPowiPass() {
  return new ExpandPowiLegacyPass();
}
