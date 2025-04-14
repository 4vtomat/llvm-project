//===- SiFive_LoopReverse.cpp - Loop Reverse Pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Loop Reverse Pass. This pass is responsible
// for reversing loops with non-infinite computable trip counts that have no
// side effects or volatile instructions, and are innermost loops in
// simple normal form and LCSSA form.  Loops that qualify will have their
// Latch rewriten and any IV related data flow.
//
//===----------------------------------------------------------------------===//

#if SIFIVE_CUSTOMIZATION

#include "llvm/Transforms/Scalar/SiFive_LoopReverse.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"

#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "loop-reverse"

STATISTIC(NumReversed, "Number of loops reversed");

static cl::opt<bool> EnableLoopReverse(
    "loop-reverse-enable", cl::Hidden, cl::init(false),
    cl::desc("Reverse loops in simple normal form and which are lcssa"));

enum class LoopReverseResult {
  Unmodified,
  Modified,
};

/// Checks if it is safe to call InductionDescriptor::isInductionPHI for \p Phi,
/// and returns true if this Phi is an induction phi in the loop. When
/// isInductionPHI returns true, \p ID will be also be set by isInductionPHI.
static bool checkIsIndPhi(PHINode *Phi, Loop *L, ScalarEvolution *SE,
                          InductionDescriptor &ID) {
  if (!Phi)
    return false;
  if (!L->getLoopPreheader())
    return false;
  if (Phi->getParent() != L->getHeader())
    return false;
  return InductionDescriptor::isInductionPHI(Phi, L, SE, ID);
}

static Value *peekThroughExtTrunc(Value *Val) {
  if (auto *ZI = dyn_cast<ZExtInst>(Val))
    Val = ZI->getOperand(0);
  else if (auto *SI = dyn_cast<SExtInst>(Val))
    Val = SI->getOperand(0);
  else if (auto *TI = dyn_cast<TruncInst>(Val))
    Val = TI->getOperand(0);
  return Val;
}

static bool isOuterLoopRelated(Loop *OuterLoop, Value *InitVal,
                               ScalarEvolution *SE) {
  if (!OuterLoop)
    return false;

  auto *InitI = dyn_cast<Instruction>(InitVal);
  if (!InitI)
    return false;

  if (!OuterLoop->contains(InitI))
    return false;

  bool CanEval = false;
  if (auto *IV = OuterLoop->getInductionVariable(*SE))
    if (auto LB = Loop::LoopBounds::getBounds(*OuterLoop, *IV, *SE)) {
      CanEval = true;
      InitVal = peekThroughExtTrunc(InitVal);
      return (InitVal == IV || InitVal == &LB->getStepInst());
    }

  // Not enough evidence to make a decision.
  if (!CanEval)
    return true;

  return false;
}

static bool hasInductionEscapes(PHINode *IV, BinaryOperator *BinOp, Loop *L) {
  // Check for escaping IV based values used outside the loop.
  for (User *U : IV->users())
    if (!L->contains(cast<Instruction>(U)))
      return true;

  // Check uses of IVBinOp to make sure none escape the loop.
  for (User *U : BinOp->users())
    if (!L->contains(cast<Instruction>(U)))
      return true;

  return false;
}

static bool
isLoopCanonical(Loop *L, PHINode *IV, ScalarEvolution *SE,
                BinaryOperator *BinOp,
                SmallDenseMap<PHINode *, BinaryOperator *> &ReferenceMap) {
  if (auto LB = Loop::LoopBounds::getBounds(*L, *IV, *SE)) {
    if (LB->getDirection() == Loop::LoopBounds::Direction::Decreasing &&
        IV == L->getInductionVariable(*SE) && LB->getStepValue()) {
      if (isOuterLoopRelated(L->getParentLoop(), &LB->getInitialIVValue(), SE))
        return false;

      if (BinOp == &LB->getStepInst() &&
          BinOp->getOpcode() == BinaryOperator::Add &&
          !hasInductionEscapes(IV, BinOp, L) && !ReferenceMap[IV] &&
          isa<ConstantInt>(LB->getStepValue()))
        return true;
    }
  }

  return false;
}

static CmpInst::Predicate EvaluatePred(Value *Condition, ScalarEvolution *SE) {
  using namespace PatternMatch;
  CmpPredicate Pred;
  Value *LeftVal, *RightVal;
  if (match(Condition, m_ICmp(Pred, m_Value(LeftVal), m_Value(RightVal)))) {
    const SCEV *LeftSCEV = SE->getSCEV(LeftVal);
    const SCEV *RightSCEV = SE->getSCEV(RightVal);

    // Check if we have a condition with one AddRec and one non AddRec
    // expression. Normalize LeftSCEV to be the AddRec.
    if (!isa<SCEVAddRecExpr>(LeftSCEV) && isa<SCEVAddRecExpr>(RightSCEV)) {
      std::swap(LeftSCEV, RightSCEV);
      Pred = ICmpInst::getSwappedPredicate(Pred);
    }
  } else {
    // This is our default evaluation for non ICmp predicates.
    Pred = CmpInst::BAD_ICMP_PREDICATE;
  }
  return Pred;
}

static bool isLoopLegalForm(Loop *L, DominatorTree &DT, ScalarEvolution *SE,
                            SmallVectorImpl<BasicBlock *> &ExitingBlocks,
                            SmallVectorImpl<BasicBlock *> &ExitBlocks,
                            BasicBlock *Latch, LoopInfo &LI) {
  if (!L->isLCSSAForm(DT) || !L->isLoopSimplifyForm() || !L->isInnermost()) {
    LLVM_DEBUG(dbgs() << "LR: Loop is not in canonical form to Reverse.\n");
    return false;
  }

  // We need to be able to compute the loop trip count in order
  // to reverse the loop and reconfigure the bounds.
  const SCEV *ExitCount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(ExitCount)) {
    LLVM_DEBUG(dbgs() << "LR: This loop is not countable\n");
    return false;
  }

  // Loop must have a single exiting block, if not return false.
  if (!L->getExitingBlock()) {
    LLVM_DEBUG(dbgs() << "LR: The loop has multiple exiting blocks\n");
    return false;
  }

  // Loop should have a single backedge, if not return false.
  if (L->getNumBackEdges() != 1) {
    LLVM_DEBUG(dbgs() << "LR: The loop has multiple backedges\n");
    return false;
  }

  LoopBlocksRPO RPOT(L);
  RPOT.perform(&LI);

  // If the loop contains an irreducible cycle, it may loop infinitely,
  // leave it in the original form.
  if (containsIrreducibleCFG<const BasicBlock *>(RPOT, LI)) {
    LLVM_DEBUG(dbgs() << "LR: CFG is Irreducible.\n");
    return false;
  }

  return true;
}

static bool
canTranslateLoop(Loop *L, ScalarEvolution *SE, ICmpInst *LatchCmp,
                 BranchInst *LatchBr, BinaryOperator *BinOp, PHINode *IV,
                 SmallDenseMap<PHINode *, BinaryOperator *> &ReferenceMap,
                 bool &IsLowerBoundInclusive, bool &IsUpperBoundInclusive,
                 bool HasPostDecrement, bool &UsesBinOpInCmp) {
  // Tie IV, BinOp and Cmp together
  if (!isLoopCanonical(L, IV, SE, BinOp, ReferenceMap)) {
    LLVM_DEBUG(dbgs() << "LR: Loop in non translatable form\n");
    return false;
  }

  // The canonical test relates the IV to loop control/direction,
  // checks for IV/StepInst inclusion/exclusion in comparator logic.
  CmpInst::Predicate Pred = EvaluatePred(LatchBr->getCondition(), SE);
  if (Pred == CmpInst::BAD_ICMP_PREDICATE) {
    LLVM_DEBUG(dbgs() << "LR: Loop divergent predicate form\n");
    return false;
  }
  ReferenceMap[IV] = BinOp;
  IsLowerBoundInclusive = !CmpInst::isNonStrictPredicate(Pred);
  if (LatchCmp->getOperand(1) == BinOp || LatchCmp->getOperand(0) == BinOp) {
    IsLowerBoundInclusive = !HasPostDecrement;
    UsesBinOpInCmp = true;
  }
  IsUpperBoundInclusive = HasPostDecrement;
  return true;
}

static bool
isInstTransformLegal(Loop *L, BasicBlock *BB, ScalarEvolution *SE,
                     Instruction &I, DependenceInfo &DI, Value *&Ptr,
                     SmallVectorImpl<Instruction *> &MemoryInstructions) {
  if (auto *Phi = dyn_cast<PHINode>(&I)) {
    if (BB == L->getHeader()) {
      RecurrenceDescriptor Rdx;
      InductionDescriptor D;
      Type *PhiType = Phi->getType();
      if (RecurrenceDescriptor::isReductionPHI(Phi, L, Rdx)) {
        if (Rdx.getExactFPMathInst() != nullptr) {
          LLVM_DEBUG(dbgs() << "LR: Reduction cannot be reorderd.\n");
          return false;
        }
      } else if (Phi != L->getInductionVariable(*SE) &&
                 !PhiType->isPointerTy()) {
        // Pessimistically decide that expressions of this type are unsafe.
        LLVM_DEBUG(dbgs() << "LR: May have a loop recursive expr.\n");
        return false;
      }
      return true;
    }
  }

  if (I.mayThrow() || !I.willReturn()) {
    LLVM_DEBUG(dbgs() << "LR: Inst: " << I << " may throw or won't return.\n");
    return false;
  }

  if (I.isAtomic() || I.isVolatile()) {
    LLVM_DEBUG(dbgs() << "LR: Instruction is volatile or atomic."
                      << "\tCannot change order.\n");
    return false;
  }

  if (auto *CurLoad = dyn_cast<LoadInst>(&I))
    Ptr = CurLoad->getPointerOperand();
  else if (auto *CurStore = dyn_cast<StoreInst>(&I))
    Ptr = CurStore->getPointerOperand();

  if (I.mayReadOrWriteMemory()) {
    for (Instruction *MemI : MemoryInstructions) {
      if (auto D = DI.depends(&I, MemI)) {
        if (D->isFlow() || D->isAnti() || D->isOutput()) {
          LLVM_DEBUG(dbgs() << "LR: unsupported dependence detected\n");
          return false;
        }
        if (D->isInput()) {
          Value *MemPtr = nullptr;
          if (auto *MemLoad = dyn_cast<LoadInst>(MemI))
            MemPtr = MemLoad->getPointerOperand();

          if (auto *MemGEP = dyn_cast_or_null<GetElementPtrInst>(MemPtr))
            if (auto *CurGEP = dyn_cast_or_null<GetElementPtrInst>(Ptr))
              if (MemGEP->getPointerOperand() == CurGEP->getPointerOperand()) {
                // TODO: Handle multiple IV related accesses on the same
                //       memory var.
                LLVM_DEBUG(dbgs() << "LR: Related input dependence detected\n");
                return false;
              }
        }
      }
    }
    MemoryInstructions.push_back(&I);
  }
  return true;
}

/// Determines if a loop is reversable.
static bool
isLoopReversable(Loop *L, DominatorTree &DT, ScalarEvolution *SE,
                 SmallVectorImpl<BasicBlock *> &ExitingBlocks,
                 SmallVectorImpl<BasicBlock *> &ExitBlocks, BasicBlock *Latch,
                 BasicBlock *Preheader, DependenceInfo &DI, LoopInfo &LI,
                 SmallDenseMap<PHINode *, BinaryOperator *> &ReferenceMap,
                 SmallDenseMap<GetElementPtrInst *, Value *> &IndexMap,
                 bool &IsLowerBoundInclusive, bool &IsUpperBoundInclusive,
                 bool &HasPostDecrement, bool &UsesBinOpInCmp) {
  if (!isLoopLegalForm(L, DT, SE, ExitingBlocks, ExitBlocks, Latch, LI))
    return false;

  auto *LatchBr = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LatchBr || LatchBr->isUnconditional()) {
    LLVM_DEBUG(dbgs() << "LR: Unsupported loop latch branch.\n");
    return false;
  }

  auto *LatchCmp = dyn_cast<ICmpInst>(LatchBr->getCondition());
  if (!LatchCmp) {
    LLVM_DEBUG(dbgs() << "LR: Latch condition is not ICmpInst.\n");
    return false;
  }

  // Examine memory and legalize contents while building info.
  bool CanTranslate = false;
  SmallVector<Instruction *, 4> MemoryInstructions;
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      Value *Ptr = nullptr;
      if (!isInstTransformLegal(L, BB, SE, I, DI, Ptr, MemoryInstructions))
        return false;

      if (!Ptr)
        continue;

      if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        if (!GEP->isInBounds() || !L->contains(GEP))
          continue;

        for (Value *Index : GEP->indices()) {
          if (isa<ConstantInt>(Index) || isa<UndefValue>(Index))
            continue;

          // Pass through artifacts.
          bool KeepLooking = true;
          while (KeepLooking) {
            Value *OldIndex = Index;
            Index = peekThroughExtTrunc(Index);
            KeepLooking = (Index != OldIndex);
          }

          IndexMap[GEP] = Index;
          BinaryOperator *IVBinOp = nullptr;
          PHINode *CurIV = nullptr;
          if (auto *BinOp = dyn_cast<BinaryOperator>(Index)) {
            if (auto *IV = dyn_cast<PHINode>(BinOp->getOperand(0))) {
              if (LI.getLoopFor(IV->getParent()) != L)
                break;

              // Check if we visited this context before.
              if (BinOp == ReferenceMap[IV])
                break;

              HasPostDecrement = false;
              IVBinOp = BinOp;
              CurIV = IV;
            }
          } else if (auto *IV = dyn_cast<PHINode>(Index)) {
            if (LI.getLoopFor(IV->getParent()) != L)
              break;

            // Now match the index flow to an IV which has a negative step,
            // this is a down counted loop.
            InductionDescriptor ID;
            if (checkIsIndPhi(IV, L, SE, ID))
              if (auto *BinOp = ID.getInductionBinOp()) {
                // Check if we visited this context before.
                if (BinOp == ReferenceMap[IV])
                  break;

                HasPostDecrement = true;
                IVBinOp = BinOp;
                CurIV = IV;
              }
          }
          if (CurIV)
            CanTranslate = canTranslateLoop(
                L, SE, LatchCmp, LatchBr, IVBinOp, CurIV, ReferenceMap,
                IsLowerBoundInclusive, IsUpperBoundInclusive, HasPostDecrement,
                UsesBinOpInCmp);
        }
      }
    }
  }

  if (CanTranslate)
    LLVM_DEBUG(dbgs() << "LR: Found a LoopReverse Candidate.\n");

  return CanTranslate;
}

static bool updateBinOp(BinaryOperator *IVBinOp, bool IsLowerBoundInclusive,
                        bool HasPostDecrement, Value *InitIndVal,
                        Value *FinalIndVal, Value *StepBy, PHINode *IV) {
  auto EmplaceInitVal = [&](PHINode *IV, Value *StepBy, Value *NewInitIndVal,
                            unsigned InitIdx) {
    if (auto *InsertPt = dyn_cast<Instruction>(NewInitIndVal)) {
      NewInitIndVal = BinaryOperator::CreateAdd(NewInitIndVal, StepBy);
      Instruction *NewInitIndInst = cast<Instruction>(NewInitIndVal);
      NewInitIndInst->insertAfter(InsertPt);
    } else {
      BasicBlock *InitValBlock = IV->getIncomingBlock(InitIdx);
      IRBuilder<> IRB(&*InitValBlock->getFirstNonPHIIt());
      NewInitIndVal = IRB.CreateAdd(NewInitIndVal, StepBy);
    }
    return NewInitIndVal;
  };
  bool FinalIndIsZero = false;
  if (auto *FinalIndCst = dyn_cast<ConstantInt>(FinalIndVal)) {
    if (FinalIndCst->isNegative()) {
      LLVM_DEBUG(dbgs() << "LR: LB in non translatable form\n");
      return false;
    }
    FinalIndIsZero = FinalIndCst->isZero();
  } else if (IsLowerBoundInclusive) {
    // The FinalIndVal is variant, pessimistically assume zero.
    FinalIndIsZero = true;
  }

  Value *NewInitIndVal = FinalIndVal;
  // Now determine which phi edge contains the init value.
  unsigned InitIdx = (IV->getIncomingValue(1) == InitIndVal) ? 1 : 0;

  // Flip the value of StepBy.
  auto *CI = cast<ConstantInt>(StepBy);
  APInt ValA = CI->getValue();
  int64_t NewStepBy = ValA.getSExtValue() * -1;
  Value *NewStepByVal = ConstantInt::get(StepBy->getType(), NewStepBy);
  if (!HasPostDecrement && !FinalIndIsZero)
    NewInitIndVal = EmplaceInitVal(IV, StepBy, NewInitIndVal, InitIdx);
  else if (!IsLowerBoundInclusive)
    NewInitIndVal = EmplaceInitVal(IV, NewStepByVal, NewInitIndVal, InitIdx);
  IV->setIncomingValue(InitIdx, NewInitIndVal);
  IVBinOp->setOperand(1, NewStepByVal);
  return true;
}

static bool updateLatchCompare(BranchInst *LatchBr, BinaryOperator *IVBinOp,
                               Value *InitIndVal, PHINode *IV,
                               bool UsesBinOpInCmp, bool IsUpperBoundInclusive,
                               bool HasPostDecrement, ScalarEvolution &SE) {
  ICmpInst *LatchCmp = cast<ICmpInst>(LatchBr->getCondition());

  // Evaluate the Latch Compare.
  CmpInst::Predicate Pred = EvaluatePred(LatchCmp, &SE);

  // Check for possible unsupported case.
  assert(Pred != CmpInst::ICMP_NE && "Unsupported cmp case");

  // Flip the logic for reversing direction of the loop unless pred is eq.
  bool AllowInverse = (Pred != CmpInst::ICMP_EQ);
  if (AllowInverse) {
    Pred = ICmpInst::getInversePredicate(Pred);
    Pred = CmpInst::getStrictPredicate(Pred);
  }
  LatchCmp->setPredicate(Pred);
  Value *IndCmpOpnd = (UsesBinOpInCmp) ? cast<Value>(IVBinOp) : cast<Value>(IV);
  // Detect the IndCmpOpnd position in LatchCmp.
  bool IsRhs = (LatchCmp->getOperand(1) == IndCmpOpnd);
  unsigned IndIdx = (IsRhs) ? 1 : 0;
  if (UsesBinOpInCmp) {
    if (IsUpperBoundInclusive)
      LatchCmp->setOperand(IndIdx, IV);
  } else if (!HasPostDecrement && !IsUpperBoundInclusive) {
    LatchCmp->setOperand(IndIdx, IVBinOp);
  }
  unsigned BndIdx = (IsRhs) ? 0 : 1;
  LatchCmp->setOperand(BndIdx, InitIndVal);
  return true;
}

static bool updateLoopAddress(GetElementPtrInst *GEP, BinaryOperator *IVBinOp,
                              unsigned OpIdx) {
  // Update qualifying addresses with the non step arg of the BinOp
  bool Changed = false;
  Value *Index = GEP->getOperand(OpIdx);
  if (isa<ConstantInt>(Index))
    return Changed;

  // pass through artifacts
  bool KeepLooking = true;
  while (KeepLooking) {
    Value *OldIndex = Index;
    Index = peekThroughExtTrunc(Index);
    KeepLooking = (Index != OldIndex);
  }
  if (OpIdx != GEP->getNumOperands()) {
    Type *NewTy = IVBinOp->getOperand(0)->getType();
    Type *IndexTy = GEP->getOperand(OpIdx)->getType();
    // If the types are the not the same we will update
    // the reaching value via the IVBinOp users.
    if (NewTy == IndexTy) {
      GEP->setOperand(OpIdx, IVBinOp->getOperand(0));
      Changed = true;
    }
  }

  return Changed;
}

static bool doLoopUpdate(Loop *L, bool HasPostDecrement,
                         BinaryOperator *IVBinOp,
                         SmallDenseMap<GetElementPtrInst *, Value *> &IndexMap,
                         BranchInst *LatchBr, PHINode *IV, ScalarEvolution &SE,
                         bool IsLowerBoundInclusive, bool IsUpperBoundInclusive,
                         bool UsesBinOpInCmp) {
  bool Changed = false;
  auto LB = Loop::LoopBounds::getBounds(*L, *IV, SE);
  if (!LB)
    return Changed;

  Value *InitIndVal = &LB->getInitialIVValue();
  Value *FinalIndVal = &LB->getFinalIVValue();
  Value *StepBy = LB->getStepValue();
  SmallVector<Instruction *, 10> Worklist;
  Worklist.push_back(LatchBr);
  if (!HasPostDecrement) {
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        auto *GEP = dyn_cast<GetElementPtrInst>(&I);
        if (!GEP)
          continue;

        Value *IndexVal = IndexMap[GEP];
        if (!IndexVal)
          continue;

        auto *BinOp = dyn_cast<BinaryOperator>(IndexVal);
        if (!BinOp)
          continue;

        // TODO: Support other binop IV cases such as a different indecies
        //       for variable memory access (a[i], b[i+1], c[i+2], etc)
        if (BinOp == IVBinOp)
          Worklist.push_back(&I);
      }
    }

    // Update reaching uses of the value of IVBinOp.
    for (User *U : IVBinOp->users())
      if (auto *TI = dyn_cast<TruncInst>(U)) {
        if (!TI->hasOneUse())
          Worklist.push_back(TI);
      } else if (isa<GetElementPtrInst, PHINode, ICmpInst>(U)) {
        continue;
      } else {
        Worklist.push_back(cast<Instruction>(U));
      }

  }

  // The first entry has special state
  Worklist.push_back(IVBinOp);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();

    auto *BinOp = dyn_cast<BinaryOperator>(I);
    if (BinOp && BinOp == IVBinOp) {
      // If we can not modify the BinOp, we do not update the loop.
      if (!updateBinOp(BinOp, IsLowerBoundInclusive, HasPostDecrement,
                       InitIndVal, FinalIndVal, StepBy, IV))
        break;
      Changed |= true;
      continue;
    } else if (auto *Branch = dyn_cast<BranchInst>(I)) {
      Changed |=
          updateLatchCompare(Branch, IVBinOp, InitIndVal, IV, UsesBinOpInCmp,
                             IsUpperBoundInclusive, HasPostDecrement, SE);
      continue;
    }
    auto *GEP = dyn_cast<GetElementPtrInst>(I);
    for (unsigned Idx = 0; Idx < I->getNumOperands(); Idx++) {
      if (GEP && Idx == 0)
        continue;

      if (GEP) {
        Changed |= updateLoopAddress(GEP, IVBinOp, Idx);
      } else if (I->getOperand(Idx) == IVBinOp) {
        I->setOperand(Idx, IV);
        Changed |= true;
      }
    }
  }
  return Changed;
}

// Transform a downcounted loop candidate to an upcounted loop.
static LoopReverseResult
doReverseLoop(Loop *L, DominatorTree &DT, ScalarEvolution &SE, LoopInfo &LI,
              MemorySSA *MSSA, BasicBlock *Latch,
              SmallDenseMap<PHINode *, BinaryOperator *> &ReferenceMap,
              SmallDenseMap<GetElementPtrInst *, Value *> &IndexMap,
              bool IsLowerBoundInclusive, bool IsUpperBoundInclusive,
              bool HasPostDecrement, bool UsesBinOpInCmp,
              OptimizationRemarkEmitter &ORE) {
  bool Changed = false;
  auto *IV = L->getInductionVariable(SE);
  if (!IV)
    return LoopReverseResult::Unmodified;

  // We should only ever have one IV to rework with its connectivity.
  auto It = ReferenceMap.find(IV);
  if (It == ReferenceMap.end())
    return LoopReverseResult::Unmodified;

  BinaryOperator *IVBinOp = It->second;
  BranchInst *LatchBr = cast<BranchInst>(Latch->getTerminator());
  Changed = doLoopUpdate(L, HasPostDecrement, IVBinOp, IndexMap, LatchBr, IV,
                         SE, IsLowerBoundInclusive, IsUpperBoundInclusive,
                         UsesBinOpInCmp);

  return Changed ? LoopReverseResult::Modified : LoopReverseResult::Unmodified;
}

///
/// Detect down counting loops with single exits at the latch,
/// which meet all legalization criteria and transform successful
/// candidates to upcounting loops, possibly optimizing the IV
/// based on inclusion/exclusion of bounds.
///
static LoopReverseResult reverseLoop(Loop *L, DominatorTree &DT,
                                     ScalarEvolution &SE, LoopInfo &LI,
                                     MemorySSA *MSSA, DependenceInfo &DI,
                                     OptimizationRemarkEmitter &ORE) {
  auto Result = LoopReverseResult::Unmodified;
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *Preheader = L->getLoopPreheader();
  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  SmallVector<BasicBlock *, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);
  SmallDenseMap<PHINode *, BinaryOperator *> ReferenceMap;
  SmallDenseMap<GetElementPtrInst *, Value *> IndexMap;
  bool IsLowerBoundInclusive = false;
  bool IsUpperBoundInclusive = false;
  bool HasPostDecrement = true;
  bool UsesBinOpInCmp = false;
  // IV may be reranged based on entry criteria.
  // We need to frame inclusion/exclusion of LB and UB.
  if (isLoopReversable(L, DT, &SE, ExitingBlocks, ExitBlocks, Latch, Preheader,
                       DI, LI, ReferenceMap, IndexMap, IsLowerBoundInclusive,
                       IsUpperBoundInclusive, HasPostDecrement,
                       UsesBinOpInCmp)) {
    Result = doReverseLoop(L, DT, SE, LI, MSSA, Latch, ReferenceMap, IndexMap,
                           IsLowerBoundInclusive, IsUpperBoundInclusive,
                           HasPostDecrement, UsesBinOpInCmp, ORE);
    if (Result == LoopReverseResult::Modified)
      NumReversed++;
  }

  return Result;
}

PreservedAnalyses LoopReversePass::run(Loop &L, LoopAnalysisManager &AM,
                                       LoopStandardAnalysisResults &AR,
                                       LPMUpdater &Updater) {
  if (!EnableLoopReverse)
    return PreservedAnalyses::all();

  LLVM_DEBUG(dbgs() << "LR: Analyzing Loop for reversal: ");
  LLVM_DEBUG(L.dump());
  // For the new PM, we can't use OptimizationRemarkEmitter as an analysis
  // pass. Function analyses need to be preserved across loop transformations
  // but ORE cannot be preserved (see comment before the pass definition).
  Function *F = L.getHeader()->getParent();
  OptimizationRemarkEmitter ORE(F);
  DependenceInfo DI(F, &AR.AA, &AR.SE, &AR.LI);
  auto Result = reverseLoop(&L, AR.DT, AR.SE, AR.LI, AR.MSSA, DI, ORE);

  if (Result == LoopReverseResult::Unmodified)
    return PreservedAnalyses::all();

  auto PA = getLoopPassPreservedAnalyses();
  if (AR.MSSA)
    PA.preserve<MemorySSAAnalysis>();
  return PA;
}

#endif // SIFIVE_CUSTOMIZATION
