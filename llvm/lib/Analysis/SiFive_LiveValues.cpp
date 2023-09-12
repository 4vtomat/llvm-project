//===- SiFive_LiveValues.cpp - Utility code -------------------------------===//
//
// Copyright (c) 2023 SiFive, Inc. -- Proprietary and Confidential All
// Rights Reserved.
//
// NOTICE: All information contained herein is, and remains the property of
// SiFive, Inc. The intellectual and technical concepts contained herein are
// proprietary to SiFive, Inc. and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// This work may not be copied, modified, re-published, uploaded, executed, or
// distributed in any way, in any medium, whether in whole or in part, without
// prior written permission from SiFive, Inc.  The copyright notice above does
// not evidence any actual or intended publication or disclosure of this source
// code, which includes information that is confidential and/or proprietary,
// and is a trade secret, of SiFive, Inc.
//
//===----------------------------------------------------------------------===//
//
// This file implements the calculating of Live Value information across
// a function and makes this information available to the caller as a utility.
//
//===----------------------------------------------------------------------===//

#if SIFIVE_CUSTOMIZATION

#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/SiFive_LiveValues.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/FormattedStream.h"

#include <cmath>
#include <type_traits>

using namespace llvm;

static cl::opt<bool>
    EnableLivenessAnnotations("liveness-annotations-enable", cl::Hidden,
                              cl::init(false),
                              cl::desc("Emit annotations for liveness in IR"));

static cl::opt<bool>
    EnableValuePressureAnalysis("value-liveness-enable", cl::Hidden,
                                cl::init(false),
                                cl::desc("Analysis of Value Liveness"));

static cl::opt<uint32_t> ThresholdKnee(
    "value-liveness-threshold-knee", cl::Hidden, cl::init(5000),
    cl::desc("Domain size(n) where we start reducing below n^2 calculations "
             "to guard compile time(default = 5000)"));

#define DEBUG_TYPE "live-values"

// Do a Backwards Analysis of the CFG to build liveness.
size_t LiveValues::doAnalysis(Workqueue &W) {
  BasicBlock *BB = *W.begin();
  W.pop_front();
  size_t NumOperations = 0;

  // Map LiveOut of this block to LiveIn of its successors.
  int NumSuccs = 0;
  for (auto SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI) {
    NumSuccs++;
    if (SI == succ_begin(BB))
      LiveOut[BB] = LiveIn[*SI];
    else
      LiveOut[BB] |= LiveIn[*SI];
  }
  NumOperations += NumSuccs;

  // For incoming values of a PHINode, map the values as LiveOut of BB.
  auto PIT = PhiValues.find(BB);
  if (PIT != PhiValues.end()) {
    LiveOut[BB] |= PIT->second;
    NumOperations++;
  }

  // Collect LiveIn of BB and Add preds to work if changed
  if (statementTransferFunction(BB, NumOperations))
    for (auto PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
      W.push_back(*PI);

  return NumOperations;
}

void LiveValues::finalizeAnalysis(Function &F) {
  for (BasicBlock &BB : F) {
    auto PIT = PhiValues.find(&BB);
    if (PIT != PhiValues.end())
      for (auto SI = succ_begin(&BB), SE = succ_end(&BB); SI != SE; SI++)
        LiveIn[*SI] |= PIT->second;
  }
}

// Configure compute threshold to limit how much compile time we use.
size_t LiveValues::configureThreshold() {
  double OperationsCalc = (double)getLivenessPoolSize();
  double Factor = log2(OperationsCalc);
  OperationsCalc = (OperationsCalc < ThresholdKnee)
                       ? (OperationsCalc * OperationsCalc)
                       : (OperationsCalc * OperationsCalc) / Factor;
  return (size_t)OperationsCalc;
}

bool LiveValues::analyzeFunction(Function &F) {
  Workqueue W;

  unsigned NumBlocks = 0;
  for (BasicBlock &BB : F) {
    NumBlocks++;
    if (succ_empty(&BB))
      W.push_back(&BB);
  }

  LLVM_DEBUG(dbgs() << "LivenessPoolSize = " << getLivenessPoolSize()
                    << "NumBlocks = " << NumBlocks << "\n");

  // Setup compute threshold to limit compile time.
  size_t TotalOperations = 0;
  auto OperationsThreshold = configureThreshold();

  // Iterate while there is work.
  while (!W.empty()) {
    // Check if computing live ranges for this CFG
    // is trending above the Operations Threshold
    if (TotalOperations >= OperationsThreshold)
      break;

    TotalOperations += doAnalysis(W);
  }

  // Successful analysis occurs when the work queue is empty.
  if (W.empty()) {
    finalizeAnalysis(F);
    LiveValuesAvailable = true;
    LLVM_DEBUG(dbgs() << "VPA: Success(" << TotalOperations
                      << " operations), threshold(" << OperationsThreshold
                      << ")\n");
  } else {
    LLVM_DEBUG(dbgs() << "VPA: Fail(" << TotalOperations
                      << " operations), threshold(" << OperationsThreshold
                      << ")\n");
  }

  return LiveValuesAvailable;
}

// Dump ValueLiveIntervals before F.
void LiveValues::emitFunctionAnnot(const Function *F,
                                   formatted_raw_ostream &OS) {
  int N = getLivenessPoolSize();
  for (int I = 0; I < N; ++I) {
    Value *V = BvIdxToValue[I];
    OS << LIs[V] << "\n";
  }
}

// Dump live values before BB.
void LiveValues::emitBasicBlockStartAnnot(const BasicBlock *BB,
                                          formatted_raw_ostream &OS) {
  OS << "LiveIn(BB): ";
  auto LiveInIt = LiveIn.find(BB);
  if (LiveInIt != LiveIn.end()) {
    for (size_t Idx : LiveInIt->second) {
      Value *V = BvIdxToValue[Idx];
      if (V->hasName())
        OS << "%" << V->getName() << ", ";
      else
        OS << *V << ", ";
    }
  }
  OS << "\n";
  OS << "LiveOut(BB): ";
  auto LiveOutIt = LiveOut.find(BB);
  if (LiveOutIt != LiveOut.end()) {
    for (size_t Idx : LiveOutIt->second) {
      Value *V = BvIdxToValue[Idx];
      if (V->hasName())
        OS << "%" << V->getName() << ", ";
      else
        OS << *V << ", ";
    }
  }
  OS << "\n";
}

// Dump live values before I.
void LiveValues::emitInstructionAnnot(const Instruction *I,
                                      formatted_raw_ostream &OS) {
  OS << "Live(I): ";
  if (!isa<PHINode>(I)) {
    for (size_t Idx : InstrLiveIn[I]) {
      Value *V = BvIdxToValue[Idx];
      if (V->hasName())
        OS << "%" << V->getName() << ", ";
      else
        OS << *V << ", ";
    }
  }
  OS << "\n";
}

// newLive[n] = use[n] U (out[n] - def[n])
bool LiveValues::statementTransferFunction(BasicBlock *BB,
                                           size_t &NumOperations) {
  SparseBitVector<> NewLiveIn = LiveOut[BB];
  bool Changed = false;

  // Bottom up walk of instructions.
  for (auto IT = BB->rbegin(); IT != BB->rend(); IT++) {
    Instruction *I = &*IT;

    // Collect (out[n] - def[n])
    NewLiveIn.reset(Indices[I].getIndex());
    NumOperations++;

    // Add uses, unless a phi node then propagate incoming values.
    if (auto *PhiNode = dyn_cast<PHINode>(&*IT)) {
      for (size_t Idx = 0; Idx < PhiNode->getNumIncomingValues(); Idx++) {
        Value *Val = PhiNode->getIncomingValue(Idx);
        if (isa<Instruction, Argument>(Val)) {
          int ValIdx = Indices[Val].getIndex();
          BasicBlock *IdxBB = PhiNode->getIncomingBlock(Idx);
          PhiValues[IdxBB].set(ValIdx);
          NumOperations++;
        }
      }
    } else {
      // use[n] U (out[n] - def[n])
      for (auto OI = I->op_begin(), OE = I->op_end(); OI != OE; ++OI)
        if (isa<Instruction, Argument>(*OI)) {
          NewLiveIn.set(Indices[*OI].getIndex());
          NumOperations++;
        }
    }

    InstrLiveIn[I] = NewLiveIn;
    NumOperations++;
  }

  if (NewLiveIn != LiveIn[BB]) {
    LiveIn[BB] = NewLiveIn;
    NumOperations++;
    Changed = true;
  }

  return Changed;
}

void LiveValues::createFullSegment(Value *Start, unsigned SlotStart,
                                   Value *End, unsigned SlotEnd, Value *V) {
  ValueSlotIndex FirstIndex(&Indices[Start], SlotStart);
  ValueSlotIndex LastIndex(&Indices[End], SlotEnd);
  ValueSlotInfo *VNI = *LIs[V].vni_begin();
  LIs[V].addSegment(ValueLiveInterval::Segment(FirstIndex, LastIndex, VNI));
}

void LiveValues::createSegmentStart(Value *Start, unsigned Slot,
                                    Value *V, ValueSlotIndex &StartIndex) {
  ValueSlotIndex DefIndex(&Indices[Start], Slot);
  StartIndex = DefIndex;
  LIs[V].createDeadDef(DefIndex, getVSInfoAllocator());
}

void LiveValues::endExistingSegment(Value *End, unsigned Slot,
                                    Value *V, ValueSlotIndex &DefIndex) {
  ValueSlotIndex EndIndex(&Indices[End], Slot);
  LIs[V].extendInBlock(DefIndex, EndIndex);
}

void LiveValues::constructLiveIntervalSegments(Value *V, Function &F) {
  if (V->getNumUses() == 0)
    return;

  BasicBlock *BB = nullptr;
  if (auto *I = dyn_cast<Instruction>(V))
    BB = I->getParent();
  else if (isa<Argument>(V))
    BB = &F.getEntryBlock();
  else
    return;

  // Build a process list of non ephermal values
  SmallVector<Instruction *, 10> Worklist;
  for (User *U : V->users()) {
    Instruction *UseI = cast<Instruction>(U);
    if (EphValues.count(UseI))
      continue;

    Worklist.push_back(UseI);
  }

  if (Worklist.empty())
    return;

  int DefIdx = Indices[V].getIndex();
  DenseMap<const BasicBlock *, ValueSlotIndex> CurIndexMap;
  unsigned Slot = (isa<PHINode, Argument>(V))
                      ? ValueSlotIndex::Slot::Slot_Block
                      : ValueSlotIndex::Slot::Slot_Register;
  createSegmentStart(V, Slot, V, CurIndexMap[BB]);

  // Extend the Def if it's live out otherwise we defer to use processing.
  if (LiveOut[BB].test(DefIdx)) {
    // Fill in a segment from the DefIndex to the EndIdx
    Instruction *I = &*BB->rbegin();
    endExistingSegment(I, ValueSlotIndex::Slot::Slot_Register,
                       V, CurIndexMap[BB]);
  }

  // Add segments as needed for uses of V.
  SmallPtrSet<BasicBlock *, 4> Visited;
  while (!Worklist.empty()) {
    Instruction *UseI = Worklist.pop_back_val();
    BasicBlock *UseBB = UseI->getParent();
    // This implies that the definition is not BB local.
    if (UseBB != BB) {
      // All the blocks that have DefIdx as LiveIn and LiveOut
      // will be processed as pass through segements later.
      if (!LiveOut[UseBB].test(DefIdx)) {
        // If this is the first time we have seen UseBB.
        if (Visited.insert(UseBB).second) {
          Instruction *FirstI = &*UseBB->begin();
          createSegmentStart(FirstI, ValueSlotIndex::Slot::Slot_Block,
                             V, CurIndexMap[UseBB]);
        }
      } else {
        // These will be handled with pass through values Segments.
        continue;
      }
    }
    endExistingSegment(UseI, ValueSlotIndex::Slot::Slot_Register,
                       V, CurIndexMap[UseBB]);
  }
}

void LiveValues::extendPassThroughLiveIntervalSegments(Function &F) {
  // Iterate the CFG and for each block where a Value is
  // both LiveIn and LiveOut, add a segment for it.
  BasicBlock *EntryBB = &F.getEntryBlock();
  for (BasicBlock &BB : F) {
    BasicBlock *CurBB = &BB;
    // Skip the entry, we processed Arguments already.
    if (EntryBB == CurBB)
      continue;

    auto LiveInIt = LiveIn.find(CurBB);
    if (LiveInIt == LiveIn.end())
      continue;

    auto LiveOutIt = LiveOut.find(CurBB);
    if (LiveOutIt == LiveOut.end())
      continue;

    // TODO: Extend support for detecting this scenario in
    //       phi based ptr/fp updates in loops.
    // Backedge based PHINodes can have holes from the phi to
    // its updater for the phi input values.  These are technically
    // backwards segments that have forward segment layout.
    for (size_t Idx : LiveInIt->second)
      if (LiveOut[CurBB].test(Idx)) {
        Value *V = BvIdxToValue[Idx];
        if (auto *BinOp = dyn_cast<BinaryOperator>(V)) {
          bool SkipToNextIdx = false;
          // Is V not a pass through value?
          for (User *U : V->users())
            if (auto *CurPhi = dyn_cast<PHINode>(U))
              if (CurPhi->getParent() == CurBB) {
                SkipToNextIdx |=
                    (any_of(BinOp->operands(), [=](const Value *Op) {
                      return (Op == CurPhi);
                    }));
                if (SkipToNextIdx)
                  break;
              }

          if (SkipToNextIdx)
            continue;
        }

        // Construct the pass through Segment for V.
        Instruction *FirstI = &*CurBB->begin();
        Instruction *LastI = &*CurBB->rbegin();
        createFullSegment(FirstI, ValueSlotIndex::Slot::Slot_Block,
                          LastI, ValueSlotIndex::Slot::Slot_Block, V);
      }
  }
}

void LiveValues::doDataFlowAnalysis(Function &F) {
  VSInfoAllocator.Reset();
  EphValues.clear();
  CodeMetrics::collectEphemeralValues(&F, AC, EphValues);

  // Give a name to any Instruction/Block without hasName.
  if (EnableLivenessAnnotations)
    nameInstructions(F);

  // Initialize SparseBitVector accessors.
  int Idx = 0;
  for (auto &ArgI : F.args()) {
    BvIdxToValue.push_back(&ArgI);
    LIs[&ArgI].setValue(&ArgI);
    Indices[&ArgI].setVal(&ArgI);
    Indices[&ArgI].setIndex(Idx++);
  }

  for (auto IT = inst_begin(F), IE = inst_end(F); IT != IE; ++IT) {
    Instruction *I = &*IT;
    BvIdxToValue.push_back(I);
    LIs[I].setValue(I);
    Indices[I].setVal(I);
    Indices[I].setIndex(Idx++);
  }

  // Do flow analysis for F.
  if (!analyzeFunction(F))
    return;

  for (int I = 0; I < Idx; ++I) {
    Value *V = BvIdxToValue[I];
    if (!V)
      continue;

    constructLiveIntervalSegments(V, F);
  }

  extendPassThroughLiveIntervalSegments(F);

  if (EnableLivenessAnnotations)
    F.print(errs(), this);
}

void LiveValues::recalcDataFlowAnalysis(Function &F) {
  LiveIn.clear();
  LiveOut.clear();
  PhiValues.clear();
  BvIdxToValue.clear();
  InstrLiveIn.clear();

  doDataFlowAnalysis(F);
}

// Borrowed from instnamer to make the annotated dumps nicer.
void LiveValues::nameInstructions(Function &F) {
  for (auto &Arg : F.args()) {
    if (!Arg.hasName())
      Arg.setName("arg");
  }

  for (BasicBlock &BB : F) {
    if (!BB.hasName())
      BB.setName("bb");

    for (Instruction &I : BB) {
      if (!I.hasName() && !I.getType()->isVoidTy())
        I.setName("i");
    }
  }
}

bool LiveValues::invalidate(Function &F, const PreservedAnalyses &PA,
                            FunctionAnalysisManager::Invalidator &) {
  // Check whether the analysis or all analyses on functions have been
  // preserved.
  auto PAC = PA.getChecker<LiveValuesAnalysis>();
  return !(PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Function>>());
}

//===----------------------------------------------------------------------===//
//  LiveValuesAnalysis and related pass implementations
//===----------------------------------------------------------------------===//
//
// This implements the LiveValuesAnalysis which is used with the new pass
// manager. It also implements some methods from utility passes.
//
//===----------------------------------------------------------------------===//

LiveValues LiveValuesAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  LiveValues LV;
  if (EnableValuePressureAnalysis) {
    LV.setAssumptionCache(&AM.getResult<AssumptionAnalysis>(F));
    LV.doDataFlowAnalysis(F);
  }

  return LV;
}

AnalysisKey LiveValuesAnalysis::Key;

#endif // SIFIVE_CUSTOMIZATION
