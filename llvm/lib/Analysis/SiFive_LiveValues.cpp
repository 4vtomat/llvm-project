//===- SiFive_LiveValues.cpp - Utility code ---------------------===//
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

#include "llvm/Analysis/SiFive_LiveValues.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/FormattedStream.h"

#include <cmath>

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
  if (PhiValues.find(BB) != PhiValues.end()) {
    LiveOut[BB] |= PhiValues[BB];
    NumOperations++;
  }

  // Collect LiveIn of BB and Add preds to work if changed
  if (statementTransferFunction(BB, NumOperations))
    for (auto PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
      W.push_back(*PI);

  return NumOperations;
}

void LiveValues::finalizeAnalysis(Function &F) {
  for (BasicBlock &BB : F)
    if (PhiValues.find(&BB) != PhiValues.end())
      for (auto SI = succ_begin(&BB), SE = succ_end(&BB); SI != SE; SI++)
        LiveIn[*SI] |= PhiValues[&BB];
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

// Dump live values before BB.
void LiveValues::emitBasicBlockStartAnnot(const BasicBlock *BB,
                                          formatted_raw_ostream &OS) {
  OS << "LiveIn(BB): ";
  if (LiveIn.find(BB) != LiveIn.end()) {
    for (auto Idx : LiveIn[BB]) {
      Value *V = BvIdxToValue[Idx];
      if (V->hasName())
        OS << "%" << V->getName() << ", ";
      else
        OS << *V << ", ";
    }
  }
  OS << "\n";
  OS << "LiveOut(BB): ";
  if (LiveOut.find(BB) != LiveOut.end()) {
    for (auto Idx : LiveOut[BB]) {
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
    for (auto Idx : InstrLiveIn[I]) {
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
    NewLiveIn.reset(ValueToBvIdx[I]);
    NumOperations++;

    // Add uses, unless a phi node then propagate incoming values.
    if (auto *PhiNode = dyn_cast<PHINode>(&*IT)) {
      for (size_t Idx = 0; Idx < PhiNode->getNumIncomingValues(); Idx++) {
        Value *Val = PhiNode->getIncomingValue(Idx);
        if (isa<Instruction, Argument>(Val)) {
          int ValIdx = ValueToBvIdx[Val];
          BasicBlock *IdxBB = PhiNode->getIncomingBlock(Idx);
          PhiValues[IdxBB].set(ValIdx);
          NumOperations++;
        }
      }
    } else {
      // use[n] U (out[n] - def[n])
      for (auto OI = I->op_begin(), OE = I->op_end(); OI != OE; ++OI)
        if (isa<Instruction, Argument>(*OI)) {
          NewLiveIn.set(ValueToBvIdx[*OI]);
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

void LiveValues::doDataFlowAnalysis(Function &F) {
  // Give a name to any Instruction/Block without hasName.
  if (EnableLivenessAnnotations)
    nameInstructions(F);

  // Initialize SparseBitVector accessors.
  int Idx = 0;
  for (auto AI = F.arg_begin(), AE = F.arg_end(); AI != AE; ++AI) {
    BvIdxToValue.push_back(&*AI);
    ValueToBvIdx[&*AI] = Idx++;
  }

  for (auto IT = inst_begin(F), IE = inst_end(F); IT != IE; ++IT) {
    BvIdxToValue.push_back(&*IT);
    ValueToBvIdx[&*IT] = Idx++;
  }

  // Do flow analysis for F.
  if (!analyzeFunction(F))
    return;

  if (EnableLivenessAnnotations)
    F.print(errs(), this);
}

void LiveValues::recalcDataFlowAnalysis(Function &F) {
  LiveIn.clear();
  LiveOut.clear();
  PhiValues.clear();
  BvIdxToValue.clear();
  ValueToBvIdx.clear();
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
  // Check whether the analysis or all analyses on functions have been preserved.
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

LiveValues LiveValuesAnalysis::run(Function &F, FunctionAnalysisManager &) {
  LiveValues LV;
  if (EnableValuePressureAnalysis)
    LV.doDataFlowAnalysis(F);

  return LV;
}

AnalysisKey LiveValuesAnalysis::Key;

#endif // SIFIVE_CUSTOMIZATION