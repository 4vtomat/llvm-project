//===- SiFive_LiveValues.h --- utility code -------------------------------===//
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
// This file provides IR based Live Value Analysis an a utility.
//
//===----------------------------------------------------------------------===//

#if SIFIVE_CUSTOMIZATION

#ifndef LLVM_ANALYSIS_LIVEVALUE_H
#define LLVM_ANALYSIS_LIVEVALUE_H

#include "llvm/ADT/SparseBitVector.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>

#include <list>

namespace llvm {

using Workqueue = std::list<BasicBlock *>;

class LiveValues : public AssemblyAnnotationWriter {
private:
  /// Liveness sets for Basic Blocks.
  DenseMap<const BasicBlock *, SparseBitVector<>> LiveIn;
  DenseMap<const BasicBlock *, SparseBitVector<>> LiveOut;
  DenseMap<const BasicBlock *, SparseBitVector<>> PhiValues;

  /// Map Values to their Instr SparseBitVector.
  SmallVector<Value *> BvIdxToValue;
  /// Map Values (args and variables) to their SparseBitVector index.
  DenseMap<Value *, int> ValueToBvIdx;
  /// Instruction In sets.
  DenseMap<const Instruction *, SparseBitVector<>> InstrLiveIn;
  bool LiveValuesAvailable = false;

public:
  /// Mark as a backward liveness analysis.
  LiveValues() = default;

  /// Do a Backwards Analysis of the CFG to build liveness.
  size_t doAnalysis(Workqueue &W);

  /// Add Phi edge values to BB's successors.
  void finalizeAnalysis(Function &F);

  /// Configure compute threshold to limit how much compile time we use.
  size_t configureThreshold();

  /// Analyse Live Values for function F.
  bool analyzeFunction(Function &F);

  /// Emit live variables before a basic block.
  void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                formatted_raw_ostream &OS) final;

  /// Emit live variables before an instruction.
  void emitInstructionAnnot(const Instruction *I,
                            formatted_raw_ostream &OS) final;

  /// Used for allocating SparseBitVectors.
  int getLivenessPoolSize(void) { return BvIdxToValue.size(); }

  /// live[n] = use[n] U (out[n] - def[n])
  bool statementTransferFunction(BasicBlock *BB, size_t &NumOperations);

  /// Function level data flow analysis.
  void doDataFlowAnalysis(Function &F);

  /// Clear and recalculate liveness.
  void recalcDataFlowAnalysis(Function &F);

  /// Give names to Instruction/Blocks where hasName is false.
  void nameInstructions(Function &F);

  /// Handle invalidation explicitly.
  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &);

  /// Get the SparseBitVector for LiveIn[BB]
  SparseBitVector<> &getLiveIn(BasicBlock *BB) {
    return LiveIn[BB];
  }

  /// Set the SparseBitVector for LiveIn[BB]
  void setLiveIn(BasicBlock *BB, const SparseBitVector<> &BV) {
    LiveIn[BB] = BV;
  }

  /// Get the SparseBitVector for LiveOut[BB]
  SparseBitVector<> &getLiveOut(BasicBlock *BB) {
    return LiveOut[BB];
  }

  /// Set the SparseBitVector for LiveOut[BB]
  void setLiveOut(BasicBlock *BB, const SparseBitVector<> &BV) {
    LiveOut[BB] = BV;
  }

  /// Get the SparseBitVector for PhiValues[BB]
  SparseBitVector<> &getPhiValues(BasicBlock *BB) {
    return PhiValues[BB];
  }

  /// Set the SparseBitVector for PhiValues[BB]
  void setPhiValues(BasicBlock *BB, const SparseBitVector<> &BV) {
    PhiValues[BB] = BV;
  }

  /// Get the SparseBitVector for InstrLiveIn[I]
  SparseBitVector<> &getInstrLiveIn(Instruction *I) {
    return InstrLiveIn[I];
  }

  /// Set the SparseBitVector for InstrLiveIn[I]
  void setInstrLiveIn(Instruction *I, const SparseBitVector<> &BV) {
    InstrLiveIn[I] = BV;
  }
};

/// Analysis pass which computes a \c LiveValues.
class LiveValuesAnalysis : public AnalysisInfoMixin<LiveValuesAnalysis> {
  friend AnalysisInfoMixin<LiveValuesAnalysis>;
  static AnalysisKey Key;

public:
  /// Provide the result typedef for this analysis pass.
  using Result = LiveValues;

  /// Run the analysis pass over a function and produce a dominator tree.
  LiveValues run(Function &F, FunctionAnalysisManager &);
};

} // end namespace llvm

#endif /* LLVM_ANALYSIS_LIVEVALUE_H */

#endif // SIFIVE_CUSTOMIZATION