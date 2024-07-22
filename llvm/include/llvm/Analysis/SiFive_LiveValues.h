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

#ifndef LLVM_ANALYSIS_LIVEVALUE_H
#define LLVM_ANALYSIS_LIVEVALUE_H

#include "llvm/ADT/SparseBitVector.h"
#include "llvm/Analysis/SiFive_LiveInterval.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <list>

#define AGGRESIVE_OPT_INSN_SIZE_THRESHOLD 5000
#define AGGRESIVE_OPT_NUM_CALLS_THRESHOLD 15

namespace llvm {

// Components for tracking Value Pressure
// attribute such as Initial, Final, Current
// pressure as well as a recorded Local Maxima.
struct PressureTracker {
  int InitPressure = 0;
  int FinalPressure = 0;
  int CurPressure = 0;
  int LocalMaxima = 0;
};

class LiveValues : public AssemblyAnnotationWriter {
private:
  // Use to track initial/final & local min/max value pressure for
  // each of the following Types.
  enum ValueDescr : unsigned {
    // Contains {char, short, int, long, etc} value types.
    Types_Integer = 0,
    // Contains {double, float, half} value types.
    Types_Float = 1,
    // Contains vector value types.
    Types_Vector = 2,
    // Num Value Descriptors
    Types_End = 3
  };

  enum TrackerDescr : unsigned {
    Calculate_InitPressure = 0,
    Calculate_CurPressure = 1,
    Calculate_FinalPressure = 2
  };

  /// Liveness sets for Basic Blocks.
  DenseMap<const BasicBlock *, SparseBitVector<>> LiveIn;
  DenseMap<const BasicBlock *, SparseBitVector<>> LiveOut;
  DenseMap<const BasicBlock *, SparseBitVector<>> PhiValues;
  DenseSet<const BasicBlock *> Visited;

  /// For collecting and comparing if values are Ephemeral.
  SmallPtrSet<const Value *, 4> EphValues;

  /// Map Values to their Instr SparseBitVector.
  SmallVector<Value *> BvIdxToValue;
  /// Instruction In sets.
  DenseMap<const Instruction *, SparseBitVector<>> InstrLiveIn;
  DenseMap<const Value *, ValueLiveInterval> LIs;
  DenseMap<const Value *, ValueIndexListEntry> Indices;
  SmallPtrSet<const Value *, 10> ResidentValues;
  bool LiveValuesAvailable = false;

  ValueSlotInfo::Allocator VSInfoAllocator;
  AssumptionCache *AC = nullptr;
  unsigned OptLevel = 0;

public:
  using Workqueue = std::list<BasicBlock *>;

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

  void emitFunctionAnnot(const Function *F,
                         formatted_raw_ostream &OS) final;

  /// Emit live variables before a basic block.
  void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                formatted_raw_ostream &OS) final;

  /// Emit live variables before an instruction.
  void emitInstructionAnnot(const Instruction *I,
                            formatted_raw_ostream &OS) final;

  /// Used for allocating SparseBitVectors.
  int getLivenessPoolSize(void) { return BvIdxToValue.size(); }

  /// Indicate if Live Value Analysis completed or not.
  bool haveLiveValueAnalysis(void) { return LiveValuesAvailable; }

  /// live[n] = use[n] U (out[n] - def[n])
  bool statementTransferFunction(BasicBlock *BB, size_t &NumOperations);

  /// Create A full Segment for V from Start to End
  void createFullSegment(Value *Start, unsigned SlotStart,
                         Value *End, unsigned SlotEnd, Value *V);

  /// Create a Segment starting at Start for V.
  void createSegmentStart(Value *Start, unsigned slot,
                          Value *V, ValueSlotIndex &DefIndex);

  /// End a Segment at End for V.
  void endExistingSegment(Value *End, unsigned slot,
                          Value *V, ValueSlotIndex &DefIndex);

  /// Find first non Ephermal value in a block.
  Instruction *FindFirstValue(BasicBlock *BB);

  /// Fill in LiveRange Segment info for V.
  void constructLiveIntervalSegments(Value *V, Function &F);

  /// Fill in pass through segments of all the LIs of F.
  void extendPassThroughLiveIntervalSegments(Function &F);

  /// Translate a given Value type as a ValueDescr.
  ValueDescr getMappedValueDesr(Value *V);

  /// Print the pressure data for each ValueDescr.
  void printPressureData(ArrayRef<PressureTracker> PT);

  /// Calculate pressure for a given value.
  void calculatePressureForValue(
      Value *CurV, unsigned TD, ValueSlotIndex *CurIndex,
      MutableArrayRef<PressureTracker> CurPT);

  /// Convience function to calculate pressure for a TrackerDescr
  void calculatePressureForBitvector(
      SparseBitVector<> &LiveData, unsigned TD,
      ValueSlotIndex *CurIndex,
      MutableArrayRef<PressureTracker> CurPT);

  /// Adjust inital block pressure with PHINode processing.
  void adjustInitialPressure(BasicBlock *BB,
                             MutableArrayRef<PressureTracker> CurPT);

  /// Compute incremental Pressure, instruction by instruction for BB.
  Instruction *calculateBlockValuePressure(
      BasicBlock *BB, Instruction *TargetI,
      MutableArrayRef<PressureTracker> MachinePT,
      MutableArrayRef<PressureTracker> InsnPT,
      MutableArrayRef<PressureTracker> CurPT,
      SmallPtrSetImpl<const Value *> &IgnoreValues,
      SmallVectorImpl<Use *> &AddValues);

  /// Find all values that are no longer used in F and decide
  /// if we need to recalculate DFA.
  bool markResidentValues(
      Function *F, ArrayRef<PressureTracker> MachinePT,
      unsigned &NumInstructions, unsigned &NumCalls);

  /// Find any local maxima which exceeds MachinePT in BB.
  Instruction *processBlock(
      BasicBlock *BB, Instruction *TargetI,
      MutableArrayRef<PressureTracker> MachinePT,
      MutableArrayRef<PressureTracker> InsnPT,
      MutableArrayRef<PressureTracker> CurPT,
      SmallPtrSetImpl<const Value *> &IgnoreValues,
      SmallVectorImpl<Use *> &AddValues);

  /// Calculate register pressure foreach block in F and
  /// determine if we exceed it for machine RCs.
  bool exceedValuePressureForFunction(
    int NumGprs, int NumFprs, int NumVrs, Function *F);

  /// Using a list of blocks, calculate the register pressure
  /// data for each block.
  bool exceedValuePressureForBlocks(
      SmallVectorImpl<BasicBlock *> &Worklist,
      SmallVectorImpl<Use *> &AddValues,
      SmallPtrSetImpl<const Value *> &IgnoreValues, DominatorTree *DT,
      BasicBlock *EndBlock, int NumGprs, int NumFprs, int NumVrs,
      Instruction *TargetI);

  /// Function level data flow analysis.
  void doDataFlowAnalysis(Function &F);

  /// Clear and recalculate liveness.
  void recalcDataFlowAnalysis(Function &F);

  /// Give names to Instruction/Blocks where hasName is false.
  void nameInstructions(Function &F);

  /// Handle invalidation explicitly.
  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &);

  /// get the local EnableValuePressureAnalysis setting
  unsigned getAndSetOptLevel();

  /// Get the SparseBitVector for LiveIn[BB]
  const SparseBitVector<> &getLiveIn(BasicBlock *BB) { return LiveIn[BB]; }

  /// Set the SparseBitVector for LiveIn[BB]
  void setLiveIn(BasicBlock *BB, const SparseBitVector<> &BV) {
    LiveIn[BB] = BV;
  }

  /// Get the SparseBitVector for LiveOut[BB]
  const SparseBitVector<> &getLiveOut(BasicBlock *BB) { return LiveOut[BB]; }

  /// Set the SparseBitVector for LiveOut[BB]
  void setLiveOut(BasicBlock *BB, const SparseBitVector<> &BV) {
    LiveOut[BB] = BV;
  }

  /// Get the SparseBitVector for PhiValues[BB]
  const SparseBitVector<> &getPhiValues(BasicBlock *BB) { return PhiValues[BB]; }

  /// Set the SparseBitVector for PhiValues[BB]
  void setPhiValues(BasicBlock *BB, const SparseBitVector<> &BV) {
    PhiValues[BB] = BV;
  }

  /// Get the SparseBitVector for InstrLiveIn[I]
  const SparseBitVector<> &getInstrLiveIn(Instruction *I) { return InstrLiveIn[I]; }

  /// Set the SparseBitVector for InstrLiveIn[I]
  void setInstrLiveIn(Instruction *I, const SparseBitVector<> &BV) {
    InstrLiveIn[I] = BV;
  }

  ValueSlotInfo::Allocator &getVSInfoAllocator() { return VSInfoAllocator; }

  void setAssumptionCache(AssumptionCache *AC) { this->AC = AC; }

  void setOptLevel(unsigned OptLevel) { this->OptLevel = OptLevel; }
};

/// Analysis pass which computes a \c LiveValues.
class LiveValuesAnalysis : public AnalysisInfoMixin<LiveValuesAnalysis> {
  friend AnalysisInfoMixin<LiveValuesAnalysis>;
  static AnalysisKey Key;

public:
  /// Provide the result typedef for this analysis pass.
  using Result = LiveValues;

  /// Run the analysis pass over a function and produce Live Values.
  LiveValues run(Function &F, FunctionAnalysisManager &);
};

} // end namespace llvm

#endif /* LLVM_ANALYSIS_LIVEVALUE_H */
