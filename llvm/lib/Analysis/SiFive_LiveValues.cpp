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

#include "llvm/Analysis/SiFive_LiveValues.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <cmath>
#include <type_traits>

using namespace llvm;

static cl::opt<bool>
    EnableLivenessAnnotations("liveness-annotations-enable", cl::Hidden,
                              cl::init(false),
                              cl::desc("Emit annotations for liveness in IR"));

// 0 is default to off
// 1 is baseline
// 2 is aggressive licm
// 3 is inline usage
// 4 is aggressive licm and inline
cl::opt<uint32_t>
    EnableValuePressureAnalysis("value-liveness-enable", cl::Hidden,
                                cl::init(0),
                                cl::desc("Analysis of Value Liveness: (0..4)"));

static cl::opt<bool>
    EnableValuePressureBypass("value-liveness-gpr-bypass", cl::Hidden,
                              cl::init(false),
                              cl::desc("Value Liveness GPR Bypass"));

static cl::opt<int32_t> MaxMissingValuesDrift(
    "value-liveness-max-drift", cl::Hidden, cl::init(4),
    cl::desc("Max missing DFA values allowed before recalc (default = 4)"));

static cl::opt<uint32_t> MaxGeneralPurposeRegs(
    "value-liveness-max-GPR-num", cl::Hidden, cl::init(6),
    cl::desc("Max num gp registers user assigned (default = 6)"));

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

  // Collect LiveIn of BB and Add preds to work if changed.
  // Ensure we visit each block at least once as simple blocks can
  // have Liveness that does not change with the transfer function,
  // causing premature termination of the flow analysis.
  bool Changed = statementTransferFunction(BB, NumOperations);
  for (auto PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
    if (Changed || !Visited.contains(*PI))
      if (!llvm::is_contained(W, *PI))
        W.push_back(*PI);

  return NumOperations;
}

void LiveValues::finalizeAnalysis(Function &F) {
  // Walk all reachable blocks.
  df_iterator_default_set<BasicBlock *> DfsSet;
  for (BasicBlock *BB : depth_first_ext(&F, DfsSet)) {
    auto PIT = PhiValues.find(BB);
    if (PIT != PhiValues.end())
      for (auto SI = succ_begin(BB), SE = succ_end(BB); SI != SE; SI++)
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

  // Walk all reachable blocks.
  df_iterator_default_set<BasicBlock *> DfsSet;
  for (BasicBlock *BB : depth_first_ext(&F, DfsSet)) {
    NumBlocks++;
    if (succ_empty(BB) && !llvm::is_contained(W, BB))
      W.push_back(BB);
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
    OS << LIs[V];
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

  // Mark each block we see visited.
  Visited.insert(BB);

  // Bottom up walk of instructions.
  for (auto It = BB->rbegin(), E = BB->rend(); It != E; ++It) {
    Instruction *I = &*It;

    // Collect (out[n] - def[n])
    NewLiveIn.reset(Indices[I].getIndex());
    NumOperations++;

    // Add uses, unless a phi node then propagate incoming values.
    if (auto *PhiNode = dyn_cast<PHINode>(I)) {
      for (unsigned Idx = 0, E = PhiNode->getNumIncomingValues(); Idx < E;
           Idx++) {
        Value *Val = PhiNode->getIncomingValue(Idx);
        if (EphValues.count(Val))
          continue;

        if (isa<Instruction, Argument>(Val)) {
          BasicBlock *IdxBB = PhiNode->getIncomingBlock(Idx);
          PhiValues[IdxBB].set(Indices[Val].getIndex());
          NumOperations++;
        }
      }
    } else {
      // use[n] U (out[n] - def[n])
      for (auto OI = I->op_begin(), OE = I->op_end(); OI != OE; ++OI) {
        if (EphValues.count(*OI))
          continue;

        if (isa<Instruction, Argument>(*OI)) {
          NewLiveIn.set(Indices[*OI].getIndex());
          NumOperations++;
        }
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

void LiveValues::createFullSegment(Value *Start, unsigned SlotStart, Value *End,
                                   unsigned SlotEnd, Value *V) {
  ValueSlotIndex FirstIndex(&Indices[Start], SlotStart);
  ValueSlotIndex LastIndex(&Indices[End], SlotEnd);
  ValueSlotInfo *VNI = *LIs[V].vni_begin();
  bool IsBlockSeg = true;
  LIs[V].addSegment(ValueLiveInterval::Segment(FirstIndex, LastIndex, VNI),
                                               IsBlockSeg);
}

void LiveValues::createSegmentStart(Value *Start, unsigned Slot, Value *V,
                                    ValueSlotIndex &StartIndex) {
  ValueSlotIndex DefIndex(&Indices[Start], Slot);
  StartIndex = DefIndex;
  LIs[V].createDeadDef(DefIndex, getVSInfoAllocator());
}

void LiveValues::endExistingSegment(Value *End, unsigned Slot, Value *V,
                                    ValueSlotIndex &DefIndex) {
  ValueSlotIndex EndIndex(&Indices[End], Slot);
  LIs[V].extendInBlock(DefIndex, EndIndex);
}

Instruction *LiveValues::FindFirstValue(BasicBlock *BB) {
  Instruction *CurI = nullptr;
  for (auto It = BB->begin(), E = BB->end(); It != E; ++It) {
    Instruction *I = &*It;
    if (EphValues.count(I))
      continue;

    CurI = I;
    break;
  }

  return CurI;
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
    endExistingSegment(I, ValueSlotIndex::Slot::Slot_Register, V,
                       CurIndexMap[BB]);
  }

  // Add segments as needed for uses of V.
  SmallPtrSet<BasicBlock *, 4> VisitedUseBB;
  while (!Worklist.empty()) {
    Instruction *UseI = Worklist.pop_back_val();
    BasicBlock *UseBB = UseI->getParent();
    // This implies that the definition is not BB local.
    if (UseBB != BB) {
      // All the blocks that have DefIdx as LiveIn and LiveOut
      // will be processed as pass through segements later.
      if (!LiveOut[UseBB].test(DefIdx)) {
        // If this is the first time we have seen UseBB.
        if (VisitedUseBB.insert(UseBB).second) {
          Instruction *FirstI = FindFirstValue(UseBB);
          assert(FirstI && "Must find a suitable value to start");
          createSegmentStart(FirstI, ValueSlotIndex::Slot::Slot_Block, V,
                             CurIndexMap[UseBB]);
        }
      } else {
        // These will be handled with pass through values Segments.
        continue;
      }
    }
    endExistingSegment(UseI, ValueSlotIndex::Slot::Slot_Register, V,
                       CurIndexMap[UseBB]);
  }
}

void LiveValues::extendPassThroughLiveIntervalSegments(Function &F) {
  // Iterate the CFG and for each reachable block where a Value is
  // both LiveIn and LiveOut, add a segment for it.
  BasicBlock *EntryBB = &F.getEntryBlock();
  df_iterator_default_set<BasicBlock *> DfsSet;
  for (BasicBlock *CurBB : depth_first_ext(&F, DfsSet)) {
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
    for (size_t Idx : LiveInIt->second) {
      // Skip uninitialized nodes, they all land on this index,
      // they have to go somewhere.
      if (Idx == EMPTY_INDEX)
        continue;

      if (LiveOut[CurBB].test(Idx)) {
        Value *V = BvIdxToValue[Idx];
        if (auto *BinOp = dyn_cast<BinaryOperator>(V)) {
          bool SkipToNextIdx = false;
          // Is V not a pass through value?
          for (User *U : V->users())
            if (auto *CurPhi = dyn_cast<PHINode>(U))
              if (CurPhi->getParent() == CurBB) {
                SkipToNextIdx |=
                    (any_of(BinOp->operands(),
                            [=](const Value *Op) { return (Op == CurPhi); }));
                if (SkipToNextIdx)
                  break;
              }

          if (SkipToNextIdx)
            continue;
        }

        // Construct the pass through Segment for V.
        Instruction *FirstI = FindFirstValue(CurBB);
        assert(FirstI && "Must find a suitable value to start");
        Instruction *LastI = &*CurBB->rbegin();
        createFullSegment(FirstI, ValueSlotIndex::Slot::Slot_Block, LastI,
                          ValueSlotIndex::Slot::Slot_Block, V);
      }
    }
  }
}

LiveValues::ValueDescr LiveValues::getMappedValueDesr(Value *V) {
  // If the value is no longer mapped, return not found.
  if (!isa<Instruction, Argument>(V))
    return ValueDescr::Types_End;

  Type *T = V->getType();
  if (T->isIntegerTy() || T->isPointerTy())
    return ValueDescr::Types_Integer;
  if (T->isFloatingPointTy())
    return ValueDescr::Types_Float;
  if (T->isVectorTy())
    return ValueDescr::Types_Vector;

  return ValueDescr::Types_End;
}

LLVM_DUMP_METHOD void
LiveValues::printPressureData(ArrayRef<PressureTracker> PT) {
  dbgs() << "ValueDescr types\n";
  for (unsigned I = ValueDescr::Types_Integer; I < ValueDescr::Types_End; ++I) {
    switch (I) {
    case ValueDescr::Types_Integer:
      dbgs() << "PT[Types_Integer]\n";
      break;
    case ValueDescr::Types_Float:
      dbgs() << "PT[Types_Float]\n";
      break;
    case ValueDescr::Types_Vector:
      dbgs() << "PT[Types_Vector]\n";
      break;
    default:
      break;
    }
    dbgs() << "\tInitPressure = " << PT[I].InitPressure << "\n";
    dbgs() << "\tCurPressure = " << PT[I].CurPressure << "\n";
    dbgs() << "\tLocalMaxima = " << PT[I].LocalMaxima << "\n";
    dbgs() << "\tFinalPressure = " << PT[I].FinalPressure << "\n";
  }
}

void LiveValues::calculatePressureForValue(
    Value *CurV, unsigned TD, ValueSlotIndex *CurIndex,
    MutableArrayRef<PressureTracker> CurPT) {
  ValueDescr VD = getMappedValueDesr(CurV);
  if (VD != ValueDescr::Types_End) {
    switch (TD) {
    case TrackerDescr::Calculate_InitPressure:
      CurPT[VD].InitPressure++;
      break;
    case TrackerDescr::Calculate_CurPressure:
      // Do a simple check that does not require LIs info.
      if (CurV->hasOneUse()) {
        CurPT[VD].CurPressure--;
        break;
      }

      // The program changes iteratively, meaning
      // the live range info may now be no longer available
      if (LIs[CurV].empty())
        break;

      // Sometimes are at the end of segment, so
      // check if there are any segments which hold
      // this value live at a later index.
      assert(CurIndex && "must provide the current loc");
      if (LIs[CurV].expiredAt(*CurIndex))
        CurPT[VD].CurPressure--;

      break;
    case TrackerDescr::Calculate_FinalPressure:
      CurPT[VD].FinalPressure++;
      break;
    }
  }
}

void LiveValues::calculatePressureForBitvector(
    SparseBitVector<> &LiveData, unsigned TD, ValueSlotIndex *CurIndex,
    MutableArrayRef<PressureTracker> CurPT) {
  for (auto Idx : LiveData) {
    // Skip unitialized indicies
    if (Idx == EMPTY_INDEX)
      continue;

    Value *CurV = BvIdxToValue[Idx];
    if (!ResidentValues.contains(CurV))
      continue;

    calculatePressureForValue(CurV, TD, CurIndex, CurPT);
  }
}

void LiveValues::adjustInitialPressure(BasicBlock *BB,
                                       MutableArrayRef<PressureTracker> CurPT) {
  // Process PHINodes to adjust side effects and set correct
  // initial conditions of block entry for value pressure.
  for (PHINode &PN : BB->phis()) {
    for (unsigned Idx = 0, E = PN.getNumIncomingValues(); Idx < E; Idx++) {
      Value *InVal = PN.getIncomingValue(Idx);
      ValueDescr VD = getMappedValueDesr(InVal);
      if (VD != ValueDescr::Types_End)
        CurPT[VD].InitPressure--;
    }

    // Now count the phi def.
    auto VD = getMappedValueDesr(&PN);
    if (VD != ValueDescr::Types_End)
      CurPT[VD].InitPressure++;
  }
}

Instruction *LiveValues::calculateBlockValuePressure(
    BasicBlock *BB, Instruction *TargetI,
    MutableArrayRef<PressureTracker> MachinePT,
    MutableArrayRef<PressureTracker> InsnPT,
    MutableArrayRef<PressureTracker> CurPT,
    SmallPtrSetImpl<const Value *> &IgnoreValues,
    SmallVectorImpl<Use *> &AddValues) {
  bool NeedFineGranularVP = (TargetI) ? (TargetI->getParent() == BB) : false;
  Instruction *FirstMaxPointI = nullptr;
  for (auto It = BB->begin(), E = BB->end(); It != E; ++It) {
    Instruction *I = &*It;
    if (isa<PHINode>(I) || EphValues.count(I))
      continue;

    // Collect the initial pressure for TargetI to calculate
    // the changed pressure of the candidate instruction.
    if (I == TargetI)
      for (unsigned Idx = ValueDescr::Types_Integer;
           Idx < ValueDescr::Types_End; ++Idx)
        InsnPT[Idx].InitPressure = CurPT[Idx].CurPressure;

    if (Indices[I].getIndex() == EMPTY_INDEX)
      continue;

    ValueSlotIndex CurIndex(&Indices[I], ValueSlotIndex::Slot::Slot_Register);
    if (!CurIndex.isValid())
      continue;

    for (Use &Op : I->operands()) {
      if (I == TargetI && IgnoreValues.contains(Op))
        continue;

      if (ResidentValues.contains(Op))
        calculatePressureForValue(Op, TrackerDescr::Calculate_CurPressure,
                                  &CurIndex, CurPT);
    }

    if (I->getNumUses() > 0) {
      auto VD = getMappedValueDesr(I);
      if (VD != ValueDescr::Types_End) {
        CurPT[VD].CurPressure++;
        CurPT[VD].LocalMaxima =
            std::max(CurPT[VD].CurPressure, CurPT[VD].LocalMaxima);
        if (NeedFineGranularVP)
          if (CurPT[VD].CurPressure > MachinePT[VD].LocalMaxima) {
            NeedFineGranularVP = false;
            FirstMaxPointI = I;
          }
      }
    }

    // Calculate the final pressure as a delta of
    // the initial pressure and the current pressure.
    if (TargetI == I) {
      // Examine any values provided that are part of
      // an optimization that must be checked.
      for (Use *Op : AddValues) {
        Value *CurV = *Op;

        if (LIs[CurV].empty())
          continue;

        // Skip Operands of I, they are already evaluated.
        if (any_of(I->operands(), [&](Use &CurOp) {return CurV == CurOp; }))
          continue;

        if (!LIs[CurV].expiredAt(CurIndex)) {
          ValueDescr VD = getMappedValueDesr(CurV);
          if (VD != ValueDescr::Types_End)
            InsnPT[VD].CurPressure++;
        }
      }
      for (unsigned Idx = ValueDescr::Types_Integer;
           Idx < ValueDescr::Types_End; ++Idx)
        InsnPT[Idx].FinalPressure =
            CurPT[Idx].CurPressure - InsnPT[Idx].InitPressure;
    }
  }

  return FirstMaxPointI;
}

bool LiveValues::markResidentValues(Function *F,
                                    ArrayRef<PressureTracker> MachinePT,
                                    unsigned &NumInstructions,
                                    unsigned &NumCalls) {
  ResidentValues.clear();
  int NumIntervalsMissing = 0;
  bool NeedRecalc = false;
  if (OptLevel == 2) {
    NumInstructions = 0;
    NumCalls = 0;
  }
  for (auto &ArgI : F->args())
    ResidentValues.insert(&ArgI);

  // Walk all reachable blocks to look for residence.
  df_iterator_default_set<BasicBlock *> DfsSet;
  for (BasicBlock *BB : depth_first_ext(F, DfsSet))
    for (auto It = BB->begin(), E = BB->end(); It != E; ++It) {
      Instruction *I = &*It;
      if (OptLevel == 2)
        NumInstructions++;

      ResidentValues.insert(I);
      if (isa<PHINode>(I) || EphValues.count(I))
        continue;

      if (Indices[I].getIndex() == EMPTY_INDEX)
        NumIntervalsMissing++;

      if (OptLevel == 2) {
        if (isa<IntrinsicInst>(I))
          continue;

        if (isa<CallInst>(I))
          NumCalls++;
      }
    }

  if (NumIntervalsMissing) {
    LLVM_DEBUG(dbgs() << "Total recorded intervals = " << getLivenessPoolSize()
                      << "\n");
    LLVM_DEBUG(dbgs() << " , Intervals missing = " << NumIntervalsMissing
                      << "\n");
    // For any RC, do we need to recalculate DFA?
    for (unsigned VD = ValueDescr::Types_Integer; VD < ValueDescr::Types_End;
         ++VD) {
      if (NumIntervalsMissing > MaxMissingValuesDrift) {
        NeedRecalc = true;
        break;
      }
    }
  }

  if (NeedRecalc)
    return true;

  int NumResidentsMissing = 0;
  for (int i = 0; i < getLivenessPoolSize(); i++) {
    Value *CurV = BvIdxToValue[i];
    if (!ResidentValues.contains(CurV))
      NumResidentsMissing++;
  }

  if (NumResidentsMissing) {
    LLVM_DEBUG(dbgs() << "Total resident intervals = " << ResidentValues.size()
                      << "\n");
    LLVM_DEBUG(dbgs() << "Residents missing from DFA = " << NumResidentsMissing
                      << "\n");
    // For any RC, do we need to recalculate DFA?
    for (unsigned VD = ValueDescr::Types_Integer; VD < ValueDescr::Types_End;
         ++VD) {
      if (NumResidentsMissing > MaxMissingValuesDrift) {
        NeedRecalc = true;
        break;
      }
    }
  }

  return NeedRecalc;
}

static bool isFlowRelated(BasicBlock *CurBB, BasicBlock *BB) {
  // Check immediate preds of CurBB for BB.
  if (is_contained(predecessors(CurBB), BB))
    return true;

  // See if BB provides a value in a phi to CurBB.
  for (PHINode &PN : CurBB->phis())
    if (PN.getBasicBlockIndex(BB) >= 0)
      return true;

  return false;
}

Instruction *LiveValues::processBlock(
    BasicBlock *BB, Instruction *TargetI,
    MutableArrayRef<PressureTracker> MachinePT,
    MutableArrayRef<PressureTracker> InsnPT,
    MutableArrayRef<PressureTracker> CurPT,
    SmallPtrSetImpl<const Value *> &IgnoreValues,
    SmallVectorImpl<Use *> &AddValues) {
  Instruction *FirstMaxPointI = nullptr;
  auto LiveInIt = LiveIn.find(BB);
  if (LiveInIt != LiveIn.end()) {
    // First calculate the initial/final pressure of BB.
    calculatePressureForBitvector(LiveInIt->second,
                                  TrackerDescr::Calculate_InitPressure,
                                  /* CurIndex */ nullptr, CurPT);

    // Adjust inital block pressure with PHINode processing
    adjustInitialPressure(BB, CurPT);

    // Initialize pressure data with Init pressure
    for (unsigned Idx = ValueDescr::Types_Integer;
         Idx < ValueDescr::Types_End; ++Idx) {
      CurPT[Idx].CurPressure = CurPT[Idx].InitPressure;
      CurPT[Idx].LocalMaxima = CurPT[Idx].InitPressure;
    }

    if (auto *I =
            calculateBlockValuePressure(BB, TargetI, MachinePT, InsnPT, CurPT,
                                        IgnoreValues, AddValues))
      FirstMaxPointI = I;

    for (unsigned Idx = ValueDescr::Types_Integer;
         Idx < ValueDescr::Types_End; ++Idx) {
      CurPT[Idx].FinalPressure = CurPT[Idx].CurPressure;
    }

    LLVM_DEBUG(dbgs() << "Block Summary\n");
    LLVM_DEBUG(printPressureData(CurPT));
  }

  return FirstMaxPointI;
}

bool LiveValues::exceedValuePressureForFunction(
    int &NumGprs, int &NumFprs, int &NumVrs, Function *F) {
  SmallVector<PressureTracker, ValueDescr::Types_End> InsnPT;
  SmallVector<PressureTracker, ValueDescr::Types_End> MachinePT;
  SmallVector<PressureTracker, ValueDescr::Types_End> UsedPT;
  DenseMap<const BasicBlock *,
          SmallVector<PressureTracker, ValueDescr::Types_End>> PressureMap;

  // Do initializations of collection PressureTrackers
  MachinePT.resize(ValueDescr::Types_End);

  MachinePT[ValueDescr::Types_Integer].LocalMaxima = NumGprs;
  MachinePT[ValueDescr::Types_Float].LocalMaxima = NumFprs;
  MachinePT[ValueDescr::Types_Vector].LocalMaxima = NumVrs;

  UsedPT.resize(ValueDescr::Types_End);

  UsedPT[ValueDescr::Types_Integer].LocalMaxima = 0;
  UsedPT[ValueDescr::Types_Float].LocalMaxima = 0;
  UsedPT[ValueDescr::Types_Vector].LocalMaxima = 0;

  SmallPtrSet<const Value *, 4> IgnoreValues;
  SmallVector<Use *> AddValues;

  // Add all params to IgnoreValues, for this usage we are
  // examining the effects of a call edge, params are already
  // counted in the caller.
  for (auto &ArgI : F->args())
    IgnoreValues.insert(&ArgI);

  Instruction *TargetI = nullptr;
  unsigned SumInstructions;
  unsigned NumCalls;
  markResidentValues(F, MachinePT, SumInstructions, NumCalls);

  // Simulate running out of GP registers if bypass enabled.
  if (OptLevel == 1 && EnableValuePressureBypass) {
    unsigned Idx = ValueDescr::Types_Integer;
    MachinePT[Idx].LocalMaxima = MaxGeneralPurposeRegs;
  }

  for (BasicBlock &BB : *F) {
    // Populate empty ValueDescr's for each type in PT.
    SmallVector<PressureTracker, ValueDescr::Types_End> &PT = PressureMap[&BB];
    PT.resize(ValueDescr::Types_End);

    // Detect Value Pressure for BB.
    processBlock(&BB, TargetI, MachinePT, InsnPT, PT, IgnoreValues, AddValues);

    for (unsigned Idx = ValueDescr::Types_Integer;
         Idx < ValueDescr::Types_End; ++Idx) {
      if (PT[Idx].LocalMaxima > MachinePT[Idx].LocalMaxima)
        return true;

      if (PT[Idx].LocalMaxima > UsedPT[Idx].LocalMaxima)
        UsedPT[Idx].LocalMaxima = PT[Idx].LocalMaxima;
    }
  }

  // Save what we used
  NumGprs = UsedPT[ValueDescr::Types_Integer].LocalMaxima;
  NumFprs = UsedPT[ValueDescr::Types_Float].LocalMaxima;
  NumVrs = UsedPT[ValueDescr::Types_Vector].LocalMaxima;

  return false;
}

bool LiveValues::exceedValuePressureForBlocks(
    SmallVectorImpl<BasicBlock *> &Worklist,
    SmallVectorImpl<Use *> &AddValues,
    SmallPtrSetImpl<const Value *> &IgnoreValues, DominatorTree *DT,
    BasicBlock *EndBlock, int NumGprs, int NumFprs, int NumVrs,
    Instruction *TargetI, bool IsHoistContext) {
  BasicBlock *TargetBB = TargetI->getParent();
  Function *F = TargetBB->getParent();
  SmallVector<PressureTracker, ValueDescr::Types_End> InsnPT;
  SmallVector<PressureTracker, ValueDescr::Types_End> MachinePT;
  DenseMap<const BasicBlock *,
          SmallVector<PressureTracker, ValueDescr::Types_End>> PressureMap;

  // Do initializations of collection PressureTrackers
  InsnPT.resize(ValueDescr::Types_End);
  MachinePT.resize(ValueDescr::Types_End);

  MachinePT[ValueDescr::Types_Integer].LocalMaxima = NumGprs;
  MachinePT[ValueDescr::Types_Float].LocalMaxima = NumFprs;
  MachinePT[ValueDescr::Types_Vector].LocalMaxima = NumVrs;

  unsigned SumInstructions;
  unsigned NumCalls;
  if (markResidentValues(F, MachinePT, SumInstructions, NumCalls)) {
    recalcDataFlowAnalysis(*F);
    markResidentValues(F, MachinePT, SumInstructions, NumCalls);
  }

  // In aggressive mode, frequent recalculation may be instigated.
  if (OptLevel == 2) {
    // When code density is high, require a minimum number of calls
    // as abi effects will cause VP to rise, else allow legacy behavior.
    if (SumInstructions > AGGRESIVE_OPT_INSN_SIZE_THRESHOLD &&
        NumCalls < AGGRESIVE_OPT_NUM_CALLS_THRESHOLD) {
      // Now remove TargetI from the Index list since it will be optimized.
      Indices[TargetI].setIndex(EMPTY_INDEX);

      return false;
    }

    // Allow more GPR usage, distance to RA and optimization will be the delta.
    MachinePT[ValueDescr::Types_Integer].LocalMaxima += NumGprs;
  }

  // Simulate running out of GP registers if bypass enabled.
  if (OptLevel == 1 && EnableValuePressureBypass) {
    unsigned Idx = ValueDescr::Types_Integer;
    MachinePT[Idx].LocalMaxima = MaxGeneralPurposeRegs;
  }

  Instruction *FirstMaxPointI = nullptr;
  for (BasicBlock *BB : Worklist) {
    // Populate empty ValueDescr's for each type in PT.
    SmallVector<PressureTracker, ValueDescr::Types_End> &PT = PressureMap[BB];
    PT.resize(ValueDescr::Types_End);

    if (auto *I = processBlock(BB, TargetI, MachinePT, InsnPT, PT,
                               IgnoreValues, AddValues))
      FirstMaxPointI = I;
  }

  // Check for BasicBlocks that have LocalMaxima in a RC that exceeds MachinePT
  for (BasicBlock *BB : Worklist) {
    SmallVector<PressureTracker, ValueDescr::Types_End> &SummaryPT =
        PressureMap[BB];
    for (unsigned Idx = ValueDescr::Types_Integer; Idx < ValueDescr::Types_End;
         ++Idx)
      if (SummaryPT[Idx].LocalMaxima > MachinePT[Idx].LocalMaxima &&
          ((!IsHoistContext) ||
           (IsHoistContext && InsnPT[Idx].FinalPressure > 0))) {
        if (BB == TargetBB) {
          if (IsHoistContext) {
            // Check if TargetI is after a local machine maxima.
            if (FirstMaxPointI && DT->dominates(FirstMaxPointI, TargetI))
              return true;

            // Do we run out ValueDescr registers right at TargetI?
            if (InsnPT[Idx].CurPressure > MachinePT[Idx].LocalMaxima)
              return true;
          } else {
            // For the general case we are out ValueDescr registers.
            return true;
          }

          continue;
        }

        // BB's VP is not in context.
        if (BB == EndBlock)
          continue;

        // If BB dominates TargetI, guard.
        if (DT->dominates(BB, TargetBB))
          return true;

        // BB's VP is not in context.
        if (DT->dominates(TargetBB, BB))
          continue;

        // TargetBB is immediately dominated by IDomBB(BB is under flow).
        // Note: There is no nice way to disambiguate sibling flow, which
        //       can have a complex relationship, making detection difficult.
        BasicBlock *IDomBB = DT->getNode(BB)->getIDom()->getBlock();
        if (DT->dominates(IDomBB, TargetBB)) {
          SmallVector<PressureTracker, ValueDescr::Types_End> &IDomPT =
              PressureMap[IDomBB];
          if (!IDomPT.empty())
            return true;
        }

        if (isFlowRelated(TargetBB, BB))
          return true;
      }
  }

  // Now remove TargetI from the Index list since it will be optimized.
  if (IsHoistContext)
    Indices[TargetI].setIndex(EMPTY_INDEX);

  return false;
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

  // Walk all reachable blocks and annotate DFA as we go.
  df_iterator_default_set<BasicBlock *> DfsSet;
  for (BasicBlock *BB : depth_first_ext(&F, DfsSet)) {
    for (auto It = BB->begin(), E = BB->end(); It != E; ++It) {
      Instruction *I = &*It;
      if (EphValues.count(I))
        continue;

      BvIdxToValue.push_back(I);
      LIs[I].setValue(I);
      Indices[I].setVal(I);
      Indices[I].setIndex(Idx++);
    }
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
  clearDataFlowAnalysis();
  doDataFlowAnalysis(F);
}

void LiveValues::clearDataFlowAnalysis() {
  if (!haveLiveValueAnalysis())
    return;

  LiveIn.clear();
  LiveOut.clear();
  PhiValues.clear();
  InstrLiveIn.clear();
  EphValues.clear();
  LIs.clear();
  BvIdxToValue.clear();
  Visited.clear();
  Indices.clear();
  ResidentValues.clear();
  VSInfoAllocator.Reset();
  LiveValuesAvailable = false;
}

// Borrowed from instnamer to make the annotated dumps nicer.
void LiveValues::nameInstructions(Function &F) {
  for (auto &Arg : F.args()) {
    if (!Arg.hasName())
      Arg.setName("arg");
  }

  // Name instructions in Dfs order
  df_iterator_default_set<BasicBlock *> DfsSet;
  for (BasicBlock *BB : depth_first_ext(&F, DfsSet)) {
    if (!BB->hasName())
      BB->setName("bb");

    for (auto It = BB->begin(), E = BB->end(); It != E; ++It) {
      Instruction *I = &*It;
      if (!I->hasName() && !I->getType()->isVoidTy())
        I->setName("i");
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

unsigned LiveValues::getAndSetOptLevel() {
  if (OptLevel == 0) {
    switch (EnableValuePressureAnalysis) {
    case LVUsageDescr::Baseline:
    case LVUsageDescr::Aggressive:
      setOptLevel(EnableValuePressureAnalysis);
      break;
    case LVUsageDescr::Inline:
      setOptLevel(LVUsageDescr::Baseline);
      break;
    case LVUsageDescr::InlineAndAggressive:
      setOptLevel(LVUsageDescr::Aggressive);
      break;
    default:
      setOptLevel(LVUsageDescr::None);
    }
  }

  return OptLevel;
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
    if (EnableValuePressureAnalysis <= LVUsageDescr::Aggressive ||
        EnableValuePressureAnalysis == LVUsageDescr::InlineAndAggressive) {
      LV.setAssumptionCache(&AM.getResult<AssumptionAnalysis>(F));
      LV.getAndSetOptLevel();
      LV.doDataFlowAnalysis(F);
    }
  }

  return LV;
}

AnalysisKey LiveValuesAnalysis::Key;
