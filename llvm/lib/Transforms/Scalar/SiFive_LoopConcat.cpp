//===- SiFive_LoopConcat.cpp - Loop Concatentation Pass -------------------===//
//
// Copyright (c) 2024 SiFive, Inc. -- Proprietary and Confidential All
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
///
/// \file
/// This file implements the loop concatenation pass.
///
/// The general approach taken is to collect sets of control flow equivalent
/// loops and test whether they can be concatenated. The necessary conditions
/// for concatenation are:
///    1. The loops must be flow dependent, where one candidate dominates the
///       other.
///    2. The bounds must be joinable, i.e. the upper bound of the first
///       candidate must be the lower bound of the second.
///    3. The loops must be control flow equivalent (if one loop executes, the
///       other is guaranteed to execute).
///    4. There cannot be any write dependencies between the loops.
///    5. The loops must be identical in form and input except for some special
///       conditions where flow via PHIs is from one loop to the other.
/// If all of these conditions are satisfied, it is safe to concatenate the
/// loops.
///
/// This implementation creates ConcatCandidates that represent the loop and the
/// necessary information needed by concatenation. It then operates on the
/// concating candidates, first confirming that the candidate is eligible for
/// concatenation. The candidates are then collected into control flow
/// equivalent sets, sorted in dominance order. Each set of control flow
/// equivalent candidates is then traversed, attempting to concatenate
/// pairs of candidates in the set. If all requirements for concatenation are
/// met, the two candidates are concatenated, creating a new loop candidate
/// which is then added back into the set to consider for additional
/// concatenation.
///
/// This implementation currently does not make any modifications to remove
/// conditions for concatenation. Code transformations to make loops
/// conform to each of the conditions for concatenation are discussed in more
/// detail below.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/SiFive_LoopConcat.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeMoverUtils.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "loop-concat"

STATISTIC(ConcatCounter, "Loops concatenated");
STATISTIC(NumConcatCandidates, "Number of candidates for loop concatenation");
STATISTIC(InvalidPreheader, "Loop has invalid preheader");
STATISTIC(InvalidHeader, "Loop has invalid header");
STATISTIC(InvalidExitingBlock, "Loop has invalid exiting blocks");
STATISTIC(InvalidExitBlock, "Loop has invalid exit block");
STATISTIC(InvalidLatch, "Loop has invalid latch");
STATISTIC(InvalidLoop, "Loop is invalid");
STATISTIC(AddressTakenBB, "Basic block has address taken");
STATISTIC(MayThrowException, "Loop may throw an exception");
STATISTIC(ContainsVolatileAccess, "Loop contains a volatile access");
STATISTIC(NotSimplifiedForm, "Loop is not in simplified form");
STATISTIC(NotSameLoopBodies, "Loops bodies not same for concatenation");
STATISTIC(InvalidDependencies, "Dependencies prevent concatenation");
STATISTIC(UnknownTripCount, "Loop has unknown trip count");
STATISTIC(UncomputableTripCount, "SCEV cannot compute trip count of loop");
STATISTIC(NonJoinableTripRange, "Loop trip counts are not the same");
STATISTIC(
    NonEmptyPreheader,
    "Loop has a non-empty preheader with instructions that cannot be moved");
STATISTIC(ConcatNotBeneficial, "Concat is not beneficial");
STATISTIC(NonIdenticalGuards, "Candidates have different guards");
STATISTIC(NonEmptyExitBlock, "Candidate has a non-empty exit block with "
                             "instructions that cannot be moved");
STATISTIC(NotRotated, "Candidate is not rotated");
STATISTIC(OnlySecondCandidateIsGuarded,
          "The second candidate is guarded while the first one is not");
STATISTIC(NumHoistedInsts, "Number of hoisted preheader instructions.");
STATISTIC(NumSunkInsts, "Number of hoisted preheader instructions.");

enum ConcatDependenceAnalysisChoice {
  CONCAT_DEPENDENCE_ANALYSIS_SCEV,
  CONCAT_DEPENDENCE_ANALYSIS_DA,
  CONCAT_DEPENDENCE_ANALYSIS_ALL,
};

static cl::opt<ConcatDependenceAnalysisChoice> ConcatDependenceAnalysis(
    "loop-concat-dependence-analysis",
    cl::desc("Which dependence analysis should loop concat use?"),
    cl::values(clEnumValN(CONCAT_DEPENDENCE_ANALYSIS_SCEV, "scev",
                          "Use the scalar evolution interface"),
               clEnumValN(CONCAT_DEPENDENCE_ANALYSIS_DA, "da",
                          "Use the dependence analysis interface"),
               clEnumValN(CONCAT_DEPENDENCE_ANALYSIS_ALL, "all",
                          "Use all available analyses")),
    cl::Hidden, cl::init(CONCAT_DEPENDENCE_ANALYSIS_ALL));

static cl::opt<bool>
    EnableLoopConcatenation("loop-concat", cl::Hidden, cl::init(false),
                            cl::desc("Enable Loop Concatenation"));

#ifndef NDEBUG
static cl::opt<bool> VerboseConcatDebugging(
    "loop-concat-verbose-debug",
    cl::desc("Enable verbose debugging for Loop Concatenation"), cl::Hidden,
    cl::init(false));
#endif

namespace {
/// This class is used to represent a candidate for loop concatenation. When
/// it is constructed, it checks the conditions for loop concatenation to
/// ensure that it represents a valid candidate. It caches several parts of
/// a loop that are used throughout loop concatenation (e.g., loop preheader,
/// loop header, etc) instead of continually querying the underlying Loop to
/// retrieve them.  It is assumed these will not change throughout loop
/// concatenation.
///
/// The invalidate method should be used to indicate that the ConcatCandidate is
/// no longer a valid candidate for concatenation. Similarly, the isValid()
/// method can be used to ensure that the ConcatCandidate is still valid for
/// concatenation.
struct ConcatCandidate {
  /// Cache of parts of the loop used throughout loop concatenation. These
  /// should not need to change throughout the analysis and transformation.
  /// These parts are cached to avoid repeatedly looking up in the Loop class.

  /// Preheader of the loop this candidate represents
  BasicBlock *Preheader;
  /// Header of the loop this candidate represents
  BasicBlock *Header;
  /// Blocks in the loop that exit the loop
  BasicBlock *ExitingBlock;
  /// The successor block of this loop (where the exiting blocks go to)
  BasicBlock *ExitBlock;
  /// Latch of the loop
  BasicBlock *Latch;
  /// The loop that this concatentation candidate represents
  Loop *L;
  /// Vector of instructions in this loop that read from memory
  SmallVector<Instruction *, 16> MemReads;
  /// Vector of instructions in this loop that write to memory
  SmallVector<Instruction *, 16> MemWrites;
  /// Are all of the members of this concatenation candidate still valid
  bool Valid;
  /// Guard branch of the loop, if it exists
  BranchInst *GuardBranch;

  /// Dominator and PostDominator trees are needed for the
  /// ConcatCandidateCompare function, required by ConcatCandidateSet to
  /// determine where the ConcatCandidate should be inserted into the set. These
  /// are used to establish ordering of the ConcatCandidates based on dominance.
  DominatorTree &DT;
  const PostDominatorTree *PDT;

  OptimizationRemarkEmitter &ORE;

  ConcatCandidate(Loop *L, DominatorTree &DT, const PostDominatorTree *PDT,
                  OptimizationRemarkEmitter &ORE)
      : Preheader(L->getLoopPreheader()), Header(L->getHeader()),
        ExitingBlock(L->getExitingBlock()), ExitBlock(L->getExitBlock()),
        Latch(L->getLoopLatch()), L(L), Valid(true),
        GuardBranch(L->getLoopGuardBranch()), DT(DT), PDT(PDT), ORE(ORE) {

    // Walk over all blocks in the loop and check for conditions that may
    // prevent concatenation. For each block, walk over all instructions and
    // collect the memory reads and writes If any instructions that prevent
    // concatenation are found, invalidate this object and return.
    for (BasicBlock *BB : L->blocks()) {
      if (BB->hasAddressTaken()) {
        invalidate();
        reportInvalidCandidate(AddressTakenBB);
        return;
      }

      for (Instruction &I : *BB) {
        if (I.mayThrow()) {
          invalidate();
          reportInvalidCandidate(MayThrowException);
          return;
        }
        if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
          if (SI->isVolatile()) {
            invalidate();
            reportInvalidCandidate(ContainsVolatileAccess);
            return;
          }
        }
        if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->isVolatile()) {
            invalidate();
            reportInvalidCandidate(ContainsVolatileAccess);
            return;
          }
        }
        if (I.mayWriteToMemory())
          MemWrites.push_back(&I);
        if (I.mayReadFromMemory())
          MemReads.push_back(&I);
      }
    }
  }

  /// Check if all members of the class are valid.
  bool isValid() const {
    return Preheader && Header && ExitingBlock && ExitBlock && Latch && L &&
           !L->isInvalid() && Valid;
  }

  /// Verify that all members are in sync with the Loop object.
  void verify() const {
    assert(isValid() && "Candidate is not valid!!");
    assert(!L->isInvalid() && "Loop is invalid!");
    assert(Preheader == L->getLoopPreheader() && "Preheader is out of sync");
    assert(Header == L->getHeader() && "Header is out of sync");
    assert(ExitingBlock == L->getExitingBlock() &&
           "Exiting Blocks is out of sync");
    assert(ExitBlock == L->getExitBlock() && "Exit block is out of sync");
    assert(Latch == L->getLoopLatch() && "Latch is out of sync");
  }

  /// Get the entry block for this concatenation candidate.
  ///
  /// If this concatenation candidate represents a guarded loop, the entry
  /// block is the loop guard block. If it represents an unguarded loop,
  /// the entry block is the preheader of the loop.
  BasicBlock *getEntryBlock() const {
    if (GuardBranch)
      return GuardBranch->getParent();
    else
      return Preheader;
  }

  /// Given a guarded loop, get the successor of the guard that is not in the
  /// loop.
  ///
  /// This method returns the successor of the loop guard that is not located
  /// within the loop (i.e., the successor of the guard that is not the
  /// preheader).
  /// This method is only valid for guarded loops.
  /// TODO: Use this as query for guarded loops.
  BasicBlock *getNonLoopBlock() const {
    assert(GuardBranch && "Only valid on guarded loops.");
    assert(GuardBranch->isConditional() &&
           "Expecting guard to be a conditional branch.");
    return (GuardBranch->getSuccessor(0) == Preheader)
               ? GuardBranch->getSuccessor(1)
               : GuardBranch->getSuccessor(0);
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const {
    dbgs() << "\tGuardBranch: ";
    if (GuardBranch)
      dbgs() << *GuardBranch;
    else
      dbgs() << "nullptr";
    dbgs() << "\n"
           << (GuardBranch ? GuardBranch->getName() : "nullptr") << "\n"
           << "\tPreheader: " << (Preheader ? Preheader->getName() : "nullptr")
           << "\n"
           << "\tHeader: " << (Header ? Header->getName() : "nullptr") << "\n"
           << "\tExitingBB: "
           << (ExitingBlock ? ExitingBlock->getName() : "nullptr") << "\n"
           << "\tExitBB: " << (ExitBlock ? ExitBlock->getName() : "nullptr")
           << "\n"
           << "\tLatch: " << (Latch ? Latch->getName() : "nullptr") << "\n"
           << "\tEntryBlock: "
           << (getEntryBlock() ? getEntryBlock()->getName() : "nullptr")
           << "\n";
  }
#endif

  /// Determine if a candidate (representing a loop) is eligible for
  /// concatenation. Note that this only checks whether a single loop can be
  /// concatenated - it does not check whether it is *legal* to concatenate
  /// two loops together.
  bool isEligibleForConcatenation(ScalarEvolution &SE) const {
    if (!isValid()) {
      LLVM_DEBUG(dbgs() << "FC has invalid CFG requirements!\n");
      if (!Preheader)
        ++InvalidPreheader;
      if (!Header)
        ++InvalidHeader;
      if (!ExitingBlock)
        ++InvalidExitingBlock;
      if (!ExitBlock)
        ++InvalidExitBlock;
      if (!Latch)
        ++InvalidLatch;
      if (L->isInvalid())
        ++InvalidLoop;

      return false;
    }

    // Require ScalarEvolution to be able to determine a trip count.
    if (!SE.hasLoopInvariantBackedgeTakenCount(L)) {
      LLVM_DEBUG(dbgs() << "Loop " << L->getName()
                        << " trip count not computable!\n");
      return reportInvalidCandidate(UnknownTripCount);
    }

    if (!L->isLoopSimplifyForm()) {
      LLVM_DEBUG(dbgs() << "Loop " << L->getName()
                        << " is not in simplified form!\n");
      return reportInvalidCandidate(NotSimplifiedForm);
    }

    if (!L->isRotatedForm()) {
      LLVM_DEBUG(dbgs() << "Loop " << L->getName() << " is not rotated!\n");
      return reportInvalidCandidate(NotRotated);
    }

    return true;
  }

private:
  // This is only used internally for now, to clear the MemWrites and MemReads
  // list and setting Valid to false. There doesn't seem to be other uses of
  // this right now, since once ConcatCandidates are put into the
  // ConcatCandidateSet they are immutable. Thus, any time we need to
  // change/update a ConcatCandidate, we must create a new one and insert it
  // into the ConcatCandidateSet to ensure the ConcatCandidateSet remains
  // ordered correctly.
  void invalidate() {
    MemWrites.clear();
    MemReads.clear();
    Valid = false;
  }

  bool reportInvalidCandidate(llvm::Statistic &Stat) const {
    using namespace ore;
    assert(L && Preheader && "Concatenate candidate not initialized properly!");
#if LLVM_ENABLE_STATS
    ++Stat;
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, Stat.getName(),
                                        L->getStartLoc(), Preheader)
             << "[" << Preheader->getParent()->getName() << "]: "
             << "Loop is not a candidate for concatenation: "
             << Stat.getDesc());
#endif
    return false;
  }
};

struct ConcatCandidateCompare {
  /// Comparison functor to sort two Control Flow Equivalent concatenation
  /// candidates into dominance order.
  /// If LHS dominates RHS and RHS post-dominates LHS, return true;
  /// If RHS dominates LHS and LHS post-dominates RHS, return false;
  /// If both LHS and RHS are not dominating each other then, non-strictly
  /// post dominate check will decide the order of candidates. If RHS
  /// non-strictly post dominates LHS then, return true. If LHS non-strictly
  /// post dominates RHS then, return false. If both are non-strictly post
  /// dominate each other then, level in the post dominator tree will decide
  /// the order of candidates.
  bool operator()(const ConcatCandidate &LHS,
                  const ConcatCandidate &RHS) const {
    const DominatorTree *DT = &(LHS.DT);

    BasicBlock *LHSEntryBlock = LHS.getEntryBlock();
    BasicBlock *RHSEntryBlock = RHS.getEntryBlock();

    // Do not save PDT to local variable as it is only used in asserts and thus
    // will trigger an unused variable warning if building without asserts.
    assert(DT && LHS.PDT && "Expecting valid dominator tree");

    // Do this compare first so if LHS == RHS, function returns false.
    if (DT->dominates(RHSEntryBlock, LHSEntryBlock)) {
      // Wrong order.
      return false;
    }

    if (DT->dominates(LHSEntryBlock, RHSEntryBlock)) {
      // Candidates ordered
      return LHS.PDT->dominates(RHSEntryBlock, LHSEntryBlock);
    }

    // If two ConcatCandidates are in the same level of dominator tree,
    // they will not dominate each other, but may still be control flow
    // equivalent. To sort those ConcatCandidates, nonStrictlyPostDominate()
    // function is needed.
    bool WrongOrder =
        nonStrictlyPostDominate(LHSEntryBlock, RHSEntryBlock, DT, LHS.PDT);
    bool RightOrder =
        nonStrictlyPostDominate(RHSEntryBlock, LHSEntryBlock, DT, LHS.PDT);
    if (WrongOrder && RightOrder) {
      // If common predecessor of LHS and RHS post dominates both
      // ConcatCandidates then, Order of ConcatCandidate can be
      // identified by its level in post dominator tree.
      DomTreeNode *LNode = LHS.PDT->getNode(LHSEntryBlock);
      DomTreeNode *RNode = LHS.PDT->getNode(RHSEntryBlock);
      return LNode->getLevel() > RNode->getLevel();
    } else if (WrongOrder)
      return false;
    else if (RightOrder)
      return true;

    // If LHS does not non-strict Postdominate RHS and RHS does not non-strict
    // Postdominate LHS then, there is no dominance relationship between the
    // two ConcatCandidates. Thus, they should not be in the same set together.
    llvm_unreachable(
        "No dominance relationship between these concatenation candidates!");
  }
};

using LoopVector = SmallVector<Loop *, 4>;

// Set of Control Flow Equivalent (CFE) Concatenation Candidates, sorted in
// dominance order. Thus, if CC0 comes *before* CC1 in a ConcatCandidateSet,
// then CC0 dominates CC1 and CC1 post-dominates CC0.
// std::set was chosen because we want a sorted data structure with stable
// iterators. A subsequent patch to loop concatenation will enable concatenating
// non-adjacent loops by moving intervening code around. When this intervening
// code contains loops, those loops will be moved also. The corresponding
// ConcatCandidates will also need to be moved accordingly. As this is done,
// having stable iterators will simplify the logic. Similarly, having an
// efficient insert that keeps the ConcatCandidateSet sorted will also simplify
// the implementation.
using ConcatCandidateSet = std::set<ConcatCandidate, ConcatCandidateCompare>;
using ConcatCandidateCollection = SmallVector<ConcatCandidateSet, 4>;

#if !defined(NDEBUG)
static llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const ConcatCandidate &FC) {
  if (FC.isValid())
    OS << FC.Preheader->getName();
  else
    OS << "<Invalid>";

  return OS;
}

static llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const ConcatCandidateSet &CandSet) {
  for (const ConcatCandidate &CC : CandSet)
    OS << CC << '\n';

  return OS;
}

static void
printConcatCandidates(const ConcatCandidateCollection &ConcatCandidates) {
  dbgs() << "Concatenation Candidates: \n";
  for (const auto &CandidateSet : ConcatCandidates) {
    dbgs() << "*** Concatenate Candidate Set ***\n";
    dbgs() << CandidateSet;
    dbgs() << "****************************\n";
  }
}
#endif

/// Collect all loops in function at the same nest level, starting at the
/// outermost level.
///
/// This data structure collects all loops at the same nest level for a
/// given function (specified by the LoopInfo object). It starts at the
/// outermost level.
struct LoopDepthTree {
  using LoopsOnLevelTy = SmallVector<LoopVector, 4>;
  using iterator = LoopsOnLevelTy::iterator;
  using const_iterator = LoopsOnLevelTy::const_iterator;

  LoopDepthTree(LoopInfo &LI) : Depth(1) {
    if (!LI.empty())
      LoopsOnLevel.emplace_back(LoopVector(LI.rbegin(), LI.rend()));
  }

  /// Test whether a given loop has been removed from the function, and thus is
  /// no longer valid.
  bool isRemovedLoop(const Loop *L) const { return RemovedLoops.count(L); }

  /// Record that a given loop has been removed from the function and is no
  /// longer valid.
  void removeLoop(const Loop *L) { RemovedLoops.insert(L); }

  /// Descend the tree to the next (inner) nesting level
  void descend() {
    LoopsOnLevelTy LoopsOnNextLevel;

    for (const LoopVector &LV : *this)
      for (Loop *L : LV)
        if (!isRemovedLoop(L) && L->begin() != L->end())
          LoopsOnNextLevel.emplace_back(LoopVector(L->begin(), L->end()));

    LoopsOnLevel = LoopsOnNextLevel;
    RemovedLoops.clear();
    Depth++;
  }

  bool empty() const { return size() == 0; }
  size_t size() const { return LoopsOnLevel.size() - RemovedLoops.size(); }
  unsigned getDepth() const { return Depth; }

  iterator begin() { return LoopsOnLevel.begin(); }
  iterator end() { return LoopsOnLevel.end(); }
  const_iterator begin() const { return LoopsOnLevel.begin(); }
  const_iterator end() const { return LoopsOnLevel.end(); }

private:
  /// Set of loops that have been removed from the function and are no longer
  /// valid.
  SmallPtrSet<const Loop *, 8> RemovedLoops;

  /// Depth of the current level, starting at 1 (outermost loops).
  unsigned Depth;

  /// Vector of loops at the current depth level that have the same parent loop
  LoopsOnLevelTy LoopsOnLevel;
};

#ifndef NDEBUG
static void printLoopVector(const LoopVector &LV) {
  dbgs() << "****************************\n";
  for (auto *L : LV)
    printLoop(*L, dbgs());
  dbgs() << "****************************\n";
}
#endif

struct LoopConcater {
private:
  // Sets of control flow equivalent concatenation candidates for a given nest
  // level.
  ConcatCandidateCollection ConcatCandidates;

  LoopDepthTree LDT;
  DomTreeUpdater DTU;

  LoopInfo &LI;
  DominatorTree &DT;
  DependenceInfo &DI;
  ScalarEvolution &SE;
  PostDominatorTree &PDT;
  OptimizationRemarkEmitter &ORE;

  SmallVector<Instruction *> LoopInsns;
  DenseMap<const Instruction *, int> CC0Map;
  DenseMap<const Instruction *, int> CC1Map;

public:
  LoopConcater(LoopInfo &LI, DominatorTree &DT, DependenceInfo &DI,
               ScalarEvolution &SE, PostDominatorTree &PDT,
               OptimizationRemarkEmitter &ORE, const DataLayout &DL)
      : LDT(LI), DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy), LI(LI),
        DT(DT), DI(DI), SE(SE), PDT(PDT), ORE(ORE) {}

  /// This is the main entry point for loop concatenation. It will traverse the
  /// specified function and collect candidate loops to concatenate, starting
  /// at the outermost nesting level and working inwards.
  bool concatLoops(Function &F) {
#ifndef NDEBUG
    if (VerboseConcatDebugging) {
      LI.print(dbgs());
    }
#endif

    LLVM_DEBUG(dbgs() << "Performing Loop Concatenation on function "
                      << F.getName() << "\n");
    bool Changed = false;

    while (!LDT.empty()) {
      LLVM_DEBUG(dbgs() << "Got " << LDT.size() << " loop sets for depth "
                        << LDT.getDepth() << "\n";);

      for (const LoopVector &LV : LDT) {
        assert(LV.size() > 0 && "Empty loop set was build!");

        // Skip singleton loop sets as they do not offer concatenation
        // opportunities on this level.
        if (LV.size() == 1)
          continue;
#ifndef NDEBUG
        if (VerboseConcatDebugging) {
          LLVM_DEBUG({
            dbgs() << "  Visit loop set (#" << LV.size() << "):\n";
            printLoopVector(LV);
          });
        }
#endif

        collectConcatCandidates(LV);
        Changed |= concatCandidates();
      }

      // Finished analyzing candidates at this level.
      // Descend to the next level and clear all of the candidates currently
      // collected. Note that it will not be possible to concatenate any of the
      // existing candidates with new candidates because the new candidates will
      // be at a different nest level and thus not be control flow equivalent
      // with all of the candidates collected so far.
      LLVM_DEBUG(dbgs() << "Descend one level!\n");
      LDT.descend();
      ConcatCandidates.clear();
    }

    if (Changed)
      LLVM_DEBUG(dbgs() << "Function after Loop Concatenation: \n"; F.dump(););

#ifndef NDEBUG
    assert(DT.verify());
    assert(PDT.verify());
    LI.verify(DT);
    SE.verify();
#endif

    LLVM_DEBUG(dbgs() << "Loop Concatenation complete\n");
    return Changed;
  }

private:
  /// Determine if two concatenation candidates are control flow equivalent.
  ///
  /// Two concatenation candidates are control flow equivalent if when one
  /// executes, the other is guaranteed to execute. This is determined using
  /// dominators and post-dominators: if A dominates B and B post-dominates
  /// A then A and B are control-flow equivalent.
  bool isControlFlowEquivalent(const ConcatCandidate &CC0,
                               const ConcatCandidate &CC1) const {
    assert(CC0.Preheader && CC1.Preheader && "Expecting valid preheaders");

    return ::isControlFlowEquivalent(*CC0.getEntryBlock(), *CC1.getEntryBlock(),
                                     DT, PDT);
  }

  /// Iterate over all loops in the given loop set and identify the loops that
  /// are eligible for concatenation. Place all eligible concatenation
  /// candidates into Control Flow Equivalent sets, sorted by dominance.
  void collectConcatCandidates(const LoopVector &LV) {
    for (Loop *L : LV) {
      ConcatCandidate CurrCand(L, DT, &PDT, ORE);
      if (!CurrCand.isEligibleForConcatenation(SE))
        continue;

      // Go through each list in ConcatCandidates and determine if L is control
      // flow equivalent with the first loop in that list. If it is, append LV.
      // If not, go to the next list.
      // If no suitable list is found, start another list and add it to
      // ConcatCandidates.
      bool FoundSet = false;

      for (auto &CurrCandSet : ConcatCandidates) {
        if (isControlFlowEquivalent(*CurrCandSet.begin(), CurrCand)) {
          // TODO: split correctness analysis before inserting
          CurrCandSet.insert(CurrCand);
          FoundSet = true;
#ifndef NDEBUG
          if (VerboseConcatDebugging)
            LLVM_DEBUG(dbgs() << "Adding " << CurrCand
                              << " to existing candidate set\n");
#endif
          break;
        }
      }
      if (!FoundSet) {
        // No set was found. Create a new set and add to ConcatCandidates
#ifndef NDEBUG
        if (VerboseConcatDebugging)
          LLVM_DEBUG(dbgs() << "Adding " << CurrCand << " to new set\n");
#endif
        ConcatCandidateSet NewCandSet;
        NewCandSet.insert(CurrCand);
        ConcatCandidates.push_back(NewCandSet);
      }
      NumConcatCandidates++;
    }
  }

  /// Determine if it is beneficial to concatenate loops.
  ///
  /// For now, this method simply returns true because we want to concatenate as
  /// much as possible (primarily to test the pass). This method will evolve,
  /// over time, to add heuristics for profitability of concatenation.
  bool isBeneficialConcatenation(const ConcatCandidate &CC0,
                                 const ConcatCandidate &CC1) {
    return true;
  }

  /// Determine if two concatenation candidates have the joinable trip ranges.
  ///
  bool haveJoinableTripRange(const ConcatCandidate &CC0,
                             const ConcatCandidate &CC1) const {
    const SCEV *TripCount0 = SE.getBackedgeTakenCount(CC0.L);
    if (isa<SCEVCouldNotCompute>(TripCount0)) {
      UncomputableTripCount++;
      LLVM_DEBUG(dbgs() << "Trip count of first loop could not be computed!");
      return false;
    }

    const SCEV *TripCount1 = SE.getBackedgeTakenCount(CC1.L);
    if (isa<SCEVCouldNotCompute>(TripCount1)) {
      UncomputableTripCount++;
      LLVM_DEBUG(dbgs() << "Trip count of second loop could not be computed!");
      return false;
    }

    // Now determine the following
    // a.) Do both loops have the same iv step.
    // b.) The start and end of the first loop.
    // c.) The start and end of the second loop.
    // d.) The end of the first loop is the start of the second loop.

    PHINode *CC0_IV = CC0.L->getInductionVariable(SE);
    PHINode *CC1_IV = CC1.L->getInductionVariable(SE);
    if (CC0_IV && CC1_IV) {
      auto CC0_LB = Loop::LoopBounds::getBounds(*CC0.L, *CC0_IV, SE);
      if (!CC0_LB.has_value())
        return false;

      auto CC1_LB = Loop::LoopBounds::getBounds(*CC1.L, *CC1_IV, SE);
      if (!CC1_LB.has_value())
        return false;

      if (CC0_LB->getDirection() == CC1_LB->getDirection() &&
          CC0_LB->getStepValue() == CC1_LB->getStepValue()) {

        Value *InitIvVal = &CC1_LB->getInitialIVValue();
        Value *FinalIvVal = &CC0_LB->getFinalIVValue();

        // Must have divergent ranges
        if (FinalIvVal == &CC1_LB->getFinalIVValue())
          return false;

        return (InitIvVal == FinalIvVal);
      }
    }

    return false;
  }

  /// Walk each set of control flow equivalent concatenation candidates and
  /// attempt to concatenate them. This does a single linear traversal of
  /// all candidates in the set. The conditions for legal concatenation are
  /// checked at this point. If a pair of concatenation candidates passes
  /// all legality checks, they are concatenated together and a new
  /// concatenation candidate is created and added to the ConcatCandidateSet.
  /// The original concatenation candidates are then removed, as they are no
  /// longer valid.
  bool concatCandidates() {
    bool Concatenated = false;
    LLVM_DEBUG(printConcatCandidates(ConcatCandidates));
    for (auto &CandidateSet : ConcatCandidates) {
      if (CandidateSet.size() < 2)
        continue;

      LLVM_DEBUG(dbgs() << "Attempting concatenation on Candidate Set:\n"
                        << CandidateSet << "\n");

      for (auto CC0 = CandidateSet.begin(); CC0 != CandidateSet.end(); ++CC0) {
        assert(!LDT.isRemovedLoop(CC0->L) &&
               "Should not have removed loops in CandidateSet!");
        auto CC1 = CC0;
        for (++CC1; CC1 != CandidateSet.end(); ++CC1) {
          assert(!LDT.isRemovedLoop(CC1->L) &&
                 "Should not have removed loops in CandidateSet!");

          LLVM_DEBUG(dbgs() << "Attempting to concatenate candidate \n";
                     CC0->dump(); dbgs() << " with\n"; CC1->dump();
                     dbgs() << "\n");

          CC0->verify();
          CC1->verify();

          // For now Guarded loop concatenation is not supported,
          // as motioning code is highly constrained.
          // TODO: Evaluate the guarded implementation further to
          //       determine if it is feasible and if not remove
          //       the guard processing code.
          if (CC0->GuardBranch || CC1->GuardBranch)
            break;

          // Check if the candidates have joinable tripcounts where the upper
          // bound of CC0 is the lower bound of CC1 and both have the same
          // step direction and size.
          bool HasJoinableTripRange = haveJoinableTripRange(*CC0, *CC1);
          if (!HasJoinableTripRange) {
            LLVM_DEBUG(
                dbgs()
                << "Concatenation candidates do not have joinable ranges."
                << "Not concatenating.\n");
            reportLoopConcatenation<OptimizationRemarkMissed>(
                *CC0, *CC1, NonJoinableTripRange);
            LLVM_DEBUG(dbgs() << "Loop trip ranges not joinable. "
                              << "Not concatenating.\n");
          }

          // Check the dependencies across the loops and do not concatenate if
          // it would violate them even if the ranges are not joinable.
          if (!dependencesAllowConcatenation(*CC0, *CC1,
                                             HasJoinableTripRange)) {
            LLVM_DEBUG(dbgs() << "Memory dependencies do not allow "
                              << "concatenation!\n");
            reportLoopConcatenation<OptimizationRemarkMissed>(
                *CC0, *CC1, InvalidDependencies);
            // This condition is special in that any candidate loop in the set
            // that posesses this attribute prevents a subsequent candidate
            // from concatenating from CC0 to any other member of the set.
            break;
          }

          // Walk both loops to check for identity between CC0 and CC1
          if (!compareAllowConcatenation(*CC0, *CC1, HasJoinableTripRange)) {
            LLVM_DEBUG(dbgs() << "Loop Bodies are not equivlant to allow "
                              << "concatenation!\n");
            reportLoopConcatenation<OptimizationRemarkMissed>(
                *CC0, *CC1, NotSameLoopBodies);
            continue;
          }

          if ((!CC0->GuardBranch && CC1->GuardBranch) ||
              (CC0->GuardBranch && !CC1->GuardBranch)) {
            LLVM_DEBUG(dbgs() << "The one of candidate is guarded while "
                              << "another one is not. Not concatenating.\n");
            reportLoopConcatenation<OptimizationRemarkMissed>(
                *CC0, *CC1, OnlySecondCandidateIsGuarded);
            continue;
          }

          // Ensure that CC0 and CC1 have identical guards.
          // If one (or both) are not guarded, this check is not necessary.
          if (CC0->GuardBranch && CC1->GuardBranch &&
              !haveIdenticalGuards(*CC0, *CC1)) {
            LLVM_DEBUG(dbgs()
                       << "Concatenation candidates do not have identical "
                       << "guards. Not Concatenating.\n");
            reportLoopConcatenation<OptimizationRemarkMissed>(
                *CC0, *CC1, NonIdenticalGuards);
            continue;
          }

          if (CC0->GuardBranch) {
            assert(CC1->GuardBranch && "Expecting valid CC1 guard branch");
            // If this happens there is state that prevents the removal of CC1.
            if (!isEmptyFlowEdge(CC1->ExitBlock)) {
              LLVM_DEBUG(dbgs() << "Concatenation candidate contains unsafe "
                                << "instructions in exit block. "
                                << "Not concatenating.\n");
              reportLoopConcatenation<OptimizationRemarkMissed>(
                  *CC0, *CC1, NonEmptyExitBlock);
              continue;
            }
          }

          // If the second loop has instructions in the pre-header, attempt to
          // hoist them up to the first loop's pre-header or sink them into the
          // body of the second loop.
          SmallVector<Instruction *, 4> SafeToHoist;
          SmallVector<Instruction *, 4> SafeToSink;
          // At this point, this is the last remaining legality check.
          // Which means if we can make this pre-header empty, we can
          // concatenate these loops
          if (!isEmptyFlowEdge(CC1->Preheader)) {
            LLVM_DEBUG(dbgs() << "Concatenation candidate does not have empty "
                              << "preheader.\n");

            // If it is not safe to hoist/sink all instructions in the
            // pre-header, we cannot concatenate these loops.
            if (!collectMovablePreheaderInsts(*CC0, *CC1, SafeToHoist,
                                              SafeToSink)) {
              LLVM_DEBUG(dbgs() << "Could not hoist/sink all instructions in "
                                << "Concatenation Candidate Pre-header.\n"
                                << "Not Concatenating.\n");
              reportLoopConcatenation<OptimizationRemarkMissed>(
                  *CC0, *CC1, NonEmptyPreheader);
              continue;
            }
          }

          bool BeneficialToConcatenate = isBeneficialConcatenation(*CC0, *CC1);
          LLVM_DEBUG(dbgs() << "\tConcatenation appears to be "
                            << (BeneficialToConcatenate ? "" : "un")
                            << "profitable!\n");
          if (!BeneficialToConcatenate) {
            reportLoopConcatenation<OptimizationRemarkMissed>(
                *CC0, *CC1, ConcatNotBeneficial);
            continue;
          }
          // All analysis has completed and has determined that concatenation is
          // legal and profitable. At this point, start transforming the code
          // and perform concatenation.

          // Execute the hoist/sink operations on preheader instructions
          movePreheaderInsts(*CC0, *CC1, SafeToHoist, SafeToSink);

          LLVM_DEBUG(dbgs() << "\tConcatenation is performed: " << *CC0
                            << " and " << *CC1 << "\n");

          ConcatCandidate CC0Copy = *CC0;

          // Report Concatenation to the Optimization Remarks.
          // Note this needs to be done *before* performConcatenation because
          // performConcatenation will change the original loops, making it not
          // possible to identify them after concatenation is complete.
          reportLoopConcatenation<OptimizationRemark>(*CC0, *CC1,
                                                      ConcatCounter);

          ConcatCandidate ConcatenatedCand(performConcatenation(*CC0, *CC1), DT,
                                           &PDT, ORE);
          ConcatenatedCand.verify();
          assert(ConcatenatedCand.isEligibleForConcatenation(SE) &&
                 "Concatenate candidate should be eligible for concatenation!");

          // Notify the loop-depth-tree that these loops are not valid objects
          LDT.removeLoop(CC1->L);

          CandidateSet.erase(CC0);
          CandidateSet.erase(CC1);

          auto InsertPos = CandidateSet.insert(ConcatenatedCand);

          assert(InsertPos.second &&
                 "Unable to insert TargetCandidate in CandidateSet!");

          // Reset CC0 and CC1 the new (concatenated) candidate. Subsequent
          // iterations of the CC1 loop will attempt to concatenate the new
          // (concatenated) loop with the remaining candidates in the current
          // candidate set.
          CC0 = CC1 = InsertPos.first;

          LLVM_DEBUG(dbgs() << "Candidate Set (after concatenation): "
                            << CandidateSet << "\n");

          Concatenated = true;
        }
      }
    }
    return Concatenated;
  }

  // Returns true if the instruction \p I can be hoisted to the end of the
  // preheader of \p CC0. \p SafeToHoist contains the instructions that are
  // known to be safe to hoist. The instructions encountered that cannot be
  // hoisted are in \p NotHoisting.
  bool canHoistInst(Instruction &I,
                    const SmallVector<Instruction *, 4> &SafeToHoist,
                    const SmallVector<Instruction *, 4> &NotHoisting,
                    const ConcatCandidate &CC0) const {
    const BasicBlock *CC0PreheaderTarget = CC0.Preheader->getSingleSuccessor();
    assert(CC0PreheaderTarget &&
           "Expected single successor for loop preheader.");

    for (Use &Op : I.operands()) {
      if (auto *OpInst = dyn_cast<Instruction>(Op)) {
        bool OpHoisted = is_contained(SafeToHoist, OpInst);
        // Check if we have already decided to hoist this operand. In this
        // case, it does not dominate CC0 *yet*, but will after we hoist it.
        if (!(OpHoisted || DT.dominates(OpInst, CC0PreheaderTarget))) {
          return false;
        }
      }
    }

    // PHIs in CC1's header only have CC0 blocks as predecessors. PHIs
    // cannot be hoisted and should be sunk to the exit of the concatenated
    // loop.
    if (isa<PHINode>(I))
      return false;

    // If this isn't a memory inst, hoisting is safe
    if (!I.mayReadOrWriteMemory())
      return true;

    LLVM_DEBUG(dbgs() << "Checking if this mem inst can be hoisted.\n");
    for (Instruction *NotHoistedInst : NotHoisting) {
      if (auto D = DI.depends(&I, NotHoistedInst, true)) {
        // Dependency is not read-before-write, write-before-read or
        // write-before-write
        if (D->isFlow() || D->isAnti() || D->isOutput()) {
          LLVM_DEBUG(dbgs() << "Inst depends on an instruction in CC1's "
                               "preheader that is not being hoisted.\n");
          return false;
        }
      }
    }

    for (Instruction *ReadInst : CC0.MemReads) {
      if (auto D = DI.depends(ReadInst, &I, true)) {
        // Dependency is not read-before-write
        if (D->isAnti()) {
          LLVM_DEBUG(dbgs() << "Inst depends on a read instruction in CC0.\n");
          return false;
        }
      }
    }

    for (Instruction *WriteInst : CC0.MemWrites) {
      if (auto D = DI.depends(WriteInst, &I, true)) {
        // Dependency is not write-before-read or write-before-write
        if (D->isFlow() || D->isOutput()) {
          LLVM_DEBUG(dbgs() << "Inst depends on a write instruction in CC0.\n");
          return false;
        }
      }
    }
    return true;
  }

  // Returns true if the instruction \p I can be sunk to the top of the exit
  // block of \p CC1.
  bool canSinkInst(Instruction &I, const ConcatCandidate &CC1) const {
    for (User *U : I.users()) {
      if (auto *UI{dyn_cast<Instruction>(U)}) {
        // Cannot sink if user in loop
        // If CC1 has phi users of this value, we cannot sink it into CC1.
        if (CC1.L->contains(UI)) {
          // Cannot hoist or sink this instruction. No hoisting/sinking
          // should take place, loops should not concatenate
          return false;
        }
      }
    }

    // If this isn't a memory inst, sinking is safe
    if (!I.mayReadOrWriteMemory())
      return true;

    for (Instruction *ReadInst : CC1.MemReads) {
      if (auto D = DI.depends(&I, ReadInst, true)) {
        // Dependency is not write-before-read
        if (D->isFlow()) {
          LLVM_DEBUG(dbgs() << "Inst depends on a read instruction in CC1.\n");
          return false;
        }
      }
    }

    for (Instruction *WriteInst : CC1.MemWrites) {
      if (auto D = DI.depends(&I, WriteInst, true)) {
        // Dependency is not write-before-write or read-before-write
        if (D->isOutput() || D->isAnti()) {
          LLVM_DEBUG(dbgs() << "Inst depends on a write instruction in CC1.\n");
          return false;
        }
      }
    }

    return true;
  }

  /// Collect instructions in the \p CC1 Preheader that can be hoisted
  /// to the \p CC0 Preheader or sunk into the \p CC1 Body
  bool collectMovablePreheaderInsts(
      const ConcatCandidate &CC0, const ConcatCandidate &CC1,
      SmallVector<Instruction *, 4> &SafeToHoist,
      SmallVector<Instruction *, 4> &SafeToSink) const {
    BasicBlock *CC1Preheader = CC1.Preheader;
    // Save the instructions that are not being hoisted, so we know not to hoist
    // mem insts that they dominate.
    SmallVector<Instruction *, 4> NotHoisting;

    for (Instruction &I : *CC1Preheader) {
      // Can't move a branch
      if (&I == CC1Preheader->getTerminator())
        continue;
      // If the instruction has side-effects, give up.
      if (I.mayThrow() || !I.willReturn()) {
        LLVM_DEBUG(dbgs() << "Inst: " << I << " may throw or won't return.\n");
        return false;
      }

      LLVM_DEBUG(dbgs() << "Checking Inst: " << I << "\n");

      // These kinds of operations cannot be reordered.
      if (I.isAtomic() || I.isVolatile()) {
        LLVM_DEBUG(
            dbgs() << "\tInstruction is volatile or atomic. Cannot move it.\n");
        return false;
      }

      if (canHoistInst(I, SafeToHoist, NotHoisting, CC0)) {
        SafeToHoist.push_back(&I);
        LLVM_DEBUG(dbgs() << "\tSafe to hoist.\n");
      } else {
        LLVM_DEBUG(dbgs() << "\tCould not hoist. Trying to sink...\n");
        NotHoisting.push_back(&I);

        if (canSinkInst(I, CC1)) {
          SafeToSink.push_back(&I);
          LLVM_DEBUG(dbgs() << "\tSafe to sink.\n");
        } else {
          LLVM_DEBUG(dbgs() << "\tCould not sink.\n");
          return false;
        }
      }
    }
    LLVM_DEBUG(
        dbgs() << "All preheader instructions could be sunk or hoisted!\n");
    return true;
  }

  /// Rewrite all additive recurrences in a SCEV to use a new loop.
  class AddRecLoopReplacer : public SCEVRewriteVisitor<AddRecLoopReplacer> {
  public:
    AddRecLoopReplacer(ScalarEvolution &SE, const Loop &OldL, const Loop &NewL,
                       bool UseMax = true)
        : SCEVRewriteVisitor(SE), Valid(true), UseMax(UseMax), OldL(OldL),
          NewL(NewL) {}

    const SCEV *visitAddRecExpr(const SCEVAddRecExpr *Expr) {
      const Loop *ExprL = Expr->getLoop();
      SmallVector<const SCEV *, 2> Operands;
      if (ExprL == &OldL) {
        append_range(Operands, Expr->operands());
        return SE.getAddRecExpr(Operands, &NewL, Expr->getNoWrapFlags());
      }

      if (OldL.contains(ExprL)) {
        bool Pos = SE.isKnownPositive(Expr->getStepRecurrence(SE));
        if (!UseMax || !Pos || !Expr->isAffine()) {
          Valid = false;
          return Expr;
        }
        return visit(Expr->getStart());
      }

      for (const SCEV *Op : Expr->operands())
        Operands.push_back(visit(Op));
      return SE.getAddRecExpr(Operands, ExprL, Expr->getNoWrapFlags());
    }

    bool wasValidSCEV() const { return Valid; }

  private:
    bool Valid, UseMax;
    const Loop &OldL, &NewL;
  };

  /// Return false if the access functions of \p I0 and \p I1 could cause
  /// a negative dependence.
  bool accessDiffIsPositive(const Loop &L0, const Loop &L1, Instruction &I0,
                            Instruction &I1, bool EqualIsInvalid) {
    Value *Ptr0 = getLoadStorePointerOperand(&I0);
    Value *Ptr1 = getLoadStorePointerOperand(&I1);
    if (!Ptr0 || !Ptr1)
      return false;

    const SCEV *SCEVPtr0 = SE.getSCEVAtScope(Ptr0, &L0);
    const SCEV *SCEVPtr1 = SE.getSCEVAtScope(Ptr1, &L1);
#ifndef NDEBUG
    if (VerboseConcatDebugging)
      LLVM_DEBUG(dbgs() << "    Access function check: " << *SCEVPtr0 << " vs "
                        << *SCEVPtr1 << "\n");
#endif
    AddRecLoopReplacer Rewriter(SE, L0, L1);
    SCEVPtr0 = Rewriter.visit(SCEVPtr0);
#ifndef NDEBUG
    if (VerboseConcatDebugging)
      LLVM_DEBUG(dbgs() << "    Access function after rewrite: " << *SCEVPtr0
                        << " [Valid: " << Rewriter.wasValidSCEV() << "]\n");
#endif
    if (!Rewriter.wasValidSCEV())
      return false;

    // TODO: isKnownPredicate doesnt work well when one SCEV is loop carried (by
    //       L0) and the other is not. We could check if it is monotone and test
    //       the beginning and end value instead.

    BasicBlock *L0Header = L0.getHeader();
    auto HasNonLinearDominanceRelation = [&](const SCEV *S) {
      const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(S);
      if (!AddRec)
        return false;
      return !DT.dominates(L0Header, AddRec->getLoop()->getHeader()) &&
             !DT.dominates(AddRec->getLoop()->getHeader(), L0Header);
    };
    if (SCEVExprContains(SCEVPtr1, HasNonLinearDominanceRelation))
      return false;

    ICmpInst::Predicate Pred =
        EqualIsInvalid ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_SGE;
    bool IsAlwaysGE = SE.isKnownPredicate(Pred, SCEVPtr0, SCEVPtr1);
#ifndef NDEBUG
    if (VerboseConcatDebugging)
      LLVM_DEBUG(dbgs() << "    Relation: " << *SCEVPtr0
                        << (IsAlwaysGE ? "  >=  " : "  may <  ") << *SCEVPtr1
                        << "\n");
#endif
    return IsAlwaysGE;
  }

  /// Return true if the dependences between @p I0 (in @p L0) and @p I1 (in
  /// @p L1) allow loop concatenation of @p L0 and @p L1. The dependence
  /// analyses specified by @p DepChoice are used to determine this.
  bool dependencesAllowConcatenation(const ConcatCandidate &CC0,
                                     const ConcatCandidate &CC1,
                                     Instruction &I0, Instruction &I1,
                                     bool AnyDep,
                                     ConcatDependenceAnalysisChoice DepChoice) {
#ifndef NDEBUG
    if (VerboseConcatDebugging) {
      LLVM_DEBUG(dbgs() << "Check dep: " << I0 << " vs " << I1 << " : "
                        << DepChoice << "\n");
    }
#endif
    switch (DepChoice) {
    case CONCAT_DEPENDENCE_ANALYSIS_SCEV:
      return accessDiffIsPositive(*CC0.L, *CC1.L, I0, I1, AnyDep);
    case CONCAT_DEPENDENCE_ANALYSIS_DA: {
      auto DepResult = DI.depends(&I0, &I1, true);
      if (!DepResult)
        return true;
#ifndef NDEBUG
      if (VerboseConcatDebugging) {
        LLVM_DEBUG(dbgs() << "DA res: "; DepResult->dump(dbgs());
                   dbgs() << " [#l: " << DepResult->getLevels() << "][Ordered: "
                          << (DepResult->isOrdered() ? "true" : "false")
                          << "]\n");
        LLVM_DEBUG(dbgs() << "DepResult Levels: " << DepResult->getLevels()
                          << "\n");
      }
#endif

      if (DepResult->getNextPredecessor() || DepResult->getNextSuccessor())
        LLVM_DEBUG(
            dbgs() << "TODO: Implement pred/succ dependence handling!\n");

      // TODO: Can we actually use the dependence info analysis here?
      return false;
    }

    case CONCAT_DEPENDENCE_ANALYSIS_ALL:
      return dependencesAllowConcatenation(CC0, CC1, I0, I1, AnyDep,
                                           CONCAT_DEPENDENCE_ANALYSIS_SCEV) ||
             dependencesAllowConcatenation(CC0, CC1, I0, I1, AnyDep,
                                           CONCAT_DEPENDENCE_ANALYSIS_DA);
    }

    llvm_unreachable("Unknown concatenation dependence analysis choice!");
  }

  bool valueMayDefFromCandidate(const ConcatCandidate &CC0,
                                Instruction *CurInst) {
    if (CC0.L->contains(CurInst))
      return true;

    if (auto *PN = dyn_cast<PHINode>(CurInst)) {
      // Walk PHINode chains to see if CC0 provides a value for PN
      Loop *CurLoop = LI.getLoopFor(CurInst->getParent());
      for (unsigned It = 0, E = PN->getNumIncomingValues(); It != E; ++It) {
        auto *InputInst = dyn_cast<Instruction>(PN->getIncomingValue(It));
        if (!InputInst)
          continue;

        // Value modified in a loop other than CC0.
        if (CurLoop && CurLoop->contains(InputInst))
          return false;

        if (valueMayDefFromCandidate(CC0, InputInst))
          return true;
      }
    }

    return false;
  }

  /// Perform a dependence check and return if @p CC0 and @p CC1 can be
  /// concatenated.
  bool dependencesAllowConcatenation(const ConcatCandidate &CC0,
                                     const ConcatCandidate &CC1,
                                     bool AllowInterCandidateDeps) {
    LLVM_DEBUG(dbgs() << "Check if " << CC0 << " can be concatenated with "
                      << CC1 << "\n");
    assert(CC0.L->getLoopDepth() == CC1.L->getLoopDepth());
    // If the candidates are not dominance ordered then we cannot
    // analyze dependences.
    if (!DT.dominates(CC0.getEntryBlock(), CC1.getEntryBlock()))
      return false;

    for (Instruction *WriteL0 : CC0.MemWrites) {
      for (Instruction *WriteL1 : CC1.MemWrites)
        if (!dependencesAllowConcatenation(CC0, CC1, *WriteL0, *WriteL1,
                                           /* AnyDep */ false,
                                           ConcatDependenceAnalysis)) {
          InvalidDependencies++;
          return false;
        }
    }

    // Walk through all uses in CC1. For each use, find the reaching def. If the
    // def is located in CC0 then it is not safe to concatenate unless it's a
    // loop input value from a PHINode in CC0 to a PHINode in CC1.
    for (BasicBlock *BB : CC1.L->blocks())
      for (Instruction &I : *BB)
        for (auto &Op : I.operands())
          if (Instruction *Def = dyn_cast<Instruction>(Op)) {
            BasicBlock *DefBlock = Def->getParent();
            if (CC0.L->contains(DefBlock)) {
              InvalidDependencies++;
              return false;
            }

            if (isa<PHINode>(&I) && I.getType()->isPointerTy())
              if (auto *PN = dyn_cast<PHINode>(Def))
                for (unsigned It = 0, E = PN->getNumIncomingValues(); It != E;
                     ++It) {
                  auto *InputVal = PN->getIncomingValue(It);
                  if (auto *InputInst = dyn_cast<Instruction>(InputVal)) {
                    // Skip backedge values of PN
                    if (CC1.L->contains(InputInst))
                      continue;

                    // If joinable, InputInst must have a may def from CC0,
                    // else it invalidates a concatenation candidate.
                    // If not joinable, any inter candidate def invalidates
                    // this candidate.
                    if (AllowInterCandidateDeps &&
                        !valueMayDefFromCandidate(CC0, InputInst)) {
                      InvalidDependencies++;
                      return false;
                    } else if (!AllowInterCandidateDeps &&
                               valueMayDefFromCandidate(CC0, InputInst)) {
                      InvalidDependencies++;
                      return false;
                    }
                  }
                }
          }

    return true;
  }

  /// Check if both candidate loops have identical code
  bool compareAllowConcatenation(const ConcatCandidate &CC0,
                                 const ConcatCandidate &CC1,
                                 bool HasJoinableTripRange) {
    // Only allow Candidates with joinable trip ranges to be checked.
    if (!HasJoinableTripRange)
      return false;

    LLVM_DEBUG(dbgs() << "Check if " << CC0 << " is identical to " << CC1);

    // Collect all CC0's instructions so we can do a side by side
    // comparison.
    CC0Map.clear();
    LoopInsns.clear();
    unsigned Idx = 0;
    for (BasicBlock *BB : CC0.L->blocks())
      for (Instruction &I : *BB) {
        LoopInsns.push_back(&I);
        CC0Map[&I] = Idx++;
      }

    // First do a quick pass checking ordered opcodes of
    // each instruction
    Idx = 0;
    CC1Map.clear();
    for (BasicBlock *BB : CC1.L->blocks())
      for (Instruction &I : *BB) {
        Instruction *CC0I = LoopInsns[Idx];
        CC1Map[&I] = Idx++;
        // Now compare I and CC0I for syntatic equivalence
        if (I.getOpcode() != CC0I->getOpcode())
          return false;

        // Operand footprint needs to match
        if (I.getNumOperands() != CC0I->getNumOperands())
          return false;
      }

    // Ensure the loop IV's are in the same place
    PHINode *CC0_IV = CC0.L->getInductionVariable(SE);
    PHINode *CC1_IV = CC1.L->getInductionVariable(SE);
    if (CC0Map[CC0_IV] != CC1Map[CC1_IV])
      return false;

    BasicBlock *Latch = CC1.L->getLoopLatch();
    auto *LatchBr = dyn_cast<BranchInst>(Latch->getTerminator());
    if (!LatchBr || LatchBr->isUnconditional()) {
      LLVM_DEBUG(dbgs() << "LC: Unsupported loop latch branch.\n");
      return false;
    }

    ICmpInst *LatchCmp = dyn_cast<ICmpInst>(LatchBr->getCondition());
    if (!LatchCmp) {
      LLVM_DEBUG(dbgs() << "LC: Unsupported loop latch branch.\n");
      return false;
    }

    // Now do a semantic comparison of CC0 and CC1
    for (BasicBlock *BB : CC1.L->blocks())
      for (Instruction &I : *BB) {
        auto *MapI = LoopInsns[CC1Map[&I]];
        // The majority of PHINode components will be evaluated in dependence
        // checks.  Here we evaluate only incoming edges that are not loop
        // carried for values that do not originate from an Instruction.
        if (auto *PN = dyn_cast<PHINode>(&I)) {
          if (I.getType()->isPointerTy())
            for (unsigned It = 0, E = PN->getNumIncomingValues(); It != E;
                 ++It) {
              auto *InputVal = PN->getIncomingValue(It);
              // Only evaluate non Instruction values here.
              if (isa<Instruction>(InputVal))
                continue;

              auto *MapPN = dyn_cast<PHINode>(MapI);
              if (MapPN && MapPN->getIncomingValue(It) != InputVal)
                return false;
            }

          continue;
        }

        if (LatchBr == dyn_cast<BranchInst>(&I))
          continue;

        if (LatchCmp == dyn_cast<ICmpInst>(&I)) {
          auto *MapCmp = dyn_cast<ICmpInst>(MapI);
          if (!MapCmp)
            return false;

          // Predicates must match.  TODO: or possibly equate...
          if (LatchCmp->getPredicate() != MapCmp->getPredicate())
            return false;

          continue;
        }

        unsigned NumOps = I.getNumOperands();
        // We already know CC0 and CC1 are ordered the same, so we
        // can use this loops map index for obtaining the mapped CC0
        // instruction.
        for (unsigned OpIdx = 0; OpIdx < NumOps; OpIdx++) {
          auto *Op = I.getOperand(OpIdx);
          auto *MapOp = MapI->getOperand(OpIdx);
          if (Instruction *Def = dyn_cast<Instruction>(Op)) {
            if (Def == &I)
              continue;

            if (CC1.L->contains(Def->getParent())) {
              // Both Op and MapOp must belong to the same mapping.
              if (auto *MapDef = dyn_cast<Instruction>(MapOp)) {
                // If Def is in CC1, MapOp must be in CC0
                if (!CC0.L->contains(MapDef->getParent()))
                  return false;

                // Both come from their loops, they must map the same way
                if (CC0Map[MapDef] != CC1Map[Def])
                  return false;
              } else {
                // Op and MapOp diverge
                return false;
              }
            } else if (Op != MapOp) {
              // If Def is from outside its loop, both must match.
              return false;
            }
          } else if (ConstantInt *OpConst = dyn_cast<ConstantInt>(Op)) {
            // Both operands must be constants and the same
            if (OpConst != dyn_cast<ConstantInt>(MapOp))
              return false;
          }
        }
      }

    return true;
  }

  bool isEmptyFlowEdge(const BasicBlock *BB) const { return BB->size() == 1; }

  /// Hoist \p CC1 Preheader instructions to \p CC0 Preheader
  /// and sink others into the body of \p CC1.
  void movePreheaderInsts(const ConcatCandidate &CC0,
                          const ConcatCandidate &CC1,
                          SmallVector<Instruction *, 4> &HoistInsts,
                          SmallVector<Instruction *, 4> &SinkInsts) const {
    // All preheader instructions except the branch must be hoisted or sunk
    assert(HoistInsts.size() + SinkInsts.size() == CC1.Preheader->size() - 1 &&
           "Attempting to sink and hoist preheader instructions, but not all "
           "the preheader instructions are accounted for.");

    NumHoistedInsts += HoistInsts.size();
    NumSunkInsts += SinkInsts.size();

    LLVM_DEBUG(if (VerboseConcatDebugging) {
      if (!HoistInsts.empty())
        dbgs() << "Hoisting: \n";
      for (Instruction *I : HoistInsts)
        dbgs() << *I << "\n";
      if (!SinkInsts.empty())
        dbgs() << "Sinking: \n";
      for (Instruction *I : SinkInsts)
        dbgs() << *I << "\n";
    });

    for (Instruction *I : HoistInsts) {
      assert(I->getParent() == CC1.Preheader);
      I->moveBefore(*CC0.Preheader,
                    CC0.Preheader->getTerminator()->getIterator());
    }
    // insert instructions in reverse order to maintain dominance relationship
    for (Instruction *I : reverse(SinkInsts)) {
      assert(I->getParent() == CC1.Preheader);
      I->moveBefore(*CC1.ExitBlock, CC1.ExitBlock->getFirstInsertionPt());
    }
  }

  /// Determine if two concatenation candidates have identical guards
  ///
  /// This method will determine if two concatenation candidates have the same
  /// guards.  The guards are considered the same if:
  ///   1. The instructions to compute the condition used in the compare are
  ///      identical.
  ///   2. The successors of the guard have the same flow into/around the loop.
  /// If the compare instructions are identical, then the first successor of the
  /// guard must go to the same place (either the preheader of the loop or the
  /// NonLoopBlock). In other words, the first successor of both loops must
  /// both go into the loop (i.e., the preheader) or go around the loop (i.e.,
  /// the NonLoopBlock). The same must be true for the second successor.
  bool haveIdenticalGuards(const ConcatCandidate &CC0,
                           const ConcatCandidate &CC1) const {
    assert(CC0.GuardBranch && CC1.GuardBranch &&
           "Expecting CC0 and CC1 to be guarded loops.");

    if (auto CC0CmpInst =
            dyn_cast<Instruction>(CC0.GuardBranch->getCondition()))
      if (auto CC1CmpInst =
              dyn_cast<Instruction>(CC1.GuardBranch->getCondition()))
        if (!CC0CmpInst->isIdenticalTo(CC1CmpInst))
          return false;

    // The compare instructions are identical.
    // Now make sure the successor of the guards have the same flow into/around
    // the loop
    if (CC0.GuardBranch->getSuccessor(0) == CC0.Preheader)
      return (CC1.GuardBranch->getSuccessor(0) == CC1.Preheader);
    else
      return (CC1.GuardBranch->getSuccessor(1) == CC1.Preheader);
  }

  /// Fetch the upper bound of CC1.Latch and apply that to CC0.Latch
  /// as its upper bound.
  void updateLatch(const ConcatCandidate &CC0, const ConcatCandidate &CC1) {
    PHINode *CC0_IV = CC0.L->getInductionVariable(SE);
    PHINode *CC1_IV = CC1.L->getInductionVariable(SE);
    auto CC0_LB = Loop::LoopBounds::getBounds(*CC0.L, *CC0_IV, SE);
    auto CC1_LB = Loop::LoopBounds::getBounds(*CC1.L, *CC1_IV, SE);
    Value *CC0FinalIvVal = &CC0_LB->getFinalIVValue();
    Value *CC1FinalIvVal = &CC1_LB->getFinalIVValue();
    BasicBlock *Latch = CC0.L->getLoopLatch();
    auto *LatchBr = cast<BranchInst>(Latch->getTerminator());
    ICmpInst *LatchCmp = cast<ICmpInst>(LatchBr->getCondition());
    unsigned NumOps = LatchCmp->getNumOperands();
    for (unsigned OpIdx = 0; OpIdx < NumOps; OpIdx++) {
      auto *Op = LatchCmp->getOperand(OpIdx);
      // Recall the loops are identical, so we can just
      // substitute the upper bound from CC1 here.
      if (Op == CC0FinalIvVal)
        LatchCmp->setOperand(OpIdx, CC1FinalIvVal);
    }
  }

  /// Concatenate two concatenation candidates, creating a new loop.
  ///
  /// This method contains the mechanics of concatenating two loops,
  /// represented by \p CC0 and \p CC1. It is assumed that \p CC0
  /// dominates \p CC1 and \p CC1 postdominates \p CC0 (making them control
  /// flow equivalent). It also assumes that the other conditions for
  /// concatenation have been met: joinable trip ranges, and no
  /// negative distance dependencies exist that would prevent concatenation.
  /// Thus, there is no checking for these conditions in this method.
  ///
  /// Recall that the body of both CC0 and CC1 are identical.
  /// The first thing we do here is move any code in CC1's preheader
  /// to end of CC0's preheader.  Then we collect all the uses of values
  /// defined in CC1 that escape it to add to their partner values in CC0
  /// as input to PHInodes added CC1's preheader.  The definitions of
  /// these PHINodes will then replace said values provided by CC1.
  /// CC1 is then removed by connecting the CC1's preheader to CC1's
  /// exit block.  Any code in CC1's exit block still needs to be executed.
  /// Concatenation is performed when CC0's latch compare is updated with
  /// CC1's compare.
  ///
  /// All of these modifications are done with dominator tree updates, thus
  /// keeping the dominator (and post dominator) information up-to-date.
  Loop *performConcatenation(const ConcatCandidate &CC0,
                             const ConcatCandidate &CC1) {
    assert(CC0.isValid() && CC1.isValid() &&
           "Expecting valid concatenation candidates");

    LLVM_DEBUG(dbgs() << "Concatenation Candidate 0: \n"; CC0.dump();
               dbgs() << "Concatenation Candidate 1: \n"; CC1.dump(););

    // Move instructions from the preheader of CC1 to the end of the preheader
    // of CC0.
    moveInstructionsToTheEnd(*CC1.Preheader, *CC0.Preheader, DT, PDT, DI);

    // Concatenating guarded loops is handled slightly differently than
    // non-guarded loops and has been broken out into a separate method instead
    // of trying to intersperse the logic within a single method.
    if (CC0.GuardBranch)
      return concatGuardedLoops(CC0, CC1);

    assert(CC1.Preheader->size() == 1 &&
           CC1.Preheader->getSingleSuccessor() == CC1.Header);

    // Process every value that escapes CC1.
    for (BasicBlock *BB : CC1.L->blocks())
      for (Instruction &I : *BB) {
        SmallVector<Instruction *, 10> Worklist;
        for (User *U : I.users()) {
          auto *UseI = cast<Instruction>(U);
          if (!CC1.L->contains(UseI))
            Worklist.push_back(UseI);
        }

        if (Worklist.empty())
          continue;

        int idx = CC1Map[&I];
        auto *CC0MapI = LoopInsns[idx];

        // Create a PHINode in CC0's ExitBlock and use the corresponding
        // mapped I from CC0 as the input, then replace each escaping
        // use at its site with the PHINode value.
        BasicBlock::iterator L1ExitIP = CC1.ExitBlock->begin();
        PHINode *L1ExitPHI = PHINode::Create(CC0MapI->getType(), 2,
                                             CC0MapI->getName() + ".afterCC1");
        L1ExitPHI->insertBefore(L1ExitIP);
        L1ExitPHI->addIncoming(CC0MapI, CC0MapI->getParent());
        while (!Worklist.empty()) {
          auto *UseI = Worklist.pop_back_val();
          unsigned NumOps = UseI->getNumOperands();
          for (unsigned OpIdx = 0; OpIdx < NumOps; OpIdx++) {
            auto *Op = UseI->getOperand(OpIdx);
            if (Op == &I)
              UseI->setOperand(OpIdx, L1ExitPHI);
          }
        }
      }

    // Udpate CC0 with CC1's latch upper bounds
    updateLatch(CC0, CC1);
    SE.forgetLoop(CC0.L);

    // Delete loop CC1.
    deleteDeadLoop(CC1.L, &DT, &SE, &LI);

#ifndef NDEBUG
    assert(!verifyFunction(*CC0.Header->getParent(), &errs()));
    assert(DT.verify(DominatorTree::VerificationLevel::Fast));
    PDT.recalculate(*CC0.Header->getParent());
    LI.verify(DT);
    SE.verify();
#endif

    LLVM_DEBUG(dbgs() << "Concatenation done:\n");

    return CC0.L;
  }

  /// Report details on loop concatenation opportunities.
  ///
  /// This template function can be used to report both successful and missed
  /// loop concatenation opportunities, based on the RemarkKind. The RemarkKind
  /// should be one of:
  ///   - OptimizationRemarkMissed to report when loop concatenation is
  ///   unsuccessful
  ///     given two valid concatenation candidates.
  ///   - OptimizationRemark to report successful concatenation of two
  ///   candidates.
  /// The remarks will be printed using the form:
  ///    <path/filename>:<line number>:<column number>: [<function name>]:
  ///       <Cand1 Preheader> and <Cand2 Preheader>: <Stat Description>
  template <typename RemarkKind>
  void reportLoopConcatenation(const ConcatCandidate &CC0,
                               const ConcatCandidate &CC1,
                               llvm::Statistic &Stat) {
    assert(CC0.Preheader && CC1.Preheader &&
           "Expecting valid concatenation candidates");
    using namespace ore;
#if LLVM_ENABLE_STATS
    ++Stat;
    ORE.emit(RemarkKind(DEBUG_TYPE, Stat.getName(), CC0.L->getStartLoc(),
                        CC0.Preheader)
             << "[" << CC0.Preheader->getParent()->getName()
             << "]: " << NV("Cand1", StringRef(CC0.Preheader->getName()))
             << " and " << NV("Cand2", StringRef(CC1.Preheader->getName()))
             << ": " << Stat.getDesc());
#endif
  }

  /// Concatenate two guarded concatenate candidates, creating a new
  /// concatenated loop.
  /// TODO: Validate this implemenation and complete.
  ///
  /// Concatenating guarded loops is handled much the same way as concatenating
  /// non-guarded loops. The rewiring of the CFG is slightly different though,
  /// because of the presence of the guards around the loops and the exit
  /// blocks after the loop body. As such, the new loop is rewired as follows:
  ///    1. Keep the guard branch from CC0 and use the non-loop block target
  /// from the CC1 guard branch.
  ///    2. Remove the exit block from CC0 (this exit block should be empty
  /// right now).
  ///    3. Remove the guard branch for CC1
  ///    4. Remove the preheader for CC1.
  /// The exit block successor for the latch of CC0 is updated to be the header
  /// of CC1 and the non-exit block successor of the latch of CC1 is updated to
  /// be the header of CC0, thus creating the concatenated loop.
  Loop *concatGuardedLoops(const ConcatCandidate &CC0,
                           const ConcatCandidate &CC1) {
    assert(CC0.GuardBranch && CC1.GuardBranch && "Expecting guarded loops");
    assert(CC1.Preheader->size() == 1 &&
           CC1.Preheader->getSingleSuccessor() == CC1.Header);

    // Process every value that escapes CC1.
    for (BasicBlock *BB : CC1.L->blocks())
      for (Instruction &I : *BB) {
        SmallVector<Instruction *, 10> Worklist;
        for (User *U : I.users()) {
          auto *UseI = cast<Instruction>(U);
          if (!CC1.L->contains(UseI))
            Worklist.push_back(UseI);
        }

        if (Worklist.empty())
          continue;

        int idx = CC1Map[&I];
        auto *CC0MapI = LoopInsns[idx];

        // Create a PHINode in CC0's ExitBlock and use the corresponding
        // mapped I from CC0 as the input.  TODO: Add phis to CC1's guard block
        // to carry the values and then replace each escaping
        // use at its site with the PHINode value.
        BasicBlock::iterator L1ExitIP = CC1.ExitBlock->begin();
        PHINode *L1ExitPHI = PHINode::Create(CC0MapI->getType(), 2,
                                             CC0MapI->getName() + ".afterCC1");
        L1ExitPHI->insertBefore(L1ExitIP);
        L1ExitPHI->addIncoming(CC0MapI, CC0MapI->getParent());
        while (!Worklist.empty()) {
          auto *UseI = Worklist.pop_back_val();
          unsigned NumOps = UseI->getNumOperands();
          for (unsigned OpIdx = 0; OpIdx < NumOps; OpIdx++) {
            auto *Op = UseI->getOperand(OpIdx);
            if (Op == &I)
              UseI->setOperand(OpIdx, L1ExitPHI);
          }
        }
      }

    // Udpate CC0 with CC1's latch upper bounds
    updateLatch(CC0, CC1);
    SE.forgetLoop(CC0.L);

    // Delete loop CC1.
    deleteDeadLoop(CC1.L, &DT, &SE, &LI);

#ifndef NDEBUG
    assert(!verifyFunction(*CC0.Header->getParent(), &errs()));
    assert(DT.verify(DominatorTree::VerificationLevel::Fast));
    PDT.recalculate(*CC0.Header->getParent());
    LI.verify(DT);
    SE.verify();
#endif

    LLVM_DEBUG(dbgs() << "Concatenation done:\n");

    return CC0.L;
  }
};
} // namespace

PreservedAnalyses LoopConcatPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &DI = AM.getResult<DependenceAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  const DataLayout &DL = F.getParent()->getDataLayout();

  if (!EnableLoopConcatenation)
    return PreservedAnalyses::all();

  // Ensure loops are in simplifed form which is a pre-requisite for loop
  // concatenation pass. Added only for new PM since the legacy PM has already
  // added LoopSimplify pass as a dependency.
  bool Changed = false;
  for (auto &L : LI) {
    Changed |=
        simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false /* PreserveLCSSA */);
  }
  if (Changed)
    PDT.recalculate(F);

  LoopConcater LF(LI, DT, DI, SE, PDT, ORE, DL);
  Changed |= LF.concatLoops(F);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}
