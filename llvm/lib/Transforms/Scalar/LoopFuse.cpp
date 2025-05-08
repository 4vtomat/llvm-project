//===- LoopFuse.cpp - Loop Fusion Pass ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the loop fusion pass.
/// The implementation is largely based on the following document:
///
///       Code Transformations to Augment the Scope of Loop Fusion in a
///         Production Compiler
///       Christopher Mark Barton
///       MSc Thesis
///       https://webdocs.cs.ualberta.ca/~amaral/thesis/ChristopherBartonMSc.pdf
///
/// The general approach taken is to collect sets of control flow equivalent
/// loops and test whether they can be fused. The necessary conditions for
/// fusion are:
///    1. The loops must be adjacent (there cannot be any statements between
///       the two loops).
///    2. The loops must be conforming (they must execute the same number of
///       iterations).
///    3. The loops must be control flow equivalent (if one loop executes, the
///       other is guaranteed to execute).
///    4. There cannot be any negative distance dependencies between the loops.
/// If all of these conditions are satisfied, it is safe to fuse the loops.
///
/// This implementation creates FusionCandidates that represent the loop and the
/// necessary information needed by fusion. It then operates on the fusion
/// candidates, first confirming that the candidate is eligible for fusion. The
/// candidates are then collected into control flow equivalent sets, sorted in
/// dominance order. Each set of control flow equivalent candidates is then
/// traversed, attempting to fuse pairs of candidates in the set. If all
/// requirements for fusion are met, the two candidates are fused, creating a
/// new (fused) candidate which is then added back into the set to consider for
/// additional fusion.
///
/// This implementation currently does not make any modifications to remove
/// conditions for fusion. Code transformations to make loops conform to each of
/// the conditions for fusion are discussed in more detail in the document
/// above. These can be added to the current implementation in the future.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopFuse.h"
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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeMoverUtils.h"
#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#if SIFIVE_CUSTOMIZATION
#include "llvm/Transforms/Utils/LoopUtils.h"
#endif

using namespace llvm;

#define DEBUG_TYPE "loop-fusion"

STATISTIC(FuseCounter, "Loops fused");
STATISTIC(NumFusionCandidates, "Number of candidates for loop fusion");
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
STATISTIC(InvalidDependencies, "Dependencies prevent fusion");
STATISTIC(UnknownTripCount, "Loop has unknown trip count");
STATISTIC(UncomputableTripCount, "SCEV cannot compute trip count of loop");
STATISTIC(NonEqualTripCount, "Loop trip counts are not the same");
STATISTIC(NonAdjacent, "Loops are not adjacent");
STATISTIC(
    NonEmptyPreheader,
    "Loop has a non-empty preheader with instructions that cannot be moved");
STATISTIC(FusionNotBeneficial, "Fusion is not beneficial");
STATISTIC(NonIdenticalGuards, "Candidates have different guards");
STATISTIC(NonEmptyExitBlock, "Candidate has a non-empty exit block with "
                             "instructions that cannot be moved");
STATISTIC(NonEmptyGuardBlock, "Candidate has a non-empty guard block with "
                              "instructions that cannot be moved");
STATISTIC(NotRotated, "Candidate is not rotated");
STATISTIC(OnlySecondCandidateIsGuarded,
          "The second candidate is guarded while the first one is not");
STATISTIC(NumHoistedInsts, "Number of hoisted preheader instructions.");
STATISTIC(NumSunkInsts, "Number of hoisted preheader instructions.");

enum FusionDependenceAnalysisChoice {
  FUSION_DEPENDENCE_ANALYSIS_SCEV,
  FUSION_DEPENDENCE_ANALYSIS_DA,
  FUSION_DEPENDENCE_ANALYSIS_ALL,
};

static cl::opt<FusionDependenceAnalysisChoice> FusionDependenceAnalysis(
    "loop-fusion-dependence-analysis",
    cl::desc("Which dependence analysis should loop fusion use?"),
    cl::values(clEnumValN(FUSION_DEPENDENCE_ANALYSIS_SCEV, "scev",
                          "Use the scalar evolution interface"),
               clEnumValN(FUSION_DEPENDENCE_ANALYSIS_DA, "da",
                          "Use the dependence analysis interface"),
               clEnumValN(FUSION_DEPENDENCE_ANALYSIS_ALL, "all",
                          "Use all available analyses")),
    cl::Hidden, cl::init(FUSION_DEPENDENCE_ANALYSIS_ALL));

#if SIFIVE_CUSTOMIZATION
static cl::opt<bool>
    EnableLoopFusion("loop-fusion", cl::Hidden, cl::init(false),
                     cl::desc("Enable Loop Fusion"));
#endif

static cl::opt<unsigned> FusionPeelMaxCount(
    "loop-fusion-peel-max-count", cl::init(0), cl::Hidden,
    cl::desc("Max number of iterations to be peeled from a loop, such that "
             "fusion can take place"));

#if SIFIVE_CUSTOMIZATION
extern cl::opt<bool> LoopConcatCanonicalize;
#endif

#ifndef NDEBUG
static cl::opt<bool>
    VerboseFusionDebugging("loop-fusion-verbose-debug",
                           cl::desc("Enable verbose debugging for Loop Fusion"),
                           cl::Hidden, cl::init(false));
#endif

namespace {
/// This class is used to represent a candidate for loop fusion. When it is
/// constructed, it checks the conditions for loop fusion to ensure that it
/// represents a valid candidate. It caches several parts of a loop that are
/// used throughout loop fusion (e.g., loop preheader, loop header, etc) instead
/// of continually querying the underlying Loop to retrieve these values. It is
/// assumed these will not change throughout loop fusion.
///
/// The invalidate method should be used to indicate that the FusionCandidate is
/// no longer a valid candidate for fusion. Similarly, the isValid() method can
/// be used to ensure that the FusionCandidate is still valid for fusion.
struct FusionCandidate {
  /// Cache of parts of the loop used throughout loop fusion. These should not
  /// need to change throughout the analysis and transformation.
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
  /// The loop that this fusion candidate represents
  Loop *L;
  /// Vector of instructions in this loop that read from memory
  SmallVector<Instruction *, 16> MemReads;
  /// Vector of instructions in this loop that write to memory
  SmallVector<Instruction *, 16> MemWrites;
  /// Are all of the members of this fusion candidate still valid
  bool Valid;
  /// Guard branch of the loop, if it exists
  BranchInst *GuardBranch;
  /// Peeling Paramaters of the Loop.
  TTI::PeelingPreferences PP;
  /// Can you Peel this Loop?
  bool AbleToPeel;
  /// Has this loop been Peeled
  bool Peeled;
#if SIFIVE_CUSTOMIZATION
  /// Recorded fuse IV
  PHINode *IV;
  /// Recorded fuse reduction IV
  PHINode *RIV;
#endif

  /// Dominator and PostDominator trees are needed for the
  /// FusionCandidateCompare function, required by FusionCandidateSet to
  /// determine where the FusionCandidate should be inserted into the set. These
  /// are used to establish ordering of the FusionCandidates based on dominance.
  DominatorTree &DT;
  const PostDominatorTree *PDT;

  OptimizationRemarkEmitter &ORE;

#if SIFIVE_CUSTOMIZATION
  FusionCandidate(Loop *L, DominatorTree &DT, const PostDominatorTree *PDT,
                  OptimizationRemarkEmitter &ORE, TTI::PeelingPreferences PP,
                  PHINode *IV = nullptr, PHINode *RIV = nullptr)
      : Preheader(L->getLoopPreheader()), Header(L->getHeader()),
        ExitingBlock(L->getExitingBlock()), ExitBlock(L->getExitBlock()),
        Latch(L->getLoopLatch()), L(L), Valid(true),
        GuardBranch(L->getLoopGuardBranch()), PP(PP), AbleToPeel(canPeel(L)),
        Peeled(false), IV(IV), RIV(RIV), DT(DT), PDT(PDT), ORE(ORE) {
#else
  FusionCandidate(Loop *L, DominatorTree &DT, const PostDominatorTree *PDT,
                  OptimizationRemarkEmitter &ORE, TTI::PeelingPreferences PP)
      : Preheader(L->getLoopPreheader()), Header(L->getHeader()),
        ExitingBlock(L->getExitingBlock()), ExitBlock(L->getExitBlock()),
        Latch(L->getLoopLatch()), L(L), Valid(true),
        GuardBranch(L->getLoopGuardBranch()), PP(PP), AbleToPeel(canPeel(L)),
        Peeled(false), DT(DT), PDT(PDT), ORE(ORE) {
#endif

    // Walk over all blocks in the loop and check for conditions that may
    // prevent fusion. For each block, walk over all instructions and collect
    // the memory reads and writes If any instructions that prevent fusion are
    // found, invalidate this object and return.
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

  /// Get the entry block for this fusion candidate.
  ///
  /// If this fusion candidate represents a guarded loop, the entry block is the
  /// loop guard block. If it represents an unguarded loop, the entry block is
  /// the preheader of the loop.
  BasicBlock *getEntryBlock() const {
    if (GuardBranch)
      return GuardBranch->getParent();
    else
      return Preheader;
  }

  /// After Peeling the loop is modified quite a bit, hence all of the Blocks
  /// need to be updated accordingly.
  void updateAfterPeeling() {
    Preheader = L->getLoopPreheader();
    Header = L->getHeader();
    ExitingBlock = L->getExitingBlock();
    ExitBlock = L->getExitBlock();
    Latch = L->getLoopLatch();
    verify();
  }

  /// Given a guarded loop, get the successor of the guard that is not in the
  /// loop.
  ///
  /// This method returns the successor of the loop guard that is not located
  /// within the loop (i.e., the successor of the guard that is not the
  /// preheader).
  /// This method is only valid for guarded loops.
  BasicBlock *getNonLoopBlock() const {
    assert(GuardBranch && "Only valid on guarded loops.");
    assert(GuardBranch->isConditional() &&
           "Expecting guard to be a conditional branch.");
    if (Peeled)
      return GuardBranch->getSuccessor(1);
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

  /// Determine if a fusion candidate (representing a loop) is eligible for
  /// fusion. Note that this only checks whether a single loop can be fused - it
  /// does not check whether it is *legal* to fuse two loops together.
  bool isEligibleForFusion(ScalarEvolution &SE) const {
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
  // list and setting Valid to false. I can't envision other uses of this right
  // now, since once FusionCandidates are put into the FusionCandidateSet they
  // are immutable. Thus, any time we need to change/update a FusionCandidate,
  // we must create a new one and insert it into the FusionCandidateSet to
  // ensure the FusionCandidateSet remains ordered correctly.
  void invalidate() {
    MemWrites.clear();
    MemReads.clear();
    Valid = false;
  }

  bool reportInvalidCandidate(llvm::Statistic &Stat) const {
    using namespace ore;
    assert(L && Preheader && "Fusion candidate not initialized properly!");
#if LLVM_ENABLE_STATS
    ++Stat;
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, Stat.getName(),
                                        L->getStartLoc(), Preheader)
             << "[" << Preheader->getParent()->getName() << "]: "
             << "Loop is not a candidate for fusion: " << Stat.getDesc());
#endif
    return false;
  }
};

struct FusionCandidateCompare {
  /// Comparison functor to sort two Control Flow Equivalent fusion candidates
  /// into dominance order.
  /// If LHS dominates RHS and RHS post-dominates LHS, return true;
  /// If RHS dominates LHS and LHS post-dominates RHS, return false;
  /// If both LHS and RHS are not dominating each other then, non-strictly
  /// post dominate check will decide the order of candidates. If RHS
  /// non-strictly post dominates LHS then, return true. If LHS non-strictly
  /// post dominates RHS then, return false. If both are non-strictly post
  /// dominate each other then, level in the post dominator tree will decide
  /// the order of candidates.
  bool operator()(const FusionCandidate &LHS,
                  const FusionCandidate &RHS) const {
    const DominatorTree *DT = &(LHS.DT);

    BasicBlock *LHSEntryBlock = LHS.getEntryBlock();
    BasicBlock *RHSEntryBlock = RHS.getEntryBlock();

    // Do not save PDT to local variable as it is only used in asserts and thus
    // will trigger an unused variable warning if building without asserts.
    assert(DT && LHS.PDT && "Expecting valid dominator tree");

    // Do this compare first so if LHS == RHS, function returns false.
#if SIFIVE_CUSTOMIZATION
    if (DT->dominates(RHSEntryBlock, LHSEntryBlock)) {
      // Wrong order.
      return false;
    }

    if (DT->dominates(LHSEntryBlock, RHSEntryBlock)) {
      // Candidates ordered
      return LHS.PDT->dominates(RHSEntryBlock, LHSEntryBlock);
    }

    if (!isControlFlowEquivalent(*LHSEntryBlock, *RHSEntryBlock, *DT, *LHS.PDT))
      return false;
#else
    if (DT->dominates(RHSEntryBlock, LHSEntryBlock)) {
      // RHS dominates LHS
      // Verify LHS post-dominates RHS
      assert(LHS.PDT->dominates(LHSEntryBlock, RHSEntryBlock));
      return false;
    }

    if (DT->dominates(LHSEntryBlock, RHSEntryBlock)) {
      // Verify RHS Postdominates LHS
      assert(LHS.PDT->dominates(RHSEntryBlock, LHSEntryBlock));
      return true;
    }
#endif

    // If two FusionCandidates are in the same level of dominator tree,
    // they will not dominate each other, but may still be control flow
    // equivalent. To sort those FusionCandidates, nonStrictlyPostDominate()
    // function is needed.
    bool WrongOrder =
        nonStrictlyPostDominate(LHSEntryBlock, RHSEntryBlock, DT, LHS.PDT);
    bool RightOrder =
        nonStrictlyPostDominate(RHSEntryBlock, LHSEntryBlock, DT, LHS.PDT);
    if (WrongOrder && RightOrder) {
      // If common predecessor of LHS and RHS post dominates both
      // FusionCandidates then, Order of FusionCandidate can be
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
    // two FusionCandidates. Thus, they should not be in the same set together.
    llvm_unreachable(
        "No dominance relationship between these fusion candidates!");
  }
};

using LoopVector = SmallVector<Loop *, 4>;

// Set of Control Flow Equivalent (CFE) Fusion Candidates, sorted in dominance
// order. Thus, if FC0 comes *before* FC1 in a FusionCandidateSet, then FC0
// dominates FC1 and FC1 post-dominates FC0.
// std::set was chosen because we want a sorted data structure with stable
// iterators. A subsequent patch to loop fusion will enable fusing non-adjacent
// loops by moving intervening code around. When this intervening code contains
// loops, those loops will be moved also. The corresponding FusionCandidates
// will also need to be moved accordingly. As this is done, having stable
// iterators will simplify the logic. Similarly, having an efficient insert that
// keeps the FusionCandidateSet sorted will also simplify the implementation.
using FusionCandidateSet = std::set<FusionCandidate, FusionCandidateCompare>;
using FusionCandidateCollection = SmallVector<FusionCandidateSet, 4>;

#if !defined(NDEBUG)
static llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const FusionCandidate &FC) {
  if (FC.isValid())
    OS << FC.Preheader->getName();
  else
    OS << "<Invalid>";

  return OS;
}

static llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const FusionCandidateSet &CandSet) {
  for (const FusionCandidate &FC : CandSet)
    OS << FC << '\n';

  return OS;
}

static void
printFusionCandidates(const FusionCandidateCollection &FusionCandidates) {
  dbgs() << "Fusion Candidates: \n";
  for (const auto &CandidateSet : FusionCandidates) {
    dbgs() << "*** Fusion Candidate Set ***\n";
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

struct LoopFuser {
private:
  // Sets of control flow equivalent fusion candidates for a given nest level.
  FusionCandidateCollection FusionCandidates;

  LoopDepthTree LDT;
  DomTreeUpdater DTU;

  LoopInfo &LI;
  DominatorTree &DT;
  DependenceInfo &DI;
  ScalarEvolution &SE;
  PostDominatorTree &PDT;
  OptimizationRemarkEmitter &ORE;
  AssumptionCache &AC;
  const TargetTransformInfo &TTI;

public:
  LoopFuser(LoopInfo &LI, DominatorTree &DT, DependenceInfo &DI,
            ScalarEvolution &SE, PostDominatorTree &PDT,
            OptimizationRemarkEmitter &ORE, const DataLayout &DL,
            AssumptionCache &AC, const TargetTransformInfo &TTI)
      : LDT(LI), DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy), LI(LI),
        DT(DT), DI(DI), SE(SE), PDT(PDT), ORE(ORE), AC(AC), TTI(TTI) {}

  /// This is the main entry point for loop fusion. It will traverse the
  /// specified function and collect candidate loops to fuse, starting at the
  /// outermost nesting level and working inwards.
  bool fuseLoops(Function &F) {
#ifndef NDEBUG
    if (VerboseFusionDebugging) {
      LI.print(dbgs());
    }
#endif

    LLVM_DEBUG(dbgs() << "Performing Loop Fusion on function " << F.getName()
                      << "\n");
    bool Changed = false;

    while (!LDT.empty()) {
      LLVM_DEBUG(dbgs() << "Got " << LDT.size() << " loop sets for depth "
                        << LDT.getDepth() << "\n";);

      for (const LoopVector &LV : LDT) {
        assert(LV.size() > 0 && "Empty loop set was build!");

        // Skip singleton loop sets as they do not offer fusion opportunities on
        // this level.
        if (LV.size() == 1)
          continue;
#ifndef NDEBUG
        if (VerboseFusionDebugging) {
          LLVM_DEBUG({
            dbgs() << "  Visit loop set (#" << LV.size() << "):\n";
            printLoopVector(LV);
          });
        }
#endif

        collectFusionCandidates(LV);
        Changed |= fuseCandidates();
      }

      // Finished analyzing candidates at this level.
      // Descend to the next level and clear all of the candidates currently
      // collected. Note that it will not be possible to fuse any of the
      // existing candidates with new candidates because the new candidates will
      // be at a different nest level and thus not be control flow equivalent
      // with all of the candidates collected so far.
      LLVM_DEBUG(dbgs() << "Descend one level!\n");
      LDT.descend();
      FusionCandidates.clear();
    }

    if (Changed)
      LLVM_DEBUG(dbgs() << "Function after Loop Fusion: \n"; F.dump(););

#ifndef NDEBUG
    assert(DT.verify());
    assert(PDT.verify());
    LI.verify(DT);
    SE.verify();
#endif

    LLVM_DEBUG(dbgs() << "Loop Fusion complete\n");
    return Changed;
  }

private:
  /// Determine if two fusion candidates are control flow equivalent.
  ///
  /// Two fusion candidates are control flow equivalent if when one executes,
  /// the other is guaranteed to execute. This is determined using dominators
  /// and post-dominators: if A dominates B and B post-dominates A then A and B
  /// are control-flow equivalent.
  bool isControlFlowEquivalent(const FusionCandidate &FC0,
                               const FusionCandidate &FC1) const {
    assert(FC0.Preheader && FC1.Preheader && "Expecting valid preheaders");

    return ::isControlFlowEquivalent(*FC0.getEntryBlock(), *FC1.getEntryBlock(),
                                     DT, PDT);
  }

  /// Iterate over all loops in the given loop set and identify the loops that
  /// are eligible for fusion. Place all eligible fusion candidates into Control
  /// Flow Equivalent sets, sorted by dominance.
  void collectFusionCandidates(const LoopVector &LV) {
    for (Loop *L : LV) {
      TTI::PeelingPreferences PP =
          gatherPeelingPreferences(L, SE, TTI, std::nullopt, std::nullopt);
      FusionCandidate CurrCand(L, DT, &PDT, ORE, PP);
      if (!CurrCand.isEligibleForFusion(SE))
        continue;

      // Go through each list in FusionCandidates and determine if L is control
      // flow equivalent with the first loop in that list. If it is, append LV.
      // If not, go to the next list.
      // If no suitable list is found, start another list and add it to
      // FusionCandidates.
      bool FoundSet = false;

      for (auto &CurrCandSet : FusionCandidates) {
        if (isControlFlowEquivalent(*CurrCandSet.begin(), CurrCand)) {
#if SIFIVE_CUSTOMIZATION
          CurrCand.IV = L->getInductionVariable(SE);
          CurrCand.RIV = findReductions(CurrCand);
#endif
          CurrCandSet.insert(CurrCand);
          FoundSet = true;
#ifndef NDEBUG
          if (VerboseFusionDebugging)
            LLVM_DEBUG(dbgs() << "Adding " << CurrCand
                              << " to existing candidate set\n");
#endif
          break;
        }
      }
      if (!FoundSet) {
        // No set was found. Create a new set and add to FusionCandidates
#ifndef NDEBUG
        if (VerboseFusionDebugging)
          LLVM_DEBUG(dbgs() << "Adding " << CurrCand << " to new set\n");
#endif
        FusionCandidateSet NewCandSet;
        NewCandSet.insert(CurrCand);
        FusionCandidates.push_back(NewCandSet);
      }
      NumFusionCandidates++;
    }
  }

  /// Determine if it is beneficial to fuse two loops.
  ///
  /// For now, this method simply returns true because we want to fuse as much
  /// as possible (primarily to test the pass). This method will evolve, over
  /// time, to add heuristics for profitability of fusion.
  bool isBeneficialFusion(const FusionCandidate &FC0,
                          const FusionCandidate &FC1) {
#if SIFIVE_CUSTOMIZATION
    if (LoopConcatCanonicalize) {
      // It is not benefitial under LoopConcatenation to fuse guarded loops.
      if (FC0.GuardBranch || FC1.GuardBranch)
        return false;

      for (int i = 0; i < 2; ++i) {
        auto *BB = (i == 0) ? FC0.Latch : FC1.Latch;
        auto *L = (i == 0) ? FC0.L : FC1.L;

        // For now only process non adjacent tight loops.
        if (!isAdjacent(FC0, FC1))
          if (BB != L->getHeader())
            return false;
      }

      // Check to see if adjacent loops contain the same amount of flow
      if (isAdjacent(FC0, FC1))
        if (FC0.L->getNumBlocks() != FC1.L->getNumBlocks())
          return false;
    }
#endif

    return true;
  }

  /// Determine if two fusion candidates have the same trip count (i.e., they
  /// execute the same number of iterations).
  ///
  /// This function will return a pair of values. The first is a boolean,
  /// stating whether or not the two candidates are known at compile time to
  /// have the same TripCount. The second is the difference in the two
  /// TripCounts. This information can be used later to determine whether or not
  /// peeling can be performed on either one of the candidates.
  std::pair<bool, std::optional<unsigned>>
  haveIdenticalTripCounts(const FusionCandidate &FC0,
                          const FusionCandidate &FC1) const {
    const SCEV *TripCount0 = SE.getBackedgeTakenCount(FC0.L);
    if (isa<SCEVCouldNotCompute>(TripCount0)) {
      UncomputableTripCount++;
      LLVM_DEBUG(dbgs() << "Trip count of first loop could not be computed!");
      return {false, std::nullopt};
    }

    const SCEV *TripCount1 = SE.getBackedgeTakenCount(FC1.L);
    if (isa<SCEVCouldNotCompute>(TripCount1)) {
      UncomputableTripCount++;
      LLVM_DEBUG(dbgs() << "Trip count of second loop could not be computed!");
      return {false, std::nullopt};
    }

    LLVM_DEBUG(dbgs() << "\tTrip counts: " << *TripCount0 << " & "
                      << *TripCount1 << " are "
                      << (TripCount0 == TripCount1 ? "identical" : "different")
                      << "\n");

    if (TripCount0 == TripCount1)
      return {true, 0};

    LLVM_DEBUG(dbgs() << "The loops do not have the same tripcount, "
                         "determining the difference between trip counts\n");

    // Currently only considering loops with a single exit point
    // and a non-constant trip count.
    const unsigned TC0 = SE.getSmallConstantTripCount(FC0.L);
    const unsigned TC1 = SE.getSmallConstantTripCount(FC1.L);

    // If any of the tripcounts are zero that means that loop(s) do not have
    // a single exit or a constant tripcount.
    if (TC0 == 0 || TC1 == 0) {
      LLVM_DEBUG(dbgs() << "Loop(s) do not have a single exit point or do not "
                           "have a constant number of iterations. Peeling "
                           "is not benefical\n");
      return {false, std::nullopt};
    }

    std::optional<unsigned> Difference;
    int Diff = TC0 - TC1;

    if (Diff > 0)
      Difference = Diff;
    else {
      LLVM_DEBUG(
          dbgs() << "Difference is less than 0. FC1 (second loop) has more "
                    "iterations than the first one. Currently not supported\n");
    }

    LLVM_DEBUG(dbgs() << "Difference in loop trip count is: " << Difference
                      << "\n");

    return {false, Difference};
  }

  void peelFusionCandidate(FusionCandidate &FC0, const FusionCandidate &FC1,
                           unsigned PeelCount) {
    assert(FC0.AbleToPeel && "Should be able to peel loop");

    LLVM_DEBUG(dbgs() << "Attempting to peel first " << PeelCount
                      << " iterations of the first loop. \n");

    ValueToValueMapTy VMap;
    FC0.Peeled = peelLoop(FC0.L, PeelCount, &LI, &SE, DT, &AC, true, VMap);
    if (FC0.Peeled) {
      LLVM_DEBUG(dbgs() << "Done Peeling\n");

#ifndef NDEBUG
      auto IdenticalTripCount = haveIdenticalTripCounts(FC0, FC1);

      assert(IdenticalTripCount.first && *IdenticalTripCount.second == 0 &&
             "Loops should have identical trip counts after peeling");
#endif

      FC0.PP.PeelCount += PeelCount;

      // Peeling does not update the PDT
      PDT.recalculate(*FC0.Preheader->getParent());

      FC0.updateAfterPeeling();

      // In this case the iterations of the loop are constant, so the first
      // loop will execute completely (will not jump from one of
      // the peeled blocks to the second loop). Here we are updating the
      // branch conditions of each of the peeled blocks, such that it will
      // branch to its successor which is not the preheader of the second loop
      // in the case of unguarded loops, or the succesors of the exit block of
      // the first loop otherwise. Doing this update will ensure that the entry
      // block of the first loop dominates the entry block of the second loop.
      BasicBlock *BB =
          FC0.GuardBranch ? FC0.ExitBlock->getUniqueSuccessor() : FC1.Preheader;
      if (BB) {
        SmallVector<DominatorTree::UpdateType, 8> TreeUpdates;
        SmallVector<Instruction *, 8> WorkList;
        for (BasicBlock *Pred : predecessors(BB)) {
          if (Pred != FC0.ExitBlock) {
            WorkList.emplace_back(Pred->getTerminator());
            TreeUpdates.emplace_back(
                DominatorTree::UpdateType(DominatorTree::Delete, Pred, BB));
          }
        }
        // Cannot modify the predecessors inside the above loop as it will cause
        // the iterators to be nullptrs, causing memory errors.
        for (Instruction *CurrentBranch : WorkList) {
          BasicBlock *Succ = CurrentBranch->getSuccessor(0);
          if (Succ == BB)
            Succ = CurrentBranch->getSuccessor(1);
          ReplaceInstWithInst(CurrentBranch, BranchInst::Create(Succ));
        }

        DTU.applyUpdates(TreeUpdates);
        DTU.flush();
      }
      LLVM_DEBUG(
          dbgs() << "Sucessfully peeled " << FC0.PP.PeelCount
                 << " iterations from the first loop.\n"
                    "Both Loops have the same number of iterations now.\n");
    }
  }

  /// Walk each set of control flow equivalent fusion candidates and attempt to
  /// fuse them. This does a single linear traversal of all candidates in the
  /// set. The conditions for legal fusion are checked at this point. If a pair
  /// of fusion candidates passes all legality checks, they are fused together
  /// and a new fusion candidate is created and added to the FusionCandidateSet.
  /// The original fusion candidates are then removed, as they are no longer
  /// valid.
  bool fuseCandidates() {
    bool Fused = false;
    LLVM_DEBUG(printFusionCandidates(FusionCandidates));
    for (auto &CandidateSet : FusionCandidates) {
      if (CandidateSet.size() < 2)
        continue;

      LLVM_DEBUG(dbgs() << "Attempting fusion on Candidate Set:\n"
                        << CandidateSet << "\n");

      for (auto FC0 = CandidateSet.begin(); FC0 != CandidateSet.end(); ++FC0) {
        assert(!LDT.isRemovedLoop(FC0->L) &&
               "Should not have removed loops in CandidateSet!");
        auto FC1 = FC0;
        for (++FC1; FC1 != CandidateSet.end(); ++FC1) {
          assert(!LDT.isRemovedLoop(FC1->L) &&
                 "Should not have removed loops in CandidateSet!");

          LLVM_DEBUG(dbgs() << "Attempting to fuse candidate \n"; FC0->dump();
                     dbgs() << " with\n"; FC1->dump(); dbgs() << "\n");

          FC0->verify();
          FC1->verify();

          // Check if the candidates have identical tripcounts (first value of
          // pair), and if not check the difference in the tripcounts between
          // the loops (second value of pair). The difference is not equal to
          // std::nullopt iff the loops iterate a constant number of times, and
          // have a single exit.
          std::pair<bool, std::optional<unsigned>> IdenticalTripCountRes =
              haveIdenticalTripCounts(*FC0, *FC1);
          bool SameTripCount = IdenticalTripCountRes.first;
          std::optional<unsigned> TCDifference = IdenticalTripCountRes.second;

          // Here we are checking that FC0 (the first loop) can be peeled, and
          // both loops have different tripcounts.
          if (FC0->AbleToPeel && !SameTripCount && TCDifference) {
            if (*TCDifference > FusionPeelMaxCount) {
              LLVM_DEBUG(dbgs()
                         << "Difference in loop trip counts: " << *TCDifference
                         << " is greater than maximum peel count specificed: "
                         << FusionPeelMaxCount << "\n");
            } else {
              // Dependent on peeling being performed on the first loop, and
              // assuming all other conditions for fusion return true.
              SameTripCount = true;
            }
          }

          if (!SameTripCount) {
            LLVM_DEBUG(dbgs() << "Fusion candidates do not have identical trip "
                                 "counts. Not fusing.\n");
            reportLoopFusion<OptimizationRemarkMissed>(*FC0, *FC1,
                                                       NonEqualTripCount);
            continue;
          }

#if SIFIVE_CUSTOMIZATION
          if (!LoopConcatCanonicalize && !isAdjacent(*FC0, *FC1)) {
            LLVM_DEBUG(dbgs()
                       << "Fusion candidates are not adjacent. Not fusing.\n");
            reportLoopFusion<OptimizationRemarkMissed>(*FC0, *FC1, NonAdjacent);
            continue;
          }
#else
          if (!isAdjacent(*FC0, *FC1)) {
            LLVM_DEBUG(dbgs()
                       << "Fusion candidates are not adjacent. Not fusing.\n");
            reportLoopFusion<OptimizationRemarkMissed>(*FC0, *FC1, NonAdjacent);
            continue;
          }
#endif

          if ((!FC0->GuardBranch && FC1->GuardBranch) ||
              (FC0->GuardBranch && !FC1->GuardBranch)) {
            LLVM_DEBUG(dbgs() << "The one of candidate is guarded while the "
                                 "another one is not. Not fusing.\n");
            reportLoopFusion<OptimizationRemarkMissed>(
                *FC0, *FC1, OnlySecondCandidateIsGuarded);
            continue;
          }

          // Ensure that FC0 and FC1 have identical guards.
          // If one (or both) are not guarded, this check is not necessary.
          if (FC0->GuardBranch && FC1->GuardBranch &&
              !haveIdenticalGuards(*FC0, *FC1) && !TCDifference) {
            LLVM_DEBUG(dbgs() << "Fusion candidates do not have identical "
                                 "guards. Not Fusing.\n");
            reportLoopFusion<OptimizationRemarkMissed>(*FC0, *FC1,
                                                       NonIdenticalGuards);
            continue;
          }

          if (FC0->GuardBranch) {
            assert(FC1->GuardBranch && "Expecting valid FC1 guard branch");

            if (!isSafeToMoveBefore(*FC0->ExitBlock,
                                    *FC1->ExitBlock->getFirstNonPHIOrDbg(), DT,
                                    &PDT, &DI)) {
              LLVM_DEBUG(dbgs() << "Fusion candidate contains unsafe "
                                   "instructions in exit block. Not fusing.\n");
              reportLoopFusion<OptimizationRemarkMissed>(*FC0, *FC1,
                                                         NonEmptyExitBlock);
              continue;
            }

            if (!isSafeToMoveBefore(
                    *FC1->GuardBranch->getParent(),
                    *FC0->GuardBranch->getParent()->getTerminator(), DT, &PDT,
                    &DI)) {
              LLVM_DEBUG(dbgs()
                         << "Fusion candidate contains unsafe "
                            "instructions in guard block. Not fusing.\n");
              reportLoopFusion<OptimizationRemarkMissed>(*FC0, *FC1,
                                                         NonEmptyGuardBlock);
              continue;
            }
          }

          // Check the dependencies across the loops and do not fuse if it would
          // violate them.
          if (!dependencesAllowFusion(*FC0, *FC1)) {
            LLVM_DEBUG(dbgs() << "Memory dependencies do not allow fusion!\n");
            reportLoopFusion<OptimizationRemarkMissed>(*FC0, *FC1,
                                                       InvalidDependencies);
            continue;
          }

          // If the second loop has instructions in the pre-header, attempt to
          // hoist them up to the first loop's pre-header or sink them into the
          // body of the second loop.
          SmallVector<Instruction *, 4> SafeToHoist;
          SmallVector<Instruction *, 4> SafeToSink;
          // At this point, this is the last remaining legality check.
          // Which means if we can make this pre-header empty, we can fuse
          // these loops
          if (!isEmptyPreheader(*FC1)) {
            LLVM_DEBUG(dbgs() << "Fusion candidate does not have empty "
                                 "preheader.\n");

            // If it is not safe to hoist/sink all instructions in the
            // pre-header, we cannot fuse these loops.
            if (!collectMovablePreheaderInsts(*FC0, *FC1, SafeToHoist,
                                              SafeToSink)) {
              LLVM_DEBUG(dbgs() << "Could not hoist/sink all instructions in "
                                   "Fusion Candidate Pre-header.\n"
                                << "Not Fusing.\n");
              reportLoopFusion<OptimizationRemarkMissed>(*FC0, *FC1,
                                                         NonEmptyPreheader);
              continue;
            }
          }

          bool BeneficialToFuse = isBeneficialFusion(*FC0, *FC1);
          LLVM_DEBUG(dbgs()
                     << "\tFusion appears to be "
                     << (BeneficialToFuse ? "" : "un") << "profitable!\n");
          if (!BeneficialToFuse) {
            reportLoopFusion<OptimizationRemarkMissed>(*FC0, *FC1,
                                                       FusionNotBeneficial);
            continue;
          }
          // All analysis has completed and has determined that fusion is legal
          // and profitable. At this point, start transforming the code and
          // perform fusion.

          // Execute the hoist/sink operations on preheader instructions
          movePreheaderInsts(*FC0, *FC1, SafeToHoist, SafeToSink);

#if SIFIVE_CUSTOMIZATION
          LLVM_DEBUG(dbgs() << "\tFusion is performed: " << *FC0 << " and "
                     << *FC1 << " adjacent: " << isAdjacent(*FC0, *FC1)
                     << "\n");
#else
          LLVM_DEBUG(dbgs() << "\tFusion is performed: " << *FC0 << " and "
                            << *FC1 << "\n");
#endif

          FusionCandidate FC0Copy = *FC0;
          // Peel the loop after determining that fusion is legal. The Loops
          // will still be safe to fuse after the peeling is performed.
#if SIFIVE_CUSTOMIZATION
          bool Peel =
              !LoopConcatCanonicalize && TCDifference && *TCDifference > 0;
#else
          bool Peel = TCDifference && *TCDifference > 0;
#endif
          if (Peel)
            peelFusionCandidate(FC0Copy, *FC1, *TCDifference);

          // Report fusion to the Optimization Remarks.
          // Note this needs to be done *before* performFusion because
          // performFusion will change the original loops, making it not
          // possible to identify them after fusion is complete.
          reportLoopFusion<OptimizationRemark>((Peel ? FC0Copy : *FC0), *FC1,
                                               FuseCounter);

#if SIFIVE_CUSTOMIZATION
          if (isAdjacent(*FC0, *FC1)) {
            FusionCandidate FusedCand(
                performFusion((Peel ? FC0Copy : *FC0), *FC1), DT, &PDT, ORE,
                FC0Copy.PP, FC0Copy.IV, FC0Copy.RIV);
            FusedCand.verify();
            assert(FusedCand.isEligibleForFusion(SE) &&
                   "Fused candidate should be eligible for fusion!");

            // Notify the loop-depth-tree that these loops are not valid objects
            LDT.removeLoop(FC1->L);

            CandidateSet.erase(FC0);
            CandidateSet.erase(FC1);

            auto InsertPos = CandidateSet.insert(FusedCand);

            assert(InsertPos.second &&
                   "Unable to insert TargetCandidate in CandidateSet!");

            // Reset FC0 and FC1 the new (fused) candidate. Subsequent
            // iterations of the FC1 loop will attempt to fuse the new (fused)
            // loop with the remaining candidates in the current candidate set.
            FC0 = FC1 = InsertPos.first;
          } else if (LoopConcatCanonicalize) {
            FusionCandidate FusedCand(performNonAdjacentFusion(*FC0, *FC1), DT,
                                      &PDT, ORE, FC0Copy.PP, FC0Copy.IV,
                                      FC0Copy.RIV);
            FusedCand.verify();
            assert(FusedCand.isEligibleForFusion(SE) &&
                   "Fused candidate should be eligible for fusion!");

            // Notify the loop-depth-tree that these loops are not valid objects
            LDT.removeLoop(FC1->L);

            CandidateSet.erase(FC0);
            CandidateSet.erase(FC1);

            auto InsertPos = CandidateSet.insert(FusedCand);

            assert(InsertPos.second &&
                   "Unable to insert TargetCandidate in CandidateSet!");

            // Reset FC0 and FC1 the new (fused) candidate. Subsequent
            // iterations of the FC1 loop will attempt to fuse the new (fused)
            // loop with the remaining candidates in the current candidate set.
            FC0 = FC1 = InsertPos.first;
          }
#else
          FusionCandidate FusedCand(
              performFusion((Peel ? FC0Copy : *FC0), *FC1), DT, &PDT, ORE,
              FC0Copy.PP);
          FusedCand.verify();
          assert(FusedCand.isEligibleForFusion(SE) &&
                 "Fused candidate should be eligible for fusion!");

          // Notify the loop-depth-tree that these loops are not valid objects
          LDT.removeLoop(FC1->L);

          CandidateSet.erase(FC0);
          CandidateSet.erase(FC1);

          auto InsertPos = CandidateSet.insert(FusedCand);

          assert(InsertPos.second &&
                 "Unable to insert TargetCandidate in CandidateSet!");

          // Reset FC0 and FC1 the new (fused) candidate. Subsequent iterations
          // of the FC1 loop will attempt to fuse the new (fused) loop with the
          // remaining candidates in the current candidate set.
          FC0 = FC1 = InsertPos.first;
#endif

#if SIFIVE_CUSTOMIZATION
          auto *L = FC0->L;
          // Mark reducing fused loops as already unrolled.
          // We do this so that the vectorizer receives these
          // loops in their current state.
          for (auto &Phi : L->getHeader()->phis()) {
            if (Phi.getNumIncomingValues() == 1)
              continue;

            RecurrenceDescriptor Rdx;
            if (RecurrenceDescriptor::isReductionPHI(&Phi, L, Rdx)) {
              L->setLoopAlreadyUnrolled();
              break;
            }
          }
#endif

          LLVM_DEBUG(dbgs() << "Candidate Set (after fusion): " << CandidateSet
                            << "\n");

          Fused = true;
        }
      }
    }
    return Fused;
  }

  // Returns true if the instruction \p I can be hoisted to the end of the
  // preheader of \p FC0. \p SafeToHoist contains the instructions that are
  // known to be safe to hoist. The instructions encountered that cannot be
  // hoisted are in \p NotHoisting.
  // TODO: Move functionality into CodeMoverUtils
  bool canHoistInst(Instruction &I,
                    const SmallVector<Instruction *, 4> &SafeToHoist,
                    const SmallVector<Instruction *, 4> &NotHoisting,
                    const FusionCandidate &FC0) const {
    const BasicBlock *FC0PreheaderTarget = FC0.Preheader->getSingleSuccessor();
    assert(FC0PreheaderTarget &&
           "Expected single successor for loop preheader.");

    for (Use &Op : I.operands()) {
      if (auto *OpInst = dyn_cast<Instruction>(Op)) {
        bool OpHoisted = is_contained(SafeToHoist, OpInst);
        // Check if we have already decided to hoist this operand. In this
        // case, it does not dominate FC0 *yet*, but will after we hoist it.
        if (!(OpHoisted || DT.dominates(OpInst, FC0PreheaderTarget))) {
          return false;
        }
      }
    }

    // PHIs in FC1's header only have FC0 blocks as predecessors. PHIs
    // cannot be hoisted and should be sunk to the exit of the fused loop.
    if (isa<PHINode>(I))
      return false;

    // If this isn't a memory inst, hoisting is safe
    if (!I.mayReadOrWriteMemory())
      return true;

    LLVM_DEBUG(dbgs() << "Checking if this mem inst can be hoisted.\n");
    for (Instruction *NotHoistedInst : NotHoisting) {
      if (auto D = DI.depends(&I, NotHoistedInst)) {
        // Dependency is not read-before-write, write-before-read or
        // write-before-write
        if (D->isFlow() || D->isAnti() || D->isOutput()) {
          LLVM_DEBUG(dbgs() << "Inst depends on an instruction in FC1's "
                               "preheader that is not being hoisted.\n");
          return false;
        }
      }
    }

    for (Instruction *ReadInst : FC0.MemReads) {
      if (auto D = DI.depends(ReadInst, &I)) {
        // Dependency is not read-before-write
        if (D->isAnti()) {
          LLVM_DEBUG(dbgs() << "Inst depends on a read instruction in FC0.\n");
          return false;
        }
      }
    }

    for (Instruction *WriteInst : FC0.MemWrites) {
      if (auto D = DI.depends(WriteInst, &I)) {
        // Dependency is not write-before-read or write-before-write
        if (D->isFlow() || D->isOutput()) {
          LLVM_DEBUG(dbgs() << "Inst depends on a write instruction in FC0.\n");
          return false;
        }
      }
    }
    return true;
  }

  // Returns true if the instruction \p I can be sunk to the top of the exit
  // block of \p FC1.
  // TODO: Move functionality into CodeMoverUtils
  bool canSinkInst(Instruction &I, const FusionCandidate &FC1) const {
    for (User *U : I.users()) {
      if (auto *UI{dyn_cast<Instruction>(U)}) {
        // Cannot sink if user in loop
        // If FC1 has phi users of this value, we cannot sink it into FC1.
        if (FC1.L->contains(UI)) {
          // Cannot hoist or sink this instruction. No hoisting/sinking
          // should take place, loops should not fuse
          return false;
        }
      }
    }

#if SIFIVE_CUSTOMIZATION
    if (I.isLifetimeStartOrEnd()) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        Intrinsic::ID ID = II->getIntrinsicID();
        // Check if this lifetime end is AllocaInst
        // related, all the related Allocas have
        // been managed during canonicalization.
        if (LoopConcatCanonicalize)
          if (ID == Intrinsic::lifetime_end) {
            auto *Def = II->getArgOperand(1);
            if (isa<AllocaInst>(Def))
              return true;
          }
      }
    }
#endif

    // If this isn't a memory inst, sinking is safe
    if (!I.mayReadOrWriteMemory())
      return true;

    for (Instruction *ReadInst : FC1.MemReads) {
      if (auto D = DI.depends(&I, ReadInst)) {
        // Dependency is not write-before-read
        if (D->isFlow()) {
          LLVM_DEBUG(dbgs() << "Inst depends on a read instruction in FC1.\n");
          return false;
        }
      }
    }

    for (Instruction *WriteInst : FC1.MemWrites) {
      if (auto D = DI.depends(&I, WriteInst)) {
        // Dependency is not write-before-write or read-before-write
        if (D->isOutput() || D->isAnti()) {
          LLVM_DEBUG(dbgs() << "Inst depends on a write instruction in FC1.\n");
          return false;
        }
      }
    }

    return true;
  }

  /// Collect instructions in the \p FC1 Preheader that can be hoisted
  /// to the \p FC0 Preheader or sunk into the \p FC1 Body
  bool collectMovablePreheaderInsts(
      const FusionCandidate &FC0, const FusionCandidate &FC1,
      SmallVector<Instruction *, 4> &SafeToHoist,
      SmallVector<Instruction *, 4> &SafeToSink) const {
    BasicBlock *FC1Preheader = FC1.Preheader;
    // Save the instructions that are not being hoisted, so we know not to hoist
    // mem insts that they dominate.
    SmallVector<Instruction *, 4> NotHoisting;

    for (Instruction &I : *FC1Preheader) {
      // Can't move a branch
      if (&I == FC1Preheader->getTerminator())
        continue;
      // If the instruction has side-effects, give up.
      // TODO: The case of mayReadFromMemory we can handle but requires
      // additional work with a dependence analysis so for now we give
      // up on memory reads.
      if (I.mayThrow() || !I.willReturn()) {
        LLVM_DEBUG(dbgs() << "Inst: " << I << " may throw or won't return.\n");
        return false;
      }

      LLVM_DEBUG(dbgs() << "Checking Inst: " << I << "\n");

      if (I.isAtomic() || I.isVolatile()) {
        LLVM_DEBUG(
            dbgs() << "\tInstruction is volatile or atomic. Cannot move it.\n");
        return false;
      }

      if (canHoistInst(I, SafeToHoist, NotHoisting, FC0)) {
        SafeToHoist.push_back(&I);
        LLVM_DEBUG(dbgs() << "\tSafe to hoist.\n");
      } else {
        LLVM_DEBUG(dbgs() << "\tCould not hoist. Trying to sink...\n");
        NotHoisting.push_back(&I);

        if (canSinkInst(I, FC1)) {
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
    if (VerboseFusionDebugging)
      LLVM_DEBUG(dbgs() << "    Access function check: " << *SCEVPtr0 << " vs "
                        << *SCEVPtr1 << "\n");
#endif
    AddRecLoopReplacer Rewriter(SE, L0, L1);
#if SIFIVE_CUSTOMIZATION
    // First validate the operands of SCEVPtr0
    for (const SCEV *Op : SCEVPtr0->operands())
      if(!SE.isAvailableAtLoopEntry(Op, &L1))
        return false;
#endif
    SCEVPtr0 = Rewriter.visit(SCEVPtr0);
#ifndef NDEBUG
    if (VerboseFusionDebugging)
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
    if (VerboseFusionDebugging)
      LLVM_DEBUG(dbgs() << "    Relation: " << *SCEVPtr0
                        << (IsAlwaysGE ? "  >=  " : "  may <  ") << *SCEVPtr1
                        << "\n");
#endif
    return IsAlwaysGE;
  }

  /// Return true if the dependences between @p I0 (in @p L0) and @p I1 (in
  /// @p L1) allow loop fusion of @p L0 and @p L1. The dependence analyses
  /// specified by @p DepChoice are used to determine this.
  bool dependencesAllowFusion(const FusionCandidate &FC0,
                              const FusionCandidate &FC1, Instruction &I0,
                              Instruction &I1, bool AnyDep,
                              FusionDependenceAnalysisChoice DepChoice) {
#ifndef NDEBUG
    if (VerboseFusionDebugging) {
      LLVM_DEBUG(dbgs() << "Check dep: " << I0 << " vs " << I1 << " : "
                        << DepChoice << "\n");
    }
#endif
    switch (DepChoice) {
    case FUSION_DEPENDENCE_ANALYSIS_SCEV:
      return accessDiffIsPositive(*FC0.L, *FC1.L, I0, I1, AnyDep);
    case FUSION_DEPENDENCE_ANALYSIS_DA: {
      auto DepResult = DI.depends(&I0, &I1);
      if (!DepResult)
        return true;
#ifndef NDEBUG
      if (VerboseFusionDebugging) {
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

    case FUSION_DEPENDENCE_ANALYSIS_ALL:
      return dependencesAllowFusion(FC0, FC1, I0, I1, AnyDep,
                                    FUSION_DEPENDENCE_ANALYSIS_SCEV) ||
             dependencesAllowFusion(FC0, FC1, I0, I1, AnyDep,
                                    FUSION_DEPENDENCE_ANALYSIS_DA);
    }

    llvm_unreachable("Unknown fusion dependence analysis choice!");
  }

#if SIFIVE_CUSTOMIZATION
  bool isFlowRelated(BasicBlock *BB, const FusionCandidate &FC0,
                     const FusionCandidate &FC1) {
    if (DT.dominates(FC0.ExitBlock, BB)) {
      if (DT.dominates(BB, FC1.Preheader))
        return true;

      BasicBlock *IDomBB = DT.getNode(BB)->getIDom()->getBlock();
      if (DT.dominates(IDomBB, FC1.Preheader))
        return true;
    }
    return false;
  }

  /// Return true if @p I is a LCSSA phi node.
  bool isLcssaPhi(Instruction *I) {
    return isa<PHINode>(I) && I->getNumOperands() == 1;
  }
#endif

  /// Perform a dependence check and return if @p FC0 and @p FC1 can be fused.
  bool dependencesAllowFusion(const FusionCandidate &FC0,
                              const FusionCandidate &FC1) {
    LLVM_DEBUG(dbgs() << "Check if " << FC0 << " can be fused with " << FC1
                      << "\n");
    assert(FC0.L->getLoopDepth() == FC1.L->getLoopDepth());
#if SIFIVE_CUSTOMIZATION
    if (!DT.dominates(FC0.getEntryBlock(), FC1.getEntryBlock())) {
      LLVM_DEBUG(dbgs() << "FC0 does not dominate FC1\n");
      return false;
    }
#else
    assert(DT.dominates(FC0.getEntryBlock(), FC1.getEntryBlock()));
#endif

    for (Instruction *WriteL0 : FC0.MemWrites) {
      for (Instruction *WriteL1 : FC1.MemWrites)
        if (!dependencesAllowFusion(FC0, FC1, *WriteL0, *WriteL1,
                                    /* AnyDep */ false,
                                    FusionDependenceAnalysis)) {
          InvalidDependencies++;
          return false;
        }
      for (Instruction *ReadL1 : FC1.MemReads)
        if (!dependencesAllowFusion(FC0, FC1, *WriteL0, *ReadL1,
                                    /* AnyDep */ false,
                                    FusionDependenceAnalysis)) {
          InvalidDependencies++;
          return false;
        }
    }

    for (Instruction *WriteL1 : FC1.MemWrites) {
      for (Instruction *WriteL0 : FC0.MemWrites)
        if (!dependencesAllowFusion(FC0, FC1, *WriteL0, *WriteL1,
                                    /* AnyDep */ false,
                                    FusionDependenceAnalysis)) {
          InvalidDependencies++;
          return false;
        }
      for (Instruction *ReadL0 : FC0.MemReads)
        if (!dependencesAllowFusion(FC0, FC1, *ReadL0, *WriteL1,
                                    /* AnyDep */ false,
                                    FusionDependenceAnalysis)) {
          InvalidDependencies++;
          return false;
        }
    }

    // Walk through all uses in FC1. For each use, find the reaching def. If the
    // def is located in FC0 then it is not safe to fuse.
    for (BasicBlock *BB : FC1.L->blocks())
      for (Instruction &I : *BB)
        for (auto &Op : I.operands())
#if SIFIVE_CUSTOMIZATION
          if (auto *Def = dyn_cast<Instruction>(Op)) {
            auto *DefBB = Def->getParent();
            if (FC0.L->contains(DefBB)) {
              InvalidDependencies++;
              return false;
            }

            if (LoopConcatCanonicalize) {
              if (FC1.L->contains(DefBB))
                continue;

              // We will handle all of FC1.Preheader as evaluated
              // as movable.
              if (DefBB == FC1.Preheader)
                continue;

              if (isFlowRelated(DefBB, FC0, FC1)) {
                InvalidDependencies++;
                return false;
              }

              // Check Ptr uses in flow.
              if (Def->getType()->isPointerTy()) {
                for (auto *U : Def->users()) {
                  auto *CurUser = cast<Instruction>(U);
                  auto *CurUserBB = CurUser->getParent();
                  // already evaluated in FusionDependenceAnalysis
                  if (FC0.L->contains(CurUserBB) || FC1.L->contains(CurUserBB))
                    continue;

                  // Defer this to preheader processing.
                  if (CurUserBB == FC1.Preheader)
                    continue;

                  // Ignore lifetime intrinsics.
                  if (CurUser->isLifetimeStartOrEnd())
                    continue;

                  // Check for dominating memops.
                  if (CurUser->mayReadOrWriteMemory())
                    if (isFlowRelated(CurUserBB, FC0, FC1)) {
                      InvalidDependencies++;
                      return false;
                    }
                }
              }
            }
          }

#else
          if (Instruction *Def = dyn_cast<Instruction>(Op))
            if (FC0.L->contains(Def->getParent())) {
              InvalidDependencies++;
              return false;
            }
#endif

#if SIFIVE_CUSTOMIZATION
    // Walk though all the defs in FC0. For each use, determine if there is a
    // use that exists between FC0.L and FC1.L.
    if (LoopConcatCanonicalize)
      for (auto *BB : FC0.L->blocks())
        for (auto &I : *BB)
          for (auto *U : I.users()) {
            auto *CurUser = cast<Instruction>(U);
            auto *CurUserBB = CurUser->getParent();
            if (FC0.L->contains(CurUserBB))
              continue;

            // We can fixup lcssa phi uses as loop carried
            // dependences in the fused loop.
            if (CurUserBB == FC1.Preheader && isLcssaPhi(CurUser))
              continue;

            // We can fixup lcssa phi uses in FC0's ExitBlock
            // by merging them into FC1's ExitBlock after
            // determining that FC1's loop does not consume the value
            if (CurUserBB == FC0.ExitBlock && isLcssaPhi(CurUser)) {
              for (auto *InteriorU : CurUser->users()) {
                auto *InteriorUser = cast<Instruction>(InteriorU);
                auto *InteriorUserBB = InteriorUser->getParent();
                if (FC1.L->contains(InteriorUserBB)) {
                  InvalidDependencies++;
                  return false;
                }
                if (isFlowRelated(InteriorUserBB, FC0, FC1)) {
                  InvalidDependencies++;
                  return false;
                }
              }
              continue;
            }

            if (isFlowRelated(CurUserBB, FC0, FC1)) {
              InvalidDependencies++;
              return false;
            }
          }
#endif

    return true;
  }

  /// Determine if two fusion candidates are adjacent in the CFG.
  ///
  /// This method will determine if there are additional basic blocks in the CFG
  /// between the exit of \p FC0 and the entry of \p FC1.
  /// If the two candidates are guarded loops, then it checks whether the
  /// non-loop successor of the \p FC0 guard branch is the entry block of \p
  /// FC1. If not, then the loops are not adjacent. If the two candidates are
  /// not guarded loops, then it checks whether the exit block of \p FC0 is the
  /// preheader of \p FC1.
  bool isAdjacent(const FusionCandidate &FC0,
                  const FusionCandidate &FC1) const {
    // If the successor of the guard branch is FC1, then the loops are adjacent
    if (FC0.GuardBranch)
      return FC0.getNonLoopBlock() == FC1.getEntryBlock();
    else
      return FC0.ExitBlock == FC1.getEntryBlock();
  }

  bool isEmptyPreheader(const FusionCandidate &FC) const {
    return FC.Preheader->size() == 1;
  }

  /// Hoist \p FC1 Preheader instructions to \p FC0 Preheader
  /// and sink others into the body of \p FC1.
  void movePreheaderInsts(const FusionCandidate &FC0,
                          const FusionCandidate &FC1,
                          SmallVector<Instruction *, 4> &HoistInsts,
                          SmallVector<Instruction *, 4> &SinkInsts) const {
    // All preheader instructions except the branch must be hoisted or sunk
    assert(HoistInsts.size() + SinkInsts.size() == FC1.Preheader->size() - 1 &&
           "Attempting to sink and hoist preheader instructions, but not all "
           "the preheader instructions are accounted for.");

    NumHoistedInsts += HoistInsts.size();
    NumSunkInsts += SinkInsts.size();

    LLVM_DEBUG(if (VerboseFusionDebugging) {
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
      assert(I->getParent() == FC1.Preheader);
      I->moveBefore(*FC0.Preheader,
                    FC0.Preheader->getTerminator()->getIterator());
    }
    // insert instructions in reverse order to maintain dominance relationship
    for (Instruction *I : reverse(SinkInsts)) {
      assert(I->getParent() == FC1.Preheader);
      I->moveBefore(*FC1.ExitBlock, FC1.ExitBlock->getFirstInsertionPt());
    }
  }

  /// Determine if two fusion candidates have identical guards
  ///
  /// This method will determine if two fusion candidates have the same guards.
  /// The guards are considered the same if:
  ///   1. The instructions to compute the condition used in the compare are
  ///      identical.
  ///   2. The successors of the guard have the same flow into/around the loop.
  /// If the compare instructions are identical, then the first successor of the
  /// guard must go to the same place (either the preheader of the loop or the
  /// NonLoopBlock). In other words, the first successor of both loops must
  /// both go into the loop (i.e., the preheader) or go around the loop (i.e.,
  /// the NonLoopBlock). The same must be true for the second successor.
  bool haveIdenticalGuards(const FusionCandidate &FC0,
                           const FusionCandidate &FC1) const {
    assert(FC0.GuardBranch && FC1.GuardBranch &&
           "Expecting FC0 and FC1 to be guarded loops.");

    if (auto FC0CmpInst =
            dyn_cast<Instruction>(FC0.GuardBranch->getCondition()))
      if (auto FC1CmpInst =
              dyn_cast<Instruction>(FC1.GuardBranch->getCondition()))
        if (!FC0CmpInst->isIdenticalTo(FC1CmpInst))
          return false;

    // The compare instructions are identical.
    // Now make sure the successor of the guards have the same flow into/around
    // the loop
    if (FC0.GuardBranch->getSuccessor(0) == FC0.Preheader)
      return (FC1.GuardBranch->getSuccessor(0) == FC1.Preheader);
    else
      return (FC1.GuardBranch->getSuccessor(1) == FC1.Preheader);
  }

  /// Modify the latch branch of FC to be unconditional since successors of the
  /// branch are the same.
  void simplifyLatchBranch(const FusionCandidate &FC) const {
    BranchInst *FCLatchBranch = dyn_cast<BranchInst>(FC.Latch->getTerminator());
    if (FCLatchBranch) {
      assert(FCLatchBranch->isConditional() &&
             FCLatchBranch->getSuccessor(0) == FCLatchBranch->getSuccessor(1) &&
             "Expecting the two successors of FCLatchBranch to be the same");
      BranchInst *NewBranch =
          BranchInst::Create(FCLatchBranch->getSuccessor(0));
      ReplaceInstWithInst(FCLatchBranch, NewBranch);
    }
  }

  /// Move instructions from FC0.Latch to FC1.Latch. If FC0.Latch has an unique
  /// successor, then merge FC0.Latch with its unique successor.
  void mergeLatch(const FusionCandidate &FC0, const FusionCandidate &FC1) {
    moveInstructionsToTheBeginning(*FC0.Latch, *FC1.Latch, DT, PDT, DI);
    if (BasicBlock *Succ = FC0.Latch->getUniqueSuccessor()) {
      MergeBlockIntoPredecessor(Succ, &DTU, &LI);
      DTU.flush();
    }
  }

#if SIFIVE_CUSTOMIZATION
  /// Determine if two fusion candidates have identical bounds
  bool loopsHaveIdenticalBounds(const FusionCandidate &FC0,
                                const FusionCandidate &FC1) {
    if (FC0.IV && FC1.IV) {
      if (FC0.IV->getNumIncomingValues() != FC1.IV->getNumIncomingValues())
       return false;

      auto FC0_LB = Loop::LoopBounds::getBounds(*FC0.L, *FC0.IV, SE);
      if (!FC0_LB.has_value())
        return false;

      auto FC1_LB = Loop::LoopBounds::getBounds(*FC1.L, *FC1.IV, SE);
      if (!FC1_LB.has_value())
        return false;

      // Only qualify loops that are upcounting with the same step.
      Loop::LoopBounds::Direction D = FC0_LB->getDirection();
      if ((D == FC1_LB->getDirection()) &&
          (D == Loop::LoopBounds::Direction::Increasing) &&
          (FC0_LB->getStepValue() == FC1_LB->getStepValue())) {

        Value *FC0InitIvVal = &FC0_LB->getInitialIVValue();
        Value *FC0FinalIvVal = &FC0_LB->getFinalIVValue();
        Value *FC1InitIvVal = &FC1_LB->getInitialIVValue();
        Value *FC1FinalIvVal = &FC1_LB->getFinalIVValue();

        if ((FC0InitIvVal == FC1InitIvVal) && (FC0FinalIvVal == FC1FinalIvVal))
          return true;
      }
    }
    return false;
  }

  /// For loops with identical bounds/trip, we can coalese the IV
  /// info and share it in the new loop, provided the trips are
  /// arithmetically modified and equavalent.
  void replaceInductionVarAndStep(PHINode *FC0_IV, PHINode *FC1_IV,
                                  Value *FC0StepVal, Value *FC1StepVal) {
    if (FC0StepVal && FC1StepVal &&
        isa<BinaryOperator>(FC0StepVal) && isa<BinaryOperator>(FC1StepVal)) {
      FC1_IV->replaceAllUsesWith(FC0_IV);
      FC1StepVal->replaceAllUsesWith(FC0StepVal);
      auto *FC1StepInst = cast<Instruction>(FC1StepVal);
      FC1StepInst->eraseFromParent();
      FC1_IV->eraseFromParent();
    }
  }

  /// Discover any reduction phis for a given Fusion Candidate
  PHINode *findReductions(const FusionCandidate &FC) {
    PHINode *FC_RIV = nullptr;
    unsigned NumRPHI = 0;
    for (auto &Phi : FC.L->getHeader()->phis()) {
      if (Phi.getNumIncomingValues() == 1)
        continue;

      // We cannot safely reform floating point reduction expressions.
      if (Phi.getType()->isFloatingPointTy())
        continue;

      if (&Phi == FC.IV)
        continue;

      RecurrenceDescriptor Rdx;
      if (RecurrenceDescriptor::isReductionPHI(&Phi, FC.L, Rdx)) {
        FC_RIV = &Phi;
        NumRPHI++;

        // Skip loops with more than one reduction phi
        if (NumRPHI > 1)
          FC_RIV = nullptr;
      }
    }
    return FC_RIV;
  }

  /// Find the LCSSA phi for a given reduction phi
  PHINode *findLoopOutput(const FusionCandidate &FC) {
    PHINode *LcssaPhi = nullptr;
    if (!FC.RIV)
      return LcssaPhi;

    Value *BackEdgeVal = FC.RIV->getIncomingValueForBlock(FC.Latch);
    for (auto *U : BackEdgeVal->users()) {
      if (FC.L->contains(cast<Instruction>(U)))
        continue;

      // We check convergence through lcssa PHIs
      if (auto *PN = dyn_cast<PHINode>(U))
        if (PN->getNumIncomingValues() == 1)
          LcssaPhi = PN;
    }
    return LcssaPhi;
  }

  /// Collect the uses of an instruction
  void collectUses(Instruction *I, SmallVectorImpl<Instruction *> &Uses) const {
    for (auto *U : I->users())
      Uses.push_back(cast<Instruction>(U));
  }

  /// Find the convergence point between two sets of instructions
  Instruction *findConvergence(const SmallVectorImpl<Instruction *> &FC0_Uses,
                               const SmallVectorImpl<Instruction *> &FC1_Uses,
                               unsigned Depth, Instruction *&LastFC1RelatedI) {
    Instruction *ConvergenceI = nullptr;
    for (auto [NumFC0Use, FC0_Use] : enumerate(FC0_Uses)) {
      // Do not evalute convergence in the presence of calls
      if (isa<CallInst>(FC0_Use))
        break;

      for (auto [NumFC1Use, FC1_Use] : enumerate(FC1_Uses)) {
        // Do not evalute convergence in the presence of calls
        if (isa<CallInst>(FC1_Use))
          break;

        if (NumFC0Use != NumFC1Use)
          continue;

        if (FC0_Use->getOpcode() != FC1_Use->getOpcode())
          continue;

        // Check for convergence point
        if (FC0_Use == FC1_Use)
          return FC0_Use;

        SmallVector<Instruction *> FC0_NextUses;
        SmallVector<Instruction *> FC1_NextUses;
        collectUses(FC0_Use, FC0_NextUses);
        collectUses(FC1_Use, FC1_NextUses);

        if (Depth < 5) {
          LastFC1RelatedI = FC1_Use;
          ConvergenceI =
              findConvergence(FC0_NextUses, FC1_NextUses, Depth + 1, LastFC1RelatedI);
        }

        if (ConvergenceI)
          break;
      }

      if (ConvergenceI)
        break;
    }
    return ConvergenceI;
  }

  /// Determine if two reduction values converge
  Instruction *reductionValuesConverge(const FusionCandidate &FC0,
                                       const FusionCandidate &FC1,
                                       Instruction *&LastFC1RelatedI,
                                       PHINode *FC0_LcssaPhi,
                                       PHINode *FC1_LcssaPhi) {
    Instruction *ConvergenceI = nullptr;
    if (!FC0.RIV || !FC1.RIV)
      return ConvergenceI;

    // If the two PHIs converge, the connecting expression must
    // be of the same form.  We only look at most 5 levels of depth.
    if (FC0_LcssaPhi && FC1_LcssaPhi) {
      if (FC1_LcssaPhi->getNumUses() == FC1_LcssaPhi->getNumUses()) {
        SmallVector<Instruction *> FC0_Lvl1Uses;
        SmallVector<Instruction *> FC1_Lvl1Uses;
        collectUses(FC0_LcssaPhi, FC0_Lvl1Uses);
        collectUses(FC1_LcssaPhi, FC1_Lvl1Uses);
        ConvergenceI =
            findConvergence(FC0_Lvl1Uses, FC1_Lvl1Uses, 0, LastFC1RelatedI);
      }
    }

    if (ConvergenceI) {
      auto *BackEdgeFC0Inst =
          cast<Instruction>(FC0.RIV->getIncomingValueForBlock(FC0.Latch));
      auto *BackEdgeFC1Inst =
          cast<Instruction>(FC1.RIV->getIncomingValueForBlock(FC1.Latch));
      // The two candidate reduction expressions must be of the same form.
      if (BackEdgeFC0Inst->getOpcode() != BackEdgeFC1Inst->getOpcode())
        return nullptr;
      // The convergence point must be of the same form as the reduction
      // expressions.
      if (ConvergenceI->getOpcode() != BackEdgeFC0Inst->getOpcode())
        return nullptr;

      LLVM_DEBUG(dbgs() << "Found convergence point: " << *ConvergenceI
                        << "\n");
    }

    return ConvergenceI;
  }

  /// Unify two reduction values
  void unifyReductions(const FusionCandidate &FC0, const FusionCandidate &FC1,
                       PHINode *FC0_LcssaPhi, PHINode *FC1_LcssaPhi) {
    assert(FC0.RIV && FC1.RIV && "Expecting valid reduction PHIs");
    assert(FC0_LcssaPhi && FC1_LcssaPhi && "Expecting valid LCSSA PHIs");
    SmallVector<Instruction *> FC1_RIVUses;
    collectUses(FC1.RIV, FC1_RIVUses);
    assert(FC1_RIVUses.size() == 1 && "Expecting single use of FC1.RIV");
    for (auto *FC1_Use : FC1_RIVUses) {
      // Replace FC1_Use with the FC0.RIV's latch value.
      Value *ConnectingValue = FC0_LcssaPhi->getIncomingValue(0);
      FC1_Use->replaceUsesOfWith(FC1.RIV, ConnectingValue);

      // Now update FC0.RIV with FC1.RIV's latch value
      FC0.RIV->setIncomingValueForBlock(
          FC0.Latch, FC1.RIV->getIncomingValueForBlock(FC0.Latch));
      FC1.RIV->eraseFromParent();
      break;
    }
  }

  /// Replace the use of the convergence value with a stack collected
  /// value that is input from FC1_LcssaPhi's expression output.
  void updateConvergence(PHINode *FC0_LcssaPhi, PHINode *FC1_LcssaPhi,
                         Instruction *ConvergenceI, Instruction *LastFC1RelatedI) {
    assert(FC0_LcssaPhi && FC1_LcssaPhi && "Expecting valid LCSSA PHIs");
    assert(ConvergenceI && "Expecting valid convergence instruction");
    assert(LastFC1RelatedI && "Expecting valid last FC1 related instruction");

    ConvergenceI->replaceAllUsesWith(LastFC1RelatedI);
    ConvergenceI->eraseFromParent();
  }
#endif

  /// Fuse two fusion candidates, creating a new fused loop.
  ///
  /// This method contains the mechanics of fusing two loops, represented by \p
  /// FC0 and \p FC1. It is assumed that \p FC0 dominates \p FC1 and \p FC1
  /// postdominates \p FC0 (making them control flow equivalent). It also
  /// assumes that the other conditions for fusion have been met: adjacent,
  /// identical trip counts, and no negative distance dependencies exist that
  /// would prevent fusion. Thus, there is no checking for these conditions in
  /// this method.
  ///
  /// Fusion is performed by rewiring the CFG to update successor blocks of the
  /// components of tho loop. Specifically, the following changes are done:
  ///
  ///   1. The preheader of \p FC1 is removed as it is no longer necessary
  ///   (because it is currently only a single statement block).
  ///   2. The latch of \p FC0 is modified to jump to the header of \p FC1.
  ///   3. The latch of \p FC1 i modified to jump to the header of \p FC0.
  ///   4. All blocks from \p FC1 are removed from FC1 and added to FC0.
  ///
  /// All of these modifications are done with dominator tree updates, thus
  /// keeping the dominator (and post dominator) information up-to-date.
  ///
  /// This can be improved in the future by actually merging blocks during
  /// fusion. For example, the preheader of \p FC1 can be merged with the
  /// preheader of \p FC0. This would allow loops with more than a single
  /// statement in the preheader to be fused. Similarly, the latch blocks of the
  /// two loops could also be fused into a single block. This will require
  /// analysis to prove it is safe to move the contents of the block past
  /// existing code, which currently has not been implemented.
  Loop *performFusion(const FusionCandidate &FC0, const FusionCandidate &FC1) {
    assert(FC0.isValid() && FC1.isValid() &&
           "Expecting valid fusion candidates");

    LLVM_DEBUG(dbgs() << "Fusion Candidate 0: \n"; FC0.dump();
               dbgs() << "Fusion Candidate 1: \n"; FC1.dump(););

    // Move instructions from the preheader of FC1 to the end of the preheader
    // of FC0.
    moveInstructionsToTheEnd(*FC1.Preheader, *FC0.Preheader, DT, PDT, DI);

    // Fusing guarded loops is handled slightly differently than non-guarded
    // loops and has been broken out into a separate method instead of trying to
    // intersperse the logic within a single method.
    if (FC0.GuardBranch)
      return fuseGuardedLoops(FC0, FC1);

    assert(FC1.Preheader ==
           (FC0.Peeled ? FC0.ExitBlock->getUniqueSuccessor() : FC0.ExitBlock));

    assert(FC1.Preheader->size() == 1 &&
           FC1.Preheader->getSingleSuccessor() == FC1.Header);

    // Remember the phi nodes originally in the header of FC0 in order to rewire
    // them later. However, this is only necessary if the new loop carried
    // values might not dominate the exiting branch. While we do not generally
    // test if this is the case but simply insert intermediate phi nodes, we
    // need to make sure these intermediate phi nodes have different
    // predecessors. To this end, we filter the special case where the exiting
    // block is the latch block of the first loop. Nothing needs to be done
    // anyway as all loop carried values dominate the latch and thereby also the
    // exiting branch.
    SmallVector<PHINode *, 8> OriginalFC0PHIs;
    if (FC0.ExitingBlock != FC0.Latch)
      for (PHINode &PHI : FC0.Header->phis())
        OriginalFC0PHIs.push_back(&PHI);

#if SIFIVE_CUSTOMIZATION
    PHINode *FC0_LcssaPhi = nullptr;
    PHINode *FC1_LcssaPhi = nullptr;
    Value *FC0StepVal = nullptr;
    Value *FC1StepVal = nullptr;

    // We are going to replace all uses of FC1's step with FC0's step, as the
    // bounds are the same, then remove FC1's IV.  As a result, FC0's
    // compare will become orphaned, we do this for simple loops only.
    if (OriginalFC0PHIs.empty() && LoopConcatCanonicalize &&
        loopsHaveIdenticalBounds(FC0, FC1)) {
      FC0StepVal = FC0.IV->getIncomingValueForBlock(FC0.Latch);
      FC1StepVal = FC1.IV->getIncomingValueForBlock(FC1.Latch);
    }

    // Now check and see if FC0.RIV and FC1.RIV converge.
    Instruction *ConvergenceI = nullptr;
    Instruction *LastFC1RelatedI = nullptr;
    if (LoopConcatCanonicalize && FC0.RIV && FC1.RIV) {
      FC0_LcssaPhi = findLoopOutput(FC0);
      FC1_LcssaPhi = findLoopOutput(FC1);
      ConvergenceI = reductionValuesConverge(FC0, FC1, LastFC1RelatedI,
                                             FC0_LcssaPhi, FC1_LcssaPhi);
    }
#endif

    // Replace incoming blocks for header PHIs first.
    FC1.Preheader->replaceSuccessorsPhiUsesWith(FC0.Preheader);
    FC0.Latch->replaceSuccessorsPhiUsesWith(FC1.Latch);

    // Then modify the control flow and update DT and PDT.
    SmallVector<DominatorTree::UpdateType, 8> TreeUpdates;

    // The old exiting block of the first loop (FC0) has to jump to the header
    // of the second as we need to execute the code in the second header block
    // regardless of the trip count. That is, if the trip count is 0, so the
    // back edge is never taken, we still have to execute both loop headers,
    // especially (but not only!) if the second is a do-while style loop.
    // However, doing so might invalidate the phi nodes of the first loop as
    // the new values do only need to dominate their latch and not the exiting
    // predicate. To remedy this potential problem we always introduce phi
    // nodes in the header of the second loop later that select the loop carried
    // value, if the second header was reached through an old latch of the
    // first, or undef otherwise. This is sound as exiting the first implies the
    // second will exit too, __without__ taking the back-edge. [Their
    // trip-counts are equal after all.
    // KB: Would this sequence be simpler to just make FC0.ExitingBlock go
    // to FC1.Header? I think this is basically what the three sequences are
    // trying to accomplish; however, doing this directly in the CFG may mean
    // the DT/PDT becomes invalid
    if (!FC0.Peeled) {
      FC0.ExitingBlock->getTerminator()->replaceUsesOfWith(FC1.Preheader,
                                                           FC1.Header);
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Delete, FC0.ExitingBlock, FC1.Preheader));
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Insert, FC0.ExitingBlock, FC1.Header));
    } else {
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Delete, FC0.ExitBlock, FC1.Preheader));

      // Remove the ExitBlock of the first Loop (also not needed)
      FC0.ExitingBlock->getTerminator()->replaceUsesOfWith(FC0.ExitBlock,
                                                           FC1.Header);
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Delete, FC0.ExitingBlock, FC0.ExitBlock));
      FC0.ExitBlock->getTerminator()->eraseFromParent();
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Insert, FC0.ExitingBlock, FC1.Header));
      new UnreachableInst(FC0.ExitBlock->getContext(), FC0.ExitBlock);
    }

    // The pre-header of L1 is not necessary anymore.
    assert(pred_empty(FC1.Preheader));
    FC1.Preheader->getTerminator()->eraseFromParent();
    new UnreachableInst(FC1.Preheader->getContext(), FC1.Preheader);
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Delete, FC1.Preheader, FC1.Header));

    // Moves the phi nodes from the second to the first loops header block.
    while (PHINode *PHI = dyn_cast<PHINode>(&FC1.Header->front())) {
      if (SE.isSCEVable(PHI->getType()))
        SE.forgetValue(PHI);
      if (PHI->hasNUsesOrMore(1))
        PHI->moveBefore(FC0.Header->getFirstInsertionPt());
      else
        PHI->eraseFromParent();
    }

    // Introduce new phi nodes in the second loop header to ensure
    // exiting the first and jumping to the header of the second does not break
    // the SSA property of the phis originally in the first loop. See also the
    // comment above.
    BasicBlock::iterator L1HeaderIP = FC1.Header->begin();
    for (PHINode *LCPHI : OriginalFC0PHIs) {
      int L1LatchBBIdx = LCPHI->getBasicBlockIndex(FC1.Latch);
      assert(L1LatchBBIdx >= 0 &&
             "Expected loop carried value to be rewired at this point!");

      Value *LCV = LCPHI->getIncomingValue(L1LatchBBIdx);

      PHINode *L1HeaderPHI =
          PHINode::Create(LCV->getType(), 2, LCPHI->getName() + ".afterFC0");
      L1HeaderPHI->insertBefore(L1HeaderIP);
      L1HeaderPHI->addIncoming(LCV, FC0.Latch);
      L1HeaderPHI->addIncoming(PoisonValue::get(LCV->getType()),
                               FC0.ExitingBlock);

      LCPHI->setIncomingValue(L1LatchBBIdx, L1HeaderPHI);
    }

    // Replace latch terminator destinations.
    FC0.Latch->getTerminator()->replaceUsesOfWith(FC0.Header, FC1.Header);
    FC1.Latch->getTerminator()->replaceUsesOfWith(FC1.Header, FC0.Header);

    // Modify the latch branch of FC0 to be unconditional as both successors of
    // the branch are the same.
    simplifyLatchBranch(FC0);

    // If FC0.Latch and FC0.ExitingBlock are the same then we have already
    // performed the updates above.
    if (FC0.Latch != FC0.ExitingBlock)
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Insert, FC0.Latch, FC1.Header));

    TreeUpdates.emplace_back(DominatorTree::UpdateType(DominatorTree::Delete,
                                                       FC0.Latch, FC0.Header));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(DominatorTree::Insert,
                                                       FC1.Latch, FC0.Header));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(DominatorTree::Delete,
                                                       FC1.Latch, FC1.Header));

    // Update DT/PDT
    DTU.applyUpdates(TreeUpdates);

    LI.removeBlock(FC1.Preheader);
    DTU.deleteBB(FC1.Preheader);

    if (FC0.Peeled) {
      LI.removeBlock(FC0.ExitBlock);
      DTU.deleteBB(FC0.ExitBlock);
    }

    DTU.flush();

    // Is there a way to keep SE up-to-date so we don't need to forget the loops
    // and rebuild the information in subsequent passes of fusion?
    // Note: Need to forget the loops before merging the loop latches, as
    // mergeLatch may remove the only block in FC1.
    SE.forgetLoop(FC1.L);
    SE.forgetLoop(FC0.L);
    // Forget block dispositions as well, so that there are no dangling
    // pointers to erased/free'ed blocks.
    SE.forgetBlockAndLoopDispositions();

    // Move instructions from FC0.Latch to FC1.Latch.
    // Note: mergeLatch requires an updated DT.
    mergeLatch(FC0, FC1);

    // Merge the loops.
    SmallVector<BasicBlock *, 8> Blocks(FC1.L->blocks());
    for (BasicBlock *BB : Blocks) {
      FC0.L->addBlockEntry(BB);
      FC1.L->removeBlockFromLoop(BB);
      if (LI.getLoopFor(BB) != FC1.L)
        continue;
      LI.changeLoopFor(BB, FC0.L);
    }
    while (!FC1.L->isInnermost()) {
      const auto &ChildLoopIt = FC1.L->begin();
      Loop *ChildLoop = *ChildLoopIt;
      FC1.L->removeChildLoop(ChildLoopIt);
      FC0.L->addChildLoop(ChildLoop);
    }

#if SIFIVE_CUSTOMIZATION
    replaceInductionVarAndStep(FC0.IV, FC1.IV, FC0StepVal, FC1StepVal);

    // Now check and see if FC0.RIV and FC1.RIV converge, if so unify.
    // Specification:
    //   Update FC0.RIV with FC1.RIV's latch value after using
    //   FC0.RIV's latch value as input to FC1.RIV's reduction expression.
    //   We keep the output value of FC1.RIV to escape the loop to
    //   FC1_LcssaPhi.
    //   Then we replace the use of the convergence value with a stack collected
    //   value that is input from FC1_LcssaPhi's expression output.
    //   The net result will still be applied to the fused output via
    //   FC1_LcssaPhi's expression output.  Last of all we remove
    //   FC1.RIV as it too is dead.  Subsequently, we will manage
    //   the RIV value in FC0.  DCE will cleanup any residual dead code.
    if (ConvergenceI) {
      unifyReductions(FC0, FC1, FC0_LcssaPhi, FC1_LcssaPhi);
      updateConvergence(FC0_LcssaPhi, FC1_LcssaPhi, ConvergenceI,
                        LastFC1RelatedI);
    }
#endif

    // Delete the now empty loop L1.
    LI.erase(FC1.L);

#ifndef NDEBUG
    assert(!verifyFunction(*FC0.Header->getParent(), &errs()));
    assert(DT.verify(DominatorTree::VerificationLevel::Fast));
    assert(PDT.verify());
    LI.verify(DT);
    SE.verify();
#endif

    LLVM_DEBUG(dbgs() << "Fusion done:\n");

    return FC0.L;
  }

#if SIFIVE_CUSTOMIZATION
  /// Fuse two fusion candidates, creating a new fused loop.
  ///
  /// This method contains the mechanics of fusing two loops, represented by \p
  /// FC0 and \p FC1. It is assumed that \p FC0 dominates \p FC1 and \p FC1
  /// postdominates \p FC0 (making them control flow equivalent). It also
  /// assumes that the other conditions for fusion have been met: non adjacent,
  /// identical trip counts, and no negative distance dependencies exist that
  /// would prevent fusion. Thus, there is no checking for these conditions in
  /// this method.  We will require that there are no intervening uses of values
  /// created in FC0 between FC0's exit and FC1.
  ///
  /// Fusion is performed by rewiring the CFG to update successor blocks of the
  /// components of tho loop. Specifically, the following changes are done:
  ///
  ///   1. The preheader of \p FC0 is emptied and left for cfg simplify to
  ///      remove based on analysis.
  ///   2. The pred of the preheader of \p FC0 is rewired to successor of
  ///      FC0's exit block.
  ///   3. The latch of \p FC0 is modified to jump to the header of \p FC1.
  ///   4. The latch of \p FC1 i modified to jump to the header of \p FC0.
  ///      as its back edge.
  ///   5. All blocks from \p FC1 are removed from FC1 and added to FC0,
  ///      subsequently \p FC1 is discarded and its loop information
  ///      forgotten.
  ///
  /// All of these modifications are done with dominator tree updates, thus
  /// keeping the dominator (and post dominator) information up-to-date.
  ///
  /// This can be improved in the future by actually merging blocks during
  /// fusion. For example, the preheader of \p FC0 can be merged with the
  /// preheader of \p FC1. This would allow loops with more than a single
  /// statement in the preheader to be fused. Similarly, the latch blocks of the
  /// two loops could also be fused into a single block. This will require
  /// analysis to prove it is safe to move the contents of the block past
  /// existing code, which currently has not been implemented.
  ///
  /// For now each candiate \p FC0 that produces escaping values, must not have
  /// uses that either dominate the preheader of \p FC1 or are
  /// dominated by the exit block of \p FC0 or have no uses until \p FC1.
  Loop *performNonAdjacentFusion(
      const FusionCandidate &FC0, const FusionCandidate &FC1) {
    assert(FC0.isValid() && FC1.isValid() &&
           "Expecting valid fusion candidates");

    LLVM_DEBUG(dbgs() << "Fusion Candidate 0: \n"; FC0.dump();
               dbgs() << "Fusion Candidate 1: \n"; FC1.dump(););

    // Move instructions from the preheader of FC0 to the beginning of the preheader
    // of FC1.
    moveInstructionsToTheBeginning(*FC0.Preheader, *FC1.Preheader, DT, PDT, DI);

    // Find all lcssa phi nodes in FC0.ExitBlock.
    SmallVector<PHINode *, 8> OriginalFC0ExitPHIs;
    for (PHINode &PHI : FC0.ExitBlock->phis()) {
      if (PHI.getNumIncomingValues() != 1)
        continue;
      if (PHI.getBasicBlockIndex(FC0.Latch) >= 0)
      OriginalFC0ExitPHIs.push_back(&PHI);
    }

    // Remember the phi nodes originally in the header of FC0 in order to rewire
    // them later. However, this is only necessary if the new loop carried
    // values might not dominate the exiting branch. While we do not generally
    // test if this is the case but simply insert intermediate phi nodes, we
    // need to make sure these intermediate phi nodes have different
    // predecessors. To this end, we filter the special case where the exiting
    // block is the latch block of the first loop. Nothing needs to be done
    // anyways as all loop carried values dominate the latch and thereby also the
    // exiting branch.
    SmallVector<PHINode *, 8> OriginalFC0PHIs;
    if (FC0.ExitingBlock != FC0.Latch)
      for (PHINode &PHI : FC0.Header->phis())
        OriginalFC0PHIs.push_back(&PHI);

    PHINode *FC0_LcssaPhi = nullptr;
    PHINode *FC1_LcssaPhi = nullptr;
    Value *FC0StepVal = nullptr;
    Value *FC1StepVal = nullptr;

    // We are going to replace all uses of FC1's step with FC0's step, as the
    // bounds/step are the same, then remove FC1's IV.  As a result, FC0's
    // compare will become orphaned, we do this for simple loops only.
    if (OriginalFC0PHIs.empty() && LoopConcatCanonicalize &&
        loopsHaveIdenticalBounds(FC0, FC1)) {
      FC0StepVal = FC0.IV->getIncomingValueForBlock(FC0.Latch);
      FC1StepVal = FC1.IV->getIncomingValueForBlock(FC1.Latch);
    }

    // Now check and see if FC0.RIV and FC1.RIV converge.
    Instruction *ConvergenceI = nullptr;
    Instruction *LastFC1RelatedI = nullptr;
    if (LoopConcatCanonicalize && FC0.RIV && FC1.RIV) {
      FC0_LcssaPhi = findLoopOutput(FC0);
      FC1_LcssaPhi = findLoopOutput(FC1);
      ConvergenceI = reductionValuesConverge(FC0, FC1, LastFC1RelatedI,
                                             FC0_LcssaPhi, FC1_LcssaPhi);
    }

    // Replace incoming blocks for header PHIs first.
    FC0.Preheader->replaceSuccessorsPhiUsesWith(FC1.Preheader);
    FC0.Latch->replaceSuccessorsPhiUsesWith(FC1.Latch);

    // Move phi nodes from FC0.ExitBlock to FC1.ExitBlock after updates.
    for (auto *PHI : OriginalFC0ExitPHIs)
        PHI->moveBefore(FC1.ExitBlock->getFirstInsertionPt());

    // Then modify the control flow and update DT and PDT.
    SmallVector<DominatorTree::UpdateType, 8> TreeUpdates;

    // The old exiting block of the first loop (FC0) has to jump to the header
    // of the second as we need to execute the code in the second header block
    // regardless of the trip count. That is, if the trip count is 0, so the
    // back edge is never taken, we still have to execute both loop headers,
    // especially (but not only!) if the second is a do-while style loop.
    // However, doing so might invalidate the phi nodes of the first loop as
    // the new values do only need to dominate their latch and not the exiting
    // predicate. To remedy this potential problem we always introduce phi
    // nodes in the header of the second loop later that select the loop carried
    // value, if the second header was reached through an old latch of the
    // first, or undef otherwise. This is sound as exiting the first implies the
    // second will exit too, __without__ taking the back-edge. [Their
    // trip-counts are equal after all.
    // KB: Would this sequence be simpler to just make FC0.ExitingBlock go
    // to FC1.Header? I think this is basically what the three sequences are
    // trying to accomplish; however, doing this directly in the CFG may mean
    // the DT/PDT becomes invalid
    if (!FC0.Peeled) {
      FC0.ExitingBlock->getTerminator()->replaceUsesOfWith(FC0.ExitBlock,
                                                           FC1.Header);
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Insert, FC0.ExitingBlock, FC1.Header));
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Delete, FC0.ExitingBlock, FC0.ExitBlock));
    } else {
      assert(0 && "Currently Peeled solutions are not supported here");
    }

    // Retarget FC0.Preheader to FC0.ExitBlock.
    FC0.Preheader->getTerminator()->replaceUsesOfWith(FC0.Header, FC0.ExitBlock);
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Insert, FC0.Preheader, FC0.ExitBlock));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Delete, FC0.Preheader, FC0.Header));

    // Retarget FC1.Preheader to FC0.Header
    FC1.Preheader->getTerminator()->replaceUsesOfWith(FC1.Header, FC0.Header);
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Insert, FC1.Preheader, FC0.Header));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Delete, FC1.Preheader, FC1.Header));

    // Moves the phi nodes from the second to the first loops header block.
    while (PHINode *PHI = dyn_cast<PHINode>(&FC1.Header->front())) {
      if (SE.isSCEVable(PHI->getType()))
        SE.forgetValue(PHI);
      if (PHI->hasNUsesOrMore(1))
        PHI->moveBefore(FC0.Header->getFirstInsertionPt());
      else
        PHI->eraseFromParent();
    }

    // Introduce new phi nodes in the second loop header to ensure
    // exiting the first and jumping to the header of the second does not break
    // the SSA property of the phis originally in the first loop. See also the
    // comment above.
    BasicBlock::iterator L1HeaderIP = FC1.Header->begin();
    for (PHINode *LCPHI : OriginalFC0PHIs) {
      int L1LatchBBIdx = LCPHI->getBasicBlockIndex(FC1.Latch);
      assert(L1LatchBBIdx >= 0 &&
             "Expected loop carried value to be rewired at this point!");

      Value *LCV = LCPHI->getIncomingValue(L1LatchBBIdx);

      PHINode *L1HeaderPHI =
          PHINode::Create(LCV->getType(), 2, LCPHI->getName() + ".afterFC0");
      L1HeaderPHI->insertBefore(L1HeaderIP);
      L1HeaderPHI->addIncoming(LCV, FC0.Latch);
      L1HeaderPHI->addIncoming(PoisonValue::get(LCV->getType()),
                               FC0.ExitingBlock);

      LCPHI->setIncomingValue(L1LatchBBIdx, L1HeaderPHI);
    }

    // Replace latch terminator destinations.
    FC0.Latch->getTerminator()->replaceUsesOfWith(FC0.Header, FC1.Header);
    if (FC0.Latch != FC0.ExitingBlock)
      FC0.Latch->getTerminator()->replaceUsesOfWith(FC0.ExitBlock, FC1.Header);
    FC1.Latch->getTerminator()->replaceUsesOfWith(FC1.Header, FC0.Header);

    // Modify the latch branch of FC0 to be unconditional as both successors of
    // the branch are the same.
    simplifyLatchBranch(FC0);

    // If FC0.Latch and FC0.ExitingBlock are the same then we have already
    // performed the updates above.
    if (FC0.Latch != FC0.ExitingBlock)
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Insert, FC0.Latch, FC1.Header));

    TreeUpdates.emplace_back(DominatorTree::UpdateType(DominatorTree::Delete,
                                                       FC0.Latch, FC0.Header));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(DominatorTree::Insert,
                                                       FC1.Latch, FC0.Header));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(DominatorTree::Delete,
                                                       FC1.Latch, FC1.Header));

    // Update DT/PDT
    DTU.applyUpdates(TreeUpdates);
    LI.removeBlock(FC0.Preheader);

    // Is there a way to keep SE up-to-date so we don't need to forget the loops
    // and rebuild the information in subsequent passes of fusion?
    // Note: Need to forget the loops before merging the loop latches, as
    // mergeLatch may remove the only block in FC1.
    SE.forgetLoop(FC1.L);
    SE.forgetLoop(FC0.L);
    // Forget block dispositions as well, so that there are no dangling
    // pointers to erased/free'ed blocks.
    SE.forgetBlockAndLoopDispositions();

    // Move instructions from FC0.Latch to FC1.Latch.
    // Note: mergeLatch requires an updated DT.
    mergeLatch(FC0, FC1);

    // Merge the loops.
    SmallVector<BasicBlock *, 8> Blocks(FC1.L->blocks());
    for (BasicBlock *BB : Blocks) {
      FC0.L->addBlockEntry(BB);
      FC1.L->removeBlockFromLoop(BB);
      if (LI.getLoopFor(BB) != FC1.L)
        continue;
      LI.changeLoopFor(BB, FC0.L);
    }
    while (!FC1.L->isInnermost()) {
      const auto &ChildLoopIt = FC1.L->begin();
      Loop *ChildLoop = *ChildLoopIt;
      FC1.L->removeChildLoop(ChildLoopIt);
      FC0.L->addChildLoop(ChildLoop);
    }

    replaceInductionVarAndStep(FC0.IV, FC1.IV, FC0StepVal, FC1StepVal);

    // Now check and see if FC0.RIV and FC1.RIV converge, if so unify.
    // Specification:
    //   Update FC0.RIV with FC1.RIV's latch value after using
    //   FC0.RIV's latch value as input to FC1.RIV's reduction expression.
    //   We keep the output value of FC1.RIV to escape the loop to
    //   FC1_LcssaPhi.
    //   Then we replace the use of the convergence value with a stack collected
    //   value that is input from FC1_LcssaPhi's expression output.
    //   The net result will still be applied to the fused output via
    //   FC1_LcssaPhi's expression output.  Last of all we remove
    //   FC1.RIV as it too is dead.  Subsequently, we will manage
    //   the RIV value in FC0.  DCE will cleanup any residual dead code.
    if (ConvergenceI) {
      unifyReductions(FC0, FC1, FC0_LcssaPhi, FC1_LcssaPhi);
      updateConvergence(FC0_LcssaPhi, FC1_LcssaPhi, ConvergenceI,
                        LastFC1RelatedI);
    }

    // Delete the now empty loop L1.
    LI.erase(FC1.L);
    DTU.flush();

#ifndef NDEBUG
    assert(!verifyFunction(*FC0.Header->getParent(), &errs()));
    assert(DT.verify(DominatorTree::VerificationLevel::Fast));
    assert(PDT.verify());
    LI.verify(DT);
    SE.verify();
#endif

    LLVM_DEBUG(dbgs() << "Fusion done:\n");

    return FC0.L;
  }
#endif

  /// Report details on loop fusion opportunities.
  ///
  /// This template function can be used to report both successful and missed
  /// loop fusion opportunities, based on the RemarkKind. The RemarkKind should
  /// be one of:
  ///   - OptimizationRemarkMissed to report when loop fusion is unsuccessful
  ///     given two valid fusion candidates.
  ///   - OptimizationRemark to report successful fusion of two fusion
  ///     candidates.
  /// The remarks will be printed using the form:
  ///    <path/filename>:<line number>:<column number>: [<function name>]:
  ///       <Cand1 Preheader> and <Cand2 Preheader>: <Stat Description>
  template <typename RemarkKind>
  void reportLoopFusion(const FusionCandidate &FC0, const FusionCandidate &FC1,
                        llvm::Statistic &Stat) {
    assert(FC0.Preheader && FC1.Preheader &&
           "Expecting valid fusion candidates");
    using namespace ore;
#if LLVM_ENABLE_STATS
    ++Stat;
    ORE.emit(RemarkKind(DEBUG_TYPE, Stat.getName(), FC0.L->getStartLoc(),
                        FC0.Preheader)
             << "[" << FC0.Preheader->getParent()->getName()
             << "]: " << NV("Cand1", StringRef(FC0.Preheader->getName()))
             << " and " << NV("Cand2", StringRef(FC1.Preheader->getName()))
             << ": " << Stat.getDesc());
#endif
  }

  /// Fuse two guarded fusion candidates, creating a new fused loop.
  ///
  /// Fusing guarded loops is handled much the same way as fusing non-guarded
  /// loops. The rewiring of the CFG is slightly different though, because of
  /// the presence of the guards around the loops and the exit blocks after the
  /// loop body. As such, the new loop is rewired as follows:
  ///    1. Keep the guard branch from FC0 and use the non-loop block target
  /// from the FC1 guard branch.
  ///    2. Remove the exit block from FC0 (this exit block should be empty
  /// right now).
  ///    3. Remove the guard branch for FC1
  ///    4. Remove the preheader for FC1.
  /// The exit block successor for the latch of FC0 is updated to be the header
  /// of FC1 and the non-exit block successor of the latch of FC1 is updated to
  /// be the header of FC0, thus creating the fused loop.
  Loop *fuseGuardedLoops(const FusionCandidate &FC0,
                         const FusionCandidate &FC1) {
    assert(FC0.GuardBranch && FC1.GuardBranch && "Expecting guarded loops");

    BasicBlock *FC0GuardBlock = FC0.GuardBranch->getParent();
    BasicBlock *FC1GuardBlock = FC1.GuardBranch->getParent();
    BasicBlock *FC0NonLoopBlock = FC0.getNonLoopBlock();
    BasicBlock *FC1NonLoopBlock = FC1.getNonLoopBlock();
    BasicBlock *FC0ExitBlockSuccessor = FC0.ExitBlock->getUniqueSuccessor();

    // Move instructions from the exit block of FC0 to the beginning of the exit
    // block of FC1, in the case that the FC0 loop has not been peeled. In the
    // case that FC0 loop is peeled, then move the instructions of the successor
    // of the FC0 Exit block to the beginning of the exit block of FC1.
    moveInstructionsToTheBeginning(
        (FC0.Peeled ? *FC0ExitBlockSuccessor : *FC0.ExitBlock), *FC1.ExitBlock,
        DT, PDT, DI);

    // Move instructions from the guard block of FC1 to the end of the guard
    // block of FC0.
    moveInstructionsToTheEnd(*FC1GuardBlock, *FC0GuardBlock, DT, PDT, DI);

    assert(FC0NonLoopBlock == FC1GuardBlock && "Loops are not adjacent");

    SmallVector<DominatorTree::UpdateType, 8> TreeUpdates;

    ////////////////////////////////////////////////////////////////////////////
    // Update the Loop Guard
    ////////////////////////////////////////////////////////////////////////////
    // The guard for FC0 is updated to guard both FC0 and FC1. This is done by
    // changing the NonLoopGuardBlock for FC0 to the NonLoopGuardBlock for FC1.
    // Thus, one path from the guard goes to the preheader for FC0 (and thus
    // executes the new fused loop) and the other path goes to the NonLoopBlock
    // for FC1 (where FC1 guard would have gone if FC1 was not executed).
    FC1NonLoopBlock->replacePhiUsesWith(FC1GuardBlock, FC0GuardBlock);
    FC0.GuardBranch->replaceUsesOfWith(FC0NonLoopBlock, FC1NonLoopBlock);

    BasicBlock *BBToUpdate = FC0.Peeled ? FC0ExitBlockSuccessor : FC0.ExitBlock;
    BBToUpdate->getTerminator()->replaceUsesOfWith(FC1GuardBlock, FC1.Header);

    // The guard of FC1 is not necessary anymore.
    FC1.GuardBranch->eraseFromParent();
    new UnreachableInst(FC1GuardBlock->getContext(), FC1GuardBlock);

    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Delete, FC1GuardBlock, FC1.Preheader));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Delete, FC1GuardBlock, FC1NonLoopBlock));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Delete, FC0GuardBlock, FC1GuardBlock));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Insert, FC0GuardBlock, FC1NonLoopBlock));

    if (FC0.Peeled) {
      // Remove the Block after the ExitBlock of FC0
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Delete, FC0ExitBlockSuccessor, FC1GuardBlock));
      FC0ExitBlockSuccessor->getTerminator()->eraseFromParent();
      new UnreachableInst(FC0ExitBlockSuccessor->getContext(),
                          FC0ExitBlockSuccessor);
    }

    assert(pred_empty(FC1GuardBlock) &&
           "Expecting guard block to have no predecessors");
    assert(succ_empty(FC1GuardBlock) &&
           "Expecting guard block to have no successors");

    // Remember the phi nodes originally in the header of FC0 in order to rewire
    // them later. However, this is only necessary if the new loop carried
    // values might not dominate the exiting branch. While we do not generally
    // test if this is the case but simply insert intermediate phi nodes, we
    // need to make sure these intermediate phi nodes have different
    // predecessors. To this end, we filter the special case where the exiting
    // block is the latch block of the first loop. Nothing needs to be done
    // anyway as all loop carried values dominate the latch and thereby also the
    // exiting branch.
    // KB: This is no longer necessary because FC0.ExitingBlock == FC0.Latch
    // (because the loops are rotated. Thus, nothing will ever be added to
    // OriginalFC0PHIs.
    SmallVector<PHINode *, 8> OriginalFC0PHIs;
    if (FC0.ExitingBlock != FC0.Latch)
      for (PHINode &PHI : FC0.Header->phis())
        OriginalFC0PHIs.push_back(&PHI);

    assert(OriginalFC0PHIs.empty() && "Expecting OriginalFC0PHIs to be empty!");

    // Replace incoming blocks for header PHIs first.
    FC1.Preheader->replaceSuccessorsPhiUsesWith(FC0.Preheader);
    FC0.Latch->replaceSuccessorsPhiUsesWith(FC1.Latch);

    // The old exiting block of the first loop (FC0) has to jump to the header
    // of the second as we need to execute the code in the second header block
    // regardless of the trip count. That is, if the trip count is 0, so the
    // back edge is never taken, we still have to execute both loop headers,
    // especially (but not only!) if the second is a do-while style loop.
    // However, doing so might invalidate the phi nodes of the first loop as
    // the new values do only need to dominate their latch and not the exiting
    // predicate. To remedy this potential problem we always introduce phi
    // nodes in the header of the second loop later that select the loop carried
    // value, if the second header was reached through an old latch of the
    // first, or undef otherwise. This is sound as exiting the first implies the
    // second will exit too, __without__ taking the back-edge (their
    // trip-counts are equal after all).
    FC0.ExitingBlock->getTerminator()->replaceUsesOfWith(FC0.ExitBlock,
                                                         FC1.Header);

    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Delete, FC0.ExitingBlock, FC0.ExitBlock));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Insert, FC0.ExitingBlock, FC1.Header));

    // Remove FC0 Exit Block
    // The exit block for FC0 is no longer needed since control will flow
    // directly to the header of FC1. Since it is an empty block, it can be
    // removed at this point.
    // TODO: In the future, we can handle non-empty exit blocks my merging any
    // instructions from FC0 exit block into FC1 exit block prior to removing
    // the block.
    assert(pred_empty(FC0.ExitBlock) && "Expecting exit block to be empty");
    FC0.ExitBlock->getTerminator()->eraseFromParent();
    new UnreachableInst(FC0.ExitBlock->getContext(), FC0.ExitBlock);

    // Remove FC1 Preheader
    // The pre-header of L1 is not necessary anymore.
    assert(pred_empty(FC1.Preheader));
    FC1.Preheader->getTerminator()->eraseFromParent();
    new UnreachableInst(FC1.Preheader->getContext(), FC1.Preheader);
    TreeUpdates.emplace_back(DominatorTree::UpdateType(
        DominatorTree::Delete, FC1.Preheader, FC1.Header));

    // Moves the phi nodes from the second to the first loops header block.
    while (PHINode *PHI = dyn_cast<PHINode>(&FC1.Header->front())) {
      if (SE.isSCEVable(PHI->getType()))
        SE.forgetValue(PHI);
      if (PHI->hasNUsesOrMore(1))
        PHI->moveBefore(FC0.Header->getFirstInsertionPt());
      else
        PHI->eraseFromParent();
    }

    // Introduce new phi nodes in the second loop header to ensure
    // exiting the first and jumping to the header of the second does not break
    // the SSA property of the phis originally in the first loop. See also the
    // comment above.
    BasicBlock::iterator L1HeaderIP = FC1.Header->begin();
    for (PHINode *LCPHI : OriginalFC0PHIs) {
      int L1LatchBBIdx = LCPHI->getBasicBlockIndex(FC1.Latch);
      assert(L1LatchBBIdx >= 0 &&
             "Expected loop carried value to be rewired at this point!");

      Value *LCV = LCPHI->getIncomingValue(L1LatchBBIdx);

      PHINode *L1HeaderPHI =
          PHINode::Create(LCV->getType(), 2, LCPHI->getName() + ".afterFC0");
      L1HeaderPHI->insertBefore(L1HeaderIP);
      L1HeaderPHI->addIncoming(LCV, FC0.Latch);
      L1HeaderPHI->addIncoming(UndefValue::get(LCV->getType()),
                               FC0.ExitingBlock);

      LCPHI->setIncomingValue(L1LatchBBIdx, L1HeaderPHI);
    }

    // Update the latches

    // Replace latch terminator destinations.
    FC0.Latch->getTerminator()->replaceUsesOfWith(FC0.Header, FC1.Header);
    FC1.Latch->getTerminator()->replaceUsesOfWith(FC1.Header, FC0.Header);

    // Modify the latch branch of FC0 to be unconditional as both successors of
    // the branch are the same.
    simplifyLatchBranch(FC0);

    // If FC0.Latch and FC0.ExitingBlock are the same then we have already
    // performed the updates above.
    if (FC0.Latch != FC0.ExitingBlock)
      TreeUpdates.emplace_back(DominatorTree::UpdateType(
          DominatorTree::Insert, FC0.Latch, FC1.Header));

    TreeUpdates.emplace_back(DominatorTree::UpdateType(DominatorTree::Delete,
                                                       FC0.Latch, FC0.Header));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(DominatorTree::Insert,
                                                       FC1.Latch, FC0.Header));
    TreeUpdates.emplace_back(DominatorTree::UpdateType(DominatorTree::Delete,
                                                       FC1.Latch, FC1.Header));

    // All done
    // Apply the updates to the Dominator Tree and cleanup.

    assert(succ_empty(FC1GuardBlock) && "FC1GuardBlock has successors!!");
    assert(pred_empty(FC1GuardBlock) && "FC1GuardBlock has predecessors!!");

    // Update DT/PDT
    DTU.applyUpdates(TreeUpdates);

    LI.removeBlock(FC1GuardBlock);
    LI.removeBlock(FC1.Preheader);
    LI.removeBlock(FC0.ExitBlock);
    if (FC0.Peeled) {
      LI.removeBlock(FC0ExitBlockSuccessor);
      DTU.deleteBB(FC0ExitBlockSuccessor);
    }
    DTU.deleteBB(FC1GuardBlock);
    DTU.deleteBB(FC1.Preheader);
    DTU.deleteBB(FC0.ExitBlock);
    DTU.flush();

    // Is there a way to keep SE up-to-date so we don't need to forget the loops
    // and rebuild the information in subsequent passes of fusion?
    // Note: Need to forget the loops before merging the loop latches, as
    // mergeLatch may remove the only block in FC1.
    SE.forgetLoop(FC1.L);
    SE.forgetLoop(FC0.L);
    // Forget block dispositions as well, so that there are no dangling
    // pointers to erased/free'ed blocks.
    SE.forgetBlockAndLoopDispositions();

    // Move instructions from FC0.Latch to FC1.Latch.
    // Note: mergeLatch requires an updated DT.
    mergeLatch(FC0, FC1);

    // Merge the loops.
    SmallVector<BasicBlock *, 8> Blocks(FC1.L->blocks());
    for (BasicBlock *BB : Blocks) {
      FC0.L->addBlockEntry(BB);
      FC1.L->removeBlockFromLoop(BB);
      if (LI.getLoopFor(BB) != FC1.L)
        continue;
      LI.changeLoopFor(BB, FC0.L);
    }
    while (!FC1.L->isInnermost()) {
      const auto &ChildLoopIt = FC1.L->begin();
      Loop *ChildLoop = *ChildLoopIt;
      FC1.L->removeChildLoop(ChildLoopIt);
      FC0.L->addChildLoop(ChildLoop);
    }

    // Delete the now empty loop L1.
    LI.erase(FC1.L);

#ifndef NDEBUG
    assert(!verifyFunction(*FC0.Header->getParent(), &errs()));
    assert(DT.verify(DominatorTree::VerificationLevel::Fast));
    assert(PDT.verify());
    LI.verify(DT);
    SE.verify();
#endif

    LLVM_DEBUG(dbgs() << "Fusion done:\n");

    return FC0.L;
  }
};
} // namespace

PreservedAnalyses LoopFusePass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &DI = AM.getResult<DependenceAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  const TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  const DataLayout &DL = F.getDataLayout();

#if SIFIVE_CUSTOMIZATION
  if (!EnableLoopFusion && !LoopConcatCanonicalize)
    return PreservedAnalyses::all();
#endif

  // Ensure loops are in simplifed form which is a pre-requisite for loop fusion
  // pass. Added only for new PM since the legacy PM has already added
  // LoopSimplify pass as a dependency.
  bool Changed = false;
  for (auto &L : LI) {
    Changed |=
        simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false /* PreserveLCSSA */);
  }
  if (Changed)
    PDT.recalculate(F);

  LoopFuser LF(LI, DT, DI, SE, PDT, ORE, DL, AC, TTI);
  Changed |= LF.fuseLoops(F);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}
