//===- SiFive_LoopDataLayout.cpp - Loop Data Layout Pass ------------------===//
//
// Copyright (c) 2021-2022 SiFive, Inc. -- Proprietary and Confidential All
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
// This file implements the Loop Data Layout Pass. This pass is responsible
// for finding Array of Structure candidates for LTO which we can translate
// without great overhead to Structure of Arrays iff the AoS instance
// resides in another Structure as an array.  This is currently limited
// to non inheritance and non excepting source languages. The allowable
// candidates must cover casts to/from structs, parameter instances
// allocations, loop references and update instances. There is a cost
// heuristic for the target struct with respect to the number of scalar members
// or flat structs of things like complex numbers, currently that is
// managed in MaxElements. This is an assertion based optimization where Casts
// to/from struct type, escaped types, addresses taken of individual fields,
// parameters, return values and semantic of constants are all considered under
// legality and alias analysis.
//
//===----------------------------------------------------------------------===//

#if SIFIVE_CUSTOMIZATION

#include "llvm/Transforms/IPO/SiFive_LoopDataLayout.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "loop-data-layout"

STATISTIC(NumLoopsAnalyzed,
          "Number of loops examined for AoS to SoA opportunities");

static cl::opt<bool> EnableLoopDataLayout(
    "loop-data-layout-enable", cl::Hidden, cl::init(false),
    cl::desc("Discover Data Layout Opportunities in Loops"));

enum class LoopDataLayoutResult {
  HasDataLayoutOpportunities,
  HasNoOpportunities,
};

static bool isScalarType(Type *Ty) {
  return (Ty->isFloatingPointTy() || Ty->isIntegerTy());
}

static bool areInternalTypesComplexPairs(Type *Ty) {
  auto *Complex = cast<StructType>(Ty);
  if (Complex->getNumElements() != 2)
    return false;

  Type *FirstTy = Complex->getElementType(0);
  Type *SecondTy = Complex->getElementType(1);
  // Complex pairs must have same type
  if (FirstTy != SecondTy)
    return false;

  if (!FirstTy->isFloatingPointTy())
    return false;

  return true;
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

static bool findInductionVariableInLoops(LoopInfo &LI, ScalarEvolution &SE,
                                         PHINode *Index) {
  return llvm::any_of(LI, [&SE, Index](Loop *L) {
    InductionDescriptor ID;
    return InductionDescriptor::isInductionPHI(Index, L, &SE, ID);
  });
}

static LoopDataLayoutResult detectArrayOfStructDataAccess(
    Loop &L, LoopInfo &LI, ScalarEvolution &SE, unsigned MaxElements,
    SmallDenseMap<std::pair<Type *, GetElementPtrInst *>, int> &CandidateMap) {
  bool FoundArrayOfStructDataAccessor = false;

  // Detect data layout opportunities for AoS variables in the loop if there is
  // a preheader and it has dedicated exits.
  BasicBlock *Preheader = L.getLoopPreheader();
  if (!Preheader || !L.hasDedicatedExits()) {
    LLVM_DEBUG(
        dbgs()
        << "Data Layout requires Loop with preheader and dedicated exits.\n");
    return LoopDataLayoutResult::HasNoOpportunities;
  }

  // Now search our loop for AoS accesses
  for (BasicBlock *BB : L.blocks()) {
    for (Instruction &I : *BB) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        // Find AoS instances
        if (!GEP->isInBounds())
          continue;

        Type *Ty = GEP->getSourceElementType();
        if (!Ty->isStructTy())
          continue;

        // Now check that the index is a loop IV
        unsigned CandidateIndices = 0;
        for (Value *Index : GEP->indices()) {
          Index = peekThroughExtTrunc(Index);
          // A singleton index can be tied to an outer loop or
          // the current loop, we only care that it is phi based.
          auto *IndexPHINode = dyn_cast<PHINode>(Index);
          if (!IndexPHINode)
            continue;

          // IndexPHINode may not be an IV in L, so we search all the loops
          // correlated in LI to the current Function for a match.
          if (findInductionVariableInLoops(LI, SE, IndexPHINode))
            CandidateIndices++;
        }

        // If CandidateIndices is 1 its a good
        // indicator that we have a 1 dim AoS Candidate.
        // TODO: add some more GEP based validation to prove
        //       we have a single dim array access.
        if (CandidateIndices != 1)
          continue;

        Value *SrcPtr = GEP->getPointerOperand();
        // TODO: possibly other patterns?
        auto *Ld = dyn_cast<LoadInst>(SrcPtr);
        if (!Ld)
          continue;

        Value *Ptr = Ld->getPointerOperand();
        auto *SrcGEP = dyn_cast<GetElementPtrInst>(Ptr);
        if (!SrcGEP)
          continue;

        Type *SrcTy = SrcGEP->getSourceElementType();
        // No recursive references.
        if (SrcTy == Ty)
          continue;

        if (SrcTy->isStructTy() && SrcGEP->isInBounds()) {
          unsigned NumQualifyingFields = 0;
          auto *ST = cast<StructType>(Ty);
          for (unsigned k = 0, e = ST->getNumElements(); k != e; ++k) {
            Type *FieldType = ST->getElementType(k);
            if (isScalarType(FieldType)) {
              NumQualifyingFields++;
              continue;
            } else if (FieldType->isStructTy()) {
              // If struct, match a complex pair.
              if (areInternalTypesComplexPairs(FieldType)) {
                NumQualifyingFields++;
                continue;
              }
            }

            // No Match
            NumQualifyingFields = 0;
            break;
          }
          if (NumQualifyingFields <= MaxElements) {
            FoundArrayOfStructDataAccessor |= true;
            CandidateMap[{Ty, SrcGEP}]++;
          }
        }
      }
    }
  }
  Function *F = L.getHeader()->getParent();
  LLVM_DEBUG(dbgs() << "Function: " << F->getName() << "\n");
  LLVM_DEBUG(for (auto &Candididates : CandidateMap) {
    GetElementPtrInst *GEP = Candididates.first.second;
    Value *Ptr = GEP->getPointerOperand();
    // Discover if the containing struct is an argument to
    // the current function.
    if (isa<Argument>(Ptr))
      dbgs() << "GEP : Param Matched with " << Candididates.second
             << " instances\n";
    else if (isa<GlobalVariable>(Ptr))
      dbgs() << "GEP : Global Matched with " << Candididates.second
             << " instances\n";
    else
      dbgs() << "GEP : Local Matched with " << Candididates.second
             << " instances\n";
  });

  ++NumLoopsAnalyzed;
  return (FoundArrayOfStructDataAccessor)
             ? LoopDataLayoutResult::HasDataLayoutOpportunities
             : LoopDataLayoutResult::HasNoOpportunities;
}

/// updateArguments - Update AoS to SoA structs, member parameters,
/// and/or data member uses as transformations to SoA instances.
static Function *
updateArguments(Function *F, function_ref<AAResults &(Function &F)> AARGetter,
                unsigned MaxElements,
                Optional<function_ref<void(CallBase &OldCS, CallBase &NewCS)>>
                    ReplaceCallSite,
                const TargetTransformInfo &TTI) {
  // TODO: here is where we replace the AoS candidates with SoA tranformations.
  // TODO: Add STATISTIC to record NumTransformed seen here.
  return nullptr;
}

static LazyCallGraph buildCG(Module &M) {
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  TargetLibraryInfo TLI(TLII);
  auto GetTLI = [&TLI](Function &F) -> TargetLibraryInfo & { return TLI; };

  LazyCallGraph CG(M, GetTLI);
  return CG;
}

static bool runOnLoops(
    LoopInfo &LI, ScalarEvolution &SE, unsigned MaxElements,
    SmallDenseMap<std::pair<Type *, GetElementPtrInst *>, int> &CandidateMap) {
  bool FoundOpportunities = false;
  for (auto &L : LI) {
    LLVM_DEBUG(L->getName());
    // For the new PM, we can't use OptimizationRemarkEmitter as an analysis
    // pass. Function analyses need to be preserved across loop transformations
    // but ORE cannot be preserved (see comment before the pass definition).
    OptimizationRemarkEmitter ORE(L->getHeader()->getParent());
    LLVM_DEBUG(L->dump());
    auto Result =
        detectArrayOfStructDataAccess(*L, LI, SE, MaxElements, CandidateMap);
    if (Result == LoopDataLayoutResult::HasDataLayoutOpportunities)
      FoundOpportunities = true;
  }
  // TODO: add TBAA and GlobalAA detection of aliased conditions.
  return FoundOpportunities;
}

static bool analyzeWholeProgram(
    function_ref<LoopInfo &(Function &)> LookupLoopInfo,
    function_ref<ScalarEvolution &(Function &)> LookupScalarEvolutionInfo,
    Module &M, unsigned MaxElements,
    SmallDenseMap<std::pair<Type *, GetElementPtrInst *>, int> &CandidateMap) {
  if (!EnableLoopDataLayout)
    return false;

  LLVM_DEBUG(
      dbgs() << "Analyzing Loop collection for data layout opportunities: ");
  bool FoundOpportunities = false;
  // During main LTO, all modules have been fused into a single module.
  // The thinLTO interface is for testing purposes only right now.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    LoopInfo &LI = LookupLoopInfo(F);
    ScalarEvolution &SE = LookupScalarEvolutionInfo(F);
    FoundOpportunities |= runOnLoops(LI, SE, MaxElements, CandidateMap);
  }
  return FoundOpportunities;
}

PreservedAnalyses LoopDataLayoutPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupScalarEvolutionInfo = [&FAM](Function &F) -> ScalarEvolution & {
    return FAM.getResult<ScalarEvolutionAnalysis>(F);
  };
  auto LookupLoopInfo = [&FAM](Function &F) -> LoopInfo & {
    return FAM.getResult<LoopAnalysis>(F);
  };
  auto LookupAssumptionCache = [&FAM](Function &F) -> AssumptionCache * {
    return FAM.getCachedResult<AssumptionAnalysis>(F);
  };

  SmallDenseMap<std::pair<Type *, GetElementPtrInst *>, int> CandidateMap;

  // find data layout candidates for SoA to AoS transformation
  if (!analyzeWholeProgram(LookupLoopInfo, LookupScalarEvolutionInfo, M,
                           MaxElements, CandidateMap))
    return PreservedAnalyses::all();

  LazyCallGraph CG = buildCG(M);
  CG.buildRefSCCs();

  bool LocalChange, Changed = false;
  do {
    LocalChange = false;

    // Once we have the candidates, walk all functions updating the
    // candidates with updated SoA references and data member array uses.
    SmallVector<Function *, 100> Worklist;
    for (Function &OldF : M) {
      // FIXME: This lambda must only be used with this function. We should
      // skip the lambda and just get the AA results directly.
      auto AARGetter = [&](Function &F) -> AAResults & {
        assert(&F == &OldF && "Called with an unexpected function!");
        return FAM.getResult<AAManager>(F);
      };

      const TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(OldF);
      // TODO: updateArguments is a stub for now until we fill it in. This
      //       will be the place where we update params and uses in the
      //       function with the updated SoA equivalents as we will be
      //       replacing the old function with a modified call signature
      //       and so have to replace it.
      Function *NewF =
          updateArguments(&OldF, AARGetter, MaxElements, None, TTI);
      if (!NewF)
        continue;
      LocalChange = true;

      LazyCallGraph::Node &N = CG.get(OldF);

      // Remove @llvm.assume calls that will be moved to the new function
      // from the old function's assumption cache.
      AssumptionCache *AC = LookupAssumptionCache(OldF);
      for (BasicBlock &Block : OldF) {
        for (Instruction &I : llvm::make_early_inc_range(Block)) {
          if (auto *AI = dyn_cast<AssumeInst>(&I)) {
            if (AC)
              AC->unregisterAssumption(AI);
            AI->eraseFromParent();
          }
        }
      }

      // Directly substitute the functions in the call graph. Note that this
      // requires the old function to be completely dead and completely
      // replaced by the new function. It does no call graph updates, it
      // merely swaps out the particular function mapped to a particular node
      // in the graph.  We do this as we are altering the function definition
      // of OldF's parameters in these cases.  We will map as many as
      // MaxElements parameters to replace the AoS instance, member usage as
      // parameters will be the same where the call site will reference an array
      // indexed field.
      CG.lookupSCC(N)->getOuterRefSCC().replaceNodeFunction(N, *NewF);
      FAM.clear(OldF, OldF.getName());
      Worklist.push_back(&OldF);

      PreservedAnalyses FuncPA;
      FuncPA.preserveSet<CFGAnalyses>();
      for (auto *U : NewF->users()) {
        auto *UserF = cast<CallBase>(U)->getFunction();
        FAM.invalidate(*UserF, FuncPA);
      }
    }
    // Now remove all worklist Functions from this Module.
    while (!Worklist.empty()) {
      Function *F = Worklist.pop_back_val();
      F->eraseFromParent();
    }
    Changed |= LocalChange;
  } while (LocalChange);

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  PA.preserveSet<AllAnalysesOn<Function>>();
  return PA;
}

namespace {

class LoopDataLayoutLegacyPass : public ModulePass {
public:
  static char ID;

  LoopDataLayoutLegacyPass(unsigned MaxElements = 2)
      : ModulePass(ID), MaxElements(MaxElements) {
    initializeLoopDataLayoutLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    getLoopAnalysisUsage(AU);
  }

  /// The maximum number of elements to map and replace AoS with.
  unsigned MaxElements;
};

} // end anonymous namespace

char LoopDataLayoutLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(LoopDataLayoutLegacyPass, "loop-data-layout",
                      "Discover data layout opportunities", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(LoopDataLayoutLegacyPass, "loop-data-layout",
                    "Discover data layout opportunities", false, false)

ModulePass *llvm::createLoopDataLayoutPass(unsigned MaxElements) {
  return new LoopDataLayoutLegacyPass(MaxElements);
}

bool LoopDataLayoutLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  bool LocalChange, Changed = false;

  auto LookupScalarEvolutionInfo =
      [this, &Changed](Function &F) -> ScalarEvolution & {
    return this->getAnalysis<ScalarEvolutionWrapperPass>(F, &Changed).getSE();
  };
  auto LookupLoopInfo = [this, &Changed](Function &F) -> LoopInfo & {
    return this->getAnalysis<LoopInfoWrapperPass>(F, &Changed).getLoopInfo();
  };
  auto LookupACT = [this](Function &F) -> AssumptionCache * {
    if (auto *ACT = this->getAnalysisIfAvailable<AssumptionCacheTracker>())
      return ACT->lookupAssumptionCache(F);
    return nullptr;
  };

  SmallDenseMap<std::pair<Type *, GetElementPtrInst *>, int> CandidateMap;

  // find data layout candidates for SoA to AoS transformation
  if (!analyzeWholeProgram(LookupLoopInfo, LookupScalarEvolutionInfo, M,
                           MaxElements, CandidateMap))
    return false;

  LazyCallGraph CG = buildCG(M);
  CG.buildRefSCCs();

  LegacyAARGetter AARGetter(*this);

  do {
    LocalChange = false;

    // Once we have the candidates, walk all functions looking for
    // container type parameters, data member parameters and local
    // instance of variables of these types to update.
    SmallVector<Function *, 100> Worklist;
    for (Function &OldF : M) {
      const TargetTransformInfo &TTI =
          getAnalysis<TargetTransformInfoWrapperPass>().getTTI(OldF);

      // TODO: updateArguments is a stub for now until we fill it in. This
      //       will be the place where we update params and uses in the
      //       function with the updated SoA equivalents as we will be
      //       replacing the old function with a modified call signature
      //       and so have to replace it.
      Function *NewF =
          updateArguments(&OldF, AARGetter, MaxElements, None, TTI);
      if (!NewF)
        continue;

      // TODO: add local instance processing here
      LocalChange = true;

      LazyCallGraph::Node &N = CG.get(OldF);

      // Remove @llvm.assume calls that will be moved to the new function
      // from the old function's assumption cache.
      AssumptionCache *AC = LookupACT(OldF);
      for (BasicBlock &Block : OldF) {
        for (Instruction &I : llvm::make_early_inc_range(Block)) {
          if (auto *AI = dyn_cast<AssumeInst>(&I)) {
            if (AC)
              AC->unregisterAssumption(AI);
            AI->eraseFromParent();
          }
        }
      }

      // Directly substitute the functions in the call graph. Note that this
      // requires the old function to be completely dead and completely
      // replaced by the new function. It does no call graph updates, it
      // merely swaps out the particular function mapped to a particular node
      // in the graph.  We do this as we are altering the function definition
      // of OldF's parameters in these cases.  We will map as many as
      // MaxElements parameters to replace the AoS instance, member usage as
      // parameters will be the same where the call site will reference an array
      // indexed field.
      CG.lookupSCC(N)->getOuterRefSCC().replaceNodeFunction(N, *NewF);
      Worklist.push_back(&OldF);
    }
    // Now remove all worklist Functions from this Module.
    while (!Worklist.empty()) {
      Function *F = Worklist.pop_back_val();
      F->eraseFromParent();
    }
    Changed |= LocalChange;
  } while (LocalChange);

  return Changed;
}

void LoopDataLayoutPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<LoopDataLayoutPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);

  OS << "<";
  OS << ">";
}

#endif // SIFIVE_CUSTOMIZATION

