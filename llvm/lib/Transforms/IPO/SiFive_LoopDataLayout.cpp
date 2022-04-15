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
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
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
  TransformationIsIllegal
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

// This is a relaxed version of InductionDescriptor::isInductionPHI.  This
// version does not require Loop Normal Form.  This algorithm will accomodate
// tight loops, the absence of preheaders, dedicated exits and the like.
static bool isInductionPHI(PHINode *Index, Loop *L, ScalarEvolution &SE) {
  auto isSuitableIV = [&](PHINode *P) {
    if (!SE.isSCEVable(P->getType()))
      return false;
    if (const SCEVAddRecExpr *Rec = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(P)))
      return Rec->isAffine() && !SE.containsUndefs(SE.getSCEV(P));
    return false;
  };

  bool FoundInductionVar = false;
  if (Index->getParent() == L->getHeader()) {
    if (isSuitableIV(Index)) {
      ICmpInst *CmpInst = L->getLatchCmpInst();
      if (!CmpInst)
        return false;

      Value *LatchCmpOp0 = peekThroughExtTrunc(CmpInst->getOperand(0));
      Value *LatchCmpOp1 = peekThroughExtTrunc(CmpInst->getOperand(1));
      Value *StepVal = Index->getIncomingValueForBlock(L->getLoopLatch());
      if (!isa<Instruction>(StepVal))
        return false;

      // Must have CurP or StepVal as one of the compare operands.
      if (Index != LatchCmpOp0 && Index != LatchCmpOp1 &&
          StepVal != LatchCmpOp0 && StepVal != LatchCmpOp1)
        return false;

      // If the current induction variable in L is updated in StepInst, is an
      // incoming value in CurP and is our Index, then we have found a match.
      // This is a relaxed case of Simple Loop Form which does not require
      // dedicate exits and preheaders. These components may have been
      // transformed away after final loop optimizations before main LTO.
      Instruction *StepInst = cast<Instruction>(StepVal);
      FoundInductionVar |= (any_of(StepInst->operands(), [=](const Value *Op) {
        return (Op == Index);
      }));
    }
  }
  return FoundInductionVar;
}

static LoopDataLayoutResult detectArrayOfStructDataAccess(
    Loop *L, LoopInfo &LI, ScalarEvolution &SE, unsigned MaxElements,
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap) {
  bool FoundArrayOfStructDataAccessor = false;

  // Now search our loop for AoS accesses
  for (BasicBlock *BB : L->blocks()) {
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
          if (isa<ConstantInt>(Index))
            continue;

          Index = peekThroughExtTrunc(Index);
          // A singleton index can be tied to an outer loop or
          // the current loop, we only care that it is phi based.
          auto *IndexPHINode = dyn_cast<PHINode>(Index);
          if (!IndexPHINode)
            continue;

          // Obtain IndexPHINode's Loop from its BasicBlock, if it
          // is in a Loop, we check that Loop for a matching IV
          Loop *CurL = LI.getLoopFor(IndexPHINode->getParent());
          if (CurL && isInductionPHI(IndexPHINode, CurL, SE))
            CandidateIndices++;
        }

        // Record invariant accesses
        if (GEP->hasAllConstantIndices())
          CandidateIndices++;

        // If CandidateIndices is 1 its a good indicator that we have a 1 dim
        // AoS Candidate, any addition constant indices are field fetch ordinals
        // if not 0.
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

        // The base ptr should not be variable indexed
        if (!SrcGEP->hasAllConstantIndices())
          continue;

        Type *SrcTy = SrcGEP->getSourceElementType();
        // No recursive references.
        if (SrcTy == Ty)
          continue;

        if (SrcTy->isStructTy() && SrcGEP->isInBounds()) {
          // Ignore containers that are complex types as these are
          // structs with no value to translate.
          if (areInternalTypesComplexPairs(Ty))
            continue;

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
          if ((NumQualifyingFields > 0) &&
              (NumQualifyingFields <= MaxElements)) {
            FoundArrayOfStructDataAccessor |= true;
            // Store the address of the array of structs
            // plus the container address that holds it.
            LocalCandidateMap[{GEP, SrcGEP}]++;
          }
        }
      }
    }
  }
  LLVM_DEBUG(Function *F = L->getHeader()->getParent();
             dbgs() << "Function: " << F->getName() << "\n");
  LLVM_DEBUG(for (auto &Candididates
                  : LocalCandidateMap) {
    GetElementPtrInst *GEP = Candididates.first.second;
    Value *Ptr = GEP->getPointerOperand();
    GEP->dump();
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
  // TODO: add return type modification here as well.
  return nullptr;
}

static bool runOnLoops(
    LoopInfo &LI, ScalarEvolution &SE, unsigned MaxElements,
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap) {
  bool FoundOpportunities = false;
  for (auto &L : LI) {
    LLVM_DEBUG(dbgs() << "Processing Loop for AoS to SoA candidates : "
                      << L->getName() << "\n");
    // For the new PM, we can't use OptimizationRemarkEmitter as an analysis
    // pass. Function analyses need to be preserved across loop transformations
    // but ORE cannot be preserved (see comment before the pass definition).
    OptimizationRemarkEmitter ORE(L->getHeader()->getParent());
    LLVM_DEBUG(L->dump());
    // build a collection of loops that reside under L and process all.
    if (!L->isInnermost()) {
      for (Loop *CurL : depth_first(L)) {
        if (CurL == L)
          continue;

        auto Result = detectArrayOfStructDataAccess(CurL, LI, SE, MaxElements,
                                                    LocalCandidateMap);
        if (Result == LoopDataLayoutResult::HasDataLayoutOpportunities)
          FoundOpportunities |= true;
      }
    }
    // Always process L regardless of loop nest context
    auto Result = detectArrayOfStructDataAccess(L, LI, SE, MaxElements,
                                                LocalCandidateMap);
    if (Result == LoopDataLayoutResult::HasDataLayoutOpportunities)
      FoundOpportunities |= true;
  }
  return FoundOpportunities;
}

static bool canFunctionUpdate(Function *F, bool ThinLTO) {
  // Don't perform translate for naked functions; otherwise we can end
  // up changing parameters that are seemingly 'not used' as they are referred
  // to in the assembly.
  if (F->hasFnAttribute(Attribute::Naked)) {
    LLVM_DEBUG(dbgs() << "Naked attribute detected\n");
    return false;
  }

  // Make sure that it is local to this module.
  if (!ThinLTO && !F->hasLocalLinkage()) {
    LLVM_DEBUG(dbgs() << "No Local Linkage detected\n");
    return false;
  }

  // Don't translate parameters for variadic functions. Adding, removing, or
  // changing non-pack parameters can change the classification of pack
  // parameters. Frontends encode that classification at the call site in the
  // IR, while in the callee the classification is determined dynamically
  // based on the number of registers consumed so far. Caveat: main LTO and
  // used.
  if (!ThinLTO && F->isVarArg()) {
    if (F->getNumUses()) {
      LLVM_DEBUG(dbgs() << "Funtion with VarArgs has uses\n");
      return false;
    }
  }

  // Don't transform functions that receive inallocas, as the transformation
  // may not be safe depending on calling convention.
  if (F->getAttributes().hasAttrSomewhere(Attribute::InAlloca)) {
    LLVM_DEBUG(dbgs() << "InAlloca attribute detected\n");
    return false;
  }

  // Make sure that all callers are direct callers.  We can't
  // transform functions that have indirect callers.  Also see if the function
  // is a musttail callee.
  for (Use &U : F->uses()) {
    CallBase *CB = dyn_cast<CallBase>(U.getUser());
    // Must be a direct call.
    if (CB == nullptr || !CB->isCallee(&U)) {
      LLVM_DEBUG(dbgs() << "Indirect call detected\n");
      return false;
    }

    // Can't change signature of musttail callee
    if (CB->isMustTailCall()) {
      LLVM_DEBUG(dbgs() << "Musttail call detected\n");
      return false;
    }
  }

  // Can't change signature of musttail caller
  for (BasicBlock &BB : *F)
    if (BB.getTerminatingMustTailCall()) {
      LLVM_DEBUG(dbgs() << "Musttail call detected\n");
      return false;
    }

  return true;
}

static bool findRelatedCandidate(
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap,
    const Value *Ptr) {
  const GetElementPtrInst *InputGEP = dyn_cast<GetElementPtrInst>(Ptr);
  if (!InputGEP)
    return false;

  return (any_of(LocalCandidateMap, [=](auto &Candididates) {
    GetElementPtrInst *GEP = Candididates.first.first;
    GetElementPtrInst *SrcGEP = Candididates.first.second;
    // Check if we matched the current ptr to a Candidate
    return ((InputGEP == GEP) || (InputGEP == SrcGEP));
  }));
}

// TODO: make this a semantic comparison
static bool compareEqualGEPs(const GetElementPtrInst *A,
                             const GetElementPtrInst *B) {
  return ::equal(A->operands(), B->operands());
}

// Check to see if a given load is aliased, if so check to see if it is
// a candidate member or is a sibling load from one of the candidate
// container types.
static bool legalLoadRelationships(
    const LoadInst *InputLd, const GetElementPtrInst *SrcGEP,
    MemoryLocation &CandLoc,
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap,
    AAResults &AAR) {
  auto CurMemLocOpt = MemoryLocation::getOrNone(InputLd);
  if (!CurMemLocOpt) {
    LLVM_DEBUG(dbgs() << "Alias scenario ambiguous with " << *InputLd << "\n");
    return false;
  }

  MemoryLocation CurMemLoc = *CurMemLocOpt;
  if (AAR.alias(CandLoc, CurMemLoc) == AliasResult::NoAlias)
    return true;

  const Value *MemPtr = InputLd->getPointerOperand();
  // Check if this load is a candidate
  if (findRelatedCandidate(LocalCandidateMap, MemPtr))
    return true;

  // If this is a sibling field load its MemPtr will relate to a
  // Candidate indirectly, however the GEPs will not match but
  // will share the same container type.
  if (const auto *CurGEP = dyn_cast<GetElementPtrInst>(MemPtr)) {
    const Type *InTy = CurGEP->getSourceElementType();
    if (!compareEqualGEPs(SrcGEP, CurGEP)) {
      // Look for a shared container type with a Candidate
      if (any_of(LocalCandidateMap, [=](auto &Candididates) {
            GetElementPtrInst *CandGEP = Candididates.first.second;
            const Type *ContainerTy = CandGEP->getSourceElementType();
            return (ContainerTy == InTy);
          }))
        return true;
    } else {
      // The GEPs are the same but used in different context, a variant
      // non phi based index variable would cause this, we will discover
      // these when we find all references.
      return true;
    }
  }

  return false;
}

static bool addReferencesForFunction(
    SmallDenseMap<std::pair<Instruction *, Type *>, int> &LocalReferenceMap,
    SmallDenseSet<std::pair<Type *, Type *>> &UniqueTypeSet, Function *F) {
  // Look for references by type if a member of our candidate map.
  bool ReferencesAsArguments = false;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      for (unsigned k = 0; k < I.getNumOperands(); ++k) {
        auto *Op = I.getOperand(k);
        if (isa<Argument>(Op) || isa<Instruction>(Op)) {
          auto *CurTy = Op->getType();

          if (auto *PtrTy = dyn_cast<PointerType>(CurTy)) {
            if (PtrTy->isOpaque()) {
              // For Opaque pointers, these will be ptr in the operand, so
              // get the type from the GEP's result.
              if (auto *GEP = dyn_cast<GetElementPtrInst>(Op))
                CurTy = GEP->getSourceElementType();
              else
                continue;
            } else {
              CurTy = PtrTy->getNonOpaquePointerElementType();
            }
          }

          if (!CurTy->isStructTy())
            continue;

          // See if this struct is a known container or an array
          // of structs that it holds.
          for (auto &TypePair : UniqueTypeSet) {
            Type *ArrayTy = TypePair.first;
            Type *ContainerTy = TypePair.second;
            if (CurTy == ArrayTy) {
              ReferencesAsArguments |= isa<Argument>(Op);
              LocalReferenceMap[{&I, CurTy}]++;
            } else if (CurTy == ContainerTy) {
              ReferencesAsArguments |= isa<Argument>(Op);
              LocalReferenceMap[{&I, CurTy}]++;
            }
          }
        }
      }
    }
  }
  return ReferencesAsArguments;
}

static bool runOnFunction(
    LoopInfo &LI, ScalarEvolution &SE, AAResults &AAR, unsigned MaxElements,
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap,
    Function *F, bool ThinLTO) {
  // Look for legality issues which would block this transformation.
  // Generally the number of candidates for a given function is small.
  for (auto &Candididates : LocalCandidateMap) {
    GetElementPtrInst *GEP = Candididates.first.first;
    Type *Ty = GEP->getSourceElementType();
    GetElementPtrInst *SrcGEP = Candididates.first.second;
    Value *SrcPtr = GEP->getPointerOperand();
    auto *Ld = cast<LoadInst>(SrcPtr);
    auto CandLocOpt = MemoryLocation::getOrNone(Ld);
    if (!CandLocOpt)
      continue;

    MemoryLocation CandLoc = *CandLocOpt;
    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        if ((Ld == &I) || (GEP == &I) || (SrcGEP == &I))
          continue;

        if (auto *CurCast = dyn_cast<CastInst>(&I)) {
          // Type Escape Analysis: No casts into or out of Ty
          if ((I.getOperand(0)->getType() == Ty) || (I.getType() == Ty)) {
            LLVM_DEBUG(dbgs() << "cast into/out of Ty : " << *Ty << "\n");
            return false;
          }
          continue;
        }

        // Skip any load/store of this SoA Candidate
        bool IsLoadStore = (isa<LoadInst>(&I) || isa<StoreInst>(&I));
        if (IsLoadStore && any_of(I.operands(), [=](const Value *Op) {
              return ((Op == GEP) || (Op == SrcGEP));
            }))
          continue;

        // If any atomics are present the transformation is deemed illegal.
        // Reason: We cannot tell if another process, which may not share
        // our code, modifies explicitly the container or the array type.
        if (isa<FenceInst>(&I) || isa<AtomicCmpXchgInst>(&I) ||
            isa<AtomicRMWInst>(&I)) {
          LLVM_DEBUG(dbgs() << "Atomic Instruction(s) detected\n");
          return false;
        }

        // Find a candidate that matches this load's ptr and ignore
        // as these do not alias.
        GetElementPtrInst *InputGEP = nullptr;
        Value *MemPtr = nullptr;
        if (auto *CurLoad = dyn_cast<LoadInst>(&I)) {
          MemPtr = CurLoad->getPointerOperand();
          if (isStrongerThan(CurLoad->getOrdering(),
                             AtomicOrdering::Unordered)) {
            LLVM_DEBUG(dbgs() << "Atomic Load(s) detected\n");
            return false;
          }
          if (findRelatedCandidate(LocalCandidateMap, MemPtr))
            continue;

          InputGEP = dyn_cast<GetElementPtrInst>(MemPtr);
        }

        // Find a candidate that matches this store's ptr and ignore
        // as these do not alias.
        if (auto *CurStore = dyn_cast<StoreInst>(&I)) {
          MemPtr = CurStore->getPointerOperand();
          if (isStrongerThan(CurStore->getOrdering(),
                             AtomicOrdering::Unordered)) {
            LLVM_DEBUG(dbgs() << "Atomic Store(s) detected\n");
            return false;
          }
          if (findRelatedCandidate(LocalCandidateMap, MemPtr))
            continue;

          InputGEP = dyn_cast<GetElementPtrInst>(MemPtr);
        }

        // Now check memory locations accessed for aliases.
        auto MemLocOpt = MemoryLocation::getOrNone(&I);
        if (!MemLocOpt)
          continue;

        MemoryLocation MemLoc = *MemLocOpt;
        if (AAR.alias(CandLoc, MemLoc) == AliasResult::NoAlias)
          continue;

        if (InputGEP) {
          // Ptr chains are different, look at InputGEP some more
          if (InputGEP->getSourceElementType() !=
              SrcGEP->getSourceElementType()) {
            Value *InputSrcPtr = InputGEP->getPointerOperand();
            if (auto *InputLd = dyn_cast<LoadInst>(InputSrcPtr)) {
              if (legalLoadRelationships(InputLd, SrcGEP, CandLoc,
                                         LocalCandidateMap, AAR))
                continue;

              LLVM_DEBUG(dbgs()
                         << "Alias detected on Candidate" << *GEP << "\n");
              return false;
            } else if (isa<Argument>(InputSrcPtr)) {
              // Check the type to see if this arg diverges from the candidate
              // type (Ty) and if so it does not alias because of type escape
              // analysis.
              if (InputSrcPtr->getType() != Ty)
                continue;
            } else {
              // If this I is a load/store we already processed the candidates
              // looking for a match, in not finding one we also have no alias
              // as we will check for failure conditions in type escape
              // analysis.
              if (IsLoadStore)
                continue;
            }
          } else {
            // We are exmaining the same base type, if the GEPs are the same
            // they do not alias, the operations could be a realloc and a
            // candidate, we will find out when we find all references. If the
            // GEPs diverge, they are still referencing the same base type, ergo
            // a sibling field operation.
            continue;
          }
        } else if (MemPtr) {
          if (isa<ExtractElementInst>(MemPtr)) {
            // We do not currently have the ability to enroll structs into
            // vectors, ergo we know indirectly the two memory locations do
            // not alias.
            continue;
          }
          if (isa<CastInst>(MemPtr)) {
            // A cast into/out of our candidate array type will
            // cause a failure condition, we differ these to type
            // based escape analysis.
            continue;
          }
          // This pointer may have an underlying objects that originate from a
          // parameter in this function, if so check the type to see if it
          // diverges from the current candidate type (Ty) and if so it does
          // not alias the current candidate, as we will check for failure
          // conditions with casts into/out of the candidate in type escape
          // analysis. Else check if it is a load for candidate or is a
          // sibling container load.
          SmallVector<const Value *, 4> Objects;
          bool NoAlias = true;

          getUnderlyingObjects(MemPtr, Objects, &LI);
          LLVM_DEBUG(dbgs()
                     << "Underlying objects for pointer " << *MemPtr << "\n");
          for (const Value *UnderlyingObj : Objects) {
            if (isa<Argument>(UnderlyingObj)) {
              if (UnderlyingObj->getType() != Ty)
                continue;
            } else if (const auto *InputLd =
                           dyn_cast<LoadInst>(UnderlyingObj)) {
              if (legalLoadRelationships(InputLd, SrcGEP, CandLoc,
                                         LocalCandidateMap, AAR))
                continue;
            } else if (isa<CallInst>(UnderlyingObj)) {
              // Lib calls like calloc and realloc we can ignore, else
              // the call graph will be visited under M and each callee
              // examined for correctness, ergo we can ignore calls
              // because of type based escape analysis.
              continue;
            } else if (isa<CastInst>(UnderlyingObj)) {
              // Defer casts to type escape analysis.
              continue;
            }

            NoAlias = false;
            LLVM_DEBUG(dbgs() << *UnderlyingObj << "\n");
            LLVM_DEBUG(dbgs() << "Alias detected on Candidate" << *GEP << "\n");
            break;
          }
          if (NoAlias)
            continue;
        }
        LLVM_DEBUG(dbgs() << "unhandled case\n");
        return false;
      }
    }
  }

  return true;
}

static bool legalUseTree(Function *F, Instruction *I, TargetLibraryInfo &TLI) {
  unsigned NumInstUses = 0;
  unsigned NumLegalUses = 0;
  for (User *U : I->users()) {
    if (auto *Inst = dyn_cast<Instruction>(U)) {
      NumInstUses++;
      if (isa<CallInst>(Inst))
        if (isReallocLikeFn(Inst, &TLI) || Inst->isLifetimeStartOrEnd() ||
            isa<MemCpyInst>(Inst)) {
          NumLegalUses++;
          continue;
        }

      if (legalUseTree(F, Inst, TLI))
        NumLegalUses++;
    }
  }
  return (NumInstUses == NumLegalUses);
}

// For each cast, examine the divergent type path and reconcile if possible
// or determine as true type escape
static bool typeBasedEscapeAnalysis(
    TargetLibraryInfo &TLI, Function *F,
    SmallDenseSet<std::pair<Type *, Type *>> &UniqueTypeSet) {
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto *CurCast = dyn_cast<CastInst>(&I)) {
        for (auto &TypePair : UniqueTypeSet) {
          Type *ArrayTy = TypePair.first;
          Type *ContainerTy = TypePair.second;

          Type *SrcTy = CurCast->getSrcTy();
          if (auto *PtrTy = dyn_cast<PointerType>(SrcTy)) {
            if (PtrTy->isOpaque())
              continue;

            SrcTy = PtrTy->getNonOpaquePointerElementType();
          }

          Type *DstTy = CurCast->getDestTy();
          if (auto *PtrTy = dyn_cast<PointerType>(DstTy)) {
            if (PtrTy->isOpaque())
              continue;

            DstTy = PtrTy->getNonOpaquePointerElementType();
          }

          // Now examine the use chain, iterating possibly to the end
          // of the function to determine if the actions are legal
          // In this case the dst is divergent as the src matches.
          if ((SrcTy == ArrayTy) || (SrcTy == ContainerTy)) {
            if (legalUseTree(F, &I, TLI))
              break;

            LLVM_DEBUG(dbgs()
                       << "cast into/out of SrcTy : " << *SrcTy << "\n");
            return true;
          }
          // Now examine input objects to see if the actions are legal.
          // In this case the input arg is divergent as the dst matches.
          if ((DstTy == ArrayTy) || (DstTy == ContainerTy)) {
            SmallVector<const Value *, 4> Objects;

            getUnderlyingObjects(CurCast->getOperand(0), Objects);
            for (const Value *UnderlyingObj : Objects) {
              if (isa<CallInst>(UnderlyingObj)) {
                if (isReallocLikeFn(UnderlyingObj, &TLI) ||
                    isMallocOrCallocLikeFn(UnderlyingObj, &TLI))
                  continue;

                LLVM_DEBUG(dbgs() << "cast into/out of DstTy : "
                                  << *DstTy << "\n");
                return true;
              }
            }
          }
        }
      }
    }
  }
  return false;
}

static LoopDataLayoutResult analyzeWholeProgram(
    function_ref<LoopInfo &(Function &)> LookupLoopInfo,
    function_ref<ScalarEvolution &(Function &)> LookupScalarEvolutionInfo,
    function_ref<TargetLibraryInfo &(Function &)> LookupTLI,
    function_ref<AAResults &(Function &F)> AARGetter, GlobalsAAResult &GAAR,
    Module &M, unsigned MaxElements, bool ThinLTO,
    DenseMap<Function *,
             SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>,
                           int>> &CandidateMap,
    DenseMap<Function *, SmallDenseMap<std::pair<Instruction *, Type *>, int>>
        &ReferenceMap) {

  LLVM_DEBUG(
      dbgs() << "Analyzing Loop collection for data layout opportunities: ");
  bool FoundOpportunities = false;
  SmallDenseSet<std::pair<Type *, Type *>, 4> UniqueTypeSet;
  // During main LTO, all modules have been fused into a single module.
  // The thinLTO interface is for testing purposes only right now.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    LoopInfo &LI = LookupLoopInfo(F);
    ScalarEvolution &SE = LookupScalarEvolutionInfo(F);
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap = CandidateMap[&F];
    if (runOnLoops(LI, SE, MaxElements, LocalCandidateMap)) {
      FoundOpportunities |= true;
      AAResults &AAR = AARGetter(F);
      // AA and other legality analysis of candidates to transform.
      if (!runOnFunction(LI, SE, AAR, MaxElements, LocalCandidateMap, &F,
                         ThinLTO))
        return LoopDataLayoutResult::TransformationIsIllegal;

      // For all local candidate maps, fill in a unique type map of SrcGEP/GEP
      // types to use for locating references in functions.
      for (auto &Candididates : LocalCandidateMap)
        UniqueTypeSet.insert({
            Candididates.first.first->getSourceElementType(),
            Candididates.first.second->getSourceElementType()});
    }
  }

  if (!FoundOpportunities)
    return LoopDataLayoutResult::HasNoOpportunities;

  // Walk every function and mark the references
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    SmallDenseMap<std::pair<Instruction *, Type *>, int> &LocalReferenceMap =
        ReferenceMap[&F];

    // TODO: check into GAAR related queries if we need them here on F.
    // See AliasAnalysis.h for a list of queries to detect behavior.

    // Add references to a local map and check if any references were
    // arguments if we can modify the function, We do this as the loop
    // processing list may be disjunct when we check this there.
    if (addReferencesForFunction(LocalReferenceMap, UniqueTypeSet, &F))
      if (!canFunctionUpdate(&F, ThinLTO)) {
        LLVM_DEBUG(dbgs() << "canFunctionUpdate() issue\n");
        return LoopDataLayoutResult::TransformationIsIllegal;
      }

    TargetLibraryInfo &TLI = LookupTLI(F);
    if (typeBasedEscapeAnalysis(TLI, &F, UniqueTypeSet)) {
      LLVM_DEBUG(dbgs() << "Type Escape Analysis found escaped types\n");
      return LoopDataLayoutResult::TransformationIsIllegal;
    }
  }

  // Check GlobalAliases that are not Functions and compare types.
  for (const GlobalAlias &GA : M.aliases()) {
    const GlobalObject *Root = GA.getAliaseeObject();
    const Function *CurF = dyn_cast_or_null<Function>(Root);
    if (CurF)
      continue;

    // Check our unique list of types of candidates against each GA,
    // if we find any occurances, it is illegal to Transform AoS to SoA.
    Type *CurTy = GA.getType();
    if (auto *PtrTy = dyn_cast<PointerType>(CurTy)) {
      if (PtrTy->isOpaque())
        continue;

      CurTy = PtrTy->getNonOpaquePointerElementType();
    }

    if (!CurTy->isStructTy())
      continue;

    if (any_of(UniqueTypeSet, [=](auto &TypePair) {
          Type *UniqueArrayTy = TypePair.first;
          Type *UniqueContainerTy = TypePair.second;
          return ((UniqueArrayTy == CurTy) || (UniqueContainerTy == CurTy));
        }))
      return LoopDataLayoutResult::TransformationIsIllegal;
  }

  return LoopDataLayoutResult::HasDataLayoutOpportunities;
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
  auto AARGetter = [&](Function &F) -> AAResults & {
    return FAM.getResult<AAManager>(F);
  };
  auto LookupTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };

  DenseMap<
      Function *,
      SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>>
      CandidateMap;

  DenseMap<Function *, SmallDenseMap<std::pair<Instruction *, Type *>, int>>
      ReferenceMap;

  // Silently exit this optimization is disabled.
  if (!EnableLoopDataLayout)
    return PreservedAnalyses::all();

  llvm::CallGraph MCG(M);
  GlobalsAAResult GAAR = GlobalsAAResult::analyzeModule(M, LookupTLI, MCG);

  // find data layout candidates for SoA to AoS transformation
  auto Result = analyzeWholeProgram(LookupLoopInfo, LookupScalarEvolutionInfo,
                                    LookupTLI, AARGetter, GAAR, M, MaxElements,
                                    false, CandidateMap, ReferenceMap);

  // Silently exit as there are no opportunites to do the transformation.
  if (Result == LoopDataLayoutResult::HasNoOpportunities)
    return PreservedAnalyses::all();

  // Exit with status, transformation is illegal to perform.
  if (Result == LoopDataLayoutResult::TransformationIsIllegal) {
    LLVM_DEBUG(dbgs() << "Not Legal to do AoS to SoA Transformation\n");
    return PreservedAnalyses::all();
  }

  LazyCallGraph CG(M, LookupTLI);
  CG.buildRefSCCs();

  bool LocalChange, Changed = false;
  do {
    LocalChange = false;

    // Once we have the candidates, walk all functions updating the
    // candidates with updated SoA references and data member array uses.
    SmallVector<Function *, 100> Worklist;
    for (Function &OldF : M) {
      const TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(OldF);
      // TODO: updateArguments is a stub for now until we fill it in. This
      //       will be the place where we update params and uses in the
      //       function with the updated SoA equivalents as we will be
      //       replacing the old function with a modified call signature
      //       and so have to replace it. We will also address return types
      //       as needed.
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
        // TODO: visit all the callers of OldF since we modify the call
        // signature, there may be side effect code to inject at each site.
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
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
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
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(LoopDataLayoutLegacyPass, "loop-data-layout",
                    "Discover data layout opportunities", false, false)

ModulePass *llvm::createLoopDataLayoutPass(unsigned MaxElements) {
  return new LoopDataLayoutLegacyPass(MaxElements);
}

bool LoopDataLayoutLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  // Silently exit this optimization is disabled.
  if (!EnableLoopDataLayout)
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
  auto LookupTLI = [this](Function &F) -> TargetLibraryInfo & {
    return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  };

  DenseMap<
      Function *,
      SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>>
      CandidateMap;
  DenseMap<Function *, SmallDenseMap<std::pair<Instruction *, Type *>, int>>
      ReferenceMap;
  LegacyAARGetter AARGetter(*this);

  llvm::CallGraph MCG(M);
  GlobalsAAResult GAAR = GlobalsAAResult::analyzeModule(M, LookupTLI, MCG);

  // find data layout candidates for SoA to AoS transformation
  auto Result = analyzeWholeProgram(LookupLoopInfo, LookupScalarEvolutionInfo,
                                    LookupTLI, AARGetter, GAAR, M, MaxElements,
                                    true, CandidateMap, ReferenceMap);

  // Silently exit as there are no opportunites to do the transformation.
  if (Result == LoopDataLayoutResult::HasNoOpportunities)
    return false;

  // Exit with status, transformation is illegal to perform.
  if (Result == LoopDataLayoutResult::TransformationIsIllegal) {
    LLVM_DEBUG(dbgs() << "Not Legal to do AoS to SoA Transformation\n");
    return false;
  }

  LazyCallGraph CG(M, LookupTLI);
  CG.buildRefSCCs();

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
