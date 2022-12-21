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

#include <atomic>
#include <type_traits>
#if SIFIVE_CUSTOMIZATION

#include "llvm/Transforms/IPO/SiFive_LoopDataLayout.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/ADT/SmallBitVector.h"
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
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "loop-data-layout"

STATISTIC(NumLoopsAnalyzed,
          "Number of loops examined for AoS to SoA opportunities");
STATISTIC(NumTransformed,
          "Number of AoS to SoA references transformed");

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
  auto FindAltLatchCmp = [&](Instruction *StepInst, Loop *L) -> ICmpInst * {
    for (User *U : StepInst->users())
      if (auto *CurCmp = dyn_cast<ICmpInst>(U))
        if (CurCmp->hasOneUse())
          if (auto *BI = dyn_cast<BranchInst>(*CurCmp->user_begin()))
            if (llvm::is_contained(BI->successors(), L->getLoopLatch()))
              return CurCmp;
    return nullptr;
  };

  bool FoundInductionVar = false;
  if (Index->getParent() == L->getHeader()) {
    Type *PhiTy = Index->getType();
    // We only handle integer inductions variables.
    if (!PhiTy->isIntegerTy())
      return false;

    if (isSuitableIV(Index)) {
      Value *StepVal = Index->getIncomingValueForBlock(L->getLoopLatch());
      auto *StepInst = dyn_cast<Instruction>(StepVal);
      if (!StepInst)
        return false;

      // If the control that exits this loop is not canonnical, the latch cmp
      // may be left empty as that control may not be in the latch block.
      ICmpInst *CmpInst = L->getLatchCmpInst();
      if (!CmpInst) {
        // Check if one of the uses of StepInst is a cmp, validate
        // that it is used in control flow which includes the latch
        // block, else the control flow is complex and we return
        // false.
        CmpInst = FindAltLatchCmp(StepInst, L);
        if (!CmpInst)
          return false;
      }

      Value *LatchCmpOp0 = peekThroughExtTrunc(CmpInst->getOperand(0));
      Value *LatchCmpOp1 = peekThroughExtTrunc(CmpInst->getOperand(1));

      // Must have CurP or StepVal as one of the compare operands.
      if (Index != LatchCmpOp0 && Index != LatchCmpOp1 &&
          StepVal != LatchCmpOp0 && StepVal != LatchCmpOp1)
        return false;

      // If the current induction variable in L is updated in StepInst, is an
      // incoming value in CurP and is our Index, then we have found a match.
      // This is a relaxed case of Simple Loop Form which does not require
      // dedicate exits and preheaders. These components may have been
      // transformed away after final loop optimizations before main LTO.
      FoundInductionVar |= (any_of(StepInst->operands(), [=](const Value *Op) {
        return (Op == Index);
      }));
    }
  }
  return FoundInductionVar;
}

static bool compareEqualGEPs(const GetElementPtrInst *A,
                             const GetElementPtrInst *B) {
  return ::equal(A->operands(), B->operands());
}

static bool matchBaseGEP(GetElementPtrInst *GEP, Type *RefTy, Value *RefPtr,
                         const SmallVectorImpl<Value *> &Indices) {
  if ((RefTy != GEP->getSourceElementType()) ||
      (RefPtr != GEP->getPointerOperand()) || (GEP->getNumIndices() != 2))
    return false;

  return ((GEP->getOperand(1) == Indices[0]) &&
          (GEP->getOperand(2) == Indices[1]));
}

static LoopDataLayoutResult detectArrayOfStructDataAccess(
    Loop *L, LoopInfo &LI, ScalarEvolution &SE, unsigned MaxElements,
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap, bool &IsVectorized) {
  bool FoundArrayOfStructDataAccessor = false;

  if (getBooleanLoopAttribute(L, "llvm.loop.isvectorized"))
    IsVectorized = true;

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
          unsigned NumComplexFields = 0;
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
                NumComplexFields++;
                continue;
              }
            }

            // No Match
            NumQualifyingFields = 0;
            break;
          }
          if ((NumQualifyingFields > 0) &&
              (NumComplexFields <= MaxElements) &&
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
    GetElementPtrInst *SrcGEP = Candididates.first.second;
    Value *Ptr = SrcGEP->getPointerOperand();
    // Discover if the containing struct is an argument to
    // the current function.
    if (isa<Argument>(Ptr))
      dbgs() << "SrcGEP : Param Matched with " << Candididates.second
             << " instances\n";
    else if (isa<GlobalVariable>(Ptr))
      dbgs() << "SrcGEP : Global Matched with " << Candididates.second
             << " instances\n";
    else
      dbgs() << "SrcGEP : Local Matched with " << Candididates.second
             << " instances\n";
  });


  ++NumLoopsAnalyzed;
  return (FoundArrayOfStructDataAccessor)
             ? LoopDataLayoutResult::HasDataLayoutOpportunities
             : LoopDataLayoutResult::HasNoOpportunities;
}

static bool updateParamTypeAttributes(
    const SmallDenseSet<std::pair<Type *, Type *>> &TranslatedTypeSet,
    AttributeList &Attrs, LLVMContext &C) {
  bool UpdatedAttrs = false;
  for (unsigned i = 0; i < Attrs.getNumAttrSets(); ++i)
    for (int AttrIdx = Attribute::FirstTypeAttr;
         AttrIdx <= Attribute::LastTypeAttr; AttrIdx++) {
      Attribute::AttrKind TypedAttr = (Attribute::AttrKind)AttrIdx;
      if (Type *ParamTy =
              Attrs.getAttributeAtIndex(i, TypedAttr).getValueAsType())
        for (auto &TypePair : TranslatedTypeSet) {
          Type *OrigTy = TypePair.first;
          Type *ReplacementTy = TypePair.second;
          if (ParamTy == OrigTy) {
            // now replace ParamTy with the ReplacementTy.
            Attrs = Attrs.replaceAttributeTypeAtIndex(C, i, TypedAttr,
                                                      ReplacementTy);
            UpdatedAttrs = true;
            break;
          }
        }
    }

  return UpdatedAttrs;
}

static bool updateDereferenceableAttributes(Instruction *Call, unsigned Bytes) {
  Function *F = Call->getParent()->getParent();
  LLVMContext &C = F->getContext();
  auto *CB = cast<CallBase>(Call);
  AttributeList CallerPAL = CB->getAttributes();
  auto AI = CB->arg_begin();
  bool UpdatedAttrs = false;
  for (unsigned i = 0, e = CB->arg_size(); i != e; ++i, ++AI) {
    AttributeSet PAS = CallerPAL.getParamAttrs(i);
    if (PAS.hasAttribute(Attribute::DereferenceableOrNull)) {
      CallerPAL = CallerPAL.removeParamAttribute(C, i, Attribute::DereferenceableOrNull);
      CallerPAL = CallerPAL.addDereferenceableOrNullParamAttr(C, i, Bytes);
      UpdatedAttrs = true;
    } else if (PAS.hasAttribute(Attribute::Dereferenceable)) {
      CallerPAL = CallerPAL.removeParamAttribute(C, i, Attribute::Dereferenceable);
      CallerPAL = CallerPAL.addDereferenceableParamAttr(C, i, Bytes);
      UpdatedAttrs = true;
    }
  }
  AttributeSet RAS = CallerPAL.getRetAttrs();
  if (RAS.hasAttribute(Attribute::DereferenceableOrNull)) {
    CallerPAL = CallerPAL.removeRetAttribute(C, Attribute::DereferenceableOrNull);
    // We take the long route as support for addDereferenceableOrNullRetAttr
    // does not exist yet.
    AttrBuilder B(C);
    B.addDereferenceableOrNullAttr(Bytes);
    CallerPAL = CallerPAL.addRetAttributes(C, B);
    UpdatedAttrs = true;
  } else if (RAS.hasAttribute(Attribute::Dereferenceable)) {
    CallerPAL = CallerPAL.removeRetAttribute(C, Attribute::Dereferenceable);
    CallerPAL = CallerPAL.addDereferenceableRetAttr(C, Bytes);
    UpdatedAttrs = true;
  }
  if (UpdatedAttrs)
    CB->setAttributes(CallerPAL);

  return UpdatedAttrs;
}

static bool updateFunctionAttributes(
    Function *F,
    const SmallDenseSet<std::pair<Type *, Type *>> &TranslatedTypeSet) {
  // Walk all the Attrs of this function and look for type updates.
  LLVMContext &C = F->getContext();
  AttributeList Attrs = F->getAttributes();
  bool UpdatedAttrs = updateParamTypeAttributes(TranslatedTypeSet, Attrs, C);
  F->setAttributes(Attrs);
  return UpdatedAttrs;
}

static uint64_t gatherIndicesForArrayAddress(SmallVectorImpl<Value *> &Indices,
                                             IRBuilder<> &IRB,
                                             StructType *ArrayST, Type *AltTy,
                                             bool &UpdateSrcGEP,
                                             bool &HasIV,
                                             uint64_t &ArrayTyIdx,
                                             const GetElementPtrInst *CurGEP) {
  bool FirstTime = true;
  UpdateSrcGEP = false;
  ArrayTyIdx = 0;
  unsigned CurIdx = 0;
  unsigned NumIdx = CurGEP->getNumIndices();
  auto *ContainerST = cast<StructType>(AltTy);
  uint64_t NumContainerTypeElements = ContainerST->getNumElements();
  uint64_t NumArrayTypeElements = ArrayST->getNumElements();
  uint64_t StartingOffset = NumContainerTypeElements - NumArrayTypeElements;
  for (Value *Index : CurGEP->indices()) {
    CurIdx++;
    ConstantInt *Offset = dyn_cast<ConstantInt>(Index);
    if (Offset && FirstTime) {
      bool IsLastIndex = (CurIdx == NumIdx);
      ArrayTyIdx = Offset->getZExtValue();
      if (!IsLastIndex) {
        // We need to update the SrcGEP to reference the new
        // field indices and the correct type of those fields
        // in the container type, but only if an IV is present.
        if (HasIV) {
          UpdateSrcGEP = true;
        } else {
          assert(CurGEP->hasAllConstantIndices() && "Unsupported GEP form");
          Indices.push_back(Index);
          continue;
        }
      } else {
        Value *NewIndex = IRB.CreateAdd(
            Offset, ConstantInt::get(Index->getType(), StartingOffset));
        Indices.push_back(NewIndex);
      }
      FirstTime = false;
      continue;
    } else {
      HasIV = true;
    }
    Indices.push_back(Index);
  }

  // no constant index was found, imply zero
  if (FirstTime)
    Indices.push_back(ConstantInt::get(IRB.getInt32Ty(), StartingOffset));

  return StartingOffset;
}

static unsigned calculateStructSizeInBytes(StructType *InputST,
                                           const DataLayout &DL) {
  return TypeSize::Fixed(DL.getStructLayout(InputST)->getSizeInBits()) >> 3;
}

static void updateAllocationSize(IRBuilder<> &IRB, LibFunc TLIFn,
                                 Instruction *I, Value *SizeVal,
                                 const unsigned SizeInBytes,
                                 const unsigned SizeArg,
                                 const unsigned SizeOfArg) {
  if (TLIFn == LibFunc_malloc) {
    I->setOperand(SizeArg, SizeVal);
  } else if (TLIFn == LibFunc_realloc) {
    I->setOperand(SizeArg, SizeVal);
  } else if (TLIFn == LibFunc_calloc) {
    Type *Ty = I->getOperand(SizeOfArg)->getType();
    I->setOperand(SizeOfArg, ConstantInt::get(Ty, SizeInBytes));
  }
}

static Value *createNewSizeArg(StructType *ArrayST, Value *SrcPtr,
                               Instruction *I, LibFunc TLIFn,
                               const DataLayout &DL, Value *SizeVal,
                               const unsigned SizeInBytes) {
  if ((TLIFn == LibFunc_malloc) || (TLIFn == LibFunc_realloc)) {
    IRBuilder<> IRB(I);
    Type *DivTy = DL.getIntPtrType(SrcPtr->getType());
    uint64_t C;
    if (match(SizeVal, m_Shl(m_Value(), m_ConstantInt(C)))) {
      // We check size of ArrayST in bytes against the shift value as a power of
      // 2 value.  If both are equal we can clone this operation and replace the
      // OldSizeInBytes with SizeInBytes, converted to the exp component, provided
      // it too is a power of 2 value.
      unsigned OldSizeInBytes = calculateStructSizeInBytes(ArrayST, DL);
      unsigned SizeInBase10 = 1 << C;
      bool CanReplaceSize = (SizeInBase10 == OldSizeInBytes);
      APInt NewSize(64, SizeInBytes, false);
      if (NewSize.isPowerOf2() && CanReplaceSize) {
        unsigned ShiftAmt = 0;
        unsigned Seed = SizeInBytes;
        while (Seed != 1) {
          Seed = Seed >> 1;
          ShiftAmt++;
        }
        Instruction *OrigI = cast<Instruction>(SizeVal);
        Instruction *NewI = OrigI->clone();
        // Place the cloned call near the old one.
        NewI->insertAfter(OrigI);
        NewI->setOperand(1, ConstantInt::get(DivTy, ShiftAmt));
        return NewI;
      }
    }

    return IRB.CreateMul(
        IRB.CreateSDiv(
            SizeVal,
            ConstantInt::get(DivTy, calculateStructSizeInBytes(ArrayST, DL))),
        ConstantInt::get(DivTy, SizeInBytes));
  }

  return SizeVal;
}

static void updateAddressWithGlossary(
    DenseMap<Type *, SmallVector<Value *>> &GEPTypeToIndices,
    GetElementPtrInst *CurGEP, Type *CurTy, Type *AltTy) {
  // Match the CurTy to a glossary entry and use its
  // GEP to update CurGEP if they have the same indices.
  IRBuilder<> Builder(CurGEP);
  assert(CurGEP->hasAllConstantIndices() && "must be constant indexed CurGEP");
  const SmallVector<Value *> &Indices = GEPTypeToIndices[CurTy];
  // If we have already modified CurGEP, the indices will be empty.
  if (Indices.empty())
    return;

  bool AllIndicesMatch = true;
  unsigned NumCurIdx = CurGEP->getNumIndices();
  unsigned NumSrcIdx = Indices.size();
  if (NumCurIdx == NumSrcIdx) {
    unsigned StartIdx = CurGEP->getNumOperands() - NumCurIdx;
    for (unsigned i = StartIdx; i < NumCurIdx + StartIdx; i++)
      if (CurGEP->getOperand(i) != Indices[i - 1]) {
        AllIndicesMatch = false;
        break;
      }

  }
  if (AllIndicesMatch) {
    auto *CurST = cast<StructType>(CurTy);
    uint64_t StartingOffset = CurST->getNumElements();
    unsigned Idx = CurGEP->getNumOperands();
    CurGEP->setOperand(Idx - 1,
                       ConstantInt::get(Builder.getInt32Ty(), StartingOffset));
  }
  // Replace the base type with AltTy as all translated and original
  // references are based off of it.
  CurGEP->setSourceElementType(AltTy);

  // Check the allocation type and update as we just updated the CurGEP
  // to AltTy, we also need the allocation type to refect that also.
  Value *SrcPtr = CurGEP->getPointerOperand();
  if (auto *AI = dyn_cast<AllocaInst>(SrcPtr)) {
    Type *AllocationTy = AI->getAllocatedType();
    // If the allocation type was already updated, we will not match here.
    if (AllocationTy == CurTy)
      AI->setAllocatedType(AltTy);
  }
}

static bool updateInputPtrForRealloc(
    Value *SrcPtr, Type *AltTy, GetElementPtrInst *SrcGEP,
    DenseMap<Type *, SmallVector<Value *>> &GEPTypeToIndices) {
  bool IsSameGEP = false;
  if (auto *CurCB = dyn_cast<CallBase>(SrcPtr)) {
    Type *SrcTy = SrcGEP->getSourceElementType();
    updateAddressWithGlossary(GEPTypeToIndices, SrcGEP, SrcTy, AltTy);
    Value *InputPtr = CurCB->getArgOperand(0);
    SmallVector<const Value *, 4> Objects;
    getUnderlyingObjects(InputPtr, Objects);
    for (const Value *UnderlyingObj : Objects)
      if (const auto *LdBase = dyn_cast<LoadInst>(UnderlyingObj)) {
        Value *LdBasePtr = const_cast<Value *>(LdBase->getPointerOperand());
        if (auto *CurGEP = dyn_cast<GetElementPtrInst>(LdBasePtr)) {
          Type *CurTy = CurGEP->getSourceElementType();
          updateAddressWithGlossary(GEPTypeToIndices, CurGEP, CurTy, AltTy);
          IsSameGEP = compareEqualGEPs(CurGEP, SrcGEP);
        }
      }

    // Now walk all the uses of SrcPtr and assert on any loads for
    // missing functionality on split processing as we need to know
    // if the realloc memory is used in this context.
    for (User *U : SrcPtr->users())
      if (isa<LoadInst>(U))
        assert(0 && "Unsupported context that requires address splitting");
  }
  return IsSameGEP;
}

static GetElementPtrInst *
updateAddressForAllocations(IRBuilder<> &IRB, SmallVectorImpl<Value *> &Indices,
                            GetElementPtrInst *CurGEP,
                            const uint64_t ArrayTyIdx, Value *SrcPtr,
                            Type *AltTy, StructType *ArrayST) {
  // Update the address to the specified field.
  Value *NewAddr = nullptr;
  unsigned N = ArrayST->getNumElements();
  if (ArrayTyIdx < N) {
    NewAddr = IRB.CreateInBoundsGEP(AltTy, SrcPtr, Indices);
    CurGEP->replaceAllUsesWith(NewAddr);
    CurGEP->eraseFromParent();
  }
  return dyn_cast<GetElementPtrInst>(NewAddr);
}

static Value *findOrCreateBaseGEP(GetElementPtrInst *SrcGEP, Type *AltTy,
                                  const SmallVectorImpl<Value *> &Indices,
                                  Instruction *RefInst) {
  Value *BasePtr = SrcGEP->getPointerOperand();
  auto FindBaseGEP = [&](Value *BasePtr) -> GetElementPtrInst * {
    for (Instruction &I : *SrcGEP->getParent()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        if (matchBaseGEP(GEP, AltTy, BasePtr, Indices))
          return GEP;
      }
      if (&I == RefInst)
        break;
    }
    return nullptr;
  };

  IRBuilder<> IRB2(SrcGEP);
  auto *GEP = FindBaseGEP(BasePtr);
  return (GEP == nullptr) ?
      IRB2.CreateInBoundsGEP(AltTy, BasePtr, Indices) : GEP;
}

static Value *findOrCreateBaseLoad(Value *BaseAddr, Instruction *InsertPt) {
  auto FindBaseLoad = [&](Value *BaseAddr) -> LoadInst * {
    for (Instruction &I : *InsertPt->getParent()) {
      if (auto *Ld = dyn_cast<LoadInst>(&I)) {
        auto *LdPtr = Ld->getPointerOperand();
        if (auto *GEP = dyn_cast<GetElementPtrInst>(LdPtr)) {
          if (GEP == BaseAddr)
            return Ld;
        }
      }
      if (&I == InsertPt)
        break;
    }
    return nullptr;
  };

  IRBuilder<> IRB2(InsertPt);
  auto *Ld = FindBaseLoad(BaseAddr);
  return (Ld == nullptr) ?
      IRB2.CreateLoad(BaseAddr->getType(), BaseAddr) : Ld;
}

static void
splitAddressStoresForFree(Value *SrcPtr, Type *AltTy, StructType *ArrayST,
                          const TargetLibraryInfo &TLI, const DataLayout &DL,
                          SmallPtrSetImpl<StoreInst *> &VisitedStores,
                          SmallPtrSetImpl<StoreInst *> &AddedStores,
                          const uint64_t StartingOffset,
                          GetElementPtrInst *CurGEP) {
  Function *F = CurGEP->getParent()->getParent();
  auto *CurCB = dyn_cast<CallBase>(SrcPtr);
  if (!CurCB)
    return;

  // We process this free if it has a null store
  StoreInst *SI = nullptr;
  Value *NullVal = nullptr;
  for (User *U : CurGEP->users()) {
    SI = dyn_cast<StoreInst>(U);
    if (SI) {
      NullVal = SI->getValueOperand();
      auto *ConstOp = dyn_cast<Constant>(NullVal);
      if (ConstOp && ConstOp->isNullValue())
        break;

      SI = nullptr;
    }
  }

  // Nothing to do.
  if (!SI)
    return;

  // Ignore stores we added.
  if (AddedStores.contains(SI))
    return;

  SmallVector<StoreInst *, 10> Worklist;
  if (VisitedStores.insert(SI).second)
    Worklist.push_back(SI);

  // Now process all the stores we found
  while (!Worklist.empty()) {
    StoreInst *SI = Worklist.pop_back_val();
    Value *BasePtr = CurGEP->getPointerOperand();
    Instruction *OrigI = cast<Instruction>(SrcPtr);
    AddedStores.insert(SI);
    IRBuilder<> IRB(CurGEP);
    for (unsigned k = 0, e = ArrayST->getNumElements(); k != e; ++k) {
      SmallVector<Value *, 10> Indices;
      // The first offset is zero indicating the start of this EltTy's array.
      Type *FirstTy = DL.getIntPtrType(CurGEP->getType());
      Indices.push_back(ConstantInt::get(FirstTy, 0));
      // The second offset is StartingOffset+k which is a 32bit int.
      Indices.push_back(ConstantInt::get(IRB.getInt32Ty(), StartingOffset + k));
      Instruction *NewI = OrigI;
      if (k > 0) {
        auto *BaseAddr = findOrCreateBaseGEP(CurGEP, AltTy, Indices, OrigI);
        // Clone the allocation call and provide its value to the store.
        NewI = OrigI->clone();
        // Place the cloned call near the old one.
        NewI->insertAfter(CurCB);
        // Create New address/store pairs for all but the first field, see
        // below for details about k == 0 (the first and original call).
        auto *BaseLd = findOrCreateBaseLoad(BaseAddr, OrigI);
        NewI->setOperand(0, BaseLd);
        IRB.SetInsertPoint(SI);
        auto *Store = IRB.CreateStore(NullVal, BaseAddr);
        AddedStores.insert(Store);
      } else {
        // Check if we have an orthogonal formed CurGEP and replace it if
        // we do, else just use the one we have.  The orthogonal formed GEPs
        // are never candidates.  These will only have type i8 and are
        // produced in places like argument promotion.  Since we already know
        // that this GEP is used in store ptr context and is connected to one
        // our candidates by ArrayTy association ergo this action is always
        // safe.
        Type *DerivedTy = CurGEP->getSourceElementType();
        if (DerivedTy == Type::getInt8Ty(F->getContext()))
          updateAddressForAllocations(IRB, Indices, CurGEP, k, BasePtr,
                                      AltTy, ArrayST);
      }
    }
  }
}

static unsigned configureBytes(const LibFunc TLIFn, Value *SizeVal, unsigned SizeInBytes) {
  if (TLIFn == LibFunc_calloc) {
    if (auto *SizeValC = dyn_cast<ConstantInt>(SizeVal)) {
      APInt ValA = SizeValC->getValue();
      return (ValA.getZExtValue() * SizeInBytes);
    }
  } else if (auto *SizeValC = dyn_cast<ConstantInt>(SizeVal)) {
    APInt ValA = SizeValC->getValue();
    return ValA.getZExtValue();
  }
  return SizeInBytes;
}

static void splitAddressStoresForAllocations(
    Value *SrcPtr, Type *AltTy, StructType *ArrayST,
    const TargetLibraryInfo &TLI, const DataLayout &DL,
    SmallPtrSetImpl<StoreInst *> &VisitedStores,
    SmallPtrSetImpl<StoreInst *> &AddedStores, const uint64_t StartingOffset,
    Instruction *RefInst, bool ReallocReuseGEPs) {
  Function *F = RefInst->getParent()->getParent();
  auto *CurCB = dyn_cast<CallBase>(SrcPtr);
  if (!CurCB)
    return;

  LibFunc TLIFn;
  TLI.getLibFunc(*CurCB->getCalledFunction(), TLIFn);
  // Only process ptr stores of allocations
  unsigned SizeArg;
  unsigned SizeOfArg;
  switch (TLIFn) {
  case LibFunc_malloc:
    SizeArg = SizeOfArg = 0;
    break;
  case LibFunc_calloc:
    SizeArg = 0;
    SizeOfArg = 1;
    break;
  case LibFunc_realloc:
    SizeArg = SizeOfArg = 1;
    break;
  default:
    return;
  }

  // Since we are going to add users to SrcPtr,
  // we need to build a Worklist to process
  // all the relevant uses we find.
  SmallVector<StoreInst *, 10> Worklist;
  for (User *U : SrcPtr->users()) {
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      // Ignore stores we added.
      if (AddedStores.contains(SI))
        continue;

      // We operate on stores of ptr values
      if (SI->getValueOperand() != SrcPtr)
        continue;

      // Only related stores with GEPs are processed.
      Value *Ptr = SI->getPointerOperand();
      if (isa<GetElementPtrInst>(Ptr)) {
        if (VisitedStores.insert(SI).second) {
          Worklist.push_back(SI);
        }
      }
    }
  }

  // Now process all the stores we found
  while (!Worklist.empty()) {
    StoreInst *SI = Worklist.pop_back_val();
    IRBuilder<> IRB(SI);
    Value *Ptr = SI->getPointerOperand();
    auto *DerivedGEP = cast<GetElementPtrInst>(Ptr);
    Value *BasePtr = DerivedGEP->getPointerOperand();
    Instruction *OrigI = cast<Instruction>(SrcPtr);
    Value *SizeVal = OrigI->getOperand(SizeArg);
    AddedStores.insert(SI);
    for (unsigned k = 0, e = ArrayST->getNumElements(); k != e; ++k) {
      // Now obtain the size of the array from the call:
      // * For calloc: The array size is the first arg
      // * For malloc: We divide the first arg by SizeInBytes to obtain
      //               size.
      SmallVector<Value *, 10> Indices;
      // The first offset is zero indicating the start of this EltTy's array.
      Type *FirstTy = DL.getIntPtrType(SrcPtr->getType());
      Indices.push_back(ConstantInt::get(FirstTy, 0));
      // The second offset is StartingOffset+k which is a 32bit int.
      Indices.push_back(ConstantInt::get(IRB.getInt32Ty(), StartingOffset + k));
      // The address we create is the stored value, the GEP is where
      // we write that memory at since this store is writing a ptr value.
      Type *EltTy = ArrayST->getElementType(k);
      unsigned SizeInBytes = DL.getTypeSizeInBits(EltTy).getFixedSize() >> 3;
      Value *NewSizeArg = createNewSizeArg(ArrayST, SrcPtr, OrigI, TLIFn, DL,
                                           SizeVal, SizeInBytes);

      Instruction *NewI = OrigI;
      // Update allocation call attributes as needed.
      unsigned Bytes = configureBytes(TLIFn, NewSizeArg, SizeInBytes);
      updateDereferenceableAttributes(NewI, Bytes);
      if (k > 0) {
        auto *BaseAddr = findOrCreateBaseGEP(DerivedGEP, AltTy, Indices, OrigI);
        // Clone the allocation call and provide its value to the store.
        NewI = OrigI->clone();
        // Place the cloned call near the old one.
        NewI->insertAfter(CurCB);
        // Create New address/store pairs for all but the first field, see
        // below for details about k == 0 (the first and original call).
        // Place the cloned load for BaseAddr in the current block.
        Instruction *InsertPt = OrigI;
        if (CurCB->getParent() == DerivedGEP->getParent())
          InsertPt = DerivedGEP;
        IRB.SetInsertPoint(InsertPt);
        // Can realloc use the same address as the store?
        if (ReallocReuseGEPs) {
          // First stage a load off of BaseAddr, then use the result as a ptr.
          auto *NewPtr = findOrCreateBaseLoad(BaseAddr, InsertPt);
          NewI->setOperand(0, NewPtr);
        }
        IRB.SetInsertPoint(SI);
        auto *Store = IRB.CreateStore(NewI, BaseAddr);
        AddedStores.insert(Store);
        // Now process all the GEP uses of the original allocation function and
        // replace its ptr via type matching so that its usage model is correct.
        SmallVector<GetElementPtrInst *, 4> GEPWorklist;
        for (User *U : OrigI->users()) {
          if (auto *UserGEP = dyn_cast<GetElementPtrInst>(U))
            if (EltTy == UserGEP->getResultElementType())
              if (UserGEP->getPointerOperand() != NewI)
                UserGEP->setOperand(0, NewI);
        }
      } else {
        // Check if we have an orthogonal formed DerivedGEP and replace it if
        // we do, else just use the one we have.  The orthogonal formed GEPs
        // are never candidates.  These will only have type i8 and are
        // produced in places like argument promotion.  Since we already know
        // that this GEP is used in store ptr context and is connected to one
        // our candidates by ArrayTy association ergo this action is always
        // safe.
        Type *DerivedTy = DerivedGEP->getSourceElementType();
        if (DerivedTy == Type::getInt8Ty(F->getContext()))
          DerivedGEP = updateAddressForAllocations(IRB, Indices, DerivedGEP, k,
                                                   BasePtr, AltTy, ArrayST);
      }
      // Update the allocation size to reflect this fields EltTy size.
      updateAllocationSize(IRB, TLIFn, NewI, NewSizeArg, SizeInBytes, SizeArg,
                           SizeOfArg);
    }
  }
}

static void
updateMemSetCall(Instruction *RefInst, Type *RefTy, Type *AltTy, bool IsBaseTy,
                 SmallPtrSetImpl<GetElementPtrInst *> &VisitedAddresses,
                 DenseMap<Type *, SmallVector<Value *>> &GEPTypeToIndices,
                 const DataLayout &DL) {
  auto *CurCB = cast<CallBase>(RefInst);
  // The first argument is a pointer to the destination to fill
  Value *InputPtr = CurCB->getArgOperand(0);
  auto *CurGEP = dyn_cast<GetElementPtrInst>(InputPtr);
  if (!CurGEP)
    return;

  if (IsBaseTy)
    return;

  Type *CurTy = CurGEP->getSourceElementType();
  if (CurTy != RefTy)
    return;

  // Memset will only be processed on ArrayTy fields wrt references,
  // so discover the necessary info to replace CurGEP.
  GetElementPtrInst *SrcGEP = nullptr;
  // Find a base pointer for this ArrayTy reference
  SmallVector<const Value *, 4> Objects;
  Value *Ptr = CurGEP->getPointerOperand();
  getUnderlyingObjects(Ptr, Objects);
  for (const Value *UnderlyingObj : Objects) {
    if (const auto *Ld = dyn_cast<LoadInst>(UnderlyingObj)) {
      auto *LdPtr = Ld->getPointerOperand();
      if (const auto *BaseGEP = dyn_cast<GetElementPtrInst>(LdPtr)) {
        SrcGEP = const_cast<GetElementPtrInst *>(BaseGEP);
        break;
      }
    }
  }
  if (!SrcGEP)
    return;

  if (VisitedAddresses.insert(CurGEP).second) {
    Value *BasePtr = SrcGEP->getPointerOperand();
    IRBuilder<> IRB(CurGEP);
    auto *ArrayST = cast<StructType>(RefTy);
    Value *SizeArg = RefInst->getOperand(2);
    Instruction *NewI = RefInst;
    SmallVector<Value *, 10> Indices;
    bool UpdateSrcGEP;
    bool HasIV = false;
    uint64_t ArrayTyIdx;
    uint64_t StartingOffset = gatherIndicesForArrayAddress(
        Indices, IRB, ArrayST, AltTy, UpdateSrcGEP, HasIV, ArrayTyIdx,
        CurGEP);
    // First we update the SrcGEP to use as a base pointer for which we
    // will indirect off of via indexing.
    Type *SrcTy = SrcGEP->getSourceElementType();
    updateAddressWithGlossary(GEPTypeToIndices, SrcGEP, SrcTy, AltTy);
    // Now iterate over the subfields of ArrayST, using the first
    // call and updating its ptr and length, then cloning the rest
    // with an appropriate ptr and length.
    for (unsigned k = 0, e = ArrayST->getNumElements(); k != e; ++k) {
      SmallVector<Value *, 10> BaseIndices;
      // The first offset is zero indicating the start of this EltTy's
      // array.
      Type *FirstTy = DL.getIntPtrType(CurGEP->getType());
      Value *Addr = nullptr;
      Type *EltTy = ArrayST->getElementType(k);
      unsigned SizeInBytes =
          DL.getTypeSizeInBits(EltTy).getFixedSize() >> 3;
      if (k > 0) {
        BaseIndices.push_back(ConstantInt::get(FirstTy, 0));
        // The second offset is StartingOffset+k which is a 32bit int.
        BaseIndices.push_back(
            ConstantInt::get(IRB.getInt32Ty(), StartingOffset + k));
        Value *BaseAddr = nullptr;
        Instruction *InsertPt = nullptr;
        if (const auto *Ld = dyn_cast<LoadInst>(Ptr)) {
          if (SrcGEP == Ld->getPointerOperand()) {
            BaseAddr = findOrCreateBaseGEP(SrcGEP, AltTy, BaseIndices, RefInst);
            InsertPt = const_cast<LoadInst *>(Ld);
          }
        }
        // Clone the memset and provide its value to the store.
        NewI = RefInst->clone();
        // Place the cloned call near the old one.
        NewI->insertAfter(CurCB);
        // If no BaseAddr is provided, place it locally.
        if (BaseAddr == nullptr) {
          BaseAddr = IRB.CreateInBoundsGEP(AltTy, BasePtr, BaseIndices);
          InsertPt = RefInst;
        }
        auto *BaseLd = findOrCreateBaseLoad(BaseAddr, InsertPt);
        Addr = IRB.CreateInBoundsGEP(EltTy, BaseLd, Indices[0]);
      } else {
        Addr = IRB.CreateInBoundsGEP(EltTy, Ptr, Indices[0]);
      }
      NewI->setOperand(0, Addr);
      NewI->setOperand(2,
                       ConstantInt::get(SizeArg->getType(), SizeInBytes));
      updateDereferenceableAttributes(NewI, SizeInBytes);
    }
  }
}

static bool isFreeCall(const Value *V, const TargetLibraryInfo *TLI) {
  if (auto *CI = dyn_cast<CallInst>(V)) {
    LibFunc TLIFn;
    TLI->getLibFunc(*CI, TLIFn);
    return (TLIFn == LibFunc_free);
  }
  return false;
}

static bool isMallocOrCallocFn(const Value *V, const TargetLibraryInfo *TLI) {
  if (auto *CI = dyn_cast<CallInst>(V)) {
    LibFunc TLIFn;
    TLI->getLibFunc(*CI, TLIFn);
    return ((TLIFn == LibFunc_calloc) || (TLIFn == LibFunc_malloc));
  }
  return false;
}

static bool isReallocFn(const Value *V, const TargetLibraryInfo *TLI) {
  if (auto *CI = dyn_cast<CallInst>(V)) {
    LibFunc TLIFn;
    TLI->getLibFunc(*CI, TLIFn);
    return (TLIFn == LibFunc_realloc);
  }
  return false;
}

static void updateCalledFunction(
    CallInst *CI, Instruction *RefInst, Type *RefTy, Type *AltTy,
    const TargetLibraryInfo &TLI, bool IsBaseTy,
    SmallPtrSetImpl<GetElementPtrInst *> &VisitedAddresses,
    DenseMap<Type *, SmallVector<Value *>> &GEPTypeToIndices,
    const SmallDenseSet<std::pair<Type *, Type *>> &TranslatedTypeSet,
    LLVMContext &C, const DataLayout &DL) {
  bool AllowUserFunctionUpdate = true;
  // We can ignore some calls - really what we want to process is user calls
  if (isAllocationFn(RefInst, &TLI) || RefInst->isLifetimeStartOrEnd() ||
      isFreeCall(RefInst, &TLI)) {
    AllowUserFunctionUpdate = false;
  } else if (isa<MemSetInst>(RefInst)) {
    updateMemSetCall(RefInst, RefTy, AltTy, IsBaseTy, VisitedAddresses,
                     GEPTypeToIndices, DL);
    AllowUserFunctionUpdate = false;
  } else if (isa<MemCpyInst>(RefInst)) {
    // Here we only update the size arg(3rd arg) in llvm.memcpy.
    Value *SizeArg = RefInst->getOperand(2);
    auto *AltST = cast<StructType>(AltTy);
    unsigned SizeInBytes = calculateStructSizeInBytes(AltST, DL);
    RefInst->setOperand(
        2, ConstantInt::get(SizeArg->getType(), SizeInBytes));
    updateDereferenceableAttributes(RefInst, SizeInBytes);
    AllowUserFunctionUpdate = false;
  }
  // look for side effects and update as needed.
  for (Use &Op : RefInst->operands()) {
    if (auto *AI = dyn_cast<AllocaInst>(Op)) {
      Type *CurTy = AI->getAllocatedType();
      if (CurTy == RefTy)
        AI->setAllocatedType(AltTy);
    }
  }
  if (AllowUserFunctionUpdate) {
    AttributeList Attrs = CI->getAttributes();
    // Update this calls attributes if any reference types match
    updateParamTypeAttributes(TranslatedTypeSet, Attrs, C);
    // TODO: consider adding support for updating dereferenceable attributes for
    // user calls.
    CI->setAttributes(Attrs);
  }
}

static void handleArrayOfStructuresAddressTranslation(IRBuilder<> &IRB,
    SmallPtrSetImpl<GetElementPtrInst *> &VisitedAddresses, StructType *ArrayST,
    SmallVectorImpl<Value *> &Indices, GetElementPtrInst *CurGEP, Value *SrcPtr,
    const DataLayout &DL, Type *AltTy, const TargetLibraryInfo &TLI,
    bool UpdateSrcGEP, uint64_t ArrayTyIdx, uint64_t StartingOffset) {
  if (VisitedAddresses.insert(CurGEP).second) {
    Type *ArrayTy = ArrayST->getElementType(ArrayTyIdx);
    Value *Addr = nullptr;
    if (ArrayTyIdx == 0) {
      if (UpdateSrcGEP)
        Addr = IRB.CreateInBoundsGEP(ArrayTy, SrcPtr, Indices);
      else
        Addr = IRB.CreateInBoundsGEP(ArrayTy, SrcPtr, Indices[0]);
    } else if (isAllocationFn(SrcPtr, &TLI)) {
      Addr = IRB.CreateInBoundsGEP(ArrayTy, SrcPtr, Indices[0]);
    } else {
      SmallVector<Value *, 4> BaseIndices;
      Type *FirstTy = DL.getIntPtrType(SrcPtr->getType());
      BaseIndices.push_back(ConstantInt::get(FirstTy, 0));
      // The second offset is StartingOffset+k which is a 32bit int.
      BaseIndices.push_back(
          ConstantInt::get(IRB.getInt32Ty(), StartingOffset + ArrayTyIdx));
      Value *BaseAddr = nullptr;
      Instruction *InsertPt = nullptr;
      if (auto *Ld = dyn_cast<LoadInst>(CurGEP->getPointerOperand())) {
        auto *LdPtr = Ld->getPointerOperand();
        if (auto *SrcGEP = dyn_cast<GetElementPtrInst>(LdPtr)) {
          if (SrcPtr == SrcGEP->getPointerOperand()) {
            BaseAddr = findOrCreateBaseGEP(SrcGEP, AltTy, BaseIndices, CurGEP);
            InsertPt = Ld;
          }
        }
      }
      // If no BaseAddr is provided, place it locally.
      if (BaseAddr == nullptr) {
        BaseAddr = IRB.CreateInBoundsGEP(AltTy, SrcPtr, BaseIndices);
        InsertPt = CurGEP;
      }
      auto *BaseLd = findOrCreateBaseLoad(BaseAddr, InsertPt);
      Addr = IRB.CreateInBoundsGEP(ArrayTy, BaseLd, Indices[0]);
    }
    CurGEP->replaceAllUsesWith(Addr);
  }
}

static void processAddressForFreeCalls(
    Type *AltTy, StructType *ArrayST, Type *RefTy, const TargetLibraryInfo &TLI,
    const DataLayout &DL, SmallPtrSetImpl<StoreInst *> &VisitedStores,
    SmallPtrSetImpl<StoreInst *> &AddedStores, GetElementPtrInst *CurGEP) {
  // Look for canonical GEPs
  uint64_t StartingOffset = cast<StructType>(RefTy)->getNumElements();
  unsigned Idx = CurGEP->getNumOperands();
  Value *LastIdx = CurGEP->getOperand(Idx - 1);
  ConstantInt *Offset = dyn_cast<ConstantInt>(LastIdx);
  if (!Offset)
    return;

  // Only process replaced GEPs
  if (Offset->getZExtValue() != StartingOffset)
    return;

  // Now find all the load ptrs of this GEP,
  // regardless of how we got here.
  SmallVector<LoadInst *, 10> Worklist;
  for (User *U : CurGEP->users()) {
    if (auto *CurLd = dyn_cast<LoadInst>(U))
      Worklist.push_back(CurLd);
  }

  while (!Worklist.empty()) {
    LoadInst *CurLd = Worklist.pop_back_val();
    for (User *U : CurLd->users())
      if (isFreeCall(U, &TLI)) {
        // We will split addresses as needed in the new container
        // type field access for address stores. Use the free input
        // ptr and update its GEP so that we have the correct
        // address.
        splitAddressStoresForFree(U, AltTy, ArrayST, TLI, DL, VisitedStores,
                                  AddedStores, StartingOffset, CurGEP);
      }
  }
}

static void doActionsForMatchedType(
    Instruction *RefInst, Type *RefTy, Type *AltTy, StructType *ArrayST,
    bool IsBaseTy,
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap,
    const DataLayout &DL, const TargetLibraryInfo &TLI,
    SmallPtrSetImpl<GetElementPtrInst *> &VisitedAddresses,
    SmallPtrSetImpl<StoreInst *> &VisitedStores,
    SmallPtrSetImpl<StoreInst *> &AddedStores,
    const SmallDenseSet<std::pair<Type *, Type *>> &TranslatedTypeSet,
    DenseMap<Type *, SmallVector<Value *>> &GEPTypeToIndices) {
  auto *Ld = dyn_cast<LoadInst>(RefInst);
  auto *St = dyn_cast<StoreInst>(RefInst);
  if (Ld || St) {
    auto *Ptr = (Ld) ? Ld->getPointerOperand() : St->getPointerOperand();
    if (auto *CurGEP = dyn_cast<GetElementPtrInst>(Ptr)) {
      Type *CurTy = CurGEP->getSourceElementType();
      if (IsBaseTy && (CurTy == RefTy)) {
        updateAddressWithGlossary(GEPTypeToIndices, CurGEP, CurTy, AltTy);
        processAddressForFreeCalls(AltTy, ArrayST, RefTy, TLI, DL,
                                   VisitedStores, AddedStores, CurGEP);
      } else if (!IsBaseTy && (CurTy == RefTy)) {
        SmallVector<Value *, 4> Indices;
        bool UpdateSrcGEP = false;
        bool HasIV = false;
        uint64_t ArrayTyIdx;
        IRBuilder<> Builder(CurGEP);
        uint64_t StartingOffset = gatherIndicesForArrayAddress(
            Indices, Builder, ArrayST, AltTy, UpdateSrcGEP, HasIV, ArrayTyIdx,
            CurGEP);

        // Find the matching candidate to obtain its SrcGEP info.
        GetElementPtrInst *SrcGEP = nullptr;
        for (auto &Candididates : LocalCandidateMap) {
          GetElementPtrInst *GEP = Candididates.first.first;
          if (GEP == CurGEP) {
            SrcGEP = Candididates.first.second;
            break;
          }
        }
        // Find our base pointer, which could be a GEP or an allocation.
        if (!SrcGEP) {
          Value *SrcPtr = CurGEP->getPointerOperand();
          if (auto *LdBase = dyn_cast<LoadInst>(SrcPtr)) {
            Value *LdBasePtr = LdBase->getPointerOperand();
            SrcGEP = dyn_cast<GetElementPtrInst>(LdBasePtr);
          } else if (St && isMallocOrCallocFn(SrcPtr, &TLI)) {
            // We split allocation calls as needed for
            // the new field arrays.
            splitAddressStoresForAllocations(
                SrcPtr, AltTy, ArrayST, TLI, DL, VisitedStores, AddedStores,
                StartingOffset, RefInst, /* ReallocReuseGEPs */ false);
            // We may have updated this GEP, so re-fetch its ptr.
            SrcPtr = CurGEP->getPointerOperand();
            handleArrayOfStructuresAddressTranslation(
                Builder, VisitedAddresses, ArrayST, Indices, CurGEP, SrcPtr, DL,
                AltTy, TLI, UpdateSrcGEP, ArrayTyIdx, StartingOffset);
            return;
          } else if (isa<PHINode>(SrcPtr)) {
            SmallVector<const Value *, 4> Objects;
            // To find a PHI base ptr, we need to converge
            // on the same or similar GEPs.
            getUnderlyingObjects(SrcPtr, Objects);
            for (const Value *UnderlyingObj : Objects) {
              if (const auto *LdBase = dyn_cast<LoadInst>(UnderlyingObj)) {
                Value *LdBasePtr =
                    const_cast<Value *>(LdBase->getPointerOperand());
                auto *LdBaseGEP = dyn_cast<GetElementPtrInst>(LdBasePtr);
                if (!LdBaseGEP)
                  continue;
                if (SrcGEP == nullptr)
                  SrcGEP = LdBaseGEP;
                else if ((SrcGEP == LdBaseGEP) ||
                         (compareEqualGEPs(SrcGEP, LdBaseGEP)))
                  continue;
                else
                  assert(0 && "non matching src pointers for address mapping");
              }
            }
          } else if (Ld && isMallocOrCallocFn(SrcPtr, &TLI)) {
            // defer calloc/malloc context to the required store for the allocation.
            return;
          } else {
            assert(0 && "unsupported case of SoA translation of GEP");
          }
        }

        if (!SrcGEP)
          return;

        // First walk the uses SrcGEP looking for Stores, checking the value
        // operand to see if a realloc defined it.
        for (User *U : SrcGEP->users())
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            // Skip stores we added for allocations
            if (AddedStores.contains(SI))
              continue;

            Value *V = SI->getValueOperand();
            if (isReallocFn(V, &TLI)) {
              bool ReallocReuseGEPs =
                  updateInputPtrForRealloc(V, AltTy, SrcGEP, GEPTypeToIndices);
              // We will split addresses as needed in the new container type
              // field access for address stores. Use the realloc input ptr and
              // update its GEP so that we have the correct address for each field.
              splitAddressStoresForAllocations(
                  V, AltTy, ArrayST, TLI, DL, VisitedStores, AddedStores,
                  StartingOffset, RefInst, ReallocReuseGEPs);
            }
          }

        // Update non standard SrcGEPs that cannot be matched.
        Value *BasePtr = SrcGEP->getPointerOperand();
        if (UpdateSrcGEP) {
          BasePtr = CurGEP->getPointerOperand();
          // Check and update the SrcGEP with AltTy blanketly.
          SrcGEP->setSourceElementType(AltTy);
          // Then update the last index with StartingOffset, the index we
          // ignored before indicates we are writing to the base of the array.
          unsigned Idx = SrcGEP->getNumOperands();
          SrcGEP->setOperand(
              Idx - 1, ConstantInt::get(Builder.getInt32Ty(), StartingOffset));
          // Now fetch the interior type to use with CurGEPs replacement.
          AltTy = ArrayST->getElementType(ArrayTyIdx);
        }
        if (ArrayTyIdx == 0)
          BasePtr = CurGEP->getPointerOperand();

        handleArrayOfStructuresAddressTranslation(
            Builder, VisitedAddresses, ArrayST, Indices, CurGEP, BasePtr, DL,
            AltTy, TLI, UpdateSrcGEP, ArrayTyIdx, StartingOffset);
      }
    }
  } else if (auto *CI = dyn_cast<CallInst>(RefInst)) {
    Function *F = RefInst->getParent()->getParent();
    updateCalledFunction(CI, RefInst, RefTy, AltTy, TLI, IsBaseTy,
                         VisitedAddresses, GEPTypeToIndices, TranslatedTypeSet,
                         F->getContext(), DL);
  }
}

/// translateReferences - Update AoS to SoA structs, member parameters,
/// and/or data member uses as transformations to SoA instances.
static bool translateReferences(
    Function *F, function_ref<AAResults &(Function &F)> AARGetter,
    unsigned MaxElements,
    Optional<function_ref<void(CallBase &OldCS, CallBase &NewCS)>>
        ReplaceCallSite,
    const TargetTransformInfo &TTI, const TargetLibraryInfo &TLI,
    SmallDenseMap<std::pair<Instruction *, Type *>, int>
        &LocalReferenceMap,
    const SmallVector<Type *> &LocalParamMap,
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap,
    const SmallDenseSet<std::pair<Type *, Type *>> &UniqueTypeSet,
    const SmallDenseSet<std::pair<Type *, Type *>> &TranslatedTypeSet,
    DenseMap<Type *, SmallVector<Value *>> &GEPTypeToIndices) {
  // With opaque pointers, all the input args we want to replace are ptr args,
  // ergo we should be able to re-interpret these pointers and replace their
  // context at each usage without creating a replacement function. We also
  // need to replace type usage for each call graph edge out of F as well.
  // Processing order does not matter now as we have all the information
  // needed to replace AoS references and SoA and its support code.
  if (updateFunctionAttributes(F, TranslatedTypeSet)) {
    LLVM_DEBUG(dbgs() << "Function: " << F->getName()
                      << "has argument side effects\n");
  }
  SmallPtrSet<GetElementPtrInst *, 8> VisitedAddresses;
  SmallPtrSet<StoreInst *, 8> VisitedStores;
  SmallPtrSet<StoreInst *, 8> AddedStores;
  DataLayout DL = F->getParent()->getDataLayout();
  bool HaveTransformations = false;
  // For each Local Reference there is an entry in UniqueTypeSet and
  // in the TranslatedTypeSet, use these to translate each
  // reference into its new usage.
  for (auto &References : LocalReferenceMap) {
    Instruction *RefInst = References.first.first;
    Type *RefTy = References.first.second;
    // Foreach unique type pair
    for (auto &TypePair : UniqueTypeSet) {
      Type *ArrayTy = TypePair.first;
      auto *ArrayST = cast<StructType>(ArrayTy);
      Type *ContainerTy = TypePair.second;
      // Find the matching replacement type
      for (auto &TypePair : TranslatedTypeSet) {
        Type *OrigTy = TypePair.first;
        Type *ReplacmentTy = TypePair.second;
        // Recall, we appended the elements
        // of ArrayTy into ReplacmentTy as arrays
        // of each local type. The ReplacmentTy looks
        // just like OrigTy up to the point where
        // the new fields are added and is used
        // identically except for the reference to
        // ArrayTy, which will become unused.
        if (OrigTy == ContainerTy) {
          if (RefTy == ContainerTy) {
            ++NumTransformed;
            HaveTransformations = true;
            doActionsForMatchedType(RefInst, ContainerTy, ReplacmentTy, ArrayST,
                                    /* IsBaseTy */ true, LocalCandidateMap, DL,
                                    TLI, VisitedAddresses, VisitedStores,
                                    AddedStores, TranslatedTypeSet,
                                    GEPTypeToIndices);
            break;
          } else if (RefTy == ArrayTy) {
            ++NumTransformed;
            HaveTransformations = true;
            doActionsForMatchedType(RefInst, ArrayTy, ReplacmentTy, ArrayST,
                                    /*IsBaseTy */ false, LocalCandidateMap, DL,
                                    TLI, VisitedAddresses, VisitedStores,
                                    AddedStores, TranslatedTypeSet,
                                    GEPTypeToIndices);
            break;
          }
        }
      }
    }
  }
  // Now cleanup GEPs we processed and orphaned
  for (auto *CurGEP : VisitedAddresses) {
    if (CurGEP->getNumUses() == 0) {
      CurGEP->eraseFromParent();
    } else {
      assert(0 && "CurGEP marked for cleanup still present");
    }
  }

  return HaveTransformations;
}

static bool runOnLoops(
    LoopInfo &LI, ScalarEvolution &SE, unsigned MaxElements,
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap, bool &MustNotProceed) {
  bool FoundOpportunities = false;
  bool IsVectorized = false;
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
                                                    LocalCandidateMap,
                                                    IsVectorized);

        // Override detection if strided vectorization is
        // not enabled when vectors are present.
        if (IsVectorized && !AdhocSkipVectorizeInPrelink) {
          MustNotProceed = true;
          return false;
        }

        if (Result == LoopDataLayoutResult::HasDataLayoutOpportunities)
          FoundOpportunities |= true;
      }
    }
    // Always process L regardless of loop nest context
    auto Result = detectArrayOfStructDataAccess(L, LI, SE, MaxElements,
                                                LocalCandidateMap,
                                                IsVectorized);

    // Override detection if strided vectorization is
    // not enabled when vectors are present.
    if (IsVectorized && !AdhocSkipVectorizeInPrelink) {
      MustNotProceed = true;
      return false;
    }

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

static void examinePhisForReferences(
    SmallDenseMap<std::pair<Instruction *, Type *>, int>
        &LocalReferenceMap,
    Value *MemPtr, Type *ContainerTy, Module *M) {
  // Build a list of objects from the ptr phi to examine
  // the pointer operands for load operations that are input,
  // if their GEPs are listed has having
  // as i8, promote to ContainerTy as the phi ptr
  // was used in a GEP to load the ArrayTy that paired
  // with it.  If opaque pointers are not enabled, we
  // will find the ContainerTy naturally as its GEP will
  // identify it.
  auto *CurTy = MemPtr->getType();
  if (auto *PtrTy = dyn_cast<PointerType>(CurTy))
    if (!PtrTy->isOpaque())
      return;

  SmallVector<const Value *, 4> Objects;
  auto *Int8Ty = Type::getInt8Ty(M->getContext());

  getUnderlyingObjects(MemPtr, Objects);
  for (const Value *UnderlyingObj : Objects)
    if (const auto *Ld = dyn_cast<LoadInst>(UnderlyingObj)) {
      auto *Ptr = Ld->getPointerOperand();
      auto *SrcGEP = dyn_cast<GetElementPtrInst>(Ptr);
      if (!SrcGEP)
        continue;

      Type *BaseTy = SrcGEP->getSourceElementType();
      // If this BaseTy was altered to i8, we can
      // still associate it, as the GEP has been reminted
      // with another way to point to ContainerTy
      // Thought: perhaps we want to translate these
      // GEPs back into a recognizable form.
      if (BaseTy == Int8Ty)
        LocalReferenceMap[{const_cast<LoadInst*>(Ld), ContainerTy}]++;
    }
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
      const Value *CurMemPtr = CurGEP->getPointerOperand();
      if (isa<Argument>(CurMemPtr)) {
        if (any_of(LocalCandidateMap, [=](auto &Candididates) {
              GetElementPtrInst *CandGEP = Candididates.first.second;
              return (CandGEP->getPointerOperand() == CurMemPtr);
            }))
          return true;
      }
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
    SmallDenseMap<std::pair<Instruction *, Type *>, int>
        &LocalReferenceMap,
    SmallVectorImpl<Type *> &LocalParamMap,
    SmallDenseSet<std::pair<Type *, Type *>> &UniqueTypeSet, Function *F) {
  Module *M = F->getParent();
  // Look for references by type if a member of our candidate map.
  bool ReferencesAsArguments = false;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      for (Use &Op : I.operands()) {
        if (isa<Argument>(Op) || isa<Instruction>(Op)) {
          auto *CurTy = Op->getType();
          GetElementPtrInst *CurGEP = nullptr;
          if (auto *PtrTy = dyn_cast<PointerType>(CurTy)) {
            if (auto *CurArg = dyn_cast<Argument>(Op)) {
              // Obtain overlayed type from param map
              CurTy = LocalParamMap[CurArg->getArgNo()];
              if (!CurTy && !PtrTy->isOpaque())
                CurTy = PtrTy->getNonOpaquePointerElementType();

              if (!CurTy)
                continue;

              if (isa<PointerType>(CurTy))
                continue;
            } else if (auto *GEP = dyn_cast<GetElementPtrInst>(Op)) {
              // Examine GEPs that are opaque
              CurGEP = GEP;
              CurTy = CurGEP->getSourceElementType();
            } else if (auto *AI = dyn_cast<AllocaInst>(Op)) {
              CurTy = AI->getAllocatedType();
            } else if (!PtrTy->isOpaque()) {
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
              if (CurGEP) {
                Value *MemPtr = CurGEP->getPointerOperand();
                if (isa<PHINode>(MemPtr))
                  examinePhisForReferences(LocalReferenceMap, MemPtr,
                                           ContainerTy, M);
              }
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
    SmallVectorImpl<Type *> &LocalParamMap, Function *F, bool ThinLTO) {
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

        if (isa<CastInst>(&I)) {
          // Type Based Escape Analysis: No casts into or out of Ty
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
            } else if (auto *CurArg = dyn_cast<Argument>(InputSrcPtr)) {
              // Check the type to see if this arg diverges from the candidate
              // type (Ty) and if so it does not alias because of type escape
              // analysis.
              Type *ArgTy = InputSrcPtr->getType();
              if (isa<PointerType>(ArgTy)) {
                // Obtain overlayed type from param map
                ArgTy = LocalParamMap[CurArg->getArgNo()];
              }
              if (ArgTy != Ty)
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
            if (auto *CurArg = dyn_cast<Argument>(UnderlyingObj)) {
              Type *ArgTy = UnderlyingObj->getType();
              if (isa<PointerType>(ArgTy)) {
                // Obtain overlayed type from the ParamMap of F.
                ArgTy = LocalParamMap[CurArg->getArgNo()];
              }
              if (ArgTy != Ty)
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
              // Defer casts to type based escape analysis.
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
        if (getReallocatedOperand(cast<CallInst>(Inst), &TLI) == U ||
            Inst->isLifetimeStartOrEnd() || isa<MemCpyInst>(Inst)) {
          NumLegalUses++;
          continue;
        }

      if (legalUseTree(F, Inst, TLI))
        NumLegalUses++;
    }
  }
  return (NumInstUses == NumLegalUses);
}

static Optional<Type *> configureParamType(
    TargetLibraryInfo &TLI, Value *Arg, Function *DCallee, Function *F,
    DenseMap<Function *, SmallVector<Type *>> &ParamMap, unsigned i) {
  SmallVector<Type *> &LocalParamMap = ParamMap[DCallee];
  Type *BaseTy = nullptr;
  if (auto *GV = dyn_cast<GlobalVariable>(Arg)) {
    if (GV->isConstant())
      return None;

    BaseTy = GV->getValueType();
  } else if (auto *CI = dyn_cast<CallInst>(Arg)) {
    if (getReallocatedOperand(CI, &TLI) != nullptr ||
        isMallocOrCallocFn(Arg, &TLI)) {
      // Look at uses of UnderlyingObj for a GEP and obtain
      // the type there as these cases return a pointer
      // and will be used in that context.
      // TODO: handle complex flow cases involving phis.
      for (const User *U : CI->users()) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          BaseTy = GEP->getSourceElementType();
          break;
        } else if (auto *PhiPtr = dyn_cast<PHINode>(U)) {
          for (const User *PhiUse : PhiPtr->users()) {
            if (auto *GEP = dyn_cast<GetElementPtrInst>(PhiUse)) {
              BaseTy = GEP->getSourceElementType();
              break;
            }
          }
          // It is sufficient to stop here as detection will
          // search for type matches from AoS context.
          if (BaseTy)
            break;
        }
      }
    } else {
      // See if this CB was already seen and params added.
      BaseTy = LocalParamMap[i];
    }
  } else if (isa<Constant>(Arg)) {
    return None;
  } else {
    SmallVector<const Value *, 4> Objects;
    getUnderlyingObjects(Arg, Objects);
    for (const Value *UnderlyingObj : Objects) {
      if (const auto *Ld = dyn_cast<LoadInst>(UnderlyingObj)) {
        auto *Ptr = Ld->getPointerOperand();
        auto *SrcGEP = dyn_cast<GetElementPtrInst>(Ptr);
        if (!SrcGEP)
          continue;

        BaseTy = SrcGEP->getSourceElementType();
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(UnderlyingObj)) {
        BaseTy = GEP->getSourceElementType();
      } else if (auto *CurArg = dyn_cast<Argument>(UnderlyingObj)) {
        // Lookup for F in overload map, since we are processing
        // the call graph in dfs order from the CG entry node in
        // main LTO, we should find an entry for F that matches
        // ArgNo.
        SmallVector<Type *> &CurParamMap = ParamMap[F];
        // Obtain overlayed type from param map
        BaseTy = CurParamMap[CurArg->getArgNo()];
      } else if (auto *AI = dyn_cast<AllocaInst>(UnderlyingObj)) {
        BaseTy = AI->getAllocatedType();
      } else if (isa<CallInst>(UnderlyingObj)) {
        // Call returns a ptr, insufficient information to determine type.
        continue;
      } else if (auto *GV = dyn_cast<GlobalVariable>(UnderlyingObj)) {
        if (GV->isConstant())
          continue;

        BaseTy = GV->getValueType();
      } else if (auto *ConstOp = dyn_cast<Constant>(UnderlyingObj)) {
        if (ConstOp->isNullValue())
          continue;
      } else if (isa<CastInst>(UnderlyingObj)) {
        continue;
      } else if (isa<ExtractElementInst>(UnderlyingObj)) {
        continue;
      } else if (isa<ExtractValueInst>(UnderlyingObj)) {
        continue;
      } else if (isa<InvokeInst>(UnderlyingObj)) {
        continue;
      } else {
        llvm_unreachable("Support needed for unhandled cases!");
      }
      if (BaseTy && BaseTy->isStructTy())
        break;
    }
  }

  if (!BaseTy)
    return None;

  return BaseTy;
}

static void processParameterMapping(SmallVector<Type *> &LocalParamMap,
                                    SmallBitVector &LocalInvalidateMap,
                                    Type *BaseTy, unsigned i) {
  if (BaseTy && BaseTy->isStructTy()) {
    Type *ParamTy = nullptr;
    // Three cases exist here:
    // * There are no entrys for Arg(i), so we add one
    // * There is an entry and Arg(i) already has BaseTy
    // * There is an entry and Arg(i) diverges from BaseTy
    // Fetch whatever we have stored in i for LocalParamMap.
    ParamTy = LocalParamMap[i];
    // The case where BaseTy is ParamTy is already included in
    // LocalParamMap as the undocumented else here.
    if (ParamTy && (BaseTy != ParamTy)) {
      // Invalidate Arg(i), we have conflicting types to overlay.
      // We mark an invalid context to avoid a race and add an
      // invalid entry for Arg(i).
      LocalParamMap[i] = nullptr;
      LocalInvalidateMap[i] = true;
    } else if (ParamTy == nullptr) {
      // If there LocalInvalidMap is false for this Arg(i),
      // it is safe to add the current entry to LocalParamMap.
      if (!LocalInvalidateMap[i])
        LocalParamMap[i] = BaseTy;
    }
  }
}

static void walkCallGraphToFillParamMap(
    CallGraph &CG, function_ref<TargetLibraryInfo &(Function &)> LookupTLI,
    DenseMap<Function *, SmallVector<Type *>> &ParamMap,
    DenseMap<Function *, SmallBitVector> &InvalidateMap, bool ThinLTO) {
  // The Root of the CG is the external calling node
  CallGraphNode *EntryNode = CG.getExternalCallingNode();
  SmallVector<CallGraphNode *, 8> CGNStack;
  // The CGNStack cannot use the depth_first iterator as the feature does not
  // properly identify the root of the graph, in our case we want to enter
  // the graph at main.  The same is true of the bfs iterator.  Are these bugs?
  for (const auto &CI : *EntryNode) {
    auto *CGN = CI.second;

    // Ignore call graph node which does not have associated function
    if (!CGN->getFunction() || CGN->getFunction()->isDeclaration())
      continue;

    if (ThinLTO) {
      CGNStack.push_back(CGN);
    } else {
      // For full LTO, main is always visible in the CG.
      if (CGN->getFunction()->getName() == "main") {
        CGNStack.push_back(CGN);

        break;
      }
    }
  }
  SmallPtrSet<CallGraphNode *, 8> VisitedCGNodes;
  while (!CGNStack.empty()) {
    auto *Node = CGNStack.pop_back_val();
    Function *F = Node->getFunction();
    TargetLibraryInfo &TLI = LookupTLI(*F);
    // Vist all the Edges of F in the CG
    for (const auto &GI : *Node) {
      auto *CurCB = cast<CallBase>(GI.first.value());
      FunctionType *FTy = CurCB->getFunctionType();
      auto *CurCGN = GI.second;

      // Ignore call graph node(s) which does not have an associated function.
      auto *DCallee = CurCGN->getFunction();
      if (!DCallee || DCallee->isDeclaration())
        continue;

      // Ignore recursive context.
      if (DCallee == F)
        continue;

      // Add any unseen CG Nodes to the stack.
      if (VisitedCGNodes.insert(CurCGN).second)
        CGNStack.push_back(CurCGN);

      // Look for args that have types of structs and add to LocalParamMap.
      SmallVector<Type *> &LocalParamMap = ParamMap[DCallee];
      SmallBitVector &LocalInvalidateMap = InvalidateMap[DCallee];
      for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i) {
        Type *ArgTy = FTy->getParamType(i);
        if (auto *PtrTy = dyn_cast<PointerType>(ArgTy)) {
          Type *BaseTy = nullptr;
          Value *Arg = CurCB->getArgOperand(i);
          Optional<Type *> OptTy =
              configureParamType(TLI, Arg, DCallee, F, ParamMap, i);

          if (OptTy != None)
            BaseTy = *OptTy;

          // Allow for maximally checking type divergence before falling back
          // to legacy pointer type mining.
          if (!BaseTy && !PtrTy->isOpaque())
            BaseTy = PtrTy->getNonOpaquePointerElementType();

          if (!BaseTy)
            continue;

          processParameterMapping(LocalParamMap, LocalInvalidateMap, BaseTy, i);
        }
      }
    }
  }
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

            LLVM_DEBUG(dbgs() << "cast into/out of SrcTy : " << *SrcTy << "\n");
            return true;
          }
          // Now examine input objects to see if the actions are legal.
          // In this case the input arg is divergent as the dst matches.
          if ((DstTy == ArrayTy) || (DstTy == ContainerTy)) {
            SmallVector<const Value *, 4> Objects;

            getUnderlyingObjects(CurCast->getOperand(0), Objects);
            for (const Value *UnderlyingObj : Objects) {
              if (isa<CallInst>(UnderlyingObj)) {
                if (getReallocatedOperand(cast<CallInst>(UnderlyingObj),
                                          &TLI) != nullptr ||
                    isMallocOrCallocFn(UnderlyingObj, &TLI))
                  continue;

                LLVM_DEBUG(dbgs()
                           << "cast into/out of DstTy : " << *DstTy << "\n");
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
    DenseMap<Function *,
             SmallDenseMap<std::pair<Instruction *, Type *>, int>>
        &ReferenceMap,
    DenseMap<Function *, SmallVector<Type *>> &ParamMap,
    DenseMap<Function *, SmallBitVector> &InvalidateMap,
    SmallDenseSet<std::pair<Type *, Type *>> &UniqueTypeSet, CallGraph &MCG) {

  LLVM_DEBUG(
      dbgs() << "Analyzing Loop collection for data layout opportunities: ");

  // First initialize the parameter info for all functions.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    SmallVector<Type *> &LocalParamMap = ParamMap[&F];
    SmallBitVector &LocalInvalidateMap = InvalidateMap[&F];
    // Now initialize with empty data so that each
    // entry for a function is correctly sized.
    // We will access both as indexed arrays.
    LocalParamMap.assign(F.arg_size(), nullptr);
    LocalInvalidateMap.resize(F.arg_size(), false);
  }

  walkCallGraphToFillParamMap(MCG, LookupTLI, ParamMap, InvalidateMap, ThinLTO);

  // During main LTO, all modules have been fused into a single module.
  // The thinLTO interface is for testing purposes only right now.
  bool FoundOpportunities = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    if (F.hasPersonalityFn()) {
      LLVM_DEBUG(dbgs() << "Exceptions detected\n");
      return LoopDataLayoutResult::TransformationIsIllegal;
    }

    bool MustNotProceed = false;
    LoopInfo &LI = LookupLoopInfo(F);
    ScalarEvolution &SE = LookupScalarEvolutionInfo(F);
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap = CandidateMap[&F];
    if (runOnLoops(LI, SE, MaxElements, LocalCandidateMap, MustNotProceed)) {
      FoundOpportunities |= true;
      AAResults &AAR = AARGetter(F);
      SmallVector<Type *> &LocalParamMap = ParamMap[&F];
      // AA and other legality analysis of candidates to transform.
      if (!runOnFunction(LI, SE, AAR, MaxElements, LocalCandidateMap,
                         LocalParamMap, &F, ThinLTO))
        return LoopDataLayoutResult::TransformationIsIllegal;

      // For all local candidate maps, fill in a unique type map of SrcGEP/GEP
      // types to use for locating references in functions.
      for (auto &Candididates : LocalCandidateMap)
        UniqueTypeSet.insert(
            {Candididates.first.first->getSourceElementType(),
             Candididates.first.second->getSourceElementType()});
    } else if (MustNotProceed) {
      return LoopDataLayoutResult::TransformationIsIllegal;
    }
  }

  if (!FoundOpportunities)
    return LoopDataLayoutResult::HasNoOpportunities;

  // Walk every function and mark the references
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    SmallDenseMap<std::pair<Instruction *, Type *>, int>
        &LocalReferenceMap = ReferenceMap[&F];
    SmallVector<Type *> &LocalParamMap = ParamMap[&F];

    // Add references to a local map and check if any references were
    // arguments and if so if we can modify the function.
    if (addReferencesForFunction(LocalReferenceMap, LocalParamMap,
                                 UniqueTypeSet, &F))
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

static void propagateNewTypeDefinitions(
    SmallDenseSet<std::pair<Type *, Type *>> &UniqueTypeSet,
    SmallDenseSet<std::pair<Type *, Type *>> &TranslatedTypeSet, Module &M) {
  for (auto &TypePair : UniqueTypeSet) {
    SmallVector<Type *> EltTys;
    Type *ArrayTy = TypePair.first;
    Type *ContainerTy = TypePair.second;
    auto *ArrayST = cast<StructType>(ArrayTy);
    auto *ContainerST = cast<StructType>(ContainerTy);
    for (unsigned i = 0, e = ContainerST->getNumElements(); i != e; ++i) {
      Type *FieldTy = ContainerST->getElementType(i);
      EltTys.push_back(FieldTy);
    }
    // We will use a unified approach for both opaque
    // pointers and the old model, ptrs to the sub fields of
    // ArrayST are appended to the new struct layout.
    for (unsigned i = 0, e = ArrayST->getNumElements(); i != e; ++i) {
      Type *FieldTy = ArrayST->getElementType(i);
      // Make a pointer to FieldTy and use that.
      PointerType *PtrTy = PointerType::getUnqual(FieldTy);
      EltTys.push_back(PtrTy);
    }
    std::string VarName((ArrayST->getName() + Twine("_soa")).str());
    StructType *NewST =
        StructType::create(M.getContext(), EltTys, VarName, false);
    // Now place the translated type in a set to be referenced when
    // we replace ContainerTy/ArrayTy instances during translation.
    TranslatedTypeSet.insert({ContainerST, NewST});
  }
}

static void mapContainerAccessToGlossary(
    SmallDenseSet<std::pair<Type *, Type *>> &UniqueTypeSet,
    DenseMap<Type *, SmallVector<Value *>> &GEPTypeToIndices,
    DenseMap<Function *,
             SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>,
                           int>> &CandidateMap,
    Module &M) {
  for (auto &TypePair : UniqueTypeSet) {
    bool FoundEntry = false;
    Type *ContainerTy = TypePair.second;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
          &LocalCandidateMap = CandidateMap[&F];

      // We construct an access glossary with a candidate connected GEP
      // that matches a reference in the UniqueTypeSet for each ContainerTy.
      // We need this as some of the references we collect are not directly
      // mapped to a candidate and we need to compare the indices to make
      // valid updates when replacing the type and access.  The first reference
      // we find is sufficient to construct a glossary entry for lookup later.
      for (auto &Candididates : LocalCandidateMap) {
        GetElementPtrInst *SrcGEP = Candididates.first.second;
        Type *RefTy = SrcGEP->getSourceElementType();
        if (RefTy == ContainerTy) {
          FoundEntry = true;
          // Now create a copy of the Indices which will not be modified
          // when the GEP is.
          SmallVector<Value *> &Indices = GEPTypeToIndices[RefTy];
          assert(Indices.empty() &&
                 "There should only be one map from RefTy to indices");
          for (const Use &Op : SrcGEP->indices())
            Indices.emplace_back(Op);

          break;
        }
      }
      // Move on to the next unique type entry
      if (FoundEntry)
        break;
    }
  }
}

PreservedAnalyses LoopDataLayoutPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  bool Changed = false;
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupScalarEvolutionInfo = [&FAM](Function &F) -> ScalarEvolution & {
    return FAM.getResult<ScalarEvolutionAnalysis>(F);
  };
  auto LookupLoopInfo = [&FAM](Function &F) -> LoopInfo & {
    return FAM.getResult<LoopAnalysis>(F);
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

  DenseMap<Function *,
           SmallDenseMap<std::pair<Instruction *, Type *>, int>>
      ReferenceMap;

  DenseMap<Function *, SmallVector<Type *>> ParamMap;
  DenseMap<Function *, SmallBitVector> InvalidateMap;
  SmallDenseSet<std::pair<Type *, Type *>, 4> UniqueTypeSet;
  DenseMap<Type *, SmallVector<Value *>> GEPTypeToIndices;
  SmallDenseSet<std::pair<Type *, Type *>, 4> TranslatedTypeSet;

  // Silently exit this optimization is disabled.
  if (!EnableLoopDataLayout)
    return PreservedAnalyses::all();

  CallGraph MCG(M);
  GlobalsAAResult GAAR = GlobalsAAResult::analyzeModule(M, LookupTLI, MCG);

  // find data layout candidates for SoA to AoS transformation
  auto Result = analyzeWholeProgram(
      LookupLoopInfo, LookupScalarEvolutionInfo, LookupTLI, AARGetter, GAAR, M,
      MaxElements, IsThinLTO, CandidateMap, ReferenceMap, ParamMap,
      InvalidateMap, UniqueTypeSet, MCG);

  // Silently exit as there are no opportunites to do the transformation.
  if (Result == LoopDataLayoutResult::HasNoOpportunities)
    return PreservedAnalyses::all();

  // Exit with status, transformation is illegal to perform.
  if (Result == LoopDataLayoutResult::TransformationIsIllegal) {
    LLVM_DEBUG(dbgs() << "Not Legal to do AoS to SoA Transformation\n");
    return PreservedAnalyses::all();
  }

  propagateNewTypeDefinitions(UniqueTypeSet, TranslatedTypeSet, M);

  mapContainerAccessToGlossary(UniqueTypeSet, GEPTypeToIndices, CandidateMap,
                               M);

  // Once we have the candidates, walk all functions updating the
  // candidates with updated SoA references and data member array uses.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    const TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);
    SmallVector<Type *> &LocalParamMap = ParamMap[&F];
    SmallDenseMap<std::pair<Instruction *, Type *>, int>
        &LocalReferenceMap = ReferenceMap[&F];
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap = CandidateMap[&F];
    TargetLibraryInfo &TLI = LookupTLI(F);
    // This is the place where we update params and uses in the
    // function with the updated SoA equivalents as we will be
    // replacing the old function with a modified call signature
    // when necessary.
    Changed =
        translateReferences(&F, AARGetter, MaxElements, None, TTI, TLI,
                            LocalReferenceMap, LocalParamMap, LocalCandidateMap,
                            UniqueTypeSet, TranslatedTypeSet, GEPTypeToIndices);
  }

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

  LoopDataLayoutLegacyPass(unsigned MaxElements = 2, bool ThinLTO = false)
      : ModulePass(ID), MaxElements(MaxElements), IsThinLTO(ThinLTO) {
    initializeLoopDataLayoutLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
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

  /// ThinLTO enabled for this pass
  bool IsThinLTO;
};

} // end anonymous namespace

char LoopDataLayoutLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(LoopDataLayoutLegacyPass, "loop-data-layout",
                      "Discover data layout opportunities", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(LoopDataLayoutLegacyPass, "loop-data-layout",
                    "Discover data layout opportunities", false, false)

ModulePass *llvm::createLoopDataLayoutPass(unsigned MaxElements, bool ThinLTO) {
  return new LoopDataLayoutLegacyPass(MaxElements, ThinLTO);
}

bool LoopDataLayoutLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  // Silently exit this optimization is disabled.
  if (!EnableLoopDataLayout)
    return false;

  bool Changed = false;
  auto LookupScalarEvolutionInfo =
      [this, &Changed](Function &F) -> ScalarEvolution & {
    return this->getAnalysis<ScalarEvolutionWrapperPass>(F, &Changed).getSE();
  };
  auto LookupLoopInfo = [this, &Changed](Function &F) -> LoopInfo & {
    return this->getAnalysis<LoopInfoWrapperPass>(F, &Changed).getLoopInfo();
  };
  auto LookupTLI = [this](Function &F) -> TargetLibraryInfo & {
    return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  };

  DenseMap<
      Function *,
      SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>>
      CandidateMap;
  DenseMap<Function *,
           SmallDenseMap<std::pair<Instruction *, Type *>, int>>
      ReferenceMap;
  DenseMap<Function *, SmallVector<Type *>> ParamMap;
  DenseMap<Function *, SmallBitVector> InvalidateMap;
  SmallDenseSet<std::pair<Type *, Type *>, 4> UniqueTypeSet;
  DenseMap<Type *, SmallVector<Value *>> GEPTypeToIndices;
  SmallDenseSet<std::pair<Type *, Type *>, 4> TranslatedTypeSet;
  LegacyAARGetter AARGetter(*this);

  CallGraph MCG(M);
  GlobalsAAResult GAAR = GlobalsAAResult::analyzeModule(M, LookupTLI, MCG);

  // find data layout candidates for SoA to AoS transformation
  auto Result = analyzeWholeProgram(
      LookupLoopInfo, LookupScalarEvolutionInfo, LookupTLI, AARGetter, GAAR, M,
      MaxElements, IsThinLTO, CandidateMap, ReferenceMap, ParamMap,
      InvalidateMap, UniqueTypeSet, MCG);

  // Silently exit as there are no opportunites to do the transformation.
  if (Result == LoopDataLayoutResult::HasNoOpportunities)
    return false;

  // Exit with status, transformation is illegal to perform.
  if (Result == LoopDataLayoutResult::TransformationIsIllegal) {
    LLVM_DEBUG(dbgs() << "Not Legal to do AoS to SoA Transformation\n");
    return false;
  }

  propagateNewTypeDefinitions(UniqueTypeSet, TranslatedTypeSet, M);

  mapContainerAccessToGlossary(UniqueTypeSet, GEPTypeToIndices, CandidateMap,
                               M);

  // Once we have the candidates, walk all functions looking for
  // container type parameters, data member parameters and local
  // instance of variables of these types to update.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    const TargetTransformInfo &TTI =
        getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    SmallVector<Type *> &LocalParamMap = ParamMap[&F];
    SmallDenseMap<std::pair<Instruction *, Type *>, int>
        &LocalReferenceMap = ReferenceMap[&F];
    SmallDenseMap<std::pair<GetElementPtrInst *, GetElementPtrInst *>, int>
        &LocalCandidateMap = CandidateMap[&F];
    TargetLibraryInfo &TLI = LookupTLI(F);
    // This is the place where we update params and uses in the
    // function with the updated SoA equivalents as we will be
    // replacing the old function with a modified call signature
    // when necessary.
    Changed =
        translateReferences(&F, AARGetter, MaxElements, None, TTI, TLI,
                            LocalReferenceMap, LocalParamMap, LocalCandidateMap,
                            UniqueTypeSet, TranslatedTypeSet, GEPTypeToIndices);
  }

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
