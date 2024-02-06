//===- SiFive_VPlanCostModel.cpp - Vectorizer Cost Model ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// VPlan-based cost model
///
//===----------------------------------------------------------------------===//
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include <unordered_map>

#include "LoopVectorizationPlanner.h"
#include "VPlan.h"
#include "VPlanValue.h"

namespace llvm {
class Type;
class LoopVectorizationLegality;
class TargetTransformInfo;
class TargetLibraryInfo;

/// The class represents (LMUL, SEW) concept in RVV. It allows to store integer
/// and fractional LMULs.
class RVVPair {
public:
  // FIXME: Unify with `RvvHintAttr` in Clang
  enum class LMULKind {
    Unsupported = INT_MIN,
    // The integer value represents encoding used by backend
    Mf8 = -3,
    Mf4 = -2,
    Mf2 = -1,
    M1 = 0,
    M2 = 1,
    M4 = 2,
    M8 = 3,
  };

  explicit RVVPair(Type *Ty, LMULKind Kind, const DataLayout &DL)
      : Ty(Ty), LMUL(Kind), DL(DL) {
    assert(Ty && !Ty->isVectorTy() && "Scalar type is expected");
  }

  /// Construct LMULType from LMUL's exponent which is 3-bit 2's complement
  /// integer value: LMULExp     LMUL      LMULKind
  /// 0b000         1         m1
  /// 0b001         2         m2
  /// 0b010         4         m4
  /// 0b011         8         m8
  /// ---------- fractional ---------
  /// 0b101         1/8      mf8
  /// 0b110         1/4      mf4
  /// 0b111         1/2      mf2
  static RVVPair getWithExponent(Type *Ty, const int LMULExp,
                                 const DataLayout &DL);

  static RVVPair get(Type *Ty, ElementCount EC, const DataLayout &DL) {
    const unsigned Numerator = DL.getTypeSizeInBits(Ty) * EC.getKnownMinValue();
    const unsigned Denominator = RISCV::RVVBitsPerBlock;
    const int LMULExp = Log2_32(Numerator / Denominator);
    return getWithExponent(Ty, LMULExp, DL);
  }

  /// Construct new RVV Pair from another RVV Pair \p RVVP and some other type
  /// \p Ty
  static RVVPair getWithType(Type *Ty, const RVVPair &RVVP);

  /// Return ElementCount that corresponds to current given parameters
  static ElementCount getElementCount(const int LMULExp, const unsigned SEW);

  /// Return a <integer, bool> pair where integer is a numerator if boolean is
  /// false or denominator if boolean is true
  std::pair<unsigned, bool> getLMUL() const { return getIntFromLMULKind(LMUL); }

  /// Return kind of LMUL
  LMULKind getLMULKind() const { return LMUL; }

  /// Return SEW
  unsigned getSEW() const { return DL.getTypeSizeInBits(Ty); }

  /// Return type of the SEW
  Type *getType() const { return Ty; }

  // Print methods
  void print(raw_ostream &O) const {
    O << '(';
    O << getStringFromLMULKind(LMUL) << ", " << *Ty << ')';
  }

  void dump(void) const { return print(llvm::dbgs()); }

  operator bool() const { return getLMULKind() != LMULKind::Unsupported; }

private:
  /// Return a <integer, bool> pair where integer is a numerator if boolean is
  /// false or denominator if boolean is true for a given \p LMUL
  static std::pair<unsigned, bool> getIntFromLMULKind(LMULKind LMUL);

  /// Convert LMULKind to string representation for a given \p LMUL
  StringRef getStringFromLMULKind(LMULKind LMUL) const;

  Type *Ty;
  LMULKind LMUL;
  const DataLayout &DL;
};

inline raw_ostream &operator<<(raw_ostream &OS, const RVVPair &RVVP) {
  RVVP.print(OS);
  return OS;
}

class VPlanCostModel {
public:
  explicit VPlanCostModel(const VPlan &Plan, LoopVectorizationLegality &Legal,
                          const TargetTransformInfo &TTI,
                          const TargetLibraryInfo &TLI)
      : Plan(Plan), Legal(Legal), TTI(TTI), TLI(TLI) {}

  /// Return cost of the VPlan for a given \p RVL
  InstructionCost getCost(const RVVPair &RVL);

  /// Return VectorType that corresponds to the specified (LMUL, SEW) pair
  static Type *getVectorType(Type *Ty, const RVVPair &RVVP);

private:
  /// Return individual cost of the \p VPBasicBlock for a given \p RVL
  InstructionCost getCost(const VPBlockBase *Block, const RVVPair &RVL);

  /// Return individual cost of the \p Recipe for a given \p RVL
  InstructionCost getCost(const VPRecipeBase *Recipe, const RVVPair &RVL);

  /// Return individual cost of the call for a given \p RVL
  InstructionCost getVectorCallCost(const CallInst *CI,
                                    const RVVPair &RVL) const;

  /// Return cost of the individual intrinsic for a given \p RVL
  InstructionCost getVectorIntrinsicCost(const CallInst *CI,
                                         const RVVPair &RVL) const;

  /// Return cost of the individual memory operation for a given \p RVL
  InstructionCost getMemoryOpCost(const VPWidenMemoryInstructionRecipe *VPWMIR,
                                  const RVVPair &RVL);

  /// Return cost of the interleavedmemory operation for a given \p RVL
  InstructionCost getInterleavedMemoryOpCost(const VPInterleaveRecipe *VPI,
                                             const RVVPair &RVL);

  /// Return cost of the individual memory operation of a instruction \p I of a
  /// given type \p Ty
  InstructionCost getMemoryOpCost(const Instruction *I, Type *Ty,
                                  bool IsConsecutive, bool IsMasked,
                                  bool IsReverse, bool IsSpeculative) const;

  /// Return individual cost of the VPInstruction \p I for a given \p RVL
  InstructionCost getInstructionCost(const VPInstruction *VPI,
                                     const RVVPair &RVL) const;

  /// Return cost of the reduction operation for the given \p RVL
  InstructionCost getReductionCost(const VPReductionRecipe *VPR,
                                   const RVVPair &RVL) const;

  /// Return individual cost of the VPReplicateRecipe \p VPR for a given \p RVL
  InstructionCost getReplicateOpCost(const VPReplicateRecipe *VPR,
                                     const RVVPair &RVL) const;

  /// Return cost to use register type \p RegID. Return 0 if no
  /// spills/reload required
  InstructionCost getRegisterPressureCost(const unsigned RegID, Type *Ty) const;

  /// Return individual cost of the VPMonotonicUpdateInstruction \p VPM for a
  /// given \p RVL
  InstructionCost
  getMonotonicUpdateCost(const VPMonotonicUpdateInstruction *VPM,
                         const RVVPair &RVL) const;

  /// Associate new registers \p NumRegs of a type \p RegID with VPValue \p VPV.
  void addRegisterUsage(const VPValue *VPV, const unsigned RegID,
                        const unsigned NumRegs);

  /// Convenient method to return cost of an intrinsic
  InstructionCost getIntrinsicCost(Intrinsic::ID Id, Type *RetTy,
                                   ArrayRef<Value *> Arguments,
                                   FastMathFlags FMF) const;

  /// VPlan for which cost is computed
  const VPlan &Plan;

  /// Vectorization legality.
  /// TODO: Consider to remove. All information from the legality should be
  /// presented in the VPlan
  LoopVectorizationLegality &Legal;

  /// Vector target information.
  const TargetTransformInfo &TTI;

  /// Target Library Info.
  const TargetLibraryInfo &TLI;

  /// Use same cost kind in the cost model
  const TargetTransformInfo::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  struct RegistersUsage {
    /// A set of VPValues and their register use that are currently live
    DenseMap<const VPValue *, DenseMap<unsigned, unsigned>> LiveRecipes;

    /// A set of registers currently in use
    DenseMap<unsigned, unsigned> LiveRegister;
  } RegistersUsage;

  /// A set of VPRecipes that were visited by the cost model
  DenseSet<const VPRecipeBase *> VisitedRecipes;
};
} // namespace llvm
