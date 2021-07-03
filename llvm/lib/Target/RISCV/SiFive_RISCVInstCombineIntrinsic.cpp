//===-- RISCVInstCombineIntrinsic.cpp - RISCV specific InstCombine pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// RISCV target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "RISCVTargetTransformInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;

#define DEBUG_TYPE "riscvtti"

// Look for a splat, ignoring whether the operand is scalar or vector.
static Value *getVSplat(Value *Op, Value *VL) {
  Type *Ty = Op->getType();
  if (Ty->isIntegerTy() || Ty->isFloatingPointTy())
    return Op;

  assert(Ty->isVectorTy() && "Unexpected type!");

  if (auto *II = dyn_cast<IntrinsicInst>(Op)) {
    if ((II->getIntrinsicID() == Intrinsic::riscv_vmv_v_x ||
         II->getIntrinsicID() == Intrinsic::riscv_vfmv_v_f) &&
        isa<UndefValue>(II->getArgOperand(0)) &&
        II->getArgOperand(2) == VL)
      return II->getOperand(1);
  }

  return nullptr;
}

// Try to match
//   (vand (vmerge 0, -1, C), A)
// To
//   (vmerge 0, A, C)
static Instruction *foldVAndWithVMerge(Value *LHS, Value *RHS, Value *VL) {
  auto *II2 = dyn_cast<IntrinsicInst>(LHS);
  if (!II2 || II2->getIntrinsicID() != Intrinsic::riscv_vmerge ||
      !isa<UndefValue>(II2->getArgOperand(0)) || II2->getArgOperand(4) != VL)
    return nullptr;

  Value *FalseVal = getVSplat(II2->getArgOperand(1), VL);
  Value *TrueVal = getVSplat(II2->getArgOperand(2), VL);
  auto *FalseC = dyn_cast_or_null<ConstantInt>(FalseVal);
  auto *TrueC = dyn_cast_or_null<ConstantInt>(TrueVal);
  if (FalseC && TrueC && FalseC->isZero() && TrueC->isMinusOne()) {
    Function *Merge = Intrinsic::getDeclaration(
        II2->getModule(), Intrinsic::riscv_vmerge,
        {II2->getType(), RHS->getType(), VL->getType()});
    return CallInst::Create(Merge, {UndefValue::get(II2->getType()),
                                    II2->getArgOperand(1), RHS,
                                    II2->getArgOperand(3), VL});
  }

  // FIXME: Handle the (vmerge -1, 0, C) case. Trickier if other AND operand
  // is a scalar since the first operand of vmerge can't be scalar.

  return nullptr;
}

// Try to match
//   (vxor (vmerge 0, (vxor A, B), C), A)
// To
//   (vmerge A, B, C)
static Instruction *foldVXorWithVMergeVXor(Value *LHS, Value *RHS, Value *VL) {
  auto *II2 = dyn_cast<IntrinsicInst>(LHS);
  if (!II2 || II2->getIntrinsicID() != Intrinsic::riscv_vmerge ||
      !isa<UndefValue>(II2->getArgOperand(0)) || II2->getArgOperand(4) != VL)
    return nullptr;

  auto MatchBitSelect = [](Value *Other, Value *Op0, Value *Op1,
                           Value *VL) -> Value * {
    // One operand of the vmerge needs to be 0.
    auto *Zero = dyn_cast_or_null<ConstantInt>(getVSplat(Op0, VL));
    if (!Zero || !Zero->isZero())
      return nullptr;

    // If this is a VXOR involving Other, return the other operand.
    auto *II3 = dyn_cast<IntrinsicInst>(Op1);
    if (II3 && II3->getIntrinsicID() == Intrinsic::riscv_vxor &&
        isa<UndefValue>(II3->getArgOperand(0)) && II3->getArgOperand(3) == VL) {
      if (II3->getOperand(1) == Other)
        return II3->getArgOperand(2);
      if (II3->getArgOperand(2) == Other)
        return II3->getArgOperand(1);
    }

    return nullptr;
  };

  Value *FalseVal = II2->getArgOperand(1);
  Value *TrueVal = II2->getArgOperand(2);
  Value *A = RHS;
  Value *C = II2->getArgOperand(3);

  // The VXOR could be either the True or False value of the vmerge, handle
  // both cases. We need to be careful not to put a scalar as the first
  // operand of vmerge.
  if (A->getType()->isVectorTy()) {
    // (vxor (vmerge 0, (vxor A, B), C), A) -> (vmerge A, B, C)
    if (Value *B = MatchBitSelect(A, FalseVal, TrueVal, VL)) {
      Function *Merge = Intrinsic::getDeclaration(
          II2->getModule(), Intrinsic::riscv_vmerge,
          {II2->getType(), B->getType(), VL->getType()});
      return CallInst::Create(Merge,
                              {UndefValue::get(II2->getType()), A, B, C, VL});
    }
  }
  // (vxor (vmerge (vxor A, B), 0, C), A) -> (vmerge B, A, C)
  if (Value *B = MatchBitSelect(A, TrueVal, FalseVal, VL)) {
    if (B->getType()->isVectorTy()) {
      Function *Merge = Intrinsic::getDeclaration(
          II2->getModule(), Intrinsic::riscv_vmerge,
          {II2->getType(), A->getType(), VL->getType()});
      return CallInst::Create(Merge,
                              {UndefValue::get(II2->getType()), B, A, C, VL});
    }
  }

  return nullptr;
}

/// This function handles following case
///
///     A  ->  B    cast to fixed
///     PHI
///     B  ->  A    cast from fixed
///
/// All the related PHI nodes can be replaced by new PHI nodes with type A.
/// The uses of \p II can be changed to the new PHI node corresponding to \p PN.
/// NOTE: This is based on optimizeBitCastFromPhi with the load/store handling
/// removed.
static Instruction *optimizeVCastFromFixedPhi(IntrinsicInst &II, PHINode *PN,
                                              InstCombiner &IC) {
  Value *Src = II.getArgOperand(0);
  Type *SrcTy = Src->getType(); // Type B
  Type *DestTy = II.getType();  // Type A

  SmallVector<PHINode *, 4> PhiWorklist;
  SmallSetVector<PHINode *, 4> OldPhiNodes;

  // Find all of the A->B casts and PHI nodes.
  // We need to inspect all related PHI nodes, but PHIs can be cyclic, so
  // OldPhiNodes is used to track all known PHI nodes, before adding a new
  // PHI to PhiWorklist, it is checked against and added to OldPhiNodes first.
  PhiWorklist.push_back(PN);
  OldPhiNodes.insert(PN);
  while (!PhiWorklist.empty()) {
    auto *OldPN = PhiWorklist.pop_back_val();
    for (Value *IncValue : OldPN->incoming_values()) {
      if (auto *PNode = dyn_cast<PHINode>(IncValue)) {
        if (OldPhiNodes.insert(PNode))
          PhiWorklist.push_back(PNode);
        continue;
      }

      auto *VCastTo = dyn_cast<IntrinsicInst>(IncValue);
      // We can't handle other instructions.
      if (!VCastTo ||
          VCastTo->getIntrinsicID() != Intrinsic::riscv_vcast_to_fixed)
        return nullptr;

      // Verify it's a A->B cast.
      Type *TyA = VCastTo->getArgOperand(0)->getType();
      Type *TyB = VCastTo->getType();
      if (TyA != DestTy || TyB != SrcTy)
        return nullptr;
    }
  }

  // Check that each user of each old PHI node is something that we can
  // rewrite, so that all of the old PHI nodes can be cleaned up afterwards.
  for (auto *OldPN : OldPhiNodes) {
    for (User *V : OldPN->users()) {
      if (auto *VCastFrom = dyn_cast<IntrinsicInst>(V)) {
        if (VCastFrom->getIntrinsicID() != Intrinsic::riscv_vcast_from_fixed)
          return nullptr;
        // Verify it's a B->A cast.
        Type *TyB = VCastFrom->getArgOperand(0)->getType();
        Type *TyA = VCastFrom->getType();
        if (TyA != DestTy || TyB != SrcTy)
          return nullptr;
      } else if (auto *PHI = dyn_cast<PHINode>(V)) {
        // As long as the user is another old PHI node, then even if we don't
        // rewrite it, the PHI web we're considering won't have any users
        // outside itself, so it'll be dead.
        if (OldPhiNodes.count(PHI) == 0)
          return nullptr;
      } else {
        return nullptr;
      }
    }
  }

  // For each old PHI node, create a corresponding new PHI node with a type A.
  SmallDenseMap<PHINode *, PHINode *> NewPNodes;
  for (auto *OldPN : OldPhiNodes) {
    IC.Builder.SetInsertPoint(OldPN);
    PHINode *NewPN = IC.Builder.CreatePHI(DestTy, OldPN->getNumOperands());
    NewPNodes[OldPN] = NewPN;
  }

  // Fill in the operands of new PHI nodes.
  for (auto *OldPN : OldPhiNodes) {
    PHINode *NewPN = NewPNodes[OldPN];
    for (unsigned j = 0, e = OldPN->getNumOperands(); j != e; ++j) {
      Value *V = OldPN->getOperand(j);
      Value *NewV = nullptr;
      if (auto *VCastTo = dyn_cast<IntrinsicInst>(V)) {
        assert(VCastTo->getIntrinsicID() == Intrinsic::riscv_vcast_to_fixed &&
               "Unexpected intrinsic");
        NewV = VCastTo->getArgOperand(0);
      } else if (auto *PrevPN = dyn_cast<PHINode>(V)) {
        NewV = NewPNodes[PrevPN];
      }
      assert(NewV);
      NewPN->addIncoming(NewV, OldPN->getIncomingBlock(j));
    }
  }

  // Traverse all accumulated PHI nodes and process its users,
  // which are vcast_from intrinsics. Without this processing
  // NewPHI nodes could be replicated and could lead to extra
  // moves generated after DeSSA.

  // Replace users of BitCast B->A with NewPHI. These will help
  // later to get rid off a closure formed by OldPHI nodes.
  Instruction *RetVal = nullptr;
  for (auto *OldPN : OldPhiNodes) {
    PHINode *NewPN = NewPNodes[OldPN];
    for (User *V : make_early_inc_range(OldPN->users())) {
      if (auto *VCastTo = dyn_cast<IntrinsicInst>(V)) {
        Type *TyB = VCastTo->getArgOperand(0)->getType();
        Type *TyA = VCastTo->getType();
        assert(TyA == DestTy && TyB == SrcTy);
        (void)TyA;
        (void)TyB;
        Instruction *I = IC.replaceInstUsesWith(*VCastTo, NewPN);
        if (VCastTo == &II)
          RetVal = I;
      } else if (auto *PHI = dyn_cast<PHINode>(V)) {
        assert(OldPhiNodes.contains(PHI));
        (void)PHI;
      } else {
        llvm_unreachable("all uses should be handled");
      }
    }
  }

  return RetVal;
}

Optional<Instruction *>
RISCVTTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  default:
    break;
  case Intrinsic::riscv_vcast_from_fixed:
    if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(0)))
      if (II2->getIntrinsicID() == Intrinsic::riscv_vcast_to_fixed &&
          II.getType() == II2->getArgOperand(0)->getType())
        return IC.replaceInstUsesWith(II, II2->getArgOperand(0));

    if (auto *PN = dyn_cast<PHINode>(II.getArgOperand(0)))
      if (auto *I = optimizeVCastFromFixedPhi(II, PN, IC))
        return I;

    break;
  case Intrinsic::riscv_vcast_to_fixed:
    if (auto *II2 = dyn_cast<IntrinsicInst>(II.getArgOperand(0)))
      if (II2->getIntrinsicID() == Intrinsic::riscv_vcast_from_fixed &&
          II.getType() == II2->getArgOperand(0)->getType())
        return IC.replaceInstUsesWith(II, II2->getArgOperand(0));
    break;
  case Intrinsic::riscv_vand: {
    if (!isa<UndefValue>(II.getArgOperand(0)))
      break;
    Value *LHS = II.getArgOperand(1);
    Value *RHS = II.getArgOperand(2);
    Value *VL = II.getArgOperand(3);
    if (Instruction *V = foldVAndWithVMerge(LHS, RHS, VL))
      return V;
    // And is commutable, try the other order.
    if (Instruction *V = foldVAndWithVMerge(RHS, LHS, VL))
      return V;
    break;
  }
  case Intrinsic::riscv_vxor: {
    if (!isa<UndefValue>(II.getArgOperand(0)))
      break;
    Value *LHS = II.getArgOperand(1);
    Value *RHS = II.getArgOperand(2);
    Value *VL = II.getArgOperand(3);

    if (Instruction *V = foldVXorWithVMergeVXor(LHS, RHS, VL))
      return V;
    // Xor is commutable, try the other order.
    if (Instruction *V = foldVXorWithVMergeVXor(RHS, LHS, VL))
      return V;

    break;
  }
  }

  return None;
}
