//===- SiFive_VPlanPredicatedInstructions.h - Vectorizer Plan -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Supports emission of the predicated versions of the vector instructions.
///
//===----------------------------------------------------------------------===//

#include "VPlan.h"
#include "VPlanValue.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/IR/Instruction.h"

namespace llvm {
void widenPredicatedInstruction(Instruction *Op, VPValue *Def, VPUser &User,
                                VPTransformState &State, VPValue *BlockInMask,
                                VPValue *EVL, unsigned Part);

/// A recipe to generate All True mask vector
class VPAllTrueMaskRecipe final : public VPRecipeBase, public VPValue {

public:
  VPAllTrueMaskRecipe(VPValue *EVL)
      : VPRecipeBase(VPRecipeBase::VPAllTrueMaskSC, {EVL}),
        VPValue(VPValue::VPVAllTrueMaskSC, nullptr, this) {}
  ~VPAllTrueMaskRecipe() override = default;

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const VPDef *D) {
    return D->getVPDefID() == VPRecipeBase::VPAllTrueMaskSC;
  }

  /// Generate the instructions to compute EVL.
  void execute(VPTransformState &State) override final;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};
} // namespace llvm
