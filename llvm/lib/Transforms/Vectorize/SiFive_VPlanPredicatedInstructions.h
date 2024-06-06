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
Value *widenPredicatedInstruction(Instruction *Op, VPValue *Def, VPUser &User,
                                  VPTransformState &State, VPValue *BlockInMask,
                                  unsigned Part);

void widenPredicatedCall(CallInst *CI, VPValue *Def, VPTransformState &State,
                         Intrinsic::ID VPID, unsigned Part);

/// Build and return either `vp.gather`/`vp.scatter` or
/// `vp.strided_load`/`vp.strided_store` if previous analysis indicated it's
/// possible to be used
Instruction *
widenPredicatedMemoryInstruction(VPWidenMemoryRecipe &VPWMIR,
                                 VPTransformState &State, unsigned Part,
                                 ArrayRef<Value *> BlockInMaskParts);

/// Build and return vp-intrinsic that corresponds to arithmetic operation \p
/// Op.
Instruction *widenPredicatedArithmeticOp(VPTransformState &State, unsigned Op,
                                         ArrayRef<Value *> Ops, unsigned Part,
                                         Value *Mask = nullptr,
                                         const Twine &Name = "");
} // namespace llvm
