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
} // namespace llvm
