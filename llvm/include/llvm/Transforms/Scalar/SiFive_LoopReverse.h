//===- SiFive_LoopReverse.h - Loop Reverse --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface for the Loop Reverse Pass.
//
//===----------------------------------------------------------------------===//

#if SIFIVE_CUSTOMIZATION

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPREVERSE_H
#define LLVM_TRANSFORMS_SCALAR_LOOPREVERSE_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Loop;
class LPMUpdater;

class LoopReversePass : public PassInfoMixin<LoopReversePass> {
public:
  LoopReversePass() = default;

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPREVERSE_H

#endif // SIFIVE_CUSTOMIZATION