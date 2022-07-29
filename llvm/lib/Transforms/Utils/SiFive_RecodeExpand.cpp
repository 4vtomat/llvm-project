//===---------------------- SiFive_RecodeExpand.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands Neon intrinsics to LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SiFive_RecodeExpand.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

bool SiFiveRecodePass::requireExpand(IntrinsicInst *II) { return false; }

PreservedAnalyses SiFiveRecodePass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  bool MadeChange = false;
  for (Instruction &Inst : llvm::make_early_inc_range(instructions(F))) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(&Inst);
    if (II && requireExpand(II)) {
      MadeChange = true;
    }
  }

  if (MadeChange) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }

  return PreservedAnalyses::all();
}
