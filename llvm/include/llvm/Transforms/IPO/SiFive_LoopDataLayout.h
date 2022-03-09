//===- SiFive_LoopDataLayout.h - Loop Data Layout -------------------------===//
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
// This file provides the interface for the Loop Data Layout Pass.
//
//===----------------------------------------------------------------------===//

#if SIFIVE_CUSTOMIZATION

#ifndef LLVM_TRANSFORMS_IPO_LOOPDATALAYOUT_H
#define LLVM_TRANSFORMS_IPO_LOOPDATALAYOUT_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Data Layout Analysis and Optimization pass.
///
/// This pass walks the functions in the merged LTO Module and for each
/// function analyzes loops and collects transformable candidate information
/// for Array of Structures to Structures of Arrays optimization.
/// It then analyzes all casts to and from the containing struct,
/// collects parameters which hold these structs and or data items
/// passed, checks TBAA locally and GlobalAA for aliasing info and
/// if all is sufficiently safe, replaces AoS entries with their
/// arrays of data members.
struct LoopDataLayoutPass : public PassInfoMixin<LoopDataLayoutPass> {
  LoopDataLayoutPass(unsigned MaxElements = 2u) : MaxElements(MaxElements) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);

private:
  unsigned MaxElements;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_LOOPDATALAYOUT_H

#endif // SIFIVE_CUSTOMIZATION
