//===- SiFive_LoopReassociate.h - Loop Reassociate Pass ---------*- C++ -*-===//
//
// Copyright (c) 2024 SiFive, Inc. -- Proprietary and Confidential All
// Rights Reserved.
//
// NOTICE: All information contained herein is, and remains the property of
// SiFive, Inc. The intellectual and technical concepts contained herein are
// proprietary to SiFive, Inc. and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// This work may not be copied, modified, re-published, uploaded, executed, or
// distributed in any way, in any medium, whether in whole or in part, without
// prior written permission from SiFive, Inc. The copyright notice above does
// not evidence any actual or intended publication or disclosure of this source
// code, which includes information that is confidential and/or proprietary,
// and is a trade secret, of SiFive, Inc.
//
//===----------------------------------------------------------------------===//
//
// The pass performs reassociation of instructions across loop.
// Currently the only optimization implemented in the pass hoists reduction
// computation out when it's possible:
//
//  for i = 0, N {
//    %r = phi %init, %r.next
//    ...
//    %red = vector.reduce ...
//
//    %r.next = add %red, %r
//  }
//  %r.lcssa = phi %r.next
//  <use-of-r.lcssa>
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SIFIVE_LOOP_REASSOCIATE_H
#define LLVM_TRANSFORMS_SCALAR_SIFIVE_LOOP_REASSOCIATE_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Loop;
class LPMUpdater;

/// Performs Loop Reassociation
class SiFiveLoopReassociatePass
    : public PassInfoMixin<SiFiveLoopReassociatePass> {
public:
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SIFIVE_LOOP_REASSOCIATE_H
