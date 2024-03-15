//=== SiFive_RISCVMaskInstDAGMutation.h - RISCVMaskInstDAGMutation -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the RISC-V definition of the DAG scheduling
/// mutation to move mask instructions away from their uses so the latency
/// of the arithmetic pipeline is not exposted to the subsequent dependent
/// instruction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVMASKINSTDAGMUTATION_H
#define LLVM_LIB_TARGET_RISCV_RISCVMASKINSTDAGMUTATION_H

#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class RISCVMaskInstDAGMutation : public ScheduleDAGMutation {

public:
  RISCVMaskInstDAGMutation() = default;

  void apply(ScheduleDAGInstrs *DAG) override;
};

/// Note that you have to add:
///   DAG.addMutation(createRISCVMaskInstDAGMutation());
/// to RISCVPassConfig::createMachineScheduler() to have an effect.
std::unique_ptr<ScheduleDAGMutation> createRISCVMaskInstDAGMutation();
} // namespace llvm

#endif
