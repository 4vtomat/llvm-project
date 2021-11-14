//===-- RISCVMacroFusion.cpp - RISCV Macro Fusion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file This file contains the RISCV implementation of the DAG scheduling
// mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "SiFive_RISCVMacroFusion.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

static bool isLUIADDI(const MachineInstr *FirstMI,
                      const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::ADDI &&
      SecondMI.getOpcode() != RISCV::ADDIW)
    return false;

  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::LUI)
    return false;

  if (!SecondMI.getOperand(1).isReg())
    return false;

  return FirstMI->getOperand(0).getReg() == SecondMI.getOperand(0).getReg() &&
         SecondMI.getOperand(0).getReg() == SecondMI.getOperand(1).getReg();
}

// \brief Check if the instr pair, FirstMI and SecondMI, should be fused
// together. Given SecondMI, when FirstMI is unspecified, then check if
// SecondMI may be part of a fused pair at all.
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const RISCVSubtarget &ST = static_cast<const RISCVSubtarget &>(TSI);

  if (ST.hasLUIADDIFusion() && isLUIADDI(FirstMI, SecondMI))
    return true;
  return false;
}

std::unique_ptr<ScheduleDAGMutation> llvm::createRISCVMacroFusionDAGMutation() {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}
