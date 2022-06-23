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

// Fuse LUI followed by ADDI or ADDIW.
// rd = imm[31:0] which decomposes to
// lui rd, imm[31:12]
// addi(w) rd, rd, imm[11:0]
static bool isLUIADDI(const MachineInstr *FirstMI,
                      const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::ADDI &&
      SecondMI.getOpcode() != RISCV::ADDIW)
    return false;

  // Assume the 1st instr to be a wildcard if it is unspecified
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::LUI)
    return false;

  // The first operand of ADDI can be a frame index.
  if (!SecondMI.getOperand(1).isReg())
    return false;

  Register FirstDest = FirstMI->getOperand(0).getReg();

  // Destination of LUI should be the ADDI(W) source register.
  if (SecondMI.getOperand(1).getReg() != FirstDest)
    return false;

  // If the FirstMI destination is non-virtual, it should match the SecondMI
  // destination.
  return FirstDest.isVirtual() ||
         SecondMI.getOperand(0).getReg() == FirstDest;
}

static bool isIndexedLoad(const MachineInstr *FirstMI,
                          const MachineInstr &SecondMI, bool FuseZba) {
  switch (SecondMI.getOpcode()) {
  default:
    return false;
  case RISCV::LB:
  case RISCV::LBU:
  case RISCV::LH:
  case RISCV::LHU:
  case RISCV::LW:
  case RISCV::LWU:
  case RISCV::LD:
  case RISCV::FLH:
  case RISCV::FLW:
  case RISCV::FLD:
    break;
  }

  // Assume the 1st instr to be a wildcard if it is unspecified
  if (!FirstMI)
    return true;

  switch (FirstMI->getOpcode()) {
  default:
    return false;
  case RISCV::ADD:
    break;
  case RISCV::SH1ADD:
  case RISCV::SH2ADD:
  case RISCV::SH3ADD:
  case RISCV::SH1ADD_UW:
  case RISCV::SH2ADD_UW:
  case RISCV::SH3ADD_UW:
  case RISCV::ADD_UW:
    if (!FuseZba)
      return false;
    break;
  }

  // The first operand might be frame index.
  if (!SecondMI.getOperand(1).isReg())
    return false;

  Register FirstDest = FirstMI->getOperand(0).getReg();

  if (SecondMI.getOperand(1).getReg() != FirstDest)
    return false;

  // If the FirstMI destination is non-virtual, it should match the SecondMI
  // destination.
  return FirstDest.isVirtual() || SecondMI.getOperand(0).getReg() == FirstDest;
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
  if (ST.hasFuseIndexedLoad() &&
      isIndexedLoad(FirstMI, SecondMI, ST.hasFuseZbaLoad()))
    return true;
  return false;
}

std::unique_ptr<ScheduleDAGMutation> llvm::createRISCVMacroFusionDAGMutation() {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}
