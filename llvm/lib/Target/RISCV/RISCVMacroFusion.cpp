//===- RISCVMacroFusion.cpp - RISCV Macro Fusion --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the RISCV implementation of the DAG scheduling
/// mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//
//
#include "RISCVMacroFusion.h"
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

  // Assume the 1st instr to be a wildcard if it is unspecified.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::LUI)
    return false;

  // The first operand of ADDI might be a frame index.
  if (!SecondMI.getOperand(1).isReg())
    return false;

  Register FirstDest = FirstMI->getOperand(0).getReg();

  // Destination of LUI should be the ADDI(W) source register.
  if (SecondMI.getOperand(1).getReg() != FirstDest)
    return false;

  // If the FirstMI destination is non-virtual, it should match the SecondMI
  // destination.
  return FirstDest.isVirtual() || SecondMI.getOperand(0).getReg() == FirstDest;
}

#if SIFIVE_CUSTOMIZATION
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

  // Immediate offset must be 0.
  if (SecondMI.getOperand(2).getImm() != 0)
    return false;

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

static bool isArithEqZ(const MachineInstr *FirstMI,
                       const MachineInstr &SecondMI) {
  unsigned SrcOpIdx;
  switch (SecondMI.getOpcode()) {
  default:
    return false;
  case RISCV::SLTIU:
    if (SecondMI.getOperand(2).getImm() != 1)
      return false;
    SrcOpIdx = 1;
    break;
  case RISCV::SLTU:
    if (SecondMI.getOperand(1).getReg() != RISCV::X0)
      return false;
    SrcOpIdx = 2;
    break;
  }

  // Assume the 1st instr to be a wildcard if it is unspecified.
  if (!FirstMI)
    return true;

  bool PreRA = false;
  switch (FirstMI->getOpcode()) {
  default:
    return false;
  case RISCV::ADDI:
  case RISCV::XOR:
    // Allow these pre-RA because they are the idioms we use for equality
    // comparisons.
    PreRA = true;
    break;
  case RISCV::SUB:
  case RISCV::OR:
    // Do not allow these pre-RA because we can't ensure that SecondMI is the
    // only user pre-RA.
    // FIXME: We probably need some pseudoinstructions and an earlier fusion
    // peephole.
    break;
  }

  Register FirstDest = FirstMI->getOperand(0).getReg();

  // The SecondMI source operand should match the FirstMI destination.
  if (SecondMI.getOperand(SrcOpIdx).getReg() != FirstDest)
    return false;

  // If the FirstMI destination is non-virtual, it should match the SecondMI
  // destination.
  return (FirstDest.isVirtual() && PreRA) ||
         SecondMI.getOperand(0).getReg() == FirstDest;
}

// \brief Check if the instr pair, FirstMI and SecondMI, should be fused
// together. Given SecondMI, when FirstMI is unspecified, then check if
// SecondMI may be part of a fused pair at all.
#endif // SIFIVE_CUSTOMIZATION
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const RISCVSubtarget &ST = static_cast<const RISCVSubtarget &>(TSI);

  if (ST.hasLUIADDIFusion() && isLUIADDI(FirstMI, SecondMI))
    return true;
#if SIFIVE_CUSTOMIZATION
  if (ST.hasFuseIndexedLoad() &&
      isIndexedLoad(FirstMI, SecondMI, ST.hasFuseZbaLoad()))
    return true;
  if (ST.hasFuseArithEqZ() && isArithEqZ(FirstMI, SecondMI))
    return true;
#endif // SIFIVE_CUSTOMIZATION
  return false;
}

std::unique_ptr<ScheduleDAGMutation> llvm::createRISCVMacroFusionDAGMutation() {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}
