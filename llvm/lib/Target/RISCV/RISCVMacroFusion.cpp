//===- RISCVMacroFusion.cpp - RISC-V Macro Fusion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the RISC-V implementation of the DAG scheduling
/// mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//
//
#include "RISCVMacroFusion.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

static bool checkRegisters(Register FirstDest, const MachineInstr &SecondMI) {
  if (!SecondMI.getOperand(1).isReg())
    return false;

  if (SecondMI.getOperand(1).getReg() != FirstDest)
    return false;

  // If the input is virtual make sure this is the only user.
  if (FirstDest.isVirtual()) {
    auto &MRI = SecondMI.getMF()->getRegInfo();
    return MRI.hasOneNonDBGUse(FirstDest);
  }

  return SecondMI.getOperand(0).getReg() == FirstDest;
}

// Fuse load with add:
// add rd, rs1, rs2
// ld rd, 0(rd)
static bool isLDADD(const MachineInstr *FirstMI, const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::LD)
    return false;

  if (!SecondMI.getOperand(2).isImm())
    return false;

  if (SecondMI.getOperand(2).getImm() != 0)
    return false;

  // Given SecondMI, when FirstMI is unspecified, we must return
  // if SecondMI may be part of a fused pair at all.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::ADD)
    return true;

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

// Fuse zero extension of halfword:
// slli rd, rs1, 48
// srli rd, rd, 48
static bool isZExtH(const MachineInstr *FirstMI, const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::SRLI)
    return false;

  if (!SecondMI.getOperand(2).isImm())
    return false;

  if (SecondMI.getOperand(2).getImm() != 48)
    return false;

  // Given SecondMI, when FirstMI is unspecified, we must return
  // if SecondMI may be part of a fused pair at all.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::SLLI)
    return false;

  if (FirstMI->getOperand(2).getImm() != 48)
    return false;

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

// Fuse zero extension of word:
// slli rd, rs1, 32
// srli rd, rd, 32
static bool isZExtW(const MachineInstr *FirstMI, const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::SRLI)
    return false;

  if (!SecondMI.getOperand(2).isImm())
    return false;

  if (SecondMI.getOperand(2).getImm() != 32)
    return false;

  // Given SecondMI, when FirstMI is unspecified, we must return
  // if SecondMI may be part of a fused pair at all.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::SLLI)
    return false;

  if (FirstMI->getOperand(2).getImm() != 32)
    return false;

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

// Fuse shifted zero extension of word:
// slli rd, rs1, 32
// srli rd, rd, x
// where 0 <= x < 32
static bool isShiftedZExtW(const MachineInstr *FirstMI,
                           const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::SRLI)
    return false;

  if (!SecondMI.getOperand(2).isImm())
    return false;

  unsigned SRLIImm = SecondMI.getOperand(2).getImm();
  if (SRLIImm >= 32)
    return false;

  // Given SecondMI, when FirstMI is unspecified, we must return
  // if SecondMI may be part of a fused pair at all.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::SLLI)
    return false;

  if (FirstMI->getOperand(2).getImm() != 32)
    return false;

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

// Fuse AUIPC followed by ADDI
// auipc rd, imm20
// addi rd, rd, imm12
static bool isAUIPCADDI(const MachineInstr *FirstMI,
                        const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::ADDI)
    return false;
  // Assume the 1st instr to be a wildcard if it is unspecified.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::AUIPC)
    return false;

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

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

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

#if SIFIVE_CUSTOMIZATION
static bool isLUILoad(const MachineInstr *FirstMI,
                      const MachineInstr &SecondMI) {
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

  if (FirstMI->getOpcode() != RISCV::LUI &&
      FirstMI->getOpcode() != RISCV::AUIPC)
    return false;

  // The first operand might be frame index.
  if (!SecondMI.getOperand(1).isReg())
    return false;

  Register FirstDest = FirstMI->getOperand(0).getReg();

  if (SecondMI.getOperand(1).getReg() != FirstDest)
    return false;

  // If the input is virtual make sure this is the only user.
  if (FirstDest.isVirtual()) {
    auto &MRI = SecondMI.getMF()->getRegInfo();
    return MRI.hasOneNonDBGUse(FirstDest);
  }

  // If the FirstMI destination is non-virtual, it should match the SecondMI
  // destination.
  return SecondMI.getOperand(0).getReg() == FirstDest;
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

  // If the input is virtual make sure this is the only user.
  if (FirstDest.isVirtual()) {
    auto &MRI = SecondMI.getMF()->getRegInfo();
    return MRI.hasOneNonDBGUse(FirstDest);
  }

  // If the FirstMI destination is non-virtual, it should match the SecondMI
  // destination.
  return SecondMI.getOperand(0).getReg() == FirstDest;
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

  // If the input is virtual make sure this is the only user.
  if (FirstDest.isVirtual()) {
    if (!PreRA)
      return false;
    auto &MRI = SecondMI.getMF()->getRegInfo();
    return MRI.hasOneNonDBGUse(FirstDest);
  }

  // If the FirstMI destination is non-virtual, it should match the SecondMI
  // destination.
  return SecondMI.getOperand(0).getReg() == FirstDest;
}

/// Check if the instr pair, FirstMI and SecondMI, should be fused
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
#endif // SIFIVE_CUSTOMIZATION
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const RISCVSubtarget &ST = static_cast<const RISCVSubtarget &>(TSI);

  if (ST.hasLUIADDIFusion() && isLUIADDI(FirstMI, SecondMI))
    return true;
#if SIFIVE_CUSTOMIZATION
  if (ST.hasFuseLUILoad() && isLUILoad(FirstMI, SecondMI))
    return true;
  if (ST.hasFuseIndexedLoad() &&
      isIndexedLoad(FirstMI, SecondMI, ST.hasFuseZbaLoad()))
    return true;
  if (ST.hasFuseArithEqZ() && isArithEqZ(FirstMI, SecondMI))
    return true;
#endif // SIFIVE_CUSTOMIZATION

  if (ST.hasAUIPCADDIFusion() && isAUIPCADDI(FirstMI, SecondMI))
    return true;

  if (ST.hasZExtHFusion() && isZExtH(FirstMI, SecondMI))
    return true;

  if (ST.hasZExtWFusion() && isZExtW(FirstMI, SecondMI))
    return true;

  if (ST.hasShiftedZExtWFusion() && isShiftedZExtW(FirstMI, SecondMI))
    return true;

  if (ST.hasLDADDFusion() && isLDADD(FirstMI, SecondMI))
    return true;

  return false;
}

std::unique_ptr<ScheduleDAGMutation> llvm::createRISCVMacroFusionDAGMutation() {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}
