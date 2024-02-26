//===-- RISCVHazardRecognizer.cpp - RISC-V Hazard Recognizers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SiFive_RISCVHazardRecognizer.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/ScheduleDAG.h"

using namespace llvm;

static bool isMaskInstruction(unsigned Opcode) {
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(Opcode);

  if (!RVV)
    return false;

  switch (RVV->BaseInstr) {
  default:
    return false;
  case RISCV::VMAND_MM:
  case RISCV::VMNAND_MM:
  case RISCV::VMANDN_MM:
  case RISCV::VMXOR_MM:
  case RISCV::VMOR_MM:
  case RISCV::VMNOR_MM:
  case RISCV::VMORN_MM:
  case RISCV::VMXNOR_MM:
  case RISCV::VMSBF_M:
  case RISCV::VMSIF_M:
  case RISCV::VMSOF_M:
  case RISCV::VMSEQ_VV:
  case RISCV::VMSEQ_VI:
  case RISCV::VMSEQ_VX:
  case RISCV::VMSNE_VV:
  case RISCV::VMSNE_VI:
  case RISCV::VMSNE_VX:
  case RISCV::VMSLTU_VV:
  case RISCV::VMSLTU_VX:
  case RISCV::VMSLT_VV:
  case RISCV::VMSLT_VX:
  case RISCV::VMSLEU_VV:
  case RISCV::VMSLEU_VI:
  case RISCV::VMSLEU_VX:
  case RISCV::VMSLE_VV:
  case RISCV::VMSLE_VI:
  case RISCV::VMSLE_VX:
  case RISCV::VMSGTU_VI:
  case RISCV::VMSGTU_VX:
  case RISCV::VMFEQ_VF:
  case RISCV::VMFEQ_VV:
  case RISCV::VMFNE_VF:
  case RISCV::VMFNE_VV:
  case RISCV::VMFLT_VF:
  case RISCV::VMFLT_VV:
  case RISCV::VMFLE_VF:
  case RISCV::VMFLE_VV:
  case RISCV::VMFGT_VF:
  case RISCV::VMFGE_VF:
    return true;
  }
}

ScheduleHazardRecognizer::HazardType
RISCVMaskInstrHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  if (!LastSU)
    return NoHazard;

  MachineInstr *MI = SU->getInstr();
  if (!isMaskInstruction(MI->getOpcode()))
    return NoHazard;

  // The hazard exists when the Mask instruction is next to one of its users.
  Register Def = MI->getOperand(0).getReg();
  if (llvm::all_of(LastSU->getInstr()->uses(),
                   [&Def](auto &U) { return U.isReg() && U.getReg() != Def; }))
    return NoHazard;

  // This is a bailout. It would be better if the mask instruction wasn't next
  // to its user, but after 8 cycles and no better candidate is found, we should
  // give up.
  // TODO: Determine the bailout number of cycles based on local instructions.
  if (CyclesSinceLastSU > 8)
    return NoHazard;

  // Check to see if the mask instruction is next to its user.
  if (SU->isSucc(LastSU))
    return Hazard;

  return NoHazard;
}

void RISCVMaskInstrHazardRecognizer::Reset() {
  LastSU = nullptr;
  CyclesSinceLastSU = 0;
}

void RISCVMaskInstrHazardRecognizer::EmitInstruction(SUnit *SU) {
  MachineInstr *MI = SU->getInstr();
  if (!MI->isDebugInstr()) {
    LastSU = SU;
    CyclesSinceLastSU = 0;
  }
}

void RISCVMaskInstrHazardRecognizer::AdvanceCycle() {
  CyclesSinceLastSU++;
}

void RISCVMaskInstrHazardRecognizer::RecedeCycle() {
  CyclesSinceLastSU++;
}
