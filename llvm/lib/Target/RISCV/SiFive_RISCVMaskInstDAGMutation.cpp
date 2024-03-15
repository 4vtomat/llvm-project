//=== SiFive_RISCVMaskInstDAGMutation.cpp - RISC-V Mask Inst DAG Mutation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the RISC-V implementation of the DAG scheduling
/// mutation to move mask instructions away from their uses so the latency
/// of the arithmetic pipeline is not exposted to the subsequent dependent
/// instruction.
//
//===----------------------------------------------------------------------===//

#include "SiFive_RISCVMaskInstDAGMutation.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/ScheduleDAG.h"

using namespace llvm;

static bool isMaskInstr(MachineInstr &MI) {
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI.getOpcode());

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

static bool isVectorInstr(MachineInstr &MI) {
  return RISCVVPseudosTable::getPseudoInfo(MI.getOpcode());
}

/// Hoist mask instructions away from there uses by adding weak edges (i.e. my
/// be violated by the scheduling strategy) in the DAG.
void RISCVMaskInstDAGMutation::apply(ScheduleDAGInstrs *DAG) {
  SmallSet<MachineInstr *, 4> MaskInstrs;
  for (MachineInstr &MI : *DAG)
    if (isMaskInstr(MI))
      MaskInstrs.insert(&MI);

  // Add edges from the mask instruction to a limited number of vector
  // instructions. The assumption is that the mask instruction only needs to be
  // hoisted a few vector instructions above to hide the latency. This allows
  // for a balance between hiding the latency but also respecting the original
  // instruction ordering.
  int MaxNumVecEdges = 3;
  int NumVecEdges = 0;

  for (MachineInstr *MaskMI : MaskInstrs) {
    // Only want to add edges to instructions before MaskMI, starting from the
    // instruction closest to MaskMI.
    auto R = llvm::reverse(*DAG);
    auto It = llvm::find_if(R, [&MaskMI](auto &MI) { return &MI == MaskMI; });
    if (It != R.end())
      ++It;

    for (; It != R.end(); ++It) {
      if (NumVecEdges >= MaxNumVecEdges)
        break;

      MachineInstr &MI = *It;
      SUnit *SU = DAG->getSUnit(&MI);
      SUnit *MaskSU = DAG->getSUnit(MaskMI);

      if (SU == MaskSU)
        continue;

      if (DAG->canAddEdge(SU, MaskSU)) {
        DAG->addEdge(SU, SDep(MaskSU, SDep::OrderKind::Weak));

        if (isVectorInstr(MI))
          NumVecEdges++;
      }
    }
  }
}

std::unique_ptr<ScheduleDAGMutation> llvm::createRISCVMaskInstDAGMutation() {
  return std::make_unique<RISCVMaskInstDAGMutation>();
}
