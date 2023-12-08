//===-- SiFive_RISCVPostRAExpandPseudoInsts.cpp - Expand pseudo instrs ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands the pseudo instruction pseudolisimm32
// into target instructions. This pass should be run during the post-regalloc
// passes, before assembly emission. It is used when the TunePseudoLISimm32
// subfeature is on.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define RISCV_POST_RA_EXPAND_PSEUDO_NAME                                       \
  "RISC-V post-regalloc pseudo instruction expansion pass"

namespace {

class RISCVPostRAExpandPseudo : public MachineFunctionPass {
public:
  const RISCVInstrInfo *TII;
  static char ID;

  RISCVPostRAExpandPseudo() : MachineFunctionPass(ID) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return RISCV_POST_RA_EXPAND_PSEUDO_NAME;
  }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandLIsimm32(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool expandLIaddr(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
};

char RISCVPostRAExpandPseudo::ID = 0;

bool RISCVPostRAExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  TII = static_cast<const RISCVInstrInfo *>(MF.getSubtarget().getInstrInfo());
  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);
  return Modified;
}

bool RISCVPostRAExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool RISCVPostRAExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI,
                                       MachineBasicBlock::iterator &NextMBBI) {
  switch (MBBI->getOpcode()) {
  case RISCV::PseudoLIsimm32:
    return expandLIsimm32(MBB, MBBI);
  case RISCV::PseudoLIaddr:
    return expandLIaddr(MBB, MBBI);
  default:
    return false;
  }
}

bool RISCVPostRAExpandPseudo::expandLIsimm32(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MBBI) {
  const RISCVSubtarget &Subtarget =
      MBB.getParent()->getSubtarget<RISCVSubtarget>();

  if (!Subtarget.usePseudoLIsimm32() || Subtarget.hasLUIADDIFusion())
    return false;

  TII->expandLIsimm32(MBB, MBBI);
  return true;
}

bool RISCVPostRAExpandPseudo::expandLIaddr(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MBBI) {
  TII->expandLIaddr(MBB, MBBI);
  return true;
}

} // end of anonymous namespace

INITIALIZE_PASS(RISCVPostRAExpandPseudo, "riscv-expand-pseudolisimm32",
                RISCV_POST_RA_EXPAND_PSEUDO_NAME, false, false)
namespace llvm {

FunctionPass *createRISCVPostRAExpandPseudoPass() {
  return new RISCVPostRAExpandPseudo();
}

} // end of namespace llvm
