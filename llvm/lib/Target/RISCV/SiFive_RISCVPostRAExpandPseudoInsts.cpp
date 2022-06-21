//===-- SiFive_RISCVPostRAExpandPseudoInsts.cpp - Expand pseudo instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands the pseudo instruction pseudolisimm32 into 
// target instructions. This pass should be run during the post-regalloc passes, 
// before assembly emission. It is used when the TunePseudoLISimm32 subfeature
// is on.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"
#include "MCTargetDesc/RISCVMatInt.h"

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define RISCV_POST_RA_EXPAND_PSEUDO_NAME "RISCV post-regalloc pseudo instruction expansion pass"

namespace {

class RISCVPostRAExpandPseudo : public MachineFunctionPass {
public:
  const RISCVInstrInfo *TII;
  static char ID;

  RISCVPostRAExpandPseudo() : MachineFunctionPass(ID) {
    initializeRISCVPostRAExpandPseudoPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return RISCV_POST_RA_EXPAND_PSEUDO_NAME; }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandLIsimm32(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
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
  // RISCVInstrInfo::getInstSizeInBytes expects that the total size of the
  // expanded instructions for each pseudo is correct in the Size field of the
  // tablegen definition for the pseudo.
  //
  // We only handle PseudoLIsimm32
  switch (MBBI->getOpcode()) {
  case RISCV::PseudoLIsimm32:
    return expandLIsimm32(MBB, MBBI);
  default:
    return false;
  }
}

bool RISCVPostRAExpandPseudo::expandLIsimm32(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI) {
  const RISCVSubtarget &Subtarget = MBB.getParent()->getSubtarget<RISCVSubtarget>();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MBBI->getDebugLoc();

  if (!Subtarget.usePseudoLIsimm32() || Subtarget.hasLUIADDIFusion())
    return false;

  int64_t Val = MI.getOperand(1).getImm();
  assert(isInt<32>(Val) && "Unexpected immediate");

  RISCVMatInt::InstSeq Seq =
      RISCVMatInt::generateInstSeq(Val, 
        MBB.getParent()->getSubtarget().getFeatureBits());
  assert(!Seq.empty());

  Register SrcReg = RISCV::X0;
  Register DstReg = MI.getOperand(0).getReg();
  bool DstIsDead = MI.getOperand(0).isDead();
  uint64_t RenamableState =
      MI.getOperand(0).isRenamable() ? RegState::Renamable : 0;
  unsigned Num = 0;

  for (RISCVMatInt::Inst &Inst : Seq) {
    bool LastItem = ++Num == Seq.size();
    if (Inst.Opc == RISCV::LUI) {
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::LUI))
          .addReg(DstReg, RegState::Define |
                  getDeadRegState(DstIsDead && LastItem) |
                  RenamableState)
          .addImm(Inst.Imm);
    } else if (Inst.Opc == RISCV::ADD_UW) {
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD_UW))
          .addReg(DstReg, RegState::Define |
                  getDeadRegState(DstIsDead && LastItem) |
                  RenamableState)
          .addReg(SrcReg, RegState::Kill)
          .addReg(RISCV::X0);
    } else {
      BuildMI(MBB, MBBI, DL, TII->get(Inst.Opc))
          .addReg(DstReg, RegState::Define |
                  getDeadRegState(DstIsDead && LastItem) |
                  RenamableState)
          .addReg(SrcReg, RegState::Kill)
          .addImm(Inst.Imm);
    }
    // Only the first instruction has X0 as its source.
    SrcReg = DstReg;
  }
  MI.eraseFromParent();
  return true;
}

} // end of anonymous namespace

INITIALIZE_PASS(RISCVPostRAExpandPseudo, "riscv-expand-pseudolisimm32",
                RISCV_POST_RA_EXPAND_PSEUDO_NAME, false, false)
namespace llvm {

FunctionPass *createRISCVPostRAExpandPseudoPass() { return new RISCVPostRAExpandPseudo(); }

} // end of namespace llvm
