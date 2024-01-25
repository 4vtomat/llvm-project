//===- RISCVMIPeepholeOpt.cpp - RISCV MI peephole optimization pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The only peephole currently supported combines SLLI+ADD to SHxADD.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-peephole"
#define RISCV_PEEPHOLE_NAME "RISC-V MI Peephole"

namespace {

class RISCVPeephole : public MachineFunctionPass {
  const RISCVInstrInfo *TII;
  MachineRegisterInfo *MRI;

public:
  static char ID;

  RISCVPeephole() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_PEEPHOLE_NAME; }

private:
  bool tryFormSHXADD(MachineInstr &MI, unsigned OpIdx1, unsigned OpIdx2);
};

} // end anonymous namespace

char RISCVPeephole::ID = 0;
INITIALIZE_PASS(RISCVPeephole, DEBUG_TYPE, RISCV_PEEPHOLE_NAME, false, false)

FunctionPass *llvm::createRISCVPeepholePass() {
  return new RISCVPeephole();
}

// Try to form SHxADD or SHxADD_UW instructions from SLLI(_UW)+ADD.
bool RISCVPeephole::tryFormSHXADD(MachineInstr &MI, unsigned OpIdx1, unsigned OpIdx2) {
  Register DstReg  = MI.getOperand(0).getReg();
  Register SrcReg1 = MI.getOperand(OpIdx1).getReg();
  Register SrcReg2 = MI.getOperand(OpIdx2).getReg();

  if (!SrcReg1.isVirtual() || !MRI->hasOneNonDBGUse(SrcReg1))
    return false;

  MachineInstr *SrcMI = MRI->getVRegDef(SrcReg1);

  bool IsSLLI_UW = SrcMI->getOpcode() == RISCV::SLLI_UW;
  if (!IsSLLI_UW && SrcMI->getOpcode() != RISCV::SLLI)
    return false;

  unsigned ShAmt = SrcMI->getOperand(2).getImm();
  unsigned Opc;
  switch (ShAmt) {
  default:
    return false;
  case 1:
    Opc = IsSLLI_UW ? RISCV::SH1ADD_UW : RISCV::SH1ADD;
    break;
  case 2:
    Opc = IsSLLI_UW ? RISCV::SH2ADD_UW : RISCV::SH2ADD;
    break;
  case 3:
    Opc = IsSLLI_UW ? RISCV::SH3ADD_UW : RISCV::SH3ADD;
    break;
  }

  Register ShiftedReg = SrcMI->getOperand(1).getReg();

  BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), TII->get(Opc), DstReg)
          .addReg(ShiftedReg)
          .addReg(SrcReg2);
  MRI->clearKillFlags(ShiftedReg);
  MI.eraseFromParent();
  SrcMI->eraseFromParent();
  return true;
}

bool RISCVPeephole::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  TII = ST.getInstrInfo();

  // The only optimization currently supported is to form SHxADD instructions.
  if (!ST.hasStdExtZba())
    return false;

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.getOpcode() != RISCV::ADD)
        continue;

      // Add has two source operands. Try both.
      MadeChange = tryFormSHXADD(MI, 2, 1) || tryFormSHXADD(MI, 1, 2);
    }
  }

  return MadeChange;
}
