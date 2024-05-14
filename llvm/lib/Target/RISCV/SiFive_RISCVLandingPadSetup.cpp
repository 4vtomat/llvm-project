//===----- SiFive_RISCVLandingPadSetup.cpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a RISC-V pass to setup landing pad labels for indirect jumps.
// Currently it is only supported fixed labels.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-lpad-setup"
#define PASS_NAME "RISC-V Landing Pad Setup"

namespace {

class RISCVLandingPadSetup : public MachineFunctionPass {
  static constexpr int FixedLabel = 1;

public:
  static char ID;

  RISCVLandingPadSetup() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

bool RISCVLandingPadSetup::runOnMachineFunction(MachineFunction &MF) {
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo &TII = *STI.getInstrInfo();

  if (!STI.hasStdExtZicfilp())
    return false;

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (!MI.isIndirectBranch() &&
          MI.getOpcode() != RISCV::PseudoCALLIndirectNonX7 &&
          MI.getOpcode() != RISCV::PseudoTAILIndirectNonX7)
        continue;
      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(RISCV::LUI), RISCV::X7)
          .addImm(FixedLabel);
      MachineInstrBuilder(MF, &MI).addUse(RISCV::X7, RegState::ImplicitKill);
      Changed = true;
    }

  return Changed;
}

INITIALIZE_PASS(RISCVLandingPadSetup, DEBUG_TYPE, PASS_NAME, false, false)

char RISCVLandingPadSetup::ID = 0;

FunctionPass *llvm::createRISCVLandingPadSetupPass() {
  return new RISCVLandingPadSetup();
}
