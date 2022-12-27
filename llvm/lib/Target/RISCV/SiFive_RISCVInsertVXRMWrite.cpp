//===- RISCVInsertVXRMWrite.cpp - Insert WriteVXRM instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts SwapVXRM/WriteVXRM
// instructions where needed.
//
// The pass consists of a single pass over each basic block looking for VXRM
// usage that requires a WriteVXRM to be inserted.
//
// To work with the intrinsics that have SideEffects, current pass always keeps
// the incoming VXRM, and recovers it for MIs which RoundModeOperand is
// specified as RISCVVXRndMode::DYN.
//
// TODO: Future enhancements to this pass will take into account VXRM from
// predecessors.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "riscv-insert-writevxrm"
#define RISCV_INSERT_WRITEVXRM_NAME "RISCV Insert required VXRM values"

namespace {

class RISCVInsertVXRMWrite : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;

public:
  static char ID;

  RISCVInsertVXRMWrite() : MachineFunctionPass(ID) {
    initializeRISCVInsertVXRMWritePass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INSERT_WRITEVXRM_NAME; }

private:
  bool emitWriteVXRM(MachineBasicBlock &MBB);
  Optional<unsigned> getRoundModeIdx(const MCInstrDesc &Desc) const;
};

} // end anonymous namespace

char RISCVInsertVXRMWrite::ID = 0;

INITIALIZE_PASS(RISCVInsertVXRMWrite, DEBUG_TYPE, RISCV_INSERT_WRITEVXRM_NAME,
                false, false)

std::optional<unsigned>
RISCVInsertVXRMWrite::getRoundModeIdx(const MCInstrDesc &Desc) const {
  uint64_t TSFlags = Desc.TSFlags;
  if (!RISCVII::hasRoundModeOp(TSFlags))
    return std::nullopt;

  return RISCVII::getRoundModeOpNum(Desc);
}

bool RISCVInsertVXRMWrite::emitWriteVXRM(MachineBasicBlock &MBB) {
  bool MadeChange = false;

  // To be defensive we keep the incoming VXRM value when round mode is changed
  // from RISCVVXRndMode::DYN
  Register SavedVXRMReg = Register();

  // For now predecessor state is unknown, we use RISCVVXRndMode::DYN to
  // represent the incoming VXRM value
  unsigned CurVXRMImm = RISCVVXRndMode::DYN;
  for (MachineInstr &MI : MBB) {
    if (auto Idx = getRoundModeIdx(MI.getDesc())) {
      MachineOperand &RoundModeOp = MI.getOperand(Idx.value());
      unsigned NewVXRMImm = RoundModeOp.getImm();
      if (NewVXRMImm == CurVXRMImm)
        continue;

      // If the current mode doesn't meet the requirement, we change it
      // accordingly.

      if (CurVXRMImm == RISCVVXRndMode::DYN) {
        assert(!SavedVXRMReg.isValid());
        SavedVXRMReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::SwapVXRMImm),
                SavedVXRMReg)
            .addImm(NewVXRMImm);
        RoundModeOp.setImm(RISCVVXRndMode::DYN);
        MI.addOperand(MachineOperand::CreateReg(RISCV::VXRM, /*isDef*/ false,
                                                /*isImp*/ true));
        CurVXRMImm = NewVXRMImm;
        MadeChange = true;
        continue;
      }

      if (NewVXRMImm != RISCVVXRndMode::DYN) {
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::WriteVXRMImm))
            .addImm(NewVXRMImm);
        RoundModeOp.setImm(RISCVVXRndMode::DYN);
        MI.addOperand(MachineOperand::CreateReg(RISCV::VXRM, /*isDef*/ false,
                                                /*isImp*/ true));
        CurVXRMImm = NewVXRMImm;
        continue;
      }
    }

    if (CurVXRMImm == RISCVVXRndMode::DYN)
      continue;

    if (MI.isCall() || MI.isInlineAsm()) {
      SavedVXRMReg = Register();
      CurVXRMImm = RISCVVXRndMode::DYN;
      continue;
    }

    if (!MI.readsRegister(RISCV::VXRM)) {
      if (MI.modifiesRegister(RISCV::VXRM)) {
        // Note: sifive-dev supports vsetvxrm intrinsic
        // so the state is set to unknown when encountering a write to VXRM
        SavedVXRMReg = Register();
        CurVXRMImm = RISCVVXRndMode::DYN;
      }
      continue;
    }

    // Here handles MIs which read VXRM and need saved VXRM

    assert(SavedVXRMReg.isValid());
    BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::WriteVXRM))
        .addReg(SavedVXRMReg);

    SavedVXRMReg = Register();
    CurVXRMImm = RISCVVXRndMode::DYN;
  }

  // Restore VXRM to previous saved value before leaving the block
  if (SavedVXRMReg.isValid()) {
    MachineInstr &MI = MBB.back();
    if (MI.isTerminator()) {
      BuildMI(MBB, MBB.getFirstTerminator(), DebugLoc(),
              TII->get(RISCV::WriteVXRM))
          .addReg(SavedVXRMReg);
    } else {
      // It is a fallthrough to next block
      BuildMI(&MBB, MI.getDebugLoc(), TII->get(RISCV::WriteVXRM))
          .addReg(SavedVXRMReg);
    }
  }

  return MadeChange;
}

bool RISCVInsertVXRMWrite::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= emitWriteVXRM(MBB);

  return Changed;
}

FunctionPass *llvm::createRISCVInsertVXRMWritePass() {
  return new RISCVInsertVXRMWrite();
}
