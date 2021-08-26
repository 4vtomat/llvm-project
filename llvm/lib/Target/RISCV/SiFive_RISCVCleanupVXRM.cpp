#include "RISCV.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-cleanup-vxrm"
#define RISCV_CLEANUP_VXRM_NAME "RISCV Cleanup VXRM pass"

namespace {

class RISCVCleanupVXRM : public MachineFunctionPass {
public:
  static char ID;

  RISCVCleanupVXRM() : MachineFunctionPass(ID) {
    initializeRISCVCleanupVXRMPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  bool runOnMachineBasicBlock(MachineBasicBlock &MBB);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_CLEANUP_VXRM_NAME; }

private:
};

static inline bool isWriteVXRM(MachineInstr &MI) {
  return MI.getOpcode() == RISCV::WriteVXRM;
}

static inline bool isReadVXRM(MachineInstr &MI) {
  return MI.getOpcode() == RISCV::ReadVXRM;
}

} // end anonymous namespace
char RISCVCleanupVXRM::ID = 0;

INITIALIZE_PASS(RISCVCleanupVXRM, DEBUG_TYPE, RISCV_CLEANUP_VXRM_NAME, false,
                false)

bool RISCVCleanupVXRM::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= runOnMachineBasicBlock(MBB);

  return Changed;
}

bool RISCVCleanupVXRM::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  bool Updated = false;

  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  do {
    Updated = false;
    MachineInstr *PrevAccess = nullptr;
    MachineInstr *LastWrite = nullptr;
    for (auto MII = MBB.begin(), MIE = MBB.end(); MII != MIE;) {
      MachineInstr &MI = *MII++;
      if (!isWriteVXRM(MI) && !isReadVXRM(MI)) {
        if (MI.modifiesRegister(RISCV::VXRM) || MI.isCall() ||
            MI.isInlineAsm()) {
          PrevAccess = nullptr;
          LastWrite = nullptr;
        } else if (MI.readsRegister(RISCV::VXRM)) {
          PrevAccess = nullptr;
        }
        continue;
      }

      if (isWriteVXRM(MI)) {
        if (LastWrite) {
          Register LastSrc = LastWrite->getOperand(0).getReg();
          Register CurSrc = MI.getOperand(0).getReg();
          if (Register::isVirtualRegister(CurSrc) && LastSrc == CurSrc) {
            // remove the write if it is writing the same value
            LLVM_DEBUG(dbgs() << "Remove WriteVXRM that uses the same source as previous WriteVXRM:"; MI.dump());
            MI.eraseFromParent();
            Updated = true;
            continue;
          }
        }
        LastWrite = &MI;
      }

      if (!PrevAccess) {
        PrevAccess = &MI;
        continue;
      }

      if (isReadVXRM(MI)) {
        // it is either Read-Read or Write-Read,
        // therefore we forward the previous value, and then eliminate this
        SmallVector<MachineOperand *, 16> Uses;
        MachineOperand &CurMode = MI.getOperand(0);
        for (auto &MO : MRI.use_nodbg_operands(CurMode.getReg())) {
          Uses.push_back(&MO);
        }
        MachineOperand &PrevMode = PrevAccess->getOperand(0);
        if (!Uses.empty()) {
          if (PrevMode.isUse())
            PrevMode.setIsKill(false);
          else
            PrevMode.setIsDead(false);
        }
        for (auto *MO : Uses) {
          MO->setReg(PrevMode.getReg());
        }
        LLVM_DEBUG(dbgs() << "Remove ReadVXRM that can reuse previous result:"; MI.dump());
        MI.eraseFromParent();

        Updated = true;
        continue;
      }

      if (isWriteVXRM(MI) && isWriteVXRM(*PrevAccess)) {
        // no reads after previous write, then eliminate it
        LLVM_DEBUG(dbgs() << "Remove WriteVXRM that has no users:";
                   PrevAccess->dump());
        PrevAccess->eraseFromParent();
        PrevAccess = &MI;
        Updated = true;
        continue;
      }

      // update PrevAccess
      PrevAccess = &MI;
    }
    if (Updated)
      Changed = true;
  } while (Updated);
  return Changed;
}

/// Returns an instance of the CleanupVXRM pass.
FunctionPass *llvm::createRISCVCleanupVXRMPass() {
  return new RISCVCleanupVXRM();
}
