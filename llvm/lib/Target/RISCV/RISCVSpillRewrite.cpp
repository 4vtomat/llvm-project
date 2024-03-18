//===--------- RISCVSpillRewrite.cpp - RISC-V Spill Rewrite ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that rewrite spills and reloads to
// reduce the instruction latency by changing full register
// store/load(VS1R/VL1R) to fractional store/load(VSE/VLE) needed and expands.
//
// This pass consists of 3 phases:
//
// Phase 1 finds and rewrites spills(VS1R) to VSE if the spilled vreg only need
// fraction of a vreg(determined by the last write instruction's LMUL). One
// important note is that the tail policy matters, e.g. if the instruction uses
// MF2 we only need to save MF2 of a vreg.
//
// Phase 2 rewrites reloads(VL1R) to VLE follows the corresponding spills in the
// spill slots.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-spill-rewrite"
#define RISCV_SPILL_REWRITE_NAME "RISC-V Spill Rewrite pass"

namespace {
static inline bool mayBePartialSpillInst(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::VS1R_V;
}

static inline bool isSpillInst(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::VS1R_V || MI.getOpcode() == RISCV::VS2R_V ||
         MI.getOpcode() == RISCV::VS4R_V || MI.getOpcode() == RISCV::VS8R_V;
}

static inline bool isReloadInst(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::VL1RE8_V ||
         MI.getOpcode() == RISCV::VL2RE8_V ||
         MI.getOpcode() == RISCV::VL4RE8_V || MI.getOpcode() == RISCV::VL8RE8_V;
}

static inline bool hasSpillSlotObject(const MachineFrameInfo *MFI,
                                      const MachineInstr &MI,
                                      bool IsReload = false) {
  unsigned MemOpIdx = IsReload ? 2 : 1;
  if (MI.getNumOperands() < (MemOpIdx + 1) || !MI.getOperand(MemOpIdx).isFI())
    return false;

  int FI = MI.getOperand(MemOpIdx).getIndex();
  return MFI->isSpillSlotObjectIndex(FI);
}

static inline RISCVII::VLMUL maxLMUL(RISCVII::VLMUL LMUL1,
                                     RISCVII::VLMUL LMUL2) {
  static std::array<int8_t, 8> Order = {3, 4, 5, 6, -1, 0, 1, 2};
  return Order[LMUL1] > Order[LMUL2] ? LMUL1 : LMUL2;
}

static inline RISCVII::VLMUL getWidenedFracLMUL(RISCVII::VLMUL LMUL) {
  if (LMUL == RISCVII::LMUL_F8)
    return RISCVII::LMUL_F4;
  if (LMUL == RISCVII::LMUL_F4)
    return RISCVII::LMUL_F2;
  if (LMUL == RISCVII::LMUL_F2)
    return RISCVII::LMUL_1;

  llvm_unreachable("The LMUL is supposed to be fractional.");
}

class RISCVSpillRewrite : public MachineFunctionPass {
  const RISCVSubtarget *ST = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineFrameInfo *MFI = nullptr;
  LiveIntervals *LIS = nullptr;

public:
  static char ID;
  RISCVSpillRewrite() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return RISCV_SPILL_REWRITE_NAME; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool tryToRewrite(MachineFunction &MF);

  RISCVII::VLMUL
  findDefiningInstUnionLMUL(MachineBasicBlock &MBB, Register Reg,
                            MachineBasicBlock::reverse_iterator BegI = nullptr);
  bool
  tryToRewriteSpill(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    std::map<int, SmallVector<RISCVII::VLMUL, 4>> &SpillLMULs);
  bool tryToRewriteReload(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator I, int FI,
      const std::map<int, SmallVector<RISCVII::VLMUL, 4>> &SpillLMULs);
};

} // end anonymous namespace

char RISCVSpillRewrite::ID = 0;

INITIALIZE_PASS(RISCVSpillRewrite, DEBUG_TYPE, RISCV_SPILL_REWRITE_NAME, false,
                false)

void RISCVSpillRewrite::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();

  AU.addPreserved<LiveIntervals>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<LiveDebugVariables>();
  AU.addPreserved<LiveStacks>();

  MachineFunctionPass::getAnalysisUsage(AU);
}

RISCVII::VLMUL RISCVSpillRewrite::findDefiningInstUnionLMUL(
    MachineBasicBlock &MBB, Register Reg,
    MachineBasicBlock::reverse_iterator BegI) {
  for (auto I = (BegI == nullptr ? MBB.rbegin() : BegI); I != MBB.rend(); ++I) {
    if (I->isDebugInstr())
      continue;

    if (I->definesRegister(Reg)) {
      if (I->registerDefIsDead(Reg))
        return RISCVII::LMUL_RESERVED;

      if (isReloadInst(*I))
        return RISCVII::LMUL_1;

      if (auto DstSrcPair = TII->isCopyInstr(*I))
        return findDefiningInstUnionLMUL(MBB, DstSrcPair->Source->getReg(), *I);

      const uint64_t TSFlags = I->getDesc().TSFlags;
      if (RISCVII::hasSEWOp(TSFlags)) {
        RISCVII::VLMUL LMUL = RISCVII::getLMul(TSFlags);
        if (RISCVII::isWiden(TSFlags))
          LMUL = getWidenedFracLMUL(LMUL);

        return LMUL;
      }

      llvm_unreachable("RVV instructions should have LMUL.");
    }
  }

  // If Reg's defining inst is not found in this BB, find it in it's
  // predecessors.
  RISCVII::VLMUL LMUL = RISCVII::LMUL_RESERVED;
  for (MachineBasicBlock *P : MBB.predecessors()) {
    RISCVII::VLMUL PredLMUL = findDefiningInstUnionLMUL(*P, Reg);
    if (PredLMUL == RISCVII::LMUL_RESERVED)
      continue;

    if (LMUL == RISCVII::LMUL_RESERVED) {
      LMUL = PredLMUL;
      continue;
    }

    LMUL = maxLMUL(LMUL, PredLMUL);
  }

  return LMUL;
}

bool RISCVSpillRewrite::tryToRewriteSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
    std::map<int, SmallVector<RISCVII::VLMUL, 4>> &SpillLMULs) {
  Register SrcReg = I->getOperand(0).getReg();
  unsigned Opcode = 0;
  // Find the nearest inst defines this spilled reg.
  RISCVII::VLMUL LMUL = findDefiningInstUnionLMUL(MBB, SrcReg, *I);
  // If the register's defined inst just uses partial of register, we only
  // need to store partial register.
  switch (LMUL) {
  case RISCVII::LMUL_F2:
    Opcode = RISCV::PseudoVSE8_V_MF2;
    break;
  case RISCVII::LMUL_F4:
    Opcode = RISCV::PseudoVSE8_V_MF4;
    break;
  case RISCVII::LMUL_F8:
    Opcode = RISCV::PseudoVSE8_V_MF8;
    break;
  default:
    break;
  }

  // No need to rewrite.
  if (!Opcode)
    return false;

  int FI = I->getOperand(1).getIndex();
  auto &LMULs = SpillLMULs[FI];

  if (Opcode == RISCV::PseudoVSE8_V_MF2)
    LMULs.push_back(RISCVII::LMUL_F2);
  else if (Opcode == RISCV::PseudoVSE8_V_MF4)
    LMULs.push_back(RISCVII::LMUL_F4);
  else if (Opcode == RISCV::PseudoVSE8_V_MF8)
    LMULs.push_back(RISCVII::LMUL_F8);

  MachineInstr *Vse = BuildMI(MBB, I, DebugLoc(), TII->get(Opcode))
                          .add(I->getOperand(0))
                          .addFrameIndex(FI)
                          .addImm(-1 /*VL Max*/)
                          .addImm(3 /*SEW = 8*/);
  LIS->InsertMachineInstrInMaps(*Vse);

  return true;
}

bool RISCVSpillRewrite::tryToRewriteReload(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, int FI,
    const std::map<int, SmallVector<RISCVII::VLMUL, 4>> &SpillLMULs) {
  // Partial reload case
  // If this frame doesn't have corresponding reload op, just skip it.
  if (!SpillLMULs.count(FI))
    return false;

  // Find the largest lmul in all of spills.
  RISCVII::VLMUL LMUL = RISCVII::LMUL_RESERVED;
  llvm::for_each(SpillLMULs.at(FI),
                 [&](RISCVII::VLMUL L) { LMUL = maxLMUL(LMUL, L); });

  unsigned Opcode = 0;
  switch (LMUL) {
  case RISCVII::LMUL_F2:
    Opcode = RISCV::PseudoVLE8_V_MF2;
    break;
  case RISCVII::LMUL_F4:
    Opcode = RISCV::PseudoVLE8_V_MF4;
    break;
  case RISCVII::LMUL_F8:
    Opcode = RISCV::PseudoVLE8_V_MF8;
    break;
  default:
    break;
  }

  if (!Opcode)
    return false;

  MachineInstr *Vle = BuildMI(MBB, I, I->getDebugLoc(), TII->get(Opcode))
                          .addReg(I->getOperand(0).getReg(),
                                  RegState::Define | RegState::Renamable)
                          .addReg(I->getOperand(0).getReg(),
                                  RegState::Undef | RegState::Renamable)
                          .addFrameIndex(FI)
                          .addImm(-1 /*VL Max*/)
                          .addImm(3 /*SEW = 8*/)
                          .addImm(3 /*TAMA*/);
  LIS->InsertMachineInstrInMaps(*Vle);

  return true;
}

bool RISCVSpillRewrite::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  // Skip if the vector extension is not enabled.
  ST = &MF.getSubtarget<RISCVSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  MFI = &MF.getFrameInfo();
  LIS = &getAnalysis<LiveIntervals>();

  return tryToRewrite(MF);
}

bool RISCVSpillRewrite::tryToRewrite(MachineFunction &MF) {
  // 1. If the MI is vector spill and need partial spill, record frame number
  // and the corresponding LMUL in SpillLMULs.
  // 2. If the MI is vector reload, change it's load instruction if found in
  // SpillLMULs.
  // Note that this pass is run before stack slot coloring pass, so it doesn't
  // need to consider stack slot reuse.
  bool Changed = false;
  std::map<int, SmallVector<RISCVII::VLMUL, 4>> SpillLMULs;
  for (MachineBasicBlock &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      int FI;
      if (hasSpillSlotObject(MFI, *MI))
        FI = MI->getOperand(1).getIndex();
      else if (hasSpillSlotObject(MFI, *MI, true))
        FI = MI->getOperand(2).getIndex();
      else
        continue;

      MachineBasicBlock::iterator NextInst = std::next(MI);
      if (mayBePartialSpillInst(*MI) &&
          tryToRewriteSpill(MBB, MI, SpillLMULs)) {
        MI->removeFromParent();
        MI = NextInst;
        Changed = true;
      } else if (tryToRewriteReload(MBB, MI, FI, SpillLMULs)) {
        MI->removeFromParent();
        MI = NextInst;
        Changed = true;
      }

      if (MI == MBB.end())
        break;
    }
  }

  return Changed;
}

/// Returns an instance of the RVV Spill Rewrite pass.
FunctionPass *llvm::createRISCVSpillRewritePass() {
  return new RISCVSpillRewrite();
}
