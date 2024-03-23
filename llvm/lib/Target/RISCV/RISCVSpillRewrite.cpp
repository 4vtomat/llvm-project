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
// (MF2, TA) we only need to save MF2 of a vreg. However if it uses (MF2, TU) we
// have to save the full vreg content.
//
// Phase 2 rewrites reloads(VL1R) to VLE follows the corresponding spills in the
// spill slots.
//
// Phase 3 fixups the VSETVLI that might be violated during rewrites.
//
//===----------------------------------------------------------------------===//

#include "../lib/CodeGen/LiveDebugVariables.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-spill-rewrite"
#define RISCV_SPILL_REWRITE_NAME "RISC-V Spill Rewrite pass"

namespace {
static inline bool isVectorConfigInst(const MachineInstr &MI) {
  unsigned Op = MI.getOpcode();
  return Op == RISCV::PseudoVSETVLI || Op == RISCV::PseudoVSETIVLI ||
         Op == RISCV::PseudoVSETVLIX0;
}

static inline bool isRVVInst(const MachineInstr &MI) {
  return RISCVII::hasSEWOp(MI.getDesc().TSFlags);
}

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

static inline bool isPartialSpillInst(const MachineFrameInfo *MFI,
                                      const MachineInstr &MI) {
  if (!hasSpillSlotObject(MFI, MI) ||
      (MI.getOpcode() != RISCV::PseudoVSE8_V_MF2 &&
       MI.getOpcode() != RISCV::PseudoVSE16_V_MF2 &&
       MI.getOpcode() != RISCV::PseudoVSE32_V_MF2 &&
       MI.getOpcode() != RISCV::PseudoVSE8_V_MF4 &&
       MI.getOpcode() != RISCV::PseudoVSE16_V_MF4 &&
       MI.getOpcode() != RISCV::PseudoVSE8_V_MF8))
    return false;
  return true;
}

static inline bool isPartialReloadInst(const MachineFrameInfo *MFI,
                                       const MachineInstr &MI) {
  if (!hasSpillSlotObject(MFI, MI, true) ||
      (MI.getOpcode() != RISCV::PseudoVLE8_V_MF2 &&
       MI.getOpcode() != RISCV::PseudoVLE16_V_MF2 &&
       MI.getOpcode() != RISCV::PseudoVLE32_V_MF2 &&
       MI.getOpcode() != RISCV::PseudoVLE8_V_MF4 &&
       MI.getOpcode() != RISCV::PseudoVLE16_V_MF4 &&
       MI.getOpcode() != RISCV::PseudoVLE8_V_MF8))
    return false;
  return true;
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
  void rewrite(MachineFunction &MF);
  void fixup(MachineFunction &MF);
  std::pair<MachineBasicBlock::reverse_iterator, bool>
  findFirstVectorConfigInstUpward(MachineBasicBlock::reverse_iterator I,
                                  MachineBasicBlock &MBB);

  bool canInsertVsetvli(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator CurI,
                        bool IsReload = false);
  bool tryToRewriteSpill(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                         std::map<int, unsigned> &RestoreOp);
  bool tryToRewriteReload(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                          int FI, const std::map<int, unsigned> &RestoreOp);
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

bool RISCVSpillRewrite::canInsertVsetvli(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator CurI,
                                         bool IsReload) {
  // First step, find the first vector instruction below the spill we want to
  // rewrite, also its vsetvli.
  MachineBasicBlock::iterator FirstVInst = nullptr, FirstVsetvli = nullptr;
  if (IsReload && isVectorConfigInst(*std::prev(CurI)))
    FirstVsetvli = std::prev(CurI);

  for (auto TempI = CurI; TempI != MBB.end(); ++TempI) {
    if (FirstVInst != nullptr)
      break;

    if (isRVVInst(*TempI))
      FirstVInst = TempI;
    else if (isVectorConfigInst(*TempI))
      FirstVsetvli = TempI;
  }

  // If there is no vector instruction, then we don't need to care about it.
  if (FirstVInst == nullptr)
    return true;

  // If the corresponding vsetvli is not found, just find it in other direction.
  for (auto TempI = CurI;; --TempI) {
    if (isVectorConfigInst(*TempI)) {
      FirstVsetvli = TempI;
      break;
    }

    assert(TempI != MBB.begin() && "Vsetvli can not be found.");
  }

  // Second step, determine whether we can modify the vl or not. If it's vsetvli
  // x0, x0..., we can't modify it because this instruction makes the vl
  // unchange, we shouldn't break that.
  if (FirstVInst != nullptr && FirstVsetvli != nullptr &&
      !(FirstVsetvli->getOperand(0).isReg() &&
        FirstVsetvli->getOperand(0).getReg() == RISCV::X0 &&
        FirstVsetvli->getOperand(1).isReg() &&
        FirstVsetvli->getOperand(1).getReg() == RISCV::X0))
    return true;
  return false;
}

// clang-format off
// Since the rewritten spills will have an addition vsetvli, so it should make
// sure it doesn't violate the original vl state. For example, there are 2 cases
// for rewritable and unrewritable correspondingly:
// case 1(rewritable):
// vsetvli a0, a1, e8, mf2, ta
// vadd v8, v8, v9
// vs1r v8, (spill_slot)   <- since the vsub below set its vl, so it's fine to rewrite this spill. vsetvli a0, a2, e8, m1,
// ta vsub v8, v8, v9
//             |
//             v
// vsetvli a0, a1, e8, mf2, ta
// vadd v8, v8, v9
// vsetvli a0, x0, e8, mf2, ta
// vse8 v8, (spill_slot)
// vsetvli a0, a2, e8, m1, ta
// vsub v8, v8, v9
//
//
// case 2(unrewritable):
// vsetvli a0, a1, e8, mf2, ta
// vadd v8, v8, v9
// vs1r v8, (spill_slot)   <- since the vsub below reuse the vl, we can't rewrite since the new inserted vsetvli could violate the original vl.
// vsetvli x0, x0, e16, m1, ta
// vsub v8, v8, v9
// TODO:: handle widen instructions
// clang-format on
bool RISCVSpillRewrite::tryToRewriteSpill(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I,
                                          std::map<int, unsigned> &RestoreOp) {
  Register SrcReg = I->getOperand(0).getReg();
  int FI = I->getOperand(1).getIndex();
  unsigned Opcode = 0;
  bool NeedToModifyVsetvli = false;
  unsigned SEW;
  // Find the nearest inst defines this spilled reg, also trace that if there is
  // any vsetvli between the define instruction and the spill instruction to
  // check if we need to insert an vsetvli before the new spill instruction.
  for (MachineBasicBlock::iterator TempI = I;; --TempI) {
    // Skip the debug insts.
    if (TempI->isDebugInstr() || !TempI->definesRegister(SrcReg)) {
      if (TempI == MBB.begin())
        return false;
      if (isVectorConfigInst(*TempI))
        NeedToModifyVsetvli = true;
      continue;
    }

    MCInstrDesc const &Desc = TempI->getDesc();
    const uint64_t TSFlags = Desc.TSFlags;
    // If the register's defined inst is tail agnostic, we only need to store
    // partial register.
    if (RISCVII::hasVecPolicyOp(TSFlags) &&
        (TempI->getOperand(RISCVII::getVecPolicyOpNum(Desc)).getImm() &
         RISCVII::TAIL_AGNOSTIC)) {
      SEW = TempI->getOperand(RISCVII::getSEWOpNum(Desc)).getImm();
      switch (RISCVII::getLMul(TSFlags)) {
      case RISCVII::LMUL_F2:
        if (SEW == 3)
          Opcode = RISCV::PseudoVSE8_V_MF2;
        else if (SEW == 4)
          Opcode = RISCV::PseudoVSE16_V_MF2;
        else if (SEW == 5)
          Opcode = RISCV::PseudoVSE32_V_MF2;
        break;
      case RISCVII::LMUL_F4:
        if (SEW == 3)
          Opcode = RISCV::PseudoVSE8_V_MF4;
        else if (SEW == 4)
          Opcode = RISCV::PseudoVSE16_V_MF4;
        break;
      case RISCVII::LMUL_F8:
        if (SEW == 3)
          Opcode = RISCV::PseudoVSE8_V_MF8;
        break;
      default:
        break;
      }
    }
    break;
  }

  if (!Opcode)
    return false;

  if (NeedToModifyVsetvli && !canInsertVsetvli(MBB, I))
    return false;

  RISCVII::VLMUL VLMUL;
  if (Opcode == RISCV::PseudoVSE8_V_MF2) {
    VLMUL = RISCVII::VLMUL::LMUL_F2;
    RestoreOp[FI] = RISCV::PseudoVLE8_V_MF2;
  } else if (Opcode == RISCV::PseudoVSE16_V_MF2) {
    VLMUL = RISCVII::VLMUL::LMUL_F2;
    RestoreOp[FI] = RISCV::PseudoVLE16_V_MF2;
  } else if (Opcode == RISCV::PseudoVSE32_V_MF2) {
    VLMUL = RISCVII::VLMUL::LMUL_F2;
    RestoreOp[FI] = RISCV::PseudoVLE32_V_MF2;
  } else if (Opcode == RISCV::PseudoVSE8_V_MF4) {
    VLMUL = RISCVII::VLMUL::LMUL_F4;
    RestoreOp[FI] = RISCV::PseudoVLE8_V_MF4;
  } else if (Opcode == RISCV::PseudoVSE16_V_MF4) {
    VLMUL = RISCVII::VLMUL::LMUL_F4;
    RestoreOp[FI] = RISCV::PseudoVLE16_V_MF4;
  } else if (Opcode == RISCV::PseudoVSE8_V_MF8) {
    VLMUL = RISCVII::VLMUL::LMUL_F8;
    RestoreOp[FI] = RISCV::PseudoVLE8_V_MF8;
  }

  if (NeedToModifyVsetvli) {
    unsigned VTypeI = RISCVVType::encodeVTYPE(VLMUL, 1 << SEW, true, true);
    Register NewReg = MRI->createVirtualRegister(&RISCV::GPRNoX0RegClass);
    MachineInstr *Vsetvli =
        BuildMI(MBB, I, DebugLoc(), TII->get(RISCV::PseudoVSETVLIX0))
            .addReg(NewReg, RegState::Define | RegState::Dead)
            .addReg(RISCV::X0)
            .addImm(VTypeI /*VtypeI*/);
    LIS->InsertMachineInstrInMaps(*Vsetvli);
    LIS->createAndComputeVirtRegInterval(NewReg);
  }
  MachineInstr *Vse = BuildMI(MBB, I, DebugLoc(), TII->get(Opcode))
                          .add(I->getOperand(0))
                          .addFrameIndex(FI)
                          .addReg(RISCV::NoRegister)
                          .addImm(SEW)
                          .addUse(RISCV::VL, RegState::Implicit)
                          .addUse(RISCV::VTYPE, RegState::Implicit);
  LIS->InsertMachineInstrInMaps(*Vse);

  return true;
}

bool RISCVSpillRewrite::tryToRewriteReload(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, int FI,
    const std::map<int, unsigned> &RestoreOp) {
  // In rewrite reload case, it always need to insert a vsetvli since it
  // doesn't have the define instruction's one to reuse.
  if (!canInsertVsetvli(MBB, I, true))
    return false;

  RISCVII::VLMUL VLMUL;
  unsigned SEW;
  if (RestoreOp.at(FI) == RISCV::PseudoVLE8_V_MF2) {
    VLMUL = RISCVII::VLMUL::LMUL_F2;
    SEW = 3;
  } else if (RestoreOp.at(FI) == RISCV::PseudoVLE16_V_MF2) {
    VLMUL = RISCVII::VLMUL::LMUL_F2;
    SEW = 4;
  } else if (RestoreOp.at(FI) == RISCV::PseudoVLE32_V_MF2) {
    VLMUL = RISCVII::VLMUL::LMUL_F2;
    SEW = 5;
  } else if (RestoreOp.at(FI) == RISCV::PseudoVLE8_V_MF4) {
    VLMUL = RISCVII::VLMUL::LMUL_F4;
    SEW = 3;
  } else if (RestoreOp.at(FI) == RISCV::PseudoVLE16_V_MF4) {
    VLMUL = RISCVII::VLMUL::LMUL_F4;
    SEW = 4;
  } else if (RestoreOp.at(FI) == RISCV::PseudoVLE8_V_MF8) {
    VLMUL = RISCVII::VLMUL::LMUL_F8;
    SEW = 3;
  }

  unsigned VTypeI = RISCVVType::encodeVTYPE(VLMUL, 8, true, true);
  Register NewReg = MRI->createVirtualRegister(&RISCV::GPRNoX0RegClass);
  MachineInstr *Vsetvli =
      BuildMI(MBB, I, DebugLoc(), TII->get(RISCV::PseudoVSETVLIX0))
          .addReg(NewReg, RegState::Define | RegState::Dead)
          .addReg(RISCV::X0)
          .addImm(VTypeI /*VtypeI*/);

  MachineInstr *Vle =
      BuildMI(MBB, I, I->getDebugLoc(), TII->get(RestoreOp.at(FI)))
          .addReg(I->getOperand(0).getReg(),
                  RegState::Define | RegState::Renamable)
          .addReg(I->getOperand(0).getReg(),
                  RegState::Undef | RegState::Renamable)
          .addFrameIndex(FI)
          .addReg(RISCV::NoRegister)
          .addImm(SEW)
          .addImm(3 /*TAMA*/)
          .addUse(RISCV::VL, RegState::Implicit)
          .addUse(RISCV::VTYPE, RegState::Implicit);
  LIS->InsertMachineInstrInMaps(*Vsetvli);
  LIS->InsertMachineInstrInMaps(*Vle);
  LIS->createAndComputeVirtRegInterval(NewReg);

  return true;
}

bool RISCVSpillRewrite::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  ST = &MF.getSubtarget<RISCVSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  MFI = &MF.getFrameInfo();
  LIS = &getAnalysis<LiveIntervals>();

  rewrite(MF);
  fixup(MF);

  return true;
}

void RISCVSpillRewrite::rewrite(MachineFunction &MF) {
  // 1. If the MI is vector spill and need partial spill, record frame number
  // and the corresponding restore instruction in RestoreOp
  // 2. If the MI is vector reload, change it's load instruction if found in
  // RestoreOp
  for (MachineBasicBlock &MBB : MF) {
    std::map<int, unsigned> RestoreOp;
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      if (!hasSpillSlotObject(MFI, *MI))
        continue;

      int FI = MI->getOperand(1).getIndex();

      // Prevent stack slot reuse.
      if (RestoreOp.count(FI) && isSpillInst(*MI))
        RestoreOp.erase(FI);

      MachineBasicBlock::iterator NextInst = std::next(MI);
      // Partial spill(store) case
      if (mayBePartialSpillInst(*MI) && tryToRewriteSpill(MBB, MI, RestoreOp)) {
        MI->removeFromParent();
        MI = NextInst;
        continue;
      }

      // Partial reload case
      // If this frame doesn't have corresponding reload op, just skip it.
      if (!RestoreOp.count(FI))
        continue;

      if (tryToRewriteReload(MBB, MI, FI, RestoreOp)) {
        MI->removeFromParent();
        MI = NextInst;
      }
    }
  }
}

std::pair<MachineBasicBlock::reverse_iterator, bool>
RISCVSpillRewrite::findFirstVectorConfigInstUpward(
    MachineBasicBlock::reverse_iterator I, MachineBasicBlock &MBB) {
  bool First = true;
  for (; I != MBB.rend(); ++I) {
    if (isVectorConfigInst(*I) && !isPartialSpillInst(MFI, *std::prev(I)) &&
        !isPartialReloadInst(MFI, *std::prev(I)))
      return std::make_pair(I, First);

    if (isRVVInst(*I) && !isSpillInst(*I) && !isReloadInst(*I))
      First = false;
  }

  llvm_unreachable("Reach the begin of MBB!");
}

// Since the new spill/reload will insert a vsetivli, it might violate the
// original user's config, so we need to either move the direct config
// instructions(if any) after spill/reload instruction or copy one for it, for
// example: case 1: direct config instruction before user
//  vsetvli a0, x0, e8, m2
//  vsetivli 31, x0, e8, mf2
//  vle8.v v9, (a1)
//  vadd.vv v8, v8, v9
//             |
//             v
//  vsetivli 31, x0, e8, mf2
//  vle8.v v8, (a1)
//  vsetvli a0, x0, e8, m2     <- move after the spill/reload
//  vadd.vv v8, v8, v9
//
//
// case 2: indirect config instruction before user
//  vsetvli a0, x0, e8, m2
//  vsub.vv v9, v9, v10
//  vadd.vv v8, v8, v9
//  vsetivli 31, x0, e8, mf2
//  vle8.v v8, (a1)
//  vadd.vv v8, v8, v9
//             |
//             v
//  vsetvli a0, x0, e8, m2      <- copy from
//  vsub.vv v9, v9, v10
//  vadd.vv v8, v8, v9
//  vsetivli 31, x0, e8, mf2
//  vle8.v v8, (a1)
//  vsetvli a0, x0, e8, m2      <- copy to
//  vadd.vv v8, v8, v9
void RISCVSpillRewrite::fixup(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock::reverse_iterator MI = MBB.rbegin();
         MI != MBB.rend(); ++MI) {
      if (!isPartialSpillInst(MFI, *MI) && !isPartialReloadInst(MFI, *MI))
        continue;
      // This spill doesn't insert any vsetvli, so it doesn't need to fixup.
      if (isPartialSpillInst(MFI, *MI) && !isVectorConfigInst(*std::next(MI)))
        continue;

      MachineBasicBlock::iterator NextInst = *std::prev(MI);
      auto needToFixup = [&](MachineBasicBlock::iterator I) {
        for (; I != MBB.end(); ++I) {
          if (isVectorConfigInst(*I))
            return false;
          if (isRVVInst(*I))
            return true;
        }

        return false;
      };
      if (!needToFixup(NextInst))
        continue;

      auto FVC = findFirstVectorConfigInstUpward(std::next(MI, 2), MBB);
      if (FVC.second) {
        // case1
        FVC.first->moveBefore(&*NextInst);
        LIS->handleMove(*FVC.first);
        // LIS->RemoveMachineInstrFromMaps(*FVC.first);
        // LIS->InsertMachineInstrInMaps(*FVC.first);

        // llvm::for_each(FVC.first->operands(), [&](const MachineOperand &MO) {
        //   if (!MO.isReg() || !MO.getReg().isVirtual())
        //     return;
        //   LIS->removeInterval(MO.getReg());
        //   LIS->createAndComputeVirtRegInterval(MO.getReg());
        // });
      } else {
        // case2
        MachineInstr &ToBeCopied = *FVC.first;
        MachineInstrBuilder Builder = BuildMI(MBB, NextInst, DebugLoc(),
                                              TII->get(ToBeCopied.getOpcode()));

        Register NewReg;
        const MachineOperand &MO0 = ToBeCopied.getOperand(0);
        MachineOperand &MO1 = ToBeCopied.getOperand(1);
        if (MO0.isReg() && MO0.getReg() != RISCV::X0) {
          NewReg = MRI->createVirtualRegister(MRI->getRegClass(MO0.getReg()));
          Builder.addDef(NewReg);
        } else {
          Builder.add(MO0);
        }
        for (unsigned i = 1; i < ToBeCopied.getNumOperands() - 2; ++i)
          Builder.add(ToBeCopied.getOperand(i));
        LIS->InsertMachineInstrInMaps(*Builder.getInstr());
        // The rs1(AVL) operand of the vsetvli might be killed. Since it's
        // copied, the life time of rs1 needs to be extended.
        if (MO1.isReg() && MO1.getReg() != RISCV::X0) {
          MO1.setIsKill(false);
          LIS->extendToIndices(
              LIS->getInterval(MO1.getReg()),
              {LIS->getInstructionIndex(*Builder.getInstr()).getRegSlot()});
        }

        if (NewReg)
          LIS->createAndComputeVirtRegInterval(NewReg);
      }
    }
  }
}

/// Returns an instance of the RVV Spill Rewrite pass.
FunctionPass *llvm::createRISCVSpillRewritePass() {
  return new RISCVSpillRewrite();
}
