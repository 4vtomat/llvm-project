//=- SiFive_BulletNopInserter.cpp -------------------------------------------=//
//
// Copyright (c) 2025 SiFive, Inc. -- Proprietary and Confidential All
// Rights Reserved.
//
// NOTICE: All information contained herein is, and remains the property of
// SiFive, Inc. The intellectual and technical concepts contained herein are
// proprietary to SiFive, Inc. and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// This work may not be copied, modified, re-published, uploaded, executed, or
// distributed in any way, in any medium, whether in whole or in part, without
// prior written permission from SiFive, Inc. The copyright notice above does
// not evidence any actual or intended publication or disclosure of this source
// code, which includes information that is confidential and/or proprietary,
// and is a trade secret, of SiFive, Inc.
//
//===----------------------------------------------------------------------===//
// This pass inserts a NOP into loops that contain only a single load and a
// conditional branch.
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-bullet-nop-inserter"

static cl::opt<bool> InsertNOPs("riscv-insert-loop-nop",
                                cl::desc("Insert NOP into short polling loops"),
                                cl::init(false), cl::Hidden);

STATISTIC(NumNopsAdded, "Number of NOPs added to load+branch loops");

namespace {
class RISCVBulletNopInserter : public MachineFunctionPass {
  const TargetInstrInfo *TII;

public:
  static char ID;
  RISCVBulletNopInserter() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  StringRef getPassName() const override {
    return "RISC-V Bullet NOP Inserter";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool runOnBasicBlock(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVBulletNopInserter::ID = 0;

INITIALIZE_PASS(RISCVBulletNopInserter, DEBUG_TYPE,
                "RISC-V Bullet NOP Inserter", false, false)

bool RISCVBulletNopInserter::runOnMachineFunction(MachineFunction &MF) {
  auto &STI = MF.getSubtarget<RISCVSubtarget>();
  if (!STI.isSiFiveBulletCPU() && !InsertNOPs)
    return false;

  TII = STI.getInstrInfo();

  bool Changed = false;
  for (auto &MBB : MF)
    Changed |= runOnBasicBlock(MBB);

  return Changed;
}

bool RISCVBulletNopInserter::runOnBasicBlock(MachineBasicBlock &MBB) {
  // Skip meta instructions
  auto FirstI = MBB.begin();
  for (; FirstI != MBB.end() && FirstI->isMetaInstruction(); ++FirstI)
    ;

  // First instruction should be a load.
  if (FirstI == MBB.end() || !FirstI->mayLoad())
    return false;

  // Skip meta instructions
  auto NextI = std::next(FirstI);
  for (; NextI != MBB.end() && NextI->isMetaInstruction(); ++NextI)
    ;

  // Second instruction should be a conditional branch to this block.
  if (NextI == MBB.end() || !NextI->isConditionalBranch())
    return false;

  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 3> Cond;
  if (TII->analyzeBranch(MBB, TBB, FBB, Cond, /*AllowModify=*/false) ||
      Cond.empty())
    return false;

  // Branch should be to this BB.
  if (TBB != &MBB)
    return false;

  LLVM_DEBUG(dbgs() << "Inserting NOP into BasicBlock " << MBB << '\n');

  BuildMI(MBB, FirstI, FirstI->getDebugLoc(), TII->get(RISCV::ADDI))
      .addReg(RISCV::X0, RegState::Dead | RegState::Define)
      .addReg(RISCV::X0)
      .addImm(0);

  ++NumNopsAdded;

  return true;
}

FunctionPass *llvm::createRISCVBulletNopInserterPass() {
  return new RISCVBulletNopInserter();
}
