//===----------- RISCVCollectMacroFusionStats.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-macro-fusion-stats"
#define PASS_NAME "Collect RISC-V Macro Fusion Statistics"

static cl::opt<unsigned>
    MacroFusionStatsWindowSize("riscv-macro-fusion-stats-window-size",
                               cl::Hidden, cl::init(1));

#define GET_RISCV_MACRO_FUSION_PRED_DECL
// GET_RISCV_MACRO_FUSION_PRED_IMPL is already in RISCVSubtarget.cpp
#define GET_RISCV_ALL_MACRO_FUSION_PRED_DECL
#define GET_RISCV_ALL_MACRO_FUSION_PRED_IMPL
#include "RISCVGenMacroFusion.inc"

namespace {
struct RISCVMacroFusionStats : public MachineFunctionPass {
  static char ID;
  RISCVMacroFusionStats();

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  ArrayRef<std::pair<const char *, MacroFusionPredTy>> AllFusions;

  // Fusion pattern name -> number of occurrences.
  StringMap<unsigned> Stats;
};
} // anonymous namespace

char RISCVMacroFusionStats::ID = 0;

RISCVMacroFusionStats::RISCVMacroFusionStats()
    : MachineFunctionPass(RISCVMacroFusionStats::ID),
      AllFusions(getAllRISCVMacroFusions()) {}

bool RISCVMacroFusionStats::runOnMachineFunction(MachineFunction &MF) {
  Stats.clear();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetInstrInfo &TII = *STI.getInstrInfo();
  auto &ORE = getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();

  // Dummy MI to obtain debug loc.
  const MachineInstr *FirstMI = nullptr;

  for (const auto &MBB : MF) {
    for (const auto &[Idx, CurMI] : enumerate(MBB)) {
      if (!FirstMI)
        FirstMI = &CurMI;
      for (unsigned Step = 1; Step <= MacroFusionStatsWindowSize; ++Step) {
        if (Step > Idx)
          continue;
        const MachineInstr &PrevMI = *std::prev(CurMI.getIterator(), Step);
        for (auto [Name, PredFunc] : AllFusions)
          if (PredFunc(TII, STI, &PrevMI, CurMI)) {
            Stats[Name]++;
            break;
          }
      }
    }
  }

  for (const auto &Entry : Stats)
    ORE.emit([&]() {
      using namespace ore;
      return MachineOptimizationRemarkAnalysis(DEBUG_TYPE, Entry.getKey(),
                                               FirstMI)
             << NV("NumOccurrences", Entry.getValue());
    });

  return false;
}

INITIALIZE_PASS_BEGIN(RISCVMacroFusionStats, DEBUG_TYPE, PASS_NAME, false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_END(RISCVMacroFusionStats, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createRISCVMacroFusionStatsPass() {
  return new RISCVMacroFusionStats();
}
