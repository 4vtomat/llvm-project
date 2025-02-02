//===-- RISCV.h - Top-level interface for RISC-V ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// RISC-V back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCV_H
#define LLVM_LIB_TARGET_RISCV_RISCV_H

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class FunctionPass;
class InstructionSelector;
class Pass; // SIFIVE
class PassRegistry;
class RISCVRegisterBankInfo;
class RISCVSubtarget;
class RISCVTargetMachine;

#if SIFIVE_CUSTOMIZATION
FunctionPass *createRISCVLandingPadSetupPass();
void initializeRISCVLandingPadSetupPass(PassRegistry &);

FunctionPass *createRISCVLateCodeGenPreparePass();
void initializeRISCVLateCodeGenPreparePass(PassRegistry &);

FunctionPass *createRISCVTypePromotionPass();
void initializeRISCVTypePromotionPass(PassRegistry &);

Pass *createRISCVWidenReductionPHIPass();
void initializeRISCVWidenReductionPHIPass(PassRegistry &);

FunctionPass *createRISCVIndirectBranchTrackingPass();
void initializeRISCVIndirectBranchTrackingPass(PassRegistry &);
#endif // SIFIVE_CUSTOMIZATION

FunctionPass *createRISCVCodeGenPreparePass();
void initializeRISCVCodeGenPreparePass(PassRegistry &);

FunctionPass *createRISCVDeadRegisterDefinitionsPass();
void initializeRISCVDeadRegisterDefinitionsPass(PassRegistry &);

FunctionPass *createRISCVIndirectBranchTrackingPass();
void initializeRISCVIndirectBranchTrackingPass(PassRegistry &);

FunctionPass *createRISCVLandingPadSetupPass();
void initializeRISCVLandingPadSetupPass(PassRegistry &);

FunctionPass *createRISCVISelDag(RISCVTargetMachine &TM,
                                 CodeGenOptLevel OptLevel);

FunctionPass *createRISCVMakeCompressibleOptPass();
void initializeRISCVMakeCompressibleOptPass(PassRegistry &);

FunctionPass *createRISCVGatherScatterLoweringPass();
void initializeRISCVGatherScatterLoweringPass(PassRegistry &);

FunctionPass *createRISCVVectorPeepholePass();
void initializeRISCVVectorPeepholePass(PassRegistry &);

FunctionPass *createRISCVOptWInstrsPass();
void initializeRISCVOptWInstrsPass(PassRegistry &);

FunctionPass *createRISCVMergeBaseOffsetOptPass();
void initializeRISCVMergeBaseOffsetOptPass(PassRegistry &);

FunctionPass *createRISCVExpandPseudoPass();
void initializeRISCVExpandPseudoPass(PassRegistry &);

FunctionPass *createRISCVPreRAExpandPseudoPass();
void initializeRISCVPreRAExpandPseudoPass(PassRegistry &);

FunctionPass *createRISCVExpandAtomicPseudoPass();
void initializeRISCVExpandAtomicPseudoPass(PassRegistry &);

FunctionPass *createRISCVInsertVSETVLIPass();
void initializeRISCVInsertVSETVLIPass(PassRegistry &);
extern char &RISCVInsertVSETVLIID;

#if SIFIVE_CUSTOMIZATION
FunctionPass *createRISCVSpillRewritePass();
void initializeRISCVSpillRewritePass(PassRegistry &);
#endif // SIFIVE_CUSTOMIZATION

FunctionPass *createRISCVPostRAExpandPseudoPass();
void initializeRISCVPostRAExpandPseudoPass(PassRegistry &);
FunctionPass *createRISCVInsertReadWriteCSRPass();
void initializeRISCVInsertReadWriteCSRPass(PassRegistry &);

FunctionPass *createRISCVInsertWriteVXRMPass();
void initializeRISCVInsertWriteVXRMPass(PassRegistry &);

FunctionPass *createRISCVRedundantCopyEliminationPass();
void initializeRISCVRedundantCopyEliminationPass(PassRegistry &);

#if SIFIVE_CUSTOMIZATION
FunctionPass *createRISCVPeepholePass();
void initializeRISCVPeepholePass(PassRegistry &);

FunctionPass *createRISCVMachineConstPropagationPass();
void initializeRISCVMachineConstPropagationPass(PassRegistry &);

FunctionPass *createSiFiveRISCVVLOptimizerPass();
void initializeSiFiveRISCVVLOptimizerPass(PassRegistry &);
#endif // SIFIVE_CUSTOMIZATION

FunctionPass *createRISCVMoveMergePass();
void initializeRISCVMoveMergePass(PassRegistry &);

FunctionPass *createRISCVPushPopOptimizationPass();
void initializeRISCVPushPopOptPass(PassRegistry &);

FunctionPass *createRISCVZacasABIFixPass();
void initializeRISCVZacasABIFixPass(PassRegistry &);

InstructionSelector *
createRISCVInstructionSelector(const RISCVTargetMachine &,
                               const RISCVSubtarget &,
                               const RISCVRegisterBankInfo &);
void initializeRISCVDAGToDAGISelLegacyPass(PassRegistry &);

FunctionPass *createRISCVPostLegalizerCombiner();
void initializeRISCVPostLegalizerCombinerPass(PassRegistry &);

FunctionPass *createRISCVO0PreLegalizerCombiner();
void initializeRISCVO0PreLegalizerCombinerPass(PassRegistry &);

FunctionPass *createRISCVPreLegalizerCombiner();
void initializeRISCVPreLegalizerCombinerPass(PassRegistry &);

FunctionPass *createRISCVVLOptimizerPass();
void initializeRISCVVLOptimizerPass(PassRegistry &);
} // namespace llvm

#endif
