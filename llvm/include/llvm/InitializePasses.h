//===- llvm/InitializePasses.h - Initialize All Passes ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for the pass initialization routines
// for the entire LLVM project.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INITIALIZEPASSES_H
#define LLVM_INITIALIZEPASSES_H

#include "llvm/Support/Compiler.h"

namespace llvm {

class PassRegistry;

/// Initialize all passes linked into the Core library.
LLVM_ABI void initializeCore(PassRegistry &);

/// Initialize all passes linked into the TransformUtils library.
LLVM_ABI void initializeTransformUtils(PassRegistry &);

/// Initialize all passes linked into the ScalarOpts library.
LLVM_ABI void initializeScalarOpts(PassRegistry &);

/// Initialize all passes linked into the Vectorize library.
LLVM_ABI void initializeVectorization(PassRegistry &);

/// Initialize all passes linked into the InstCombine library.
LLVM_ABI void initializeInstCombine(PassRegistry &);

/// Initialize all passes linked into the IPO library.
LLVM_ABI void initializeIPO(PassRegistry &);

/// Initialize all passes linked into the Analysis library.
LLVM_ABI void initializeAnalysis(PassRegistry &);

/// Initialize all passes linked into the CodeGen library.
LLVM_ABI void initializeCodeGen(PassRegistry &);

/// Initialize all passes linked into the GlobalISel library.
LLVM_ABI void initializeGlobalISel(PassRegistry &);

/// Initialize all passes linked into the CodeGen library.
LLVM_ABI void initializeTarget(PassRegistry &);

<<<<<<< HEAD
void initializeAAResultsWrapperPassPass(PassRegistry &);
void initializeAlwaysInlinerLegacyPassPass(PassRegistry &);
void initializeAssignmentTrackingAnalysisPass(PassRegistry &);
void initializeAssumptionCacheTrackerPass(PassRegistry &);
void initializeAtomicExpandLegacyPass(PassRegistry &);
void initializeBasicBlockPathCloningPass(PassRegistry &);
void initializeBasicBlockSectionsProfileReaderWrapperPassPass(PassRegistry &);
void initializeBasicBlockSectionsPass(PassRegistry &);
void initializeBarrierNoopPass(PassRegistry &);
void initializeBasicAAWrapperPassPass(PassRegistry &);
void initializeBlockFrequencyInfoWrapperPassPass(PassRegistry &);
void initializeBranchFolderLegacyPass(PassRegistry &);
void initializeBranchProbabilityInfoWrapperPassPass(PassRegistry &);
void initializeBranchRelaxationLegacyPass(PassRegistry &);
void initializeBreakCriticalEdgesPass(PassRegistry &);
void initializeBreakFalseDepsPass(PassRegistry &);
void initializeCanonicalizeFreezeInLoopsPass(PassRegistry &);
void initializeCFGSimplifyPassPass(PassRegistry &);
void initializeCFGuardPass(PassRegistry &);
void initializeCFGuardLongjmpPass(PassRegistry &);
void initializeCFIFixupPass(PassRegistry &);
void initializeCFIInstrInserterPass(PassRegistry &);
void initializeCallBrPreparePass(PassRegistry &);
void initializeCallGraphDOTPrinterPass(PassRegistry &);
void initializeCallGraphViewerPass(PassRegistry &);
void initializeCallGraphWrapperPassPass(PassRegistry &);
void initializeCheckDebugMachineModulePass(PassRegistry &);
void initializeCodeGenPrepareLegacyPassPass(PassRegistry &);
void initializeComplexDeinterleavingLegacyPassPass(PassRegistry &);
void initializeConstantHoistingLegacyPassPass(PassRegistry &);
void initializeCycleInfoWrapperPassPass(PassRegistry &);
void initializeDAEPass(PassRegistry &);
void initializeDAHPass(PassRegistry &);
void initializeDCELegacyPassPass(PassRegistry &);
void initializeDXILMetadataAnalysisWrapperPassPass(PassRegistry &);
void initializeDXILMetadataAnalysisWrapperPrinterPass(PassRegistry &);
void initializeDXILResourceBindingWrapperPassPass(PassRegistry &);
void initializeDXILResourceImplicitBindingLegacyPass(PassRegistry &);
void initializeDXILResourceTypeWrapperPassPass(PassRegistry &);
void initializeDXILResourceWrapperPassPass(PassRegistry &);
void initializeDeadMachineInstructionElimPass(PassRegistry &);
void initializeDebugifyMachineModulePass(PassRegistry &);
void initializeDependenceAnalysisWrapperPassPass(PassRegistry &);
void initializeDetectDeadLanesLegacyPass(PassRegistry &);
void initializeDomOnlyPrinterWrapperPassPass(PassRegistry &);
void initializeDomOnlyViewerWrapperPassPass(PassRegistry &);
void initializeDomPrinterWrapperPassPass(PassRegistry &);
void initializeDomViewerWrapperPassPass(PassRegistry &);
void initializeDominanceFrontierWrapperPassPass(PassRegistry &);
void initializeDominatorTreeWrapperPassPass(PassRegistry &);
void initializeDwarfEHPrepareLegacyPassPass(PassRegistry &);
void initializeEarlyCSELegacyPassPass(PassRegistry &);
void initializeEarlyCSEMemSSALegacyPassPass(PassRegistry &);
void initializeEarlyIfConverterLegacyPass(PassRegistry &);
void initializeEarlyIfPredicatorPass(PassRegistry &);
void initializeEarlyMachineLICMPass(PassRegistry &);
void initializeEarlyTailDuplicateLegacyPass(PassRegistry &);
void initializeEdgeBundlesWrapperLegacyPass(PassRegistry &);
void initializeEHContGuardTargetsPass(PassRegistry &);
void initializeExpandFpLegacyPassPass(PassRegistry &);
void initializeExpandLargeDivRemLegacyPassPass(PassRegistry &);
void initializeExpandMemCmpLegacyPassPass(PassRegistry &);
void initializeExpandPostRAPass(PassRegistry &);
#if SIFIVE_CUSTOMIZATION
void initializeExpandPowiLegacyPassPass(PassRegistry &);
void initializeExpandVPReductionLegacyPassPass(PassRegistry &);
#endif // SIFIVE_CUSTOMIZATION
void initializeExpandPostRALegacyPass(PassRegistry &);
void initializeExpandReductionsPass(PassRegistry &);
void initializeExpandVariadicsPass(PassRegistry &);
void initializeExternalAAWrapperPassPass(PassRegistry &);
void initializeFEntryInserterLegacyPass(PassRegistry &);
void initializeFinalizeISelPass(PassRegistry &);
void initializeFinalizeMachineBundlesPass(PassRegistry &);
void initializeFixIrreduciblePass(PassRegistry &);
void initializeFixupStatepointCallerSavedLegacyPass(PassRegistry &);
void initializeFlattenCFGLegacyPassPass(PassRegistry &);
void initializeFuncletLayoutPass(PassRegistry &);
void initializeGCEmptyBasicBlocksPass(PassRegistry &);
void initializeGCMachineCodeAnalysisPass(PassRegistry &);
void initializeGCModuleInfoPass(PassRegistry &);
void initializeGVNLegacyPassPass(PassRegistry &);
void initializeGlobalMergeFuncPassWrapperPass(PassRegistry &);
void initializeGlobalMergePass(PassRegistry &);
void initializeGlobalsAAWrapperPassPass(PassRegistry &);
void initializeHardwareLoopsLegacyPass(PassRegistry &);
void initializeMIRProfileLoaderPassPass(PassRegistry &);
void initializeIRSimilarityIdentifierWrapperPassPass(PassRegistry &);
void initializeIRTranslatorPass(PassRegistry &);
void initializeIVUsersWrapperPassPass(PassRegistry &);
void initializeIfConverterPass(PassRegistry &);
void initializeImmutableModuleSummaryIndexWrapperPassPass(PassRegistry &);
void initializeImplicitNullChecksPass(PassRegistry &);
void initializeIndirectBrExpandLegacyPassPass(PassRegistry &);
void initializeInferAddressSpacesPass(PassRegistry &);
void initializeInstSimplifyLegacyPassPass(PassRegistry &);
void initializeInstructionCombiningPassPass(PassRegistry &);
void initializeInstructionSelectPass(PassRegistry &);
void initializeInterleavedAccessPass(PassRegistry &);
void initializeInterleavedLoadCombinePass(PassRegistry &);
void initializeJMCInstrumenterPass(PassRegistry &);
void initializeKCFIPass(PassRegistry &);
void initializeLCSSAVerificationPassPass(PassRegistry &);
void initializeLCSSAWrapperPassPass(PassRegistry &);
void initializeLazyBFIPassPass(PassRegistry &);
void initializeLazyBlockFrequencyInfoPassPass(PassRegistry &);
void initializeLazyBranchProbabilityInfoPassPass(PassRegistry &);
void initializeLazyMachineBlockFrequencyInfoPassPass(PassRegistry &);
void initializeLazyValueInfoWrapperPassPass(PassRegistry &);
void initializeLegacyLICMPassPass(PassRegistry &);
void initializeLegalizerPass(PassRegistry &);
void initializeGISelCSEAnalysisWrapperPassPass(PassRegistry &);
void initializeGISelValueTrackingAnalysisLegacyPass(PassRegistry &);
void initializeLiveDebugValuesLegacyPass(PassRegistry &);
void initializeLiveDebugVariablesWrapperLegacyPass(PassRegistry &);
void initializeLiveIntervalsWrapperPassPass(PassRegistry &);
void initializeLiveRangeShrinkPass(PassRegistry &);
void initializeLiveRegMatrixWrapperLegacyPass(PassRegistry &);
void initializeLiveStacksWrapperLegacyPass(PassRegistry &);
void initializeLiveVariablesWrapperPassPass(PassRegistry &);
void initializeLoadStoreOptPass(PassRegistry &);
void initializeLoadStoreVectorizerLegacyPassPass(PassRegistry &);
void initializeLocalStackSlotPassPass(PassRegistry &);
void initializeLocalizerPass(PassRegistry &);
void initializeLoopDataPrefetchLegacyPassPass(PassRegistry &);
#if SIFIVE_CUSTOMIZATION
void initializeLoopDataLayoutLegacyPassPass(PassRegistry &);
#endif
void initializeLoopExtractorLegacyPassPass(PassRegistry &);
void initializeLoopInfoWrapperPassPass(PassRegistry &);
void initializeLoopPassPass(PassRegistry &);
void initializeLoopSimplifyPass(PassRegistry &);
void initializeLoopStrengthReducePass(PassRegistry &);
void initializeLoopTermFoldPass(PassRegistry &);
void initializeLoopUnrollPass(PassRegistry &);
void initializeLowerAtomicLegacyPassPass(PassRegistry &);
void initializeLowerEmuTLSPass(PassRegistry &);
void initializeLowerGlobalDtorsLegacyPassPass(PassRegistry &);
void initializeLowerIntrinsicsPass(PassRegistry &);
void initializeLowerInvokeLegacyPassPass(PassRegistry &);
void initializeLowerSwitchLegacyPassPass(PassRegistry &);
void initializeKCFIPass(PassRegistry &);
void initializeMIRAddFSDiscriminatorsPass(PassRegistry &);
void initializeMIRCanonicalizerPass(PassRegistry &);
void initializeMIRNamerPass(PassRegistry &);
void initializeMIRPrintingPassPass(PassRegistry &);
void initializeMachineBlockFrequencyInfoWrapperPassPass(PassRegistry &);
void initializeMachineBlockPlacementLegacyPass(PassRegistry &);
void initializeMachineBlockPlacementStatsLegacyPass(PassRegistry &);
void initializeMachineBranchProbabilityInfoWrapperPassPass(PassRegistry &);
void initializeMachineCFGPrinterPass(PassRegistry &);
void initializeMachineCSELegacyPass(PassRegistry &);
void initializeMachineCombinerPass(PassRegistry &);
void initializeMachineCopyPropagationLegacyPass(PassRegistry &);
void initializeMachineCycleInfoPrinterLegacyPass(PassRegistry &);
void initializeMachineCycleInfoWrapperPassPass(PassRegistry &);
void initializeMachineDominanceFrontierPass(PassRegistry &);
void initializeMachineDominatorTreeWrapperPassPass(PassRegistry &);
void initializeMachineFunctionPrinterPassPass(PassRegistry &);
void initializeMachineFunctionSplitterPass(PassRegistry &);
void initializeMachineLateInstrsCleanupLegacyPass(PassRegistry &);
void initializeMachineLICMPass(PassRegistry &);
void initializeMachineLoopInfoWrapperPassPass(PassRegistry &);
void initializeMachineModuleInfoWrapperPassPass(PassRegistry &);
void initializeMachineOptimizationRemarkEmitterPassPass(PassRegistry &);
void initializeMachineOutlinerPass(PassRegistry &);
void initializeStaticDataProfileInfoWrapperPassPass(PassRegistry &);
void initializeStaticDataAnnotatorPass(PassRegistry &);
void initializeMachinePipelinerPass(PassRegistry &);
void initializeMachinePostDominatorTreeWrapperPassPass(PassRegistry &);
void initializeMachineRegionInfoPassPass(PassRegistry &);
void initializeMachineSanitizerBinaryMetadataLegacyPass(PassRegistry &);
void initializeMachineSchedulerLegacyPass(PassRegistry &);
void initializeMachineSinkingLegacyPass(PassRegistry &);
void initializeMachineTraceMetricsWrapperPassPass(PassRegistry &);
void initializeMachineUniformityInfoPrinterPassPass(PassRegistry &);
void initializeMachineUniformityAnalysisPassPass(PassRegistry &);
void initializeMachineVerifierLegacyPassPass(PassRegistry &);
void initializeMemoryDependenceWrapperPassPass(PassRegistry &);
void initializeMemorySSAWrapperPassPass(PassRegistry &);
void initializeMergeICmpsLegacyPassPass(PassRegistry &);
void initializeModuleSummaryIndexWrapperPassPass(PassRegistry &);
void initializeModuloScheduleTestPass(PassRegistry &);
void initializeNaryReassociateLegacyPassPass(PassRegistry &);
void initializeObjCARCContractLegacyPassPass(PassRegistry &);
void initializeOptimizationRemarkEmitterWrapperPassPass(PassRegistry &);
void initializeOptimizePHIsLegacyPass(PassRegistry &);
void initializePEILegacyPass(PassRegistry &);
void initializePHIEliminationPass(PassRegistry &);
void initializePartiallyInlineLibCallsLegacyPassPass(PassRegistry &);
void initializePatchableFunctionLegacyPass(PassRegistry &);
void initializePeepholeOptimizerLegacyPass(PassRegistry &);
void initializePhiValuesWrapperPassPass(PassRegistry &);
void initializePhysicalRegisterUsageInfoWrapperLegacyPass(PassRegistry &);
void initializePlaceBackedgeSafepointsLegacyPassPass(PassRegistry &);
void initializePostDomOnlyPrinterWrapperPassPass(PassRegistry &);
void initializePostDomOnlyViewerWrapperPassPass(PassRegistry &);
void initializePostDomPrinterWrapperPassPass(PassRegistry &);
void initializePostDomViewerWrapperPassPass(PassRegistry &);
void initializePostDominatorTreeWrapperPassPass(PassRegistry &);
void initializePostInlineEntryExitInstrumenterPass(PassRegistry &);
void initializePostMachineSchedulerLegacyPass(PassRegistry &);
void initializePostRAHazardRecognizerLegacyPass(PassRegistry &);
void initializePostRAMachineSinkingPass(PassRegistry &);
void initializePostRASchedulerLegacyPass(PassRegistry &);
void initializePreISelIntrinsicLoweringLegacyPassPass(PassRegistry &);
void initializePrintFunctionPassWrapperPass(PassRegistry &);
void initializePrintModulePassWrapperPass(PassRegistry &);
void initializeProcessImplicitDefsPass(PassRegistry &);
void initializeProfileSummaryInfoWrapperPassPass(PassRegistry &);
void initializePromoteLegacyPassPass(PassRegistry &);
void initializeRABasicPass(PassRegistry &);
void initializePseudoProbeInserterPass(PassRegistry &);
void initializeRAGreedyLegacyPass(PassRegistry &);
void initializeReachingDefAnalysisPass(PassRegistry &);
void initializeReassociateLegacyPassPass(PassRegistry &);
void initializeRegAllocEvictionAdvisorAnalysisLegacyPass(PassRegistry &);
void initializeRegAllocFastPass(PassRegistry &);
void initializeRegAllocPriorityAdvisorAnalysisLegacyPass(PassRegistry &);
void initializeRegAllocScoringPass(PassRegistry &);
void initializeRegBankSelectPass(PassRegistry &);
void initializeRegToMemWrapperPassPass(PassRegistry &);
void initializeRegUsageInfoCollectorLegacyPass(PassRegistry &);
void initializeRegUsageInfoPropagationLegacyPass(PassRegistry &);
void initializeRegionInfoPassPass(PassRegistry &);
void initializeRegionOnlyPrinterPass(PassRegistry &);
void initializeRegionOnlyViewerPass(PassRegistry &);
void initializeRegionPrinterPass(PassRegistry &);
void initializeRegionViewerPass(PassRegistry &);
void initializeRegisterCoalescerLegacyPass(PassRegistry &);
void initializeRemoveLoadsIntoFakeUsesLegacyPass(PassRegistry &);
void initializeRemoveRedundantDebugValuesLegacyPass(PassRegistry &);
void initializeRenameIndependentSubregsLegacyPass(PassRegistry &);
void initializeReplaceWithVeclibLegacyPass(PassRegistry &);
void initializeResetMachineFunctionPass(PassRegistry &);
void initializeSCEVAAWrapperPassPass(PassRegistry &);
void initializeSROALegacyPassPass(PassRegistry &);
void initializeSafeStackLegacyPassPass(PassRegistry &);
void initializeSafepointIRVerifierPass(PassRegistry &);
void initializeSelectOptimizePass(PassRegistry &);
void initializeScalarEvolutionWrapperPassPass(PassRegistry &);
void initializeScalarizeMaskedMemIntrinLegacyPassPass(PassRegistry &);
void initializeScalarizerLegacyPassPass(PassRegistry &);
void initializeScavengerTestPass(PassRegistry &);
void initializeScopedNoAliasAAWrapperPassPass(PassRegistry &);
void initializeSeparateConstOffsetFromGEPLegacyPassPass(PassRegistry &);
void initializeShadowStackGCLoweringPass(PassRegistry &);
void initializeShrinkWrapLegacyPass(PassRegistry &);
void initializeSingleLoopExtractorPass(PassRegistry &);
void initializeSinkingLegacyPassPass(PassRegistry &);
void initializeSjLjEHPreparePass(PassRegistry &);
void initializeSlotIndexesWrapperPassPass(PassRegistry &);
void initializeSpeculativeExecutionLegacyPassPass(PassRegistry &);
void initializeSpillPlacementWrapperLegacyPass(PassRegistry &);
void initializeStackColoringLegacyPass(PassRegistry &);
void initializeStackFrameLayoutAnalysisLegacyPass(PassRegistry &);
void initializeStaticDataSplitterPass(PassRegistry &);
void initializeStackMapLivenessPass(PassRegistry &);
void initializeStackProtectorPass(PassRegistry &);
void initializeStackSafetyGlobalInfoWrapperPassPass(PassRegistry &);
void initializeStackSafetyInfoWrapperPassPass(PassRegistry &);
void initializeStackSlotColoringLegacyPass(PassRegistry &);
void initializeStraightLineStrengthReduceLegacyPassPass(PassRegistry &);
void initializeStripDebugMachineModulePass(PassRegistry &);
void initializeStructurizeCFGLegacyPassPass(PassRegistry &);
void initializeTailCallElimPass(PassRegistry &);
void initializeTailDuplicateLegacyPass(PassRegistry &);
void initializeTargetLibraryInfoWrapperPassPass(PassRegistry &);
void initializeTargetPassConfigPass(PassRegistry &);
void initializeTargetTransformInfoWrapperPassPass(PassRegistry &);
void initializeTwoAddressInstructionLegacyPassPass(PassRegistry &);
void initializeTypeBasedAAWrapperPassPass(PassRegistry &);
void initializeTypePromotionLegacyPass(PassRegistry &);
void initializeInitUndefPass(PassRegistry &);
void initializeUniformityInfoWrapperPassPass(PassRegistry &);
void initializeUnifyLoopExitsLegacyPassPass(PassRegistry &);
void initializeUnpackMachineBundlesPass(PassRegistry &);
void initializeUnreachableBlockElimLegacyPassPass(PassRegistry &);
void initializeUnreachableMachineBlockElimLegacyPass(PassRegistry &);
void initializeVerifierLegacyPassPass(PassRegistry &);
void initializeVirtRegMapWrapperLegacyPass(PassRegistry &);
void initializeVirtRegRewriterLegacyPass(PassRegistry &);
void initializeWasmEHPreparePass(PassRegistry &);
void initializeWinEHPreparePass(PassRegistry &);
void initializeWriteBitcodePassPass(PassRegistry &);
void initializeXRayInstrumentationLegacyPass(PassRegistry &);
=======
LLVM_ABI void initializeAAResultsWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeAlwaysInlinerLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeAssignmentTrackingAnalysisPass(PassRegistry &);
LLVM_ABI void initializeAssumptionCacheTrackerPass(PassRegistry &);
LLVM_ABI void initializeAtomicExpandLegacyPass(PassRegistry &);
LLVM_ABI void initializeBasicBlockPathCloningPass(PassRegistry &);
LLVM_ABI void
initializeBasicBlockSectionsProfileReaderWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeBasicBlockSectionsPass(PassRegistry &);
LLVM_ABI void initializeBarrierNoopPass(PassRegistry &);
LLVM_ABI void initializeBasicAAWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeBlockFrequencyInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeBranchFolderLegacyPass(PassRegistry &);
LLVM_ABI void initializeBranchProbabilityInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeBranchRelaxationLegacyPass(PassRegistry &);
LLVM_ABI void initializeBreakCriticalEdgesPass(PassRegistry &);
LLVM_ABI void initializeBreakFalseDepsPass(PassRegistry &);
LLVM_ABI void initializeCanonicalizeFreezeInLoopsPass(PassRegistry &);
LLVM_ABI void initializeCFGSimplifyPassPass(PassRegistry &);
LLVM_ABI void initializeCFGuardPass(PassRegistry &);
LLVM_ABI void initializeCFGuardLongjmpPass(PassRegistry &);
LLVM_ABI void initializeCFIFixupPass(PassRegistry &);
LLVM_ABI void initializeCFIInstrInserterPass(PassRegistry &);
LLVM_ABI void initializeCallBrPreparePass(PassRegistry &);
LLVM_ABI void initializeCallGraphDOTPrinterPass(PassRegistry &);
LLVM_ABI void initializeCallGraphViewerPass(PassRegistry &);
LLVM_ABI void initializeCallGraphWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeCheckDebugMachineModulePass(PassRegistry &);
LLVM_ABI void initializeCodeGenPrepareLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeComplexDeinterleavingLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeConstantHoistingLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeCycleInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDAEPass(PassRegistry &);
LLVM_ABI void initializeDAHPass(PassRegistry &);
LLVM_ABI void initializeDCELegacyPassPass(PassRegistry &);
LLVM_ABI void initializeDXILMetadataAnalysisWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDXILMetadataAnalysisWrapperPrinterPass(PassRegistry &);
LLVM_ABI void initializeDXILResourceBindingWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDXILResourceImplicitBindingLegacyPass(PassRegistry &);
LLVM_ABI void initializeDXILResourceTypeWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDXILResourceWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDeadMachineInstructionElimPass(PassRegistry &);
LLVM_ABI void initializeDebugifyMachineModulePass(PassRegistry &);
LLVM_ABI void initializeDependenceAnalysisWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDetectDeadLanesLegacyPass(PassRegistry &);
LLVM_ABI void initializeDomOnlyPrinterWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDomOnlyViewerWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDomPrinterWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDomViewerWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDominanceFrontierWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDominatorTreeWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeDwarfEHPrepareLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeEarlyCSELegacyPassPass(PassRegistry &);
LLVM_ABI void initializeEarlyCSEMemSSALegacyPassPass(PassRegistry &);
LLVM_ABI void initializeEarlyIfConverterLegacyPass(PassRegistry &);
LLVM_ABI void initializeEarlyIfPredicatorPass(PassRegistry &);
LLVM_ABI void initializeEarlyMachineLICMPass(PassRegistry &);
LLVM_ABI void initializeEarlyTailDuplicateLegacyPass(PassRegistry &);
LLVM_ABI void initializeEdgeBundlesWrapperLegacyPass(PassRegistry &);
LLVM_ABI void initializeEHContGuardTargetsPass(PassRegistry &);
LLVM_ABI void initializeExpandFpLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeExpandLargeDivRemLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeExpandMemCmpLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeExpandPostRALegacyPass(PassRegistry &);
LLVM_ABI void initializeExpandReductionsPass(PassRegistry &);
LLVM_ABI void initializeExpandVariadicsPass(PassRegistry &);
LLVM_ABI void initializeExternalAAWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeFEntryInserterLegacyPass(PassRegistry &);
LLVM_ABI void initializeFinalizeISelPass(PassRegistry &);
LLVM_ABI void initializeFinalizeMachineBundlesPass(PassRegistry &);
LLVM_ABI void initializeFixIrreduciblePass(PassRegistry &);
LLVM_ABI void initializeFixupStatepointCallerSavedLegacyPass(PassRegistry &);
LLVM_ABI void initializeFlattenCFGLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeFuncletLayoutPass(PassRegistry &);
LLVM_ABI void initializeGCEmptyBasicBlocksPass(PassRegistry &);
LLVM_ABI void initializeGCMachineCodeAnalysisPass(PassRegistry &);
LLVM_ABI void initializeGCModuleInfoPass(PassRegistry &);
LLVM_ABI void initializeGVNLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeGlobalMergeFuncPassWrapperPass(PassRegistry &);
LLVM_ABI void initializeGlobalMergePass(PassRegistry &);
LLVM_ABI void initializeGlobalsAAWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeHardwareLoopsLegacyPass(PassRegistry &);
LLVM_ABI void initializeMIRProfileLoaderPassPass(PassRegistry &);
LLVM_ABI void initializeIRSimilarityIdentifierWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeIRTranslatorPass(PassRegistry &);
LLVM_ABI void initializeIVUsersWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeIfConverterPass(PassRegistry &);
LLVM_ABI void
initializeImmutableModuleSummaryIndexWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeImplicitNullChecksPass(PassRegistry &);
LLVM_ABI void initializeIndirectBrExpandLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeInferAddressSpacesPass(PassRegistry &);
LLVM_ABI void initializeInstSimplifyLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeInstructionCombiningPassPass(PassRegistry &);
LLVM_ABI void initializeInstructionSelectPass(PassRegistry &);
LLVM_ABI void initializeInterleavedAccessPass(PassRegistry &);
LLVM_ABI void initializeInterleavedLoadCombinePass(PassRegistry &);
LLVM_ABI void initializeJMCInstrumenterPass(PassRegistry &);
LLVM_ABI void initializeKCFIPass(PassRegistry &);
LLVM_ABI void initializeLCSSAVerificationPassPass(PassRegistry &);
LLVM_ABI void initializeLCSSAWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeLazyBFIPassPass(PassRegistry &);
LLVM_ABI void initializeLazyBlockFrequencyInfoPassPass(PassRegistry &);
LLVM_ABI void initializeLazyBranchProbabilityInfoPassPass(PassRegistry &);
LLVM_ABI void initializeLazyMachineBlockFrequencyInfoPassPass(PassRegistry &);
LLVM_ABI void initializeLazyValueInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeLegacyLICMPassPass(PassRegistry &);
LLVM_ABI void initializeLegalizerPass(PassRegistry &);
LLVM_ABI void initializeGISelCSEAnalysisWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeGISelValueTrackingAnalysisLegacyPass(PassRegistry &);
LLVM_ABI void initializeLiveDebugValuesLegacyPass(PassRegistry &);
LLVM_ABI void initializeLiveDebugVariablesWrapperLegacyPass(PassRegistry &);
LLVM_ABI void initializeLiveIntervalsWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeLiveRangeShrinkPass(PassRegistry &);
LLVM_ABI void initializeLiveRegMatrixWrapperLegacyPass(PassRegistry &);
LLVM_ABI void initializeLiveStacksWrapperLegacyPass(PassRegistry &);
LLVM_ABI void initializeLiveVariablesWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeLoadStoreOptPass(PassRegistry &);
LLVM_ABI void initializeLoadStoreVectorizerLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeLocalStackSlotPassPass(PassRegistry &);
LLVM_ABI void initializeLocalizerPass(PassRegistry &);
LLVM_ABI void initializeLoopDataPrefetchLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeLoopExtractorLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeLoopInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeLoopPassPass(PassRegistry &);
LLVM_ABI void initializeLoopSimplifyPass(PassRegistry &);
LLVM_ABI void initializeLoopStrengthReducePass(PassRegistry &);
LLVM_ABI void initializeLoopTermFoldPass(PassRegistry &);
LLVM_ABI void initializeLoopUnrollPass(PassRegistry &);
LLVM_ABI void initializeLowerAtomicLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeLowerEmuTLSPass(PassRegistry &);
LLVM_ABI void initializeLowerGlobalDtorsLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeLowerIntrinsicsPass(PassRegistry &);
LLVM_ABI void initializeLowerInvokeLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeLowerSwitchLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeMIRAddFSDiscriminatorsPass(PassRegistry &);
LLVM_ABI void initializeMIRCanonicalizerPass(PassRegistry &);
LLVM_ABI void initializeMIRNamerPass(PassRegistry &);
LLVM_ABI void initializeMIRPrintingPassPass(PassRegistry &);
LLVM_ABI void
initializeMachineBlockFrequencyInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeMachineBlockPlacementLegacyPass(PassRegistry &);
LLVM_ABI void initializeMachineBlockPlacementStatsLegacyPass(PassRegistry &);
LLVM_ABI void
initializeMachineBranchProbabilityInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeMachineCFGPrinterPass(PassRegistry &);
LLVM_ABI void initializeMachineCSELegacyPass(PassRegistry &);
LLVM_ABI void initializeMachineCombinerPass(PassRegistry &);
LLVM_ABI void initializeMachineCopyPropagationLegacyPass(PassRegistry &);
LLVM_ABI void initializeMachineCycleInfoPrinterLegacyPass(PassRegistry &);
LLVM_ABI void initializeMachineCycleInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeMachineDominanceFrontierPass(PassRegistry &);
LLVM_ABI void initializeMachineDominatorTreeWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeMachineFunctionPrinterPassPass(PassRegistry &);
LLVM_ABI void initializeMachineFunctionSplitterPass(PassRegistry &);
LLVM_ABI void initializeMachineLateInstrsCleanupLegacyPass(PassRegistry &);
LLVM_ABI void initializeMachineLICMPass(PassRegistry &);
LLVM_ABI void initializeMachineLoopInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeMachineModuleInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void
initializeMachineOptimizationRemarkEmitterPassPass(PassRegistry &);
LLVM_ABI void initializeMachineOutlinerPass(PassRegistry &);
LLVM_ABI void initializeStaticDataProfileInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeStaticDataAnnotatorPass(PassRegistry &);
LLVM_ABI void initializeMachinePipelinerPass(PassRegistry &);
LLVM_ABI void initializeMachinePostDominatorTreeWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeMachineRegionInfoPassPass(PassRegistry &);
LLVM_ABI void
initializeMachineSanitizerBinaryMetadataLegacyPass(PassRegistry &);
LLVM_ABI void initializeMachineSchedulerLegacyPass(PassRegistry &);
LLVM_ABI void initializeMachineSinkingLegacyPass(PassRegistry &);
LLVM_ABI void initializeMachineTraceMetricsWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeMachineUniformityInfoPrinterPassPass(PassRegistry &);
LLVM_ABI void initializeMachineUniformityAnalysisPassPass(PassRegistry &);
LLVM_ABI void initializeMachineVerifierLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeMemoryDependenceWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeMemorySSAWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeMergeICmpsLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeModuleSummaryIndexWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeModuloScheduleTestPass(PassRegistry &);
LLVM_ABI void initializeNaryReassociateLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeObjCARCContractLegacyPassPass(PassRegistry &);
LLVM_ABI void
initializeOptimizationRemarkEmitterWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeOptimizePHIsLegacyPass(PassRegistry &);
LLVM_ABI void initializePEILegacyPass(PassRegistry &);
LLVM_ABI void initializePHIEliminationPass(PassRegistry &);
LLVM_ABI void initializePartiallyInlineLibCallsLegacyPassPass(PassRegistry &);
LLVM_ABI void initializePatchableFunctionLegacyPass(PassRegistry &);
LLVM_ABI void initializePeepholeOptimizerLegacyPass(PassRegistry &);
LLVM_ABI void initializePhiValuesWrapperPassPass(PassRegistry &);
LLVM_ABI void
initializePhysicalRegisterUsageInfoWrapperLegacyPass(PassRegistry &);
LLVM_ABI void initializePlaceBackedgeSafepointsLegacyPassPass(PassRegistry &);
LLVM_ABI void initializePostDomOnlyPrinterWrapperPassPass(PassRegistry &);
LLVM_ABI void initializePostDomOnlyViewerWrapperPassPass(PassRegistry &);
LLVM_ABI void initializePostDomPrinterWrapperPassPass(PassRegistry &);
LLVM_ABI void initializePostDomViewerWrapperPassPass(PassRegistry &);
LLVM_ABI void initializePostDominatorTreeWrapperPassPass(PassRegistry &);
LLVM_ABI void initializePostInlineEntryExitInstrumenterPass(PassRegistry &);
LLVM_ABI void initializePostMachineSchedulerLegacyPass(PassRegistry &);
LLVM_ABI void initializePostRAHazardRecognizerLegacyPass(PassRegistry &);
LLVM_ABI void initializePostRAMachineSinkingPass(PassRegistry &);
LLVM_ABI void initializePostRASchedulerLegacyPass(PassRegistry &);
LLVM_ABI void initializePreISelIntrinsicLoweringLegacyPassPass(PassRegistry &);
LLVM_ABI void initializePrintFunctionPassWrapperPass(PassRegistry &);
LLVM_ABI void initializePrintModulePassWrapperPass(PassRegistry &);
LLVM_ABI void initializeProcessImplicitDefsPass(PassRegistry &);
LLVM_ABI void initializeProfileSummaryInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializePromoteLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeRABasicPass(PassRegistry &);
LLVM_ABI void initializePseudoProbeInserterPass(PassRegistry &);
LLVM_ABI void initializeRAGreedyLegacyPass(PassRegistry &);
LLVM_ABI void initializeReachingDefAnalysisPass(PassRegistry &);
LLVM_ABI void initializeReassociateLegacyPassPass(PassRegistry &);
LLVM_ABI void
initializeRegAllocEvictionAdvisorAnalysisLegacyPass(PassRegistry &);
LLVM_ABI void initializeRegAllocFastPass(PassRegistry &);
LLVM_ABI void
initializeRegAllocPriorityAdvisorAnalysisLegacyPass(PassRegistry &);
LLVM_ABI void initializeRegAllocScoringPass(PassRegistry &);
LLVM_ABI void initializeRegBankSelectPass(PassRegistry &);
LLVM_ABI void initializeRegToMemWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeRegUsageInfoCollectorLegacyPass(PassRegistry &);
LLVM_ABI void initializeRegUsageInfoPropagationLegacyPass(PassRegistry &);
LLVM_ABI void initializeRegionInfoPassPass(PassRegistry &);
LLVM_ABI void initializeRegionOnlyPrinterPass(PassRegistry &);
LLVM_ABI void initializeRegionOnlyViewerPass(PassRegistry &);
LLVM_ABI void initializeRegionPrinterPass(PassRegistry &);
LLVM_ABI void initializeRegionViewerPass(PassRegistry &);
LLVM_ABI void initializeRegisterCoalescerLegacyPass(PassRegistry &);
LLVM_ABI void initializeRemoveLoadsIntoFakeUsesLegacyPass(PassRegistry &);
LLVM_ABI void initializeRemoveRedundantDebugValuesLegacyPass(PassRegistry &);
LLVM_ABI void initializeRenameIndependentSubregsLegacyPass(PassRegistry &);
LLVM_ABI void initializeReplaceWithVeclibLegacyPass(PassRegistry &);
LLVM_ABI void initializeResetMachineFunctionPass(PassRegistry &);
LLVM_ABI void initializeSCEVAAWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeSROALegacyPassPass(PassRegistry &);
LLVM_ABI void initializeSafeStackLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeSafepointIRVerifierPass(PassRegistry &);
LLVM_ABI void initializeSelectOptimizePass(PassRegistry &);
LLVM_ABI void initializeScalarEvolutionWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeScalarizeMaskedMemIntrinLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeScalarizerLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeScavengerTestPass(PassRegistry &);
LLVM_ABI void initializeScopedNoAliasAAWrapperPassPass(PassRegistry &);
LLVM_ABI void
initializeSeparateConstOffsetFromGEPLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeShadowStackGCLoweringPass(PassRegistry &);
LLVM_ABI void initializeShrinkWrapLegacyPass(PassRegistry &);
LLVM_ABI void initializeSingleLoopExtractorPass(PassRegistry &);
LLVM_ABI void initializeSinkingLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeSjLjEHPreparePass(PassRegistry &);
LLVM_ABI void initializeSlotIndexesWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeSpeculativeExecutionLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeSpillPlacementWrapperLegacyPass(PassRegistry &);
LLVM_ABI void initializeStackColoringLegacyPass(PassRegistry &);
LLVM_ABI void initializeStackFrameLayoutAnalysisLegacyPass(PassRegistry &);
LLVM_ABI void initializeStaticDataSplitterPass(PassRegistry &);
LLVM_ABI void initializeStackMapLivenessPass(PassRegistry &);
LLVM_ABI void initializeStackProtectorPass(PassRegistry &);
LLVM_ABI void initializeStackSafetyGlobalInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeStackSafetyInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeStackSlotColoringLegacyPass(PassRegistry &);
LLVM_ABI void
initializeStraightLineStrengthReduceLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeStripDebugMachineModulePass(PassRegistry &);
LLVM_ABI void initializeStructurizeCFGLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeTailCallElimPass(PassRegistry &);
LLVM_ABI void initializeTailDuplicateLegacyPass(PassRegistry &);
LLVM_ABI void initializeTargetLibraryInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeTargetPassConfigPass(PassRegistry &);
LLVM_ABI void initializeTargetTransformInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeTwoAddressInstructionLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeTypeBasedAAWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeTypePromotionLegacyPass(PassRegistry &);
LLVM_ABI void initializeInitUndefPass(PassRegistry &);
LLVM_ABI void initializeUniformityInfoWrapperPassPass(PassRegistry &);
LLVM_ABI void initializeUnifyLoopExitsLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeUnpackMachineBundlesPass(PassRegistry &);
LLVM_ABI void initializeUnreachableBlockElimLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeUnreachableMachineBlockElimLegacyPass(PassRegistry &);
LLVM_ABI void initializeVerifierLegacyPassPass(PassRegistry &);
LLVM_ABI void initializeVirtRegMapWrapperLegacyPass(PassRegistry &);
LLVM_ABI void initializeVirtRegRewriterLegacyPass(PassRegistry &);
LLVM_ABI void initializeWasmEHPreparePass(PassRegistry &);
LLVM_ABI void initializeWinEHPreparePass(PassRegistry &);
LLVM_ABI void initializeWriteBitcodePassPass(PassRegistry &);
LLVM_ABI void initializeXRayInstrumentationLegacyPass(PassRegistry &);
>>>>>>> 836201f1177c38f3ca0457de019bb179a04afe3c

} // end namespace llvm

#endif // LLVM_INITIALIZEPASSES_H
