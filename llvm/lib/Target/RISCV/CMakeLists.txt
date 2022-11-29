add_llvm_component_group(RISCV)

set(LLVM_TARGET_DEFINITIONS RISCV.td)

tablegen(LLVM RISCVGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM RISCVGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM RISCVGenCompressInstEmitter.inc -gen-compress-inst-emitter)
tablegen(LLVM RISCVGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM RISCVGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM RISCVGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM RISCVGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM RISCVGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM RISCVGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM RISCVGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM RISCVGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM RISCVGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM RISCVGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(RISCVCommonTableGen)

add_llvm_target(RISCVCodeGen
  RISCVAsmPrinter.cpp
  RISCVCodeGenPrepare.cpp
  RISCVMakeCompressible.cpp
  RISCVExpandAtomicPseudoInsts.cpp
  RISCVExpandPseudoInsts.cpp
  RISCVFrameLowering.cpp
  RISCVGatherScatterLowering.cpp
  SiFive_RISCVInsertNTLHInsts.cpp
  RISCVInsertVSETVLI.cpp
  SiFive_RISCVInsertVXRMWrite.cpp
  SiFive_RISCVCleanupVXRM.cpp
  RISCVInstrInfo.cpp
<<<<<<< HEAD
  RISCVInstructionSelector.cpp
  SiFive_RISCVInstCombineIntrinsic.cpp
  RISCVISelDAGToDAG.cpp
  RISCVISelLowering.cpp
  SiFive_RISCVLateCodeGenPrepare.cpp
  RISCVLegalizerInfo.cpp
  SiFive_RISCVMachineConstProp.cpp
=======
  RISCVISelDAGToDAG.cpp
  RISCVISelLowering.cpp
>>>>>>> upstream/main
  RISCVMachineFunctionInfo.cpp
  RISCVMacroFusion.cpp
  RISCVMCInstLower.cpp
  RISCVMergeBaseOffset.cpp
  RISCVRedundantCopyElimination.cpp
  RISCVRegisterInfo.cpp
  RISCVSExtWRemoval.cpp
  RISCVSubtarget.cpp
  RISCVTargetMachine.cpp
  RISCVTargetObjectFile.cpp
  RISCVTargetTransformInfo.cpp
<<<<<<< HEAD
  SiFive_RISCVPostRAExpandPseudoInsts.cpp
  SiFive_RISCVTypePromotion.cpp
=======
  GISel/RISCVCallLowering.cpp
  GISel/RISCVInstructionSelector.cpp
  GISel/RISCVLegalizerInfo.cpp
  GISel/RISCVRegisterBankInfo.cpp
>>>>>>> upstream/main

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  Core
  IPO
  CodeGen
  MC
  Scalar
  RISCVDesc
  RISCVInfo
  SelectionDAG
  Support
  Target
  TransformUtils
  GlobalISel

  ADD_TO_COMPONENT
  RISCV
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(MCA)
add_subdirectory(TargetInfo)
