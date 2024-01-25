//===- opt.cpp - The LLVM Modular Optimizer -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimizations may be specified an arbitrary number of times on the command
// line, They are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include <functional>

namespace llvm {
class PassBuilder;
}

extern "C" int optMain(int argc, char **argv,
                       llvm::ArrayRef<std::function<void(llvm::PassBuilder &)>>
                           PassBuilderCallbacks);

<<<<<<< HEAD
// For use in NPM transition. Currently this contains most codegen-specific
// passes. Remove passes from here when porting to the NPM.
// TODO: use a codegen version of PassRegistry.def/PassBuilder::is*Pass() once
// it exists.
static bool shouldPinPassToLegacyPM(StringRef Pass) {
  std::vector<StringRef> PassNameExactToIgnore = {
      "nvvm-reflect",
      "nvvm-intr-range",
      "amdgpu-simplifylib",
      "amdgpu-image-intrinsic-opt",
      "amdgpu-usenative",
      "amdgpu-promote-alloca",
      "amdgpu-promote-alloca-to-vector",
      "amdgpu-lower-kernel-attributes",
      "amdgpu-propagate-attributes-early",
      "amdgpu-propagate-attributes-late",
      "amdgpu-unify-metadata",
      "amdgpu-printf-runtime-binding",
      "amdgpu-always-inline"};
  if (llvm::is_contained(PassNameExactToIgnore, Pass))
    return false;

  std::vector<StringRef> PassNamePrefix = {
      "x86-",    "xcore-", "wasm-",  "systemz-", "ppc-",    "nvvm-",
      "nvptx-",  "mips-",  "lanai-", "hexagon-", "bpf-",    "avr-",
      "thumb2-", "arm-",   "si-",    "gcn-",     "amdgpu-", "aarch64-",
      "amdgcn-", "polly-", "riscv-", "dxil-"};
  std::vector<StringRef> PassNameContain = {"-eh-prepare"};
  std::vector<StringRef> PassNameExact = {
      "safe-stack",
      "cost-model",
      "codegenprepare",
      "interleaved-load-combine",
      "unreachableblockelim",
      "verify-safepoint-ir",
      "atomic-expand",
      "expandvp",
      "mve-tail-predication",
      "interleaved-access",
      "global-merge",
      "pre-isel-intrinsic-lowering",
      "expand-reductions",
      "indirectbr-expand",
      "generic-to-nvvm",
      "expand-memcmp",
      "loop-reduce",
      "lower-amx-type",
      "lower-amx-intrinsics",
      "polyhedral-info",
      "print-polyhedral-info",
      "replace-with-veclib",
      "jmc-instrumenter",
      "dot-regions",
      "dot-regions-only",
      "view-regions",
      "view-regions-only",
      "select-optimize",
      "expand-large-div-rem",
      "structurizecfg",
      "fix-irreducible",
      "expand-large-fp-convert",
      "callbrprepare",
      "expand-powi",
  };
  for (const auto &P : PassNamePrefix)
    if (Pass.starts_with(P))
      return true;
  for (const auto &P : PassNameContain)
    if (Pass.contains(P))
      return true;
  return llvm::is_contained(PassNameExact, Pass);
}

// For use in NPM transition.
static bool shouldForceLegacyPM() {
  for (const auto &P : PassList) {
    StringRef Arg = P->getPassArgument();
    if (shouldPinPassToLegacyPM(Arg))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// main for opt
//
int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeScalarOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeTarget(Registry);
  // For codegen passes, only passes that do IR to IR transformation are
  // supported.
  initializeExpandLargeDivRemLegacyPassPass(Registry);
  initializeExpandLargeFpConvertLegacyPassPass(Registry);
  initializeExpandMemCmpLegacyPassPass(Registry);
#if SIFIVE_CUSTOMIZATION
  initializeExpandPowiLegacyPassPass(Registry);
#endif // SIFIVE_CUSTOMIZATION
  initializeScalarizeMaskedMemIntrinLegacyPassPass(Registry);
  initializeSelectOptimizePass(Registry);
  initializeCallBrPreparePass(Registry);
  initializeCodeGenPrepareLegacyPassPass(Registry);
  initializeAtomicExpandPass(Registry);
  initializeWinEHPreparePass(Registry);
  initializeDwarfEHPrepareLegacyPassPass(Registry);
  initializeSafeStackLegacyPassPass(Registry);
  initializeSjLjEHPreparePass(Registry);
  initializePreISelIntrinsicLoweringLegacyPassPass(Registry);
  initializeGlobalMergePass(Registry);
  initializeIndirectBrExpandLegacyPassPass(Registry);
  initializeInterleavedLoadCombinePass(Registry);
  initializeInterleavedAccessPass(Registry);
  initializeUnreachableBlockElimLegacyPassPass(Registry);
  initializeExpandReductionsPass(Registry);
  initializeExpandVectorPredicationPass(Registry);
  initializeWasmEHPreparePass(Registry);
  initializeWriteBitcodePassPass(Registry);
  initializeReplaceWithVeclibLegacyPass(Registry);
  initializeJMCInstrumenterPass(Registry);

  SmallVector<PassPlugin, 1> PluginList;
  PassPlugins.setCallback([&](const std::string &PluginPath) {
    auto Plugin = PassPlugin::Load(PluginPath);
    if (!Plugin)
      report_fatal_error(Plugin.takeError(), /*gen_crash_diag=*/false);
    PluginList.emplace_back(Plugin.get());
  });

  // Register the Target and CPU printer for --version.
  cl::AddExtraVersionPrinter(sys::printDefaultTargetAndDetectedCPU);

  cl::ParseCommandLineOptions(argc, argv,
    "llvm .bc -> .bc modular optimizer and analysis printer\n");

  // RemoveDIs debug-info transition: tests may request that we /try/ to use the
  // new debug-info format, if it's built in.
#ifdef EXPERIMENTAL_DEBUGINFO_ITERATORS
  if (TryUseNewDbgInfoFormat) {
    // If LLVM was built with support for this, turn the new debug-info format
    // on.
    UseNewDbgInfoFormat = true;
  }
#endif
  (void)TryUseNewDbgInfoFormat;

  LLVMContext Context;

  // TODO: remove shouldForceLegacyPM().
  const bool UseNPM = (!EnableLegacyPassManager && !shouldForceLegacyPM()) ||
                      PassPipeline.getNumOccurrences() > 0;

  if (UseNPM && !PassList.empty()) {
    errs() << "The `opt -passname` syntax for the new pass manager is "
              "not supported, please use `opt -passes=<pipeline>` (or the `-p` "
              "alias for a more concise version).\n";
    errs() << "See https://llvm.org/docs/NewPassManager.html#invoking-opt "
              "for more details on the pass pipeline syntax.\n\n";
    return 1;
  }

  if (!UseNPM && PluginList.size()) {
    errs() << argv[0] << ": " << PassPlugins.ArgStr
           << " specified with legacy PM.\n";
    return 1;
  }

  // FIXME: once the legacy PM code is deleted, move runPassPipeline() here and
  // construct the PassBuilder before parsing IR so we can reuse the same
  // PassBuilder for print passes.
  if (PrintPasses) {
    printPasses(outs());
    return 0;
  }

  TimeTracerRAII TimeTracer(argv[0]);

  SMDiagnostic Err;

  Context.setDiscardValueNames(DiscardValueNames);
  if (!DisableDITypeMap)
    Context.enableDebugTypeODRUniquing();

  Expected<std::unique_ptr<ToolOutputFile>> RemarksFileOrErr =
      setupLLVMOptimizationRemarks(Context, RemarksFilename, RemarksPasses,
                                   RemarksFormat, RemarksWithHotness,
                                   RemarksHotnessThreshold);
  if (Error E = RemarksFileOrErr.takeError()) {
    errs() << toString(std::move(E)) << '\n';
    return 1;
  }
  std::unique_ptr<ToolOutputFile> RemarksFile = std::move(*RemarksFileOrErr);

  // Load the input module...
  auto SetDataLayout = [&](StringRef IRTriple,
                           StringRef IRLayout) -> std::optional<std::string> {
    // Data layout specified on the command line has the highest priority.
    if (!ClDataLayout.empty())
      return ClDataLayout;
    // If an explicit data layout is already defined in the IR, don't infer.
    if (!IRLayout.empty())
      return std::nullopt;

    // If an explicit triple was specified (either in the IR or on the
    // command line), use that to infer the default data layout. However, the
    // command line target triple should override the IR file target triple.
    std::string TripleStr =
        TargetTriple.empty() ? IRTriple.str() : Triple::normalize(TargetTriple);
    // If the triple string is still empty, we don't fall back to
    // sys::getDefaultTargetTriple() since we do not want to have differing
    // behaviour dependent on the configured default triple. Therefore, if the
    // user did not pass -mtriple or define an explicit triple/datalayout in
    // the IR, we should default to an empty (default) DataLayout.
    if (TripleStr.empty())
      return std::nullopt;
    // Otherwise we infer the DataLayout from the target machine.
    Expected<std::unique_ptr<TargetMachine>> ExpectedTM =
        codegen::createTargetMachineForTriple(TripleStr, GetCodeGenOptLevel());
    if (!ExpectedTM) {
      errs() << argv[0] << ": warning: failed to infer data layout: "
             << toString(ExpectedTM.takeError()) << "\n";
      return std::nullopt;
    }
    return (*ExpectedTM)->createDataLayout().getStringRepresentation();
  };
  std::unique_ptr<Module> M;
  if (NoUpgradeDebugInfo)
    M = parseAssemblyFileWithIndexNoUpgradeDebugInfo(
            InputFilename, Err, Context, nullptr, SetDataLayout)
            .Mod;
  else
    M = parseIRFile(InputFilename, Err, Context,
                    ParserCallbacks(SetDataLayout));

  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  // Strip debug info before running the verifier.
  if (StripDebug)
    StripDebugInfo(*M);

  // Erase module-level named metadata, if requested.
  if (StripNamedMetadata) {
    while (!M->named_metadata_empty()) {
      NamedMDNode *NMD = &*M->named_metadata_begin();
      M->eraseNamedMetadata(NMD);
    }
  }

  // If we are supposed to override the target triple, do so now.
  if (!TargetTriple.empty())
    M->setTargetTriple(Triple::normalize(TargetTriple));

  // Immediately run the verifier to catch any problems before starting up the
  // pass pipelines.  Otherwise we can crash on broken code during
  // doInitialization().
  if (!NoVerify && verifyModule(*M, &errs())) {
    errs() << argv[0] << ": " << InputFilename
           << ": error: input module is broken!\n";
    return 1;
  }

  // Enable testing of whole program devirtualization on this module by invoking
  // the facility for updating public visibility to linkage unit visibility when
  // specified by an internal option. This is normally done during LTO which is
  // not performed via opt.
  updateVCallVisibilityInModule(
      *M,
      /*WholeProgramVisibilityEnabledInLTO=*/false,
      // FIXME: These need linker information via a
      // TBD new interface.
      /*DynamicExportSymbols=*/{},
      /*ValidateAllVtablesHaveTypeInfos=*/false,
      /*IsVisibleToRegularObj=*/[](StringRef) { return true; });

  // Figure out what stream we are supposed to write to...
  std::unique_ptr<ToolOutputFile> Out;
  std::unique_ptr<ToolOutputFile> ThinLinkOut;
  if (NoOutput) {
    if (!OutputFilename.empty())
      errs() << "WARNING: The -o (output filename) option is ignored when\n"
                "the --disable-output option is used.\n";
  } else {
    // Default to standard output.
    if (OutputFilename.empty())
      OutputFilename = "-";

    std::error_code EC;
    sys::fs::OpenFlags Flags =
        OutputAssembly ? sys::fs::OF_TextWithCRLF : sys::fs::OF_None;
    Out.reset(new ToolOutputFile(OutputFilename, EC, Flags));
    if (EC) {
      errs() << EC.message() << '\n';
      return 1;
    }

    if (!ThinLinkBitcodeFile.empty()) {
      ThinLinkOut.reset(
          new ToolOutputFile(ThinLinkBitcodeFile, EC, sys::fs::OF_None));
      if (EC) {
        errs() << EC.message() << '\n';
        return 1;
      }
    }
  }

  Triple ModuleTriple(M->getTargetTriple());
  std::string CPUStr, FeaturesStr;
  std::unique_ptr<TargetMachine> TM;
  if (ModuleTriple.getArch()) {
    CPUStr = codegen::getCPUStr();
    FeaturesStr = codegen::getFeaturesStr();
    Expected<std::unique_ptr<TargetMachine>> ExpectedTM =
        codegen::createTargetMachineForTriple(ModuleTriple.str(),
                                              GetCodeGenOptLevel());
    if (auto E = ExpectedTM.takeError()) {
      errs() << argv[0] << ": WARNING: failed to create target machine for '"
             << ModuleTriple.str() << "': " << toString(std::move(E)) << "\n";
    } else {
      TM = std::move(*ExpectedTM);
    }
  } else if (ModuleTriple.getArchName() != "unknown" &&
             ModuleTriple.getArchName() != "") {
    errs() << argv[0] << ": unrecognized architecture '"
           << ModuleTriple.getArchName() << "' provided.\n";
    return 1;
  }

  // Override function attributes based on CPUStr, FeaturesStr, and command line
  // flags.
  codegen::setFunctionAttributes(CPUStr, FeaturesStr, *M);

  // If the output is set to be emitted to standard out, and standard out is a
  // console, print out a warning message and refuse to do it.  We don't
  // impress anyone by spewing tons of binary goo to a terminal.
  if (!Force && !NoOutput && !OutputAssembly)
    if (CheckBitcodeOutputToConsole(Out->os()))
      NoOutput = true;

  if (OutputThinLTOBC) {
    M->addModuleFlag(Module::Error, "EnableSplitLTOUnit", SplitLTOUnit);
    if (UnifiedLTO)
      M->addModuleFlag(Module::Error, "UnifiedLTO", 1);
  }

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(ModuleTriple);

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (DisableSimplifyLibCalls)
    TLII.disableAllFunctions();
  else {
    // Disable individual builtin functions in TargetLibraryInfo.
    LibFunc F;
    for (auto &FuncName : DisableBuiltins)
      if (TLII.getLibFunc(FuncName, F))
        TLII.setUnavailable(F);
      else {
        errs() << argv[0] << ": cannot disable nonexistent builtin function "
               << FuncName << '\n';
        return 1;
      }
  }

  if (UseNPM) {
    if (legacy::debugPassSpecified()) {
      errs() << "-debug-pass does not work with the new PM, either use "
                "-debug-pass-manager, or use the legacy PM\n";
      return 1;
    }
    auto NumOLevel = OptLevelO0 + OptLevelO1 + OptLevelO2 + OptLevelO3 +
                     OptLevelOs + OptLevelOz;
    if (NumOLevel > 1) {
      errs() << "Cannot specify multiple -O#\n";
      return 1;
    }
    if (NumOLevel > 0 && (PassPipeline.getNumOccurrences() > 0)) {
      errs() << "Cannot specify -O# and --passes=/--foo-pass, use "
                "-passes='default<O#>,other-pass'\n";
      return 1;
    }
    std::string Pipeline = PassPipeline;

    if (OptLevelO0)
      Pipeline = "default<O0>";
    if (OptLevelO1)
      Pipeline = "default<O1>";
    if (OptLevelO2)
      Pipeline = "default<O2>";
    if (OptLevelO3)
      Pipeline = "default<O3>";
    if (OptLevelOs)
      Pipeline = "default<Os>";
    if (OptLevelOz)
      Pipeline = "default<Oz>";
    OutputKind OK = OK_NoOutput;
    if (!NoOutput)
      OK = OutputAssembly
               ? OK_OutputAssembly
               : (OutputThinLTOBC ? OK_OutputThinLTOBitcode : OK_OutputBitcode);

    VerifierKind VK = VK_VerifyOut;
    if (NoVerify)
      VK = VK_NoVerifier;
    else if (VerifyEach)
      VK = VK_VerifyEachPass;

    // The user has asked to use the new pass manager and provided a pipeline
    // string. Hand off the rest of the functionality to the new code for that
    // layer.
    return runPassPipeline(argv[0], *M, TM.get(), &TLII, Out.get(),
                           ThinLinkOut.get(), RemarksFile.get(), Pipeline,
                           PluginList, OK, VK, PreserveAssemblyUseListOrder,
                           PreserveBitcodeUseListOrder, EmitSummaryIndex,
                           EmitModuleHash, EnableDebugify,
                           VerifyDebugInfoPreserve, UnifiedLTO)
               ? 0
               : 1;
  }

  if (OptLevelO0 || OptLevelO1 || OptLevelO2 || OptLevelOs || OptLevelOz ||
      OptLevelO3) {
    errs() << "Cannot use -O# with legacy PM.\n";
    return 1;
  }
  if (EmitSummaryIndex) {
    errs() << "Cannot use -module-summary with legacy PM.\n";
    return 1;
  }
  if (EmitModuleHash) {
    errs() << "Cannot use -module-hash with legacy PM.\n";
    return 1;
  }
  if (OutputThinLTOBC) {
    errs() << "Cannot use -thinlto-bc with legacy PM.\n";
    return 1;
  }
  // Create a PassManager to hold and optimize the collection of passes we are
  // about to build. If the -debugify-each option is set, wrap each pass with
  // the (-check)-debugify passes.
  DebugifyCustomPassManager Passes;
  DebugifyStatsMap DIStatsMap;
  DebugInfoPerPass DebugInfoBeforePass;
  if (DebugifyEach) {
    Passes.setDebugifyMode(DebugifyMode::SyntheticDebugInfo);
    Passes.setDIStatsMap(DIStatsMap);
  } else if (VerifyEachDebugInfoPreserve) {
    Passes.setDebugifyMode(DebugifyMode::OriginalDebugInfo);
    Passes.setDebugInfoBeforePass(DebugInfoBeforePass);
    if (!VerifyDIPreserveExport.empty())
      Passes.setOrigDIVerifyBugsReportFilePath(VerifyDIPreserveExport);
  }

  bool AddOneTimeDebugifyPasses =
      (EnableDebugify && !DebugifyEach) ||
      (VerifyDebugInfoPreserve && !VerifyEachDebugInfoPreserve);

  Passes.add(new TargetLibraryInfoWrapperPass(TLII));

  // Add internal analysis passes from the target machine.
  Passes.add(createTargetTransformInfoWrapperPass(TM ? TM->getTargetIRAnalysis()
                                                     : TargetIRAnalysis()));

  if (AddOneTimeDebugifyPasses) {
    if (EnableDebugify) {
      Passes.setDIStatsMap(DIStatsMap);
      Passes.add(createDebugifyModulePass());
    } else if (VerifyDebugInfoPreserve) {
      Passes.setDebugInfoBeforePass(DebugInfoBeforePass);
      Passes.add(createDebugifyModulePass(
          DebugifyMode::OriginalDebugInfo, "",
          &(Passes.getDebugInfoPerPass())));
    }
  }

  if (TM) {
    // FIXME: We should dyn_cast this when supported.
    auto &LTM = static_cast<LLVMTargetMachine &>(*TM);
    Pass *TPC = LTM.createPassConfig(Passes);
    Passes.add(TPC);
  }

  // Create a new optimization pass for each one specified on the command line
  for (unsigned i = 0; i < PassList.size(); ++i) {
    const PassInfo *PassInf = PassList[i];
    if (PassInf->getNormalCtor()) {
      Pass *P = PassInf->getNormalCtor()();
      if (P) {
        // Add the pass to the pass manager.
        Passes.add(P);
        // If we are verifying all of the intermediate steps, add the verifier.
        if (VerifyEach)
          Passes.add(createVerifierPass());
      }
    } else
      errs() << argv[0] << ": cannot create pass: "
             << PassInf->getPassName() << "\n";
  }

  // Check that the module is well formed on completion of optimization
  if (!NoVerify && !VerifyEach)
    Passes.add(createVerifierPass());

  if (AddOneTimeDebugifyPasses) {
    if (EnableDebugify)
      Passes.add(createCheckDebugifyModulePass(false));
    else if (VerifyDebugInfoPreserve) {
      if (!VerifyDIPreserveExport.empty())
        Passes.setOrigDIVerifyBugsReportFilePath(VerifyDIPreserveExport);
      Passes.add(createCheckDebugifyModulePass(
          false, "", nullptr, DebugifyMode::OriginalDebugInfo,
          &(Passes.getDebugInfoPerPass()), VerifyDIPreserveExport));
    }
  }

  // In run twice mode, we want to make sure the output is bit-by-bit
  // equivalent if we run the pass manager again, so setup two buffers and
  // a stream to write to them. Note that llc does something similar and it
  // may be worth to abstract this out in the future.
  SmallVector<char, 0> Buffer;
  SmallVector<char, 0> FirstRunBuffer;
  std::unique_ptr<raw_svector_ostream> BOS;
  raw_ostream *OS = nullptr;

  const bool ShouldEmitOutput = !NoOutput;

  // Write bitcode or assembly to the output as the last step...
  if (ShouldEmitOutput || RunTwice) {
    assert(Out);
    OS = &Out->os();
    if (RunTwice) {
      BOS = std::make_unique<raw_svector_ostream>(Buffer);
      OS = BOS.get();
    }
    if (OutputAssembly)
      Passes.add(createPrintModulePass(*OS, "", PreserveAssemblyUseListOrder));
    else
      Passes.add(createBitcodeWriterPass(*OS, PreserveBitcodeUseListOrder));
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  if (!RunTwice) {
    // Now that we have all of the passes ready, run them.
    Passes.run(*M);
  } else {
    // If requested, run all passes twice with the same pass manager to catch
    // bugs caused by persistent state in the passes.
    std::unique_ptr<Module> M2(CloneModule(*M));
    // Run all passes on the original module first, so the second run processes
    // the clone to catch CloneModule bugs.
    Passes.run(*M);
    FirstRunBuffer = Buffer;
    Buffer.clear();

    Passes.run(*M2);

    // Compare the two outputs and make sure they're the same
    assert(Out);
    if (Buffer.size() != FirstRunBuffer.size() ||
        (memcmp(Buffer.data(), FirstRunBuffer.data(), Buffer.size()) != 0)) {
      errs()
          << "Running the pass manager twice changed the output.\n"
             "Writing the result of the second run to the specified output.\n"
             "To generate the one-run comparison binary, just run without\n"
             "the compile-twice option\n";
      if (ShouldEmitOutput) {
        Out->os() << BOS->str();
        Out->keep();
      }
      if (RemarksFile)
        RemarksFile->keep();
      return 1;
    }
    if (ShouldEmitOutput)
      Out->os() << BOS->str();
  }

  if (DebugifyEach && !DebugifyExport.empty())
    exportDebugifyStats(DebugifyExport, Passes.getDebugifyStatsMap());

  // Declare success.
  if (!NoOutput)
    Out->keep();

  if (RemarksFile)
    RemarksFile->keep();

  if (ThinLinkOut)
    ThinLinkOut->keep();

  return 0;
}
=======
int main(int argc, char **argv) { return optMain(argc, argv, {}); }
>>>>>>> llvm/main
