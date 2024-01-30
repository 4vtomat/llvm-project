//===--- RISCVToolchain.cpp - RISC-V ToolChain Implementations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVToolchain.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static void addMultilibsFilePaths(const Driver &D, const MultilibSet &Multilibs,
                                  const Multilib &Multilib,
                                  StringRef InstallPath,
                                  ToolChain::path_list &Paths) {
  if (const auto &PathsCallback = Multilibs.filePathsCallback())
    for (const auto &Path : PathsCallback(Multilib))
      addPathIfExists(D, InstallPath + Path, Paths);
}

// This function tests whether a gcc installation is present either
// through gcc-toolchain argument or in the same prefix where clang
// is installed. This helps decide whether to instantiate this toolchain
// or Baremetal toolchain.
bool RISCVToolChain::hasGCCToolchain(const Driver &D,
                                     const llvm::opt::ArgList &Args) {
  if (Args.getLastArg(options::OPT_gcc_toolchain))
    return true;

  SmallString<128> GCCDir;
  llvm::sys::path::append(GCCDir, D.Dir, "..", D.getTargetTriple(),
                          "lib/crt0.o");
  return llvm::sys::fs::exists(GCCDir);
}

/// RISC-V Toolchain
RISCVToolChain::RISCVToolChain(const Driver &D, const llvm::Triple &Triple,
                               const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  GCCInstallation.init(Triple, Args);
  if (GCCInstallation.isValid()) {
    Multilibs = GCCInstallation.getMultilibs();
    SelectedMultilibs.assign({GCCInstallation.getMultilib()});
    path_list &Paths = getFilePaths();
    // Add toolchain/multilib specific file paths.
    addMultilibsFilePaths(D, Multilibs, SelectedMultilibs.back(),
                          GCCInstallation.getInstallPath(), Paths);
    getFilePaths().push_back(GCCInstallation.getInstallPath().str());
    ToolChain::path_list &PPaths = getProgramPaths();
    // Multilib cross-compiler GCC installations put ld in a triple-prefixed
    // directory off of the parent of the GCC installation.
    PPaths.push_back(Twine(GCCInstallation.getParentLibPath() + "/../" +
                           GCCInstallation.getTriple().str() + "/bin")
                         .str());
    PPaths.push_back((GCCInstallation.getParentLibPath() + "/../bin").str());
  } else {
    getProgramPaths().push_back(D.Dir);
  }
  getFilePaths().push_back(computeSysRoot() + "/lib");

#if SIFIVE_CUSTOMIZATION
  // Check if we are using special libc base on --specs option.
  // The 'SpecialLibc' variable is used to hack flags at different stages.
  SpecialLibc = LibcType::None;
  for (auto Specs : Args.getAllArgValues(clang::driver::options::OPT_specs_EQ)) {
    if (Specs == "nano.specs")
      SpecialLibc = LibcType::NewlibNano;
    else if (Specs == "gloss-segger.specs")
      SpecialLibc = LibcType::SeggerGloss;
    else if (Specs == "metal-segger.specs")
      SpecialLibc = LibcType::SeggerMetal;
  }
#endif
}

Tool *RISCVToolChain::buildLinker() const {
#if SIFIVE_CUSTOMIZATION
  return new tools::RISCV::Linker(*this, SpecialLibc);
#else
  return new tools::RISCV::Linker(*this);
#endif
}

ToolChain::RuntimeLibType RISCVToolChain::GetDefaultRuntimeLibType() const {
  return GCCInstallation.isValid() ?
    ToolChain::RLT_Libgcc : ToolChain::RLT_CompilerRT;
}

ToolChain::UnwindLibType
RISCVToolChain::GetUnwindLibType(const llvm::opt::ArgList &Args) const {
  return ToolChain::UNW_None;
}

void RISCVToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind) const {
  CC1Args.push_back("-nostdsysteminc");

#if SIFIVE_CUSTOMIZATION
  switch (SpecialLibc) {
    case LibcType::SeggerGloss:
    case LibcType::SeggerMetal:
        CC1Args.push_back("-D__SEGGER_LIBC__");
      break;
    default:
      break;
  }
#endif
}

void RISCVToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                               ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

#if SIFIVE_CUSTOMIZATION
  // Segger includ path should be searched firstly.
  SmallString<128> SeggerDir(computeSysRoot());
  switch (SpecialLibc) {
    case LibcType::SeggerGloss:
    case LibcType::SeggerMetal:
      llvm::sys::path::append(SeggerDir, "include/segger");
      addSystemInclude(DriverArgs, CC1Args, SeggerDir.str());
      break;
    default:
      break;
  }
#endif

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> Dir(getDriver().ResourceDir);
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc)) {
    SmallString<128> Dir(computeSysRoot());
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }
}

void RISCVToolChain::addLibStdCxxIncludePaths(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  const GCCVersion &Version = GCCInstallation.getVersion();
  StringRef TripleStr = GCCInstallation.getTriple().str();
  const Multilib &Multilib = GCCInstallation.getMultilib();
  addLibStdCXXIncludePaths(computeSysRoot() + "/include/c++/" + Version.Text,
                           TripleStr, Multilib.includeSuffix(), DriverArgs,
                           CC1Args);
}

std::string RISCVToolChain::computeSysRoot() const {
  if (!getDriver().SysRoot.empty())
    return getDriver().SysRoot;

  SmallString<128> SysRootDir;
  if (GCCInstallation.isValid()) {
    StringRef LibDir = GCCInstallation.getParentLibPath();
    StringRef TripleStr = GCCInstallation.getTriple().str();
    llvm::sys::path::append(SysRootDir, LibDir, "..", TripleStr);
  } else {
    // Use the triple as provided to the driver. Unlike the parsed triple
    // this has not been normalized to always contain every field.
    llvm::sys::path::append(SysRootDir, getDriver().Dir, "..",
                            getDriver().getTargetTriple());
  }

  if (!llvm::sys::fs::exists(SysRootDir))
    return std::string();

  return std::string(SysRootDir);
}

void RISCV::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const ToolChain &ToolChain = getToolChain();
  const Driver &D = ToolChain.getDriver();
  ArgStringList CmdArgs;

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (Args.hasArg(options::OPT_mno_relax))
    CmdArgs.push_back("--no-relax");

  bool IsRV64 = ToolChain.getArch() == llvm::Triple::riscv64;
  CmdArgs.push_back("-m");
  if (IsRV64) {
    CmdArgs.push_back("elf64lriscv");
  } else {
    CmdArgs.push_back("elf32lriscv");
  }
  CmdArgs.push_back("-X");

#if SIFIVE_CUSTOMIZATION
  if (D.isUsingLTO()) {
    assert(!Inputs.empty() && "Must have at least one input.");
    addLTOOptions(ToolChain, Args, CmdArgs, Output, Inputs[0],
                  D.getLTOMode() == LTOK_Thin);
  }

  if (!Args.getLastArgValue(options::OPT_fuse_ld_EQ).equals_insensitive("lld")) {
    if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
      if (A->getOption().matches(options::OPT_O)) {
        StringRef OOpt = A->getValue();
        if (OOpt == "s" || OOpt == "z") {
          CmdArgs.push_back("--relax-zcmt");
        }
      }
    }
  }
#endif // SIFIVE_CUSTOMIZATION

  std::string Linker = getToolChain().GetLinkerPath();

  bool WantCRTs =
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles);

  const char *crtbegin, *crtend;
  auto RuntimeLib = ToolChain.GetRuntimeLibType(Args);
  if (RuntimeLib == ToolChain::RLT_Libgcc) {
    crtbegin = "crtbegin.o";
    crtend = "crtend.o";
  } else {
    assert (RuntimeLib == ToolChain::RLT_CompilerRT);
    crtbegin = ToolChain.getCompilerRTArgString(Args, "crtbegin",
                                                ToolChain::FT_Object);
    crtend = ToolChain.getCompilerRTArgString(Args, "crtend",
                                              ToolChain::FT_Object);
  }

  if (WantCRTs) {
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt0.o")));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtbegin)));
  }

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);

  Args.addAllArgs(CmdArgs, {options::OPT_L, options::OPT_u});

  ToolChain.AddFilePathLibArgs(Args, CmdArgs);
  Args.addAllArgs(CmdArgs, {options::OPT_T_Group, options::OPT_s,
                            options::OPT_t, options::OPT_r});

  // TODO: add C++ includes and libs if compiling C++.

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    if (D.CCCIsCXX()) {
      if (ToolChain.ShouldLinkCXXStdlib(Args))
        ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-lm");
    }
    CmdArgs.push_back("--start-group");
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lgloss");
    CmdArgs.push_back("--end-group");
    AddRunTimeLibs(ToolChain, ToolChain.getDriver(), CmdArgs, Args);
  }


#if SIFIVE_CUSTOMIZATION
  auto *GlossIdx = CmdArgs.begin();

  switch (SpecialLibc) {
    case LibcType::NewlibNano:
        for (size_t i = 0; i < CmdArgs.size(); ++i) {
          StringRef Arg = CmdArgs[i];
          if (Arg == "-lc")
            CmdArgs[i] = "-lc_nano";
          if (Arg == "-lgloss")
            CmdArgs[i] = "-lgloss_nano";
          if (Arg == "-lm")
            CmdArgs[i] = "-lm_nano";
          if (Arg == "-lg")
            CmdArgs[i] = "-lg_nano";
        }
      break;
    case LibcType::SeggerGloss:
        for (size_t i = 0; i < CmdArgs.size(); ++i) {
          StringRef Arg = CmdArgs[i];
          if (Arg == "-lc")
            CmdArgs[i] = "-lc_segger";
          if (Arg == "-lgloss")
            CmdArgs[i] = "-lgloss-segger";
          // libg.a is the same as libc.a.
          // See https://www.cygwin.com/bugzilla/show_bug.cgi?id=26102#c1
          if (Arg == "-lg")
            CmdArgs[i] = "lc_segger";
        }
      break;
    case LibcType::SeggerMetal:
        for (size_t i = 0; i < CmdArgs.size(); ++i) {
          StringRef Arg = CmdArgs[i];
          if (Arg == "-lc")
            CmdArgs[i] = "-lc_segger";
          if (Arg == "-lgloss") {
            CmdArgs[i] = "-lmetal";
            GlossIdx += i;
          }
          // libg.a is the same as libc.a.
          // See https://www.cygwin.com/bugzilla/show_bug.cgi?id=26102#c1
          if (Arg == "-lg")
            CmdArgs[i] = "lc_segger";
        }
        // Insert metal-segger in group
        if (GlossIdx != CmdArgs.begin())
          CmdArgs.insert(GlossIdx, "-lmetal-segger");
      break;
    default:
      break;
  }
#endif

  if (WantCRTs)
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtend)));

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Args.MakeArgString(Linker),
      CmdArgs, Inputs, Output));
}
// RISCV tools end.
