//===--- RISCVToolchain.h - RISC-V ToolChain Implementations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_RISCVTOOLCHAIN_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_RISCVTOOLCHAIN_H

#include "Gnu.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {

#if SIFIVE_CUSTOMIZATION
// For hacking compile options base on --specs options.
enum class LibcType {
  None,
  NewlibNano,
  SeggerGloss, // Should match https://github.com/sifive/segger_libc/blob/sifive-dev/src/gloss-segger.specs
  SeggerMetal // Should match https://github.com/sifive/segger_libc/blob/sifive-dev/src/metal-segger.specs
};
#endif

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY RISCVToolChain : public Generic_ELF {
public:
  RISCVToolChain(const Driver &D, const llvm::Triple &Triple,
                 const llvm::opt::ArgList &Args);

  static bool hasGCCToolchain(const Driver &D, const llvm::opt::ArgList &Args);
  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args,
                             Action::OffloadKind) const override;
  RuntimeLibType GetDefaultRuntimeLibType() const override;
  UnwindLibType
  GetUnwindLibType(const llvm::opt::ArgList &Args) const override;
#if SIFIVE_CUSTOMIZATION
  bool HasNativeLLVMSupport() const override { return true; }
#endif
  UnwindTableLevel
  getDefaultUnwindTableLevel(const llvm::opt::ArgList &Args) const override;
  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void
  addLibStdCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                           llvm::opt::ArgStringList &CC1Args) const override;
#if SIFIVE_CUSTOMIZATION
  LibcType SpecialLibc;
#endif

protected:
  Tool *buildLinker() const override;

private:
  std::string computeSysRoot() const override;
};

} // end namespace toolchains

namespace tools {
namespace RISCV {
class LLVM_LIBRARY_VISIBILITY Linker final : public Tool {
public:
#if SIFIVE_CUSTOMIZATION
  Linker(const ToolChain &TC, LibcType libc) : Tool("RISCV::Linker", "ld", TC), SpecialLibc(libc) {}
#else
  Linker(const ToolChain &TC, LibcType libc) : Tool("RISCV::Linker", "ld", TC)
#endif
  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
#if SIFIVE_CUSTOMIZATION
  LibcType SpecialLibc;
#endif
};
} // end namespace RISCV
} // end namespace tools

} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_RISCVTOOLCHAIN_H
