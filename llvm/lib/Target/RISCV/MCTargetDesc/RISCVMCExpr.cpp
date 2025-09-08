//===-- RISCVMCExpr.cpp - RISC-V specific MC expression classes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the assembly expression modifiers
// accepted by the RISC-V architecture (e.g. ":lo12:", ":gottprel_g1:", ...).
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVAsmBackend.h"
#include "MCTargetDesc/RISCVMCAsmInfo.h"
#include "RISCVFixupKinds.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "riscvmcexpr"

RISCV::Specifier RISCV::parseSpecifierName(StringRef name) {
  return StringSwitch<RISCV::Specifier>(name)
      .Case("lo", RISCV::S_LO)
      .Case("hi", ELF::R_RISCV_HI20)
      .Case("pcrel_lo", RISCV::S_PCREL_LO)
      .Case("pcrel_hi", ELF::R_RISCV_PCREL_HI20)
      .Case("got_pcrel_hi", ELF::R_RISCV_GOT_HI20)
      .Case("tprel_lo", RISCV::S_TPREL_LO)
      .Case("tprel_hi", ELF::R_RISCV_TPREL_HI20)
      .Case("tprel_add", ELF::R_RISCV_TPREL_ADD)
      .Case("tls_ie_pcrel_hi", ELF::R_RISCV_TLS_GOT_HI20)
      .Case("tls_gd_pcrel_hi", ELF::R_RISCV_TLS_GD_HI20)
      .Case("tlsdesc_hi", ELF::R_RISCV_TLSDESC_HI20)
      .Case("tlsdesc_load_lo", ELF::R_RISCV_TLSDESC_LOAD_LO12)
      .Case("tlsdesc_add_lo", ELF::R_RISCV_TLSDESC_ADD_LO12)
      .Case("tlsdesc_call", ELF::R_RISCV_TLSDESC_CALL)
#if SIFIVE_CUSTOMIZATION
      .Case("gprel_lo", RISCV::S_GPREL_LO)
      .Case("gprel_hi", RISCV::S_GPREL_HI)
      .Case("gprel", RISCV::S_GPREL_ADD)
      .Case("got_gprel_lo", RISCV::S_GOT_GPREL_LO)
      .Case("got_gprel_hi", RISCV::S_GOT_GPREL_HI)
      .Case("got_gprel", RISCV::S_GOT_GPREL_ADD)
      .Case("tls_ie_gprel_lo", RISCV::S_TLS_GOT_GPREL_LO)
      .Case("tls_ie_gprel_hi", RISCV::S_TLS_GOT_GPREL_HI)
      .Case("tls_ie_gprel", RISCV::S_TLS_GOT_GPREL_ADD)
      .Case("tls_gd_gprel_lo", RISCV::S_TLS_GD_GPREL_LO)
      .Case("tls_gd_gprel_hi", RISCV::S_TLS_GD_GPREL_HI)
      .Case("tls_gd_gprel", RISCV::S_TLS_GD_GPREL_ADD)
#endif // SIFIVE_CUSTOMIZATION
      .Case("qc.abs20", RISCV::S_QC_ABS20)
      // Used in data directives
      .Case("pltpcrel", ELF::R_RISCV_PLT32)
      .Case("gotpcrel", ELF::R_RISCV_GOT32_PCREL)
      .Default(0);
}

StringRef RISCV::getSpecifierName(Specifier S) {
  switch (S) {
  case RISCV::S_None:
    llvm_unreachable("not used as %specifier()");
  case RISCV::S_LO:
    return "lo";
  case ELF::R_RISCV_HI20:
    return "hi";
  case RISCV::S_PCREL_LO:
    return "pcrel_lo";
  case ELF::R_RISCV_PCREL_HI20:
    return "pcrel_hi";
  case ELF::R_RISCV_GOT_HI20:
    return "got_pcrel_hi";
  case RISCV::S_TPREL_LO:
    return "tprel_lo";
  case ELF::R_RISCV_TPREL_HI20:
    return "tprel_hi";
  case ELF::R_RISCV_TPREL_ADD:
    return "tprel_add";
  case ELF::R_RISCV_TLS_GOT_HI20:
    return "tls_ie_pcrel_hi";
  case ELF::R_RISCV_TLSDESC_HI20:
    return "tlsdesc_hi";
  case ELF::R_RISCV_TLSDESC_LOAD_LO12:
    return "tlsdesc_load_lo";
  case ELF::R_RISCV_TLSDESC_ADD_LO12:
    return "tlsdesc_add_lo";
  case ELF::R_RISCV_TLSDESC_CALL:
    return "tlsdesc_call";
  case ELF::R_RISCV_TLS_GD_HI20:
    return "tls_gd_pcrel_hi";
  case ELF::R_RISCV_CALL_PLT:
    return "call_plt";
  case ELF::R_RISCV_32_PCREL:
    return "32_pcrel";
#if SIFIVE_CUSTOMIZATION
  case RISCV::S_GPREL_LO:
    return "gprel_lo";
  case RISCV::S_GPREL_HI:
    return "gprel_hi";
  case RISCV::S_GPREL_ADD:
    return "gprel";
  case RISCV::S_GOT_GPREL_LO:
    return "got_gprel_lo";
  case RISCV::S_GOT_GPREL_HI:
    return "got_gprel_hi";
  case RISCV::S_GOT_GPREL_ADD:
    return "got_gprel";
  case RISCV::S_TLS_GOT_GPREL_LO:
    return "tls_ie_gprel_lo";
  case RISCV::S_TLS_GOT_GPREL_HI:
    return "tls_ie_gprel_hi";
  case RISCV::S_TLS_GOT_GPREL_ADD:
    return "tls_ie_gprel";
  case RISCV::S_TLS_GD_GPREL_LO:
    return "tls_gd_gprel_lo";
  case RISCV::S_TLS_GD_GPREL_HI:
    return "tls_gd_gprel_hi";
  case RISCV::S_TLS_GD_GPREL_ADD:
    return "tls_gd_gprel";
#endif // SIFIVE_CUSTOMIZATION
  case ELF::R_RISCV_GOT32_PCREL:
    return "gotpcrel";
  case ELF::R_RISCV_PLT32:
    return "pltpcrel";
  case RISCV::S_QC_ABS20:
    return "qc.abs20";
  }
  llvm_unreachable("Invalid ELF symbol kind");
}
