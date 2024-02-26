//===- SiFive_RISCVHazardRecognizer.h - RISCV Hazard Recognizers *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling RISCV functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVHAZARDRECOGNIZER_H
#define LLVM_LIB_TARGET_RISCV_RISCVHAZARDRECOGNIZER_H

#include "llvm/CodeGen/ScheduleHazardRecognizer.h"

namespace llvm {

class RISCVMaskInstrHazardRecognizer : public ScheduleHazardRecognizer {
private:
  SUnit *LastSU = nullptr;
  int CyclesSinceLastSU = 0;

public:
  RISCVMaskInstrHazardRecognizer() {
    // Needed for isEnabled to return true.
    MaxLookAhead = 1;
  }
  HazardType getHazardType(SUnit *SU, int Stalls) override;
  void Reset() override;
  void EmitInstruction(SUnit *SU) override;
  void AdvanceCycle() override;
  void RecedeCycle() override;
};

} // end namespace llvm

#endif
