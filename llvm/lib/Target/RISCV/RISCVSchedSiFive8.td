//==- RISCVSchedSiFive8.td - SiFive8 Scheduling Definitions --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// SiFive8 machine model for scheduling and other instruction cost heuristics.
// https://github.com/sifive/federation/blob/master/specs/design/standard/u84.json
def SiFive8Model : SchedMachineModel {
  let IssueWidth = 3;         // 3 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 56; // Max micro-ops that can be buffered.
  let LoadLatency = 4;        // Cycles for loads to access the cache.
  let MispredictPenalty = 9;  // Extra cycles for a mispredicted branch.
  let PostRAScheduler = true;
  let UnsupportedFeatures = [HasVInstructions];
  let CompleteModel = false;
}

// The SiFive8 microarchitecure has 6 pipelines:
// Three pipelines for integer operations.
// Two pipelines for FPU operations.
// One pipeline for Load/Store operations.
let SchedModel = SiFive8Model in {

def SiFive8IEXQ0       : ProcResource<1>;
def SiFive8IEXQ1       : ProcResource<1>;
def SiFive8IEXQ2       : ProcResource<1>;
def SiFive8FEXQ0       : ProcResource<1>;
def SiFive8FEXQ1       : ProcResource<1>;
// Define two resources for the load/store pipeline. It is better to schedule
// load/store interchange for U8.
def SiFive8LDST0       : ProcResource<1>;
def SiFive8LDST1       : ProcResource<1>;

def SiFive8IntArith    : ProcResGroup<[SiFive8IEXQ0, SiFive8IEXQ1, SiFive8IEXQ2]>;
def SiFive8Branch      : ProcResGroup<[SiFive8IEXQ0]>;
def SiFive8SYS         : ProcResGroup<[SiFive8IEXQ1]>;
def SiFive8MulI2F      : ProcResGroup<[SiFive8IEXQ2]>;
def SiFive8Div         : ProcResource<1>;

def SiFive8FloatArith  : ProcResGroup<[SiFive8FEXQ0, SiFive8FEXQ1]>;
def SiFive8F2I         : ProcResGroup<[SiFive8FEXQ0]>;
def SiFive8Load        : ProcResGroup<[SiFive8LDST0]>;
def SiFive8Store       : ProcResGroup<[SiFive8LDST1]>;
def SiFive8FloatDiv    : ProcResource<1>;

let Latency = 1 in {
// Integer arithmetic and logic
def : WriteRes<WriteIALU, [SiFive8IntArith]>;
def : WriteRes<WriteIALU32, [SiFive8IntArith]>;
def : WriteRes<WriteShiftImm, [SiFive8IntArith]>;
def : WriteRes<WriteShiftImm32, [SiFive8IntArith]>;
def : WriteRes<WriteShiftReg, [SiFive8IntArith]>;
def : WriteRes<WriteShiftReg32, [SiFive8IntArith]>;
// Branching
def : WriteRes<WriteJmp, [SiFive8Branch]>;
def : WriteRes<WriteJal, [SiFive8Branch]>;
def : WriteRes<WriteJalr, [SiFive8Branch]>;
def : WriteRes<WriteJmpReg, [SiFive8Branch]>;
}

let Latency = 3 in {
// Integer multiplication
def : WriteRes<WriteIMul, [SiFive8MulI2F]>;
def : WriteRes<WriteIMul32, [SiFive8MulI2F]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFive8IEXQ2, SiFive8Div]> {
  let Latency = 35;
  let ResourceCycles = [1, 34];
}
def : WriteRes<WriteIDiv32,  [SiFive8IEXQ2, SiFive8Div]> {
  let Latency = 35;
  let ResourceCycles = [1, 34];
}

let Latency = 1 in {
// Bitmanip
def : WriteRes<WriteRotateImm, [SiFive8IntArith]>;
def : WriteRes<WriteRotateImm32, [SiFive8IntArith]>;
def : WriteRes<WriteRotateReg, [SiFive8IntArith]>;
def : WriteRes<WriteRotateReg32, [SiFive8IntArith]>;

def : WriteRes<WriteCLZ, [SiFive8IntArith]>;
def : WriteRes<WriteCLZ32, [SiFive8IntArith]>;
def : WriteRes<WriteCTZ, [SiFive8IntArith]>;
def : WriteRes<WriteCTZ32, [SiFive8IntArith]>;

def : WriteRes<WriteORCB, [SiFive8IntArith]>;

def : WriteRes<WriteREV8, [SiFive8IntArith]>;

def : WriteRes<WriteSHXADD, [SiFive8IntArith]>;
def : WriteRes<WriteSHXADD32, [SiFive8IntArith]>;
}

let Latency = 3 in {
// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFive8IntArith]>;
def : WriteRes<WriteCPOP32, [SiFive8IntArith]>;
}

// Memory
let Latency = 1 in {
def : WriteRes<WriteSTB, [SiFive8Store]>;
def : WriteRes<WriteSTH, [SiFive8Store]>;
def : WriteRes<WriteSTW, [SiFive8Store]>;
def : WriteRes<WriteSTD, [SiFive8Store]>;
def : WriteRes<WriteFST16, [SiFive8Store]>;
def : WriteRes<WriteFST32, [SiFive8Store]>;
def : WriteRes<WriteFST64, [SiFive8Store]>;
}
let Latency = 4 in {
def : WriteRes<WriteLDB, [SiFive8Load]>;
def : WriteRes<WriteLDH, [SiFive8Load]>;
}
let Latency = 4 in {
def : WriteRes<WriteLDW, [SiFive8Load]>;
def : WriteRes<WriteLDWU, [SiFive8Load]>;
def : WriteRes<WriteLDD, [SiFive8Load]>;
}

let Latency = 6 in {
def : WriteRes<WriteFLD16, [SiFive8Load]>;
def : WriteRes<WriteFLD32, [SiFive8Load]>;
def : WriteRes<WriteFLD64, [SiFive8Load]>;
}

// Atomic memory
let Latency = 3 in {
def : WriteRes<WriteAtomicSTW, [SiFive8Store]>;
def : WriteRes<WriteAtomicSTD, [SiFive8Store]>;
def : WriteRes<WriteAtomicW, [SiFive8Load]>;
def : WriteRes<WriteAtomicD, [SiFive8Load]>;
def : WriteRes<WriteAtomicLDW, [SiFive8Load]>;
def : WriteRes<WriteAtomicLDD, [SiFive8Load]>;
}

// Half precision.
let Latency = 4 in {
def : WriteRes<WriteFALU16, [SiFive8FloatArith]>;
def : WriteRes<WriteFMul16, [SiFive8FloatArith]>;
def : WriteRes<WriteFMA16, [SiFive8FloatArith]>;
}
let Latency = 2 in {
def : WriteRes<WriteFSGNJ16, [SiFive8FloatArith]>;
def : WriteRes<WriteFMinMax16, [SiFive8FloatArith]>;
}

def : WriteRes<WriteFDiv16, [SiFive8FEXQ1, SiFive8FloatDiv]> {
  let Latency = 19;
  let ResourceCycles = [1, 18];
}
def : WriteRes<WriteFSqrt16, [SiFive8FEXQ1, SiFive8FloatDiv]> {
  let Latency = 18;
  let ResourceCycles = [1, 17];
}

// Single precision.
let Latency = 4 in {
def : WriteRes<WriteFALU32, [SiFive8FloatArith]>;
def : WriteRes<WriteFMul32, [SiFive8FloatArith]>;
def : WriteRes<WriteFMA32, [SiFive8FloatArith]>;
}
let Latency = 2 in {
def : WriteRes<WriteFSGNJ32, [SiFive8FloatArith]>;
def : WriteRes<WriteFMinMax32, [SiFive8FloatArith]>;
}

def : WriteRes<WriteFDiv32, [SiFive8FEXQ1, SiFive8FloatDiv]> {
  let Latency = 19;
  let ResourceCycles = [1, 18];
}
def : WriteRes<WriteFSqrt32, [SiFive8FEXQ1, SiFive8FloatDiv]> {
  let Latency = 18;
  let ResourceCycles = [1, 17];
}

// Double precision
let Latency = 4 in {
def : WriteRes<WriteFALU64, [SiFive8FloatArith]>;
def : WriteRes<WriteFMul64, [SiFive8FloatArith]>;
def : WriteRes<WriteFMA64, [SiFive8FloatArith]>;
}
let Latency = 2 in {
def : WriteRes<WriteFSGNJ64, [SiFive8FloatArith]>;
def : WriteRes<WriteFMinMax64, [SiFive8FloatArith]>;
}

def : WriteRes<WriteFDiv64, [SiFive8FEXQ1, SiFive8FloatDiv]> {
  let Latency = 33;
  let ResourceCycles = [1, 32];
}
def : WriteRes<WriteFSqrt64, [SiFive8FEXQ1, SiFive8FloatDiv]> {
  let Latency = 33;
  let ResourceCycles = [1, 32];
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFive8MulI2F]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFive8MulI2F]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFive8MulI2F]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFive8MulI2F]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFive8MulI2F]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFive8MulI2F]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFive8F2I]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFive8F2I]>;

def : WriteRes<WriteFClass16, [SiFive8FloatArith]>;
def : WriteRes<WriteFClass32, [SiFive8FloatArith]>;
def : WriteRes<WriteFClass64, [SiFive8FloatArith]>;
def : WriteRes<WriteFCmp16, [SiFive8FloatArith]>;
def : WriteRes<WriteFCmp32, [SiFive8FloatArith]>;
def : WriteRes<WriteFCmp64, [SiFive8FloatArith]>;
def : WriteRes<WriteFMovI16ToF16, [SiFive8FloatArith]>;
def : WriteRes<WriteFMovF16ToI16, [SiFive8FloatArith]>;
def : WriteRes<WriteFMovI32ToF32, [SiFive8FloatArith]>;
def : WriteRes<WriteFMovF32ToI32, [SiFive8FloatArith]>;
def : WriteRes<WriteFMovI64ToF64, [SiFive8FloatArith]>;
def : WriteRes<WriteFMovF64ToI64, [SiFive8FloatArith]>;
}

// Others
def : WriteRes<WriteCSR, [SiFive8SYS]>;
def : WriteRes<WriteNop, []>;

// FIXME: This could be better modeled by looking at the regclasses of the operands.
def : InstRW<[WriteIALU, ReadIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFALU16, 0>;
def : ReadAdvance<ReadFALU32, 0>;
def : ReadAdvance<ReadFALU64, 0>;
def : ReadAdvance<ReadFMul16, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFDiv16, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt16, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp16, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ16, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax16, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF16ToI32, 0>;
def : ReadAdvance<ReadFCvtF16ToI64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFCvtF16ToF32, 0>;
def : ReadAdvance<ReadFCvtF32ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF16, 0>;
def : ReadAdvance<ReadFMovF16ToI16, 0>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

// Bitmanip
def : ReadAdvance<ReadRotateImm, 0>;
def : ReadAdvance<ReadRotateImm32, 0>;
def : ReadAdvance<ReadRotateReg, 0>;
def : ReadAdvance<ReadRotateReg32, 0>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : ReadAdvance<ReadORCB, 0>;
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbe;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbf;
defm : UnsupportedSchedZbm;
defm : UnsupportedSchedZbp;
defm : UnsupportedSchedZbr;
defm : UnsupportedSchedZbt;
defm : UnsupportedSchedSFB;
}
