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
let BufferSize = 0 in  {
def SiFive8Div         : ProcResGroup<[SiFive8IEXQ2]>;
}
def SiFive8FloatArith  : ProcResGroup<[SiFive8FEXQ0, SiFive8FEXQ1]>;
def SiFive8F2I         : ProcResGroup<[SiFive8FEXQ0]>;
def SiFive8Load        : ProcResGroup<[SiFive8LDST0]>;
def SiFive8Store       : ProcResGroup<[SiFive8LDST1]>;
let BufferSize = 0 in {
def SiFive8FloatDiv    : ProcResGroup<[SiFive8FEXQ1]>;
}

let Latency = 9 in {
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

let Latency = 11 in {
// Integer multiplication
def : WriteRes<WriteIMul, [SiFive8MulI2F]>;
def : WriteRes<WriteIMul32, [SiFive8MulI2F]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFive8Div]> {
  let Latency = 43;
  let ResourceCycles = [34];
}
def : WriteRes<WriteIDiv32,  [SiFive8Div]> {
  let Latency = 43;
  let ResourceCycles = [34];
}

let Latency = 9 in {
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

let Latency = 11 in {
// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFive8IntArith]>;
def : WriteRes<WriteCPOP32, [SiFive8IntArith]>;
}

// Memory
def : WriteRes<WriteSTB, [SiFive8Store]>;
def : WriteRes<WriteSTH, [SiFive8Store]>;
def : WriteRes<WriteSTW, [SiFive8Store]>;
def : WriteRes<WriteSTD, [SiFive8Store]>;
def : WriteRes<WriteFST16, [SiFive8Store]>;
def : WriteRes<WriteFST32, [SiFive8Store]>;
def : WriteRes<WriteFST64, [SiFive8Store]>;

let Latency = 12 in {
def : WriteRes<WriteLDB, [SiFive8Load]>;
def : WriteRes<WriteLDH, [SiFive8Load]>;
}
let Latency = 12 in {
def : WriteRes<WriteLDW, [SiFive8Load]>;
def : WriteRes<WriteLDWU, [SiFive8Load]>;
def : WriteRes<WriteLDD, [SiFive8Load]>;
}

let Latency = 14 in {
def : WriteRes<WriteFLD16, [SiFive8Load]>;
def : WriteRes<WriteFLD32, [SiFive8Load]>;
def : WriteRes<WriteFLD64, [SiFive8Load]>;
}

// Atomic memory
def : WriteRes<WriteAtomicSTW, [SiFive8Store]>;
def : WriteRes<WriteAtomicSTD, [SiFive8Store]>;

let Latency = 11 in {
def : WriteRes<WriteAtomicW, [SiFive8Load]>;
def : WriteRes<WriteAtomicD, [SiFive8Load]>;
def : WriteRes<WriteAtomicLDW, [SiFive8Load]>;
def : WriteRes<WriteAtomicLDD, [SiFive8Load]>;
}

// Half precision.
let Latency = 12 in {
def : WriteRes<WriteFALU16, [SiFive8FloatArith]>;
def : WriteRes<WriteFMul16, [SiFive8FloatArith]>;
def : WriteRes<WriteFMA16, [SiFive8FloatArith]>;
}
let Latency = 10 in {
def : WriteRes<WriteFSGNJ16, [SiFive8FloatArith]>;
def : WriteRes<WriteFMinMax16, [SiFive8FloatArith]>;
}

def : WriteRes<WriteFDiv16, [SiFive8FloatDiv]> { let Latency = 27;
                                                 let ResourceCycles = [18]; }
def : WriteRes<WriteFSqrt16, [SiFive8FloatDiv]> { let Latency = 26;
                                                  let ResourceCycles = [17]; }

// Single precision.
let Latency = 12 in {
def : WriteRes<WriteFALU32, [SiFive8FloatArith]>;
def : WriteRes<WriteFMul32, [SiFive8FloatArith]>;
def : WriteRes<WriteFMA32, [SiFive8FloatArith]>;
}
let Latency = 10 in {
def : WriteRes<WriteFSGNJ32, [SiFive8FloatArith]>;
def : WriteRes<WriteFMinMax32, [SiFive8FloatArith]>;
}

def : WriteRes<WriteFDiv32, [SiFive8FloatDiv]> { let Latency = 27;
                                                 let ResourceCycles = [18]; }
def : WriteRes<WriteFSqrt32, [SiFive8FloatDiv]> { let Latency = 26;
                                                  let ResourceCycles = [17]; }

// Double precision
let Latency = 12 in {
def : WriteRes<WriteFALU64, [SiFive8FloatArith]>;
def : WriteRes<WriteFMul64, [SiFive8FloatArith]>;
def : WriteRes<WriteFMA64, [SiFive8FloatArith]>;
}
let Latency = 10 in {
def : WriteRes<WriteFSGNJ64, [SiFive8FloatArith]>;
def : WriteRes<WriteFMinMax64, [SiFive8FloatArith]>;
}

def : WriteRes<WriteFDiv64, [SiFive8FloatDiv]> { let Latency = 41;
                                                 let ResourceCycles = [32]; }
def : WriteRes<WriteFSqrt64, [SiFive8FloatDiv]> { let Latency = 41;
                                                  let ResourceCycles = [32]; }

// Conversions
let Latency = 10 in {
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
def : ReadAdvance<ReadJmp, 8>;
def : ReadAdvance<ReadJalr, 8>;
def : ReadAdvance<ReadCSR, 8>;
def : ReadAdvance<ReadStoreData, 10>;
def : ReadAdvance<ReadMemBase, 8>;
def : ReadAdvance<ReadIALU, 8>;
def : ReadAdvance<ReadIALU32, 8>;
def : ReadAdvance<ReadShiftImm, 8>;
def : ReadAdvance<ReadShiftImm32, 8>;
def : ReadAdvance<ReadShiftReg, 8>;
def : ReadAdvance<ReadShiftReg32, 8>;
def : ReadAdvance<ReadIDiv, 8>;
def : ReadAdvance<ReadIDiv32, 8>;
def : ReadAdvance<ReadIMul, 8>;
def : ReadAdvance<ReadIMul32, 8>;
def : ReadAdvance<ReadAtomicWA, 8>;
def : ReadAdvance<ReadAtomicWD, 8>;
def : ReadAdvance<ReadAtomicDA, 8>;
def : ReadAdvance<ReadAtomicDD, 8>;
def : ReadAdvance<ReadAtomicLDW, 8>;
def : ReadAdvance<ReadAtomicLDD, 8>;
def : ReadAdvance<ReadAtomicSTW, 8>;
def : ReadAdvance<ReadAtomicSTD, 8>;
def : ReadAdvance<ReadFMemBase, 8>;
def : ReadAdvance<ReadFALU16, 8>;
def : ReadAdvance<ReadFALU32, 8>;
def : ReadAdvance<ReadFALU64, 8>;
def : ReadAdvance<ReadFMul16, 8>;
def : ReadAdvance<ReadFMA16, 8>;
def : ReadAdvance<ReadFMul32, 8>;
def : ReadAdvance<ReadFMA32, 8>;
def : ReadAdvance<ReadFMul64, 8>;
def : ReadAdvance<ReadFMA64, 8>;
def : ReadAdvance<ReadFDiv16, 8>;
def : ReadAdvance<ReadFDiv32, 8>;
def : ReadAdvance<ReadFDiv64, 8>;
def : ReadAdvance<ReadFSqrt16, 8>;
def : ReadAdvance<ReadFSqrt32, 8>;
def : ReadAdvance<ReadFSqrt64, 8>;
def : ReadAdvance<ReadFCmp16, 8>;
def : ReadAdvance<ReadFCmp32, 8>;
def : ReadAdvance<ReadFCmp64, 8>;
def : ReadAdvance<ReadFSGNJ16, 8>;
def : ReadAdvance<ReadFSGNJ32, 8>;
def : ReadAdvance<ReadFSGNJ64, 8>;
def : ReadAdvance<ReadFMinMax16, 8>;
def : ReadAdvance<ReadFMinMax32, 8>;
def : ReadAdvance<ReadFMinMax64, 8>;
def : ReadAdvance<ReadFCvtF16ToI32, 8>;
def : ReadAdvance<ReadFCvtF16ToI64, 8>;
def : ReadAdvance<ReadFCvtF32ToI32, 8>;
def : ReadAdvance<ReadFCvtF32ToI64, 8>;
def : ReadAdvance<ReadFCvtF64ToI32, 8>;
def : ReadAdvance<ReadFCvtF64ToI64, 8>;
def : ReadAdvance<ReadFCvtI32ToF16, 8>;
def : ReadAdvance<ReadFCvtI32ToF32, 8>;
def : ReadAdvance<ReadFCvtI32ToF64, 8>;
def : ReadAdvance<ReadFCvtI64ToF16, 8>;
def : ReadAdvance<ReadFCvtI64ToF32, 8>;
def : ReadAdvance<ReadFCvtI64ToF64, 8>;
def : ReadAdvance<ReadFCvtF32ToF64, 8>;
def : ReadAdvance<ReadFCvtF64ToF32, 8>;
def : ReadAdvance<ReadFCvtF16ToF32, 8>;
def : ReadAdvance<ReadFCvtF32ToF16, 8>;
def : ReadAdvance<ReadFCvtF16ToF64, 8>;
def : ReadAdvance<ReadFCvtF64ToF16, 8>;
def : ReadAdvance<ReadFMovF16ToI16, 8>;
def : ReadAdvance<ReadFMovI16ToF16, 8>;
def : ReadAdvance<ReadFMovF32ToI32, 8>;
def : ReadAdvance<ReadFMovI32ToF32, 8>;
def : ReadAdvance<ReadFMovF64ToI64, 8>;
def : ReadAdvance<ReadFMovI64ToF64, 8>;
def : ReadAdvance<ReadFClass16, 8>;
def : ReadAdvance<ReadFClass32, 8>;
def : ReadAdvance<ReadFClass64, 8>;

// Bitmanip
def : ReadAdvance<ReadRotateImm, 8>;
def : ReadAdvance<ReadRotateImm32, 8>;
def : ReadAdvance<ReadRotateReg, 8>;
def : ReadAdvance<ReadRotateReg32, 8>;
def : ReadAdvance<ReadCLZ, 8>;
def : ReadAdvance<ReadCLZ32, 8>;
def : ReadAdvance<ReadCTZ, 8>;
def : ReadAdvance<ReadCTZ32, 8>;
def : ReadAdvance<ReadCPOP, 8>;
def : ReadAdvance<ReadCPOP32, 8>;
def : ReadAdvance<ReadORCB, 8>;
def : ReadAdvance<ReadREV8, 8>;
def : ReadAdvance<ReadSHXADD, 8>;
def : ReadAdvance<ReadSHXADD32, 8>;

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
}
