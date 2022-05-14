//==- RISCVSchedSiFive9.td - SiFive9 Scheduling Definitions --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// SiFive9 machine model for scheduling and other instruction cost heuristics.
// https://sifive.atlassian.net/wiki/spaces/AR/pages/1844513085/FLC+Draft+Proposal+-+Bordeaux+SPECint2006+Performance+Uplift
def SiFive9Model : SchedMachineModel {
  let IssueWidth = 4;         // 4 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 56; // Max micro-ops that can be buffered.
  let LoadLatency = 4;        // Cycles for loads to access the cache.
  let MispredictPenalty = 9;  // Extra cycles for a mispredicted branch.
  let PostRAScheduler = true;
  let UnsupportedFeatures = [HasVInstructions];
  let CompleteModel = false;
}

let SchedModel = SiFive9Model in {

def SiFive9IEXQ0       : ProcResource<1>;
def SiFive9IEXQ1       : ProcResource<1>;
def SiFive9IEXQ2       : ProcResource<1>;
def SiFive9IEXQ3       : ProcResource<1>;
def SiFive9FEXQ0       : ProcResource<1>;
def SiFive9FEXQ1       : ProcResource<1>;

// Two Load/Store ports that can issue either two loads, two stores, or one load
// and one store (P550 has one load and one separate store pipe).
def SiFive9LDST       : ProcResource<2>;

// 4-wide pipeline with 4 ALU pipes.
def SiFive9IntArith    : ProcResGroup<[SiFive9IEXQ0, SiFive9IEXQ1, SiFive9IEXQ2, SiFive9IEXQ3]>;
def SiFive9Branch      : ProcResGroup<[SiFive9IEXQ0]>;
def SiFive9SYS         : ProcResGroup<[SiFive9IEXQ1]>;
def SiFive9MulI2F      : ProcResGroup<[SiFive9IEXQ2]>;
let BufferSize = 0 in  {
def SiFive9Div         : ProcResGroup<[SiFive9IEXQ2]>;
}
def SiFive9FloatArith  : ProcResGroup<[SiFive9FEXQ0, SiFive9FEXQ1]>;
def SiFive9F2I         : ProcResGroup<[SiFive9FEXQ0]>;
let BufferSize = 0 in {
def SiFive9FloatDiv    : ProcResGroup<[SiFive9FEXQ1]>;
}

let Latency = 9 in {
// Integer arithmetic and logic
def : WriteRes<WriteIALU, [SiFive9IntArith]>;
def : WriteRes<WriteIALU32, [SiFive9IntArith]>;
def : WriteRes<WriteShiftImm, [SiFive9IntArith]>;
def : WriteRes<WriteShiftImm32, [SiFive9IntArith]>;
def : WriteRes<WriteShiftReg, [SiFive9IntArith]>;
def : WriteRes<WriteShiftReg32, [SiFive9IntArith]>;
// Branching
def : WriteRes<WriteJmp, [SiFive9Branch]>;
def : WriteRes<WriteJal, [SiFive9Branch]>;
def : WriteRes<WriteJalr, [SiFive9Branch]>;
def : WriteRes<WriteJmpReg, [SiFive9Branch]>;
}

let Latency = 11 in {
// Integer multiplication
def : WriteRes<WriteIMul, [SiFive9MulI2F]>;
def : WriteRes<WriteIMul32, [SiFive9MulI2F]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFive9Div]> {
  let Latency = 43;
  let ResourceCycles = [34];
}
def : WriteRes<WriteIDiv32,  [SiFive9Div]> {
  let Latency = 43;
  let ResourceCycles = [34];
}

let Latency = 9 in {
// Bitmanip
def : WriteRes<WriteRotateImm, [SiFive9IntArith]>;
def : WriteRes<WriteRotateImm32, [SiFive9IntArith]>;
def : WriteRes<WriteRotateReg, [SiFive9IntArith]>;
def : WriteRes<WriteRotateReg32, [SiFive9IntArith]>;

def : WriteRes<WriteCLZ, [SiFive9IntArith]>;
def : WriteRes<WriteCLZ32, [SiFive9IntArith]>;
def : WriteRes<WriteCTZ, [SiFive9IntArith]>;
def : WriteRes<WriteCTZ32, [SiFive9IntArith]>;

def : WriteRes<WriteORCB, [SiFive9IntArith]>;

def : WriteRes<WriteREV8, [SiFive9IntArith]>;

def : WriteRes<WriteSHXADD, [SiFive9IntArith]>;
def : WriteRes<WriteSHXADD32, [SiFive9IntArith]>;
}

let Latency = 11 in {
// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFive9IntArith]>;
def : WriteRes<WriteCPOP32, [SiFive9IntArith]>;
}

// Memory
def : WriteRes<WriteSTB, [SiFive9LDST]>;
def : WriteRes<WriteSTH, [SiFive9LDST]>;
def : WriteRes<WriteSTW, [SiFive9LDST]>;
def : WriteRes<WriteSTD, [SiFive9LDST]>;
def : WriteRes<WriteFST16, [SiFive9LDST]>;
def : WriteRes<WriteFST32, [SiFive9LDST]>;
def : WriteRes<WriteFST64, [SiFive9LDST]>;

let Latency = 12 in {
def : WriteRes<WriteLDB, [SiFive9LDST]>;
def : WriteRes<WriteLDH, [SiFive9LDST]>;
}
let Latency = 12 in {
def : WriteRes<WriteLDW, [SiFive9LDST]>;
def : WriteRes<WriteLDWU, [SiFive9LDST]>;
def : WriteRes<WriteLDD, [SiFive9LDST]>;
}

let Latency = 14 in {
def : WriteRes<WriteFLD16, [SiFive9LDST]>;
def : WriteRes<WriteFLD32, [SiFive9LDST]>;
def : WriteRes<WriteFLD64, [SiFive9LDST]>;
}

// Atomic memory
def : WriteRes<WriteAtomicSTW, [SiFive9LDST]>;
def : WriteRes<WriteAtomicSTD, [SiFive9LDST]>;

let Latency = 11 in {
def : WriteRes<WriteAtomicW, [SiFive9LDST]>;
def : WriteRes<WriteAtomicD, [SiFive9LDST]>;
def : WriteRes<WriteAtomicLDW, [SiFive9LDST]>;
def : WriteRes<WriteAtomicLDD, [SiFive9LDST]>;
}

// Half precision.
let Latency = 12 in {
def : WriteRes<WriteFALU16, [SiFive9FloatArith]>;
def : WriteRes<WriteFMul16, [SiFive9FloatArith]>;
def : WriteRes<WriteFMA16, [SiFive9FloatArith]>;
}
let Latency = 10 in {
def : WriteRes<WriteFSGNJ16, [SiFive9FloatArith]>;
def : WriteRes<WriteFMinMax16, [SiFive9FloatArith]>;
}

def : WriteRes<WriteFDiv16, [SiFive9FloatDiv]> { let Latency = 27;
                                                 let ResourceCycles = [18]; }
def : WriteRes<WriteFSqrt16, [SiFive9FloatDiv]> { let Latency = 26;
                                                  let ResourceCycles = [17]; }

// Single precision.
let Latency = 12 in {
def : WriteRes<WriteFALU32, [SiFive9FloatArith]>;
def : WriteRes<WriteFMul32, [SiFive9FloatArith]>;
def : WriteRes<WriteFMA32, [SiFive9FloatArith]>;
}
let Latency = 10 in {
def : WriteRes<WriteFSGNJ32, [SiFive9FloatArith]>;
def : WriteRes<WriteFMinMax32, [SiFive9FloatArith]>;
}

def : WriteRes<WriteFDiv32, [SiFive9FloatDiv]> { let Latency = 27;
                                                 let ResourceCycles = [18]; }
def : WriteRes<WriteFSqrt32, [SiFive9FloatDiv]> { let Latency = 26;
                                                  let ResourceCycles = [17]; }

// Double precision
let Latency = 12 in {
def : WriteRes<WriteFALU64, [SiFive9FloatArith]>;
def : WriteRes<WriteFMul64, [SiFive9FloatArith]>;
def : WriteRes<WriteFMA64, [SiFive9FloatArith]>;
}
let Latency = 10 in {
def : WriteRes<WriteFSGNJ64, [SiFive9FloatArith]>;
def : WriteRes<WriteFMinMax64, [SiFive9FloatArith]>;
}

def : WriteRes<WriteFDiv64, [SiFive9FloatDiv]> { let Latency = 41;
                                                 let ResourceCycles = [32]; }
def : WriteRes<WriteFSqrt64, [SiFive9FloatDiv]> { let Latency = 41;
                                                  let ResourceCycles = [32]; }

// Conversions
let Latency = 10 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFive9MulI2F]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFive9MulI2F]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFive9MulI2F]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFive9MulI2F]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFive9MulI2F]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFive9MulI2F]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFive9F2I]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFive9F2I]>;

def : WriteRes<WriteFClass16, [SiFive9FloatArith]>;
def : WriteRes<WriteFClass32, [SiFive9FloatArith]>;
def : WriteRes<WriteFClass64, [SiFive9FloatArith]>;
def : WriteRes<WriteFCmp16, [SiFive9FloatArith]>;
def : WriteRes<WriteFCmp32, [SiFive9FloatArith]>;
def : WriteRes<WriteFCmp64, [SiFive9FloatArith]>;
def : WriteRes<WriteFMovI16ToF16, [SiFive9FloatArith]>;
def : WriteRes<WriteFMovF16ToI16, [SiFive9FloatArith]>;
def : WriteRes<WriteFMovI32ToF32, [SiFive9FloatArith]>;
def : WriteRes<WriteFMovF32ToI32, [SiFive9FloatArith]>;
def : WriteRes<WriteFMovI64ToF64, [SiFive9FloatArith]>;
def : WriteRes<WriteFMovF64ToI64, [SiFive9FloatArith]>;
}

// Others
def : WriteRes<WriteCSR, [SiFive9SYS]>;
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
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbe;
defm : UnsupportedSchedZbf;
defm : UnsupportedSchedZbm;
defm : UnsupportedSchedZbp;
defm : UnsupportedSchedZbr;
defm : UnsupportedSchedZbt;
defm : UnsupportedSchedSFB;
}
