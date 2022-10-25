//==- RISCVSchedSiFiveP600.td - SiFiveP600 Scheduling Defs ---*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// SiFiveP600 machine model for scheduling and other instruction cost heuristics.
// https://sifive.atlassian.net/wiki/spaces/AR/pages/1844513085/FLC+Draft+Proposal+-+Bordeaux+SPECint2006+Performance+Uplift
def SiFiveP600Model : SchedMachineModel {
  let IssueWidth = 4;         // 4 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 160; // Max micro-ops that can be buffered.
  let LoadLatency = 4;        // Cycles for loads to access the cache.
  let MispredictPenalty = 9;  // Extra cycles for a mispredicted branch.
  let PostRAScheduler = true;
  let UnsupportedFeatures = [HasVInstructions];
  let CompleteModel = false;
}

let SchedModel = SiFiveP600Model in {

def SiFiveP600IEXQ0       : ProcResource<1>;
def SiFiveP600IEXQ1       : ProcResource<1>;
def SiFiveP600IEXQ2       : ProcResource<1>;
def SiFiveP600IEXQ3       : ProcResource<1>;
def SiFiveP600FEXQ0       : ProcResource<1>;
def SiFiveP600FEXQ1       : ProcResource<1>;

// Two Load/Store ports that can issue either two loads, two stores, or one load
// and one store (P550 has one load and one separate store pipe).
def SiFiveP600LDST       : ProcResource<2>;

// 4-wide pipeline with 4 ALU pipes.
def SiFiveP600IntArith    : ProcResGroup<[SiFiveP600IEXQ0, SiFiveP600IEXQ1, SiFiveP600IEXQ2, SiFiveP600IEXQ3]>;
defvar SiFiveP600SYS      = SiFiveP600IEXQ0;
defvar SiFiveP600CMOV     = SiFiveP600IEXQ0;
defvar SiFiveP600MulI2F   = SiFiveP600IEXQ1;
def SiFiveP600Branch      : ProcResGroup<[SiFiveP600IEXQ2, SiFiveP600IEXQ3]>;
def SiFiveP600Div         : ProcResource<1>;

def SiFiveP600FloatArith  : ProcResGroup<[SiFiveP600FEXQ0, SiFiveP600FEXQ1]>;
defvar SiFiveP600F2I      = SiFiveP600FEXQ0;
def SiFiveP600FloatDiv    : ProcResource<1>;

let Latency = 1 in {
// Integer arithmetic and logic
def : WriteRes<WriteIALU, [SiFiveP600IntArith]>;
def : WriteRes<WriteIALU32, [SiFiveP600IntArith]>;
def : WriteRes<WriteShiftImm, [SiFiveP600IntArith]>;
def : WriteRes<WriteShiftImm32, [SiFiveP600IntArith]>;
def : WriteRes<WriteShiftReg, [SiFiveP600IntArith]>;
def : WriteRes<WriteShiftReg32, [SiFiveP600IntArith]>;
// Branching
def : WriteRes<WriteJmp, [SiFiveP600Branch]>;
def : WriteRes<WriteJal, [SiFiveP600Branch]>;
def : WriteRes<WriteJalr, [SiFiveP600Branch]>;
def : WriteRes<WriteJmpReg, [SiFiveP600Branch]>;
}

// CMOV
def : WriteRes<WriteCMOV, [SiFiveP600Branch, SiFiveP600CMOV]> {
  let Latency = 2;
  let NumMicroOps = 2;
}

let Latency = 3 in {
// Integer multiplication
def : WriteRes<WriteIMul, [SiFiveP600MulI2F]>;
def : WriteRes<WriteIMul32, [SiFiveP600MulI2F]>;
// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFiveP600MulI2F]>;
def : WriteRes<WriteCPOP32, [SiFiveP600MulI2F]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFiveP600MulI2F, SiFiveP600Div]> {
  let Latency = 35;
  let ResourceCycles = [1, 34];
}
def : WriteRes<WriteIDiv32,  [SiFiveP600MulI2F, SiFiveP600Div]> {
  let Latency = 20;
  let ResourceCycles = [1, 19];
}

let Latency = 1 in {
// Bitmanip
def : WriteRes<WriteRotateImm, [SiFiveP600IntArith]>;
def : WriteRes<WriteRotateImm32, [SiFiveP600IntArith]>;
def : WriteRes<WriteRotateReg, [SiFiveP600IntArith]>;
def : WriteRes<WriteRotateReg32, [SiFiveP600IntArith]>;

def : WriteRes<WriteCLZ, [SiFiveP600IntArith]>;
def : WriteRes<WriteCLZ32, [SiFiveP600IntArith]>;
def : WriteRes<WriteCTZ, [SiFiveP600IntArith]>;
def : WriteRes<WriteCTZ32, [SiFiveP600IntArith]>;

def : WriteRes<WriteORCB, [SiFiveP600IntArith]>;

def : WriteRes<WriteREV8, [SiFiveP600IntArith]>;

def : WriteRes<WriteSHXADD, [SiFiveP600IntArith]>;
def : WriteRes<WriteSHXADD32, [SiFiveP600IntArith]>;
}

// Memory
let Latency = 1 in {
def : WriteRes<WriteSTB, [SiFiveP600LDST]>;
def : WriteRes<WriteSTH, [SiFiveP600LDST]>;
def : WriteRes<WriteSTW, [SiFiveP600LDST]>;
def : WriteRes<WriteSTD, [SiFiveP600LDST]>;
def : WriteRes<WriteFST16, [SiFiveP600LDST]>;
def : WriteRes<WriteFST32, [SiFiveP600LDST]>;
def : WriteRes<WriteFST64, [SiFiveP600LDST]>;
}
let Latency = 4 in {
def : WriteRes<WriteLDB, [SiFiveP600LDST]>;
def : WriteRes<WriteLDH, [SiFiveP600LDST]>;
}
let Latency = 4 in {
def : WriteRes<WriteLDW, [SiFiveP600LDST]>;
def : WriteRes<WriteLDWU, [SiFiveP600LDST]>;
def : WriteRes<WriteLDD, [SiFiveP600LDST]>;
}

let Latency = 6 in {
def : WriteRes<WriteFLD16, [SiFiveP600LDST]>;
def : WriteRes<WriteFLD32, [SiFiveP600LDST]>;
def : WriteRes<WriteFLD64, [SiFiveP600LDST]>;
}

// Atomic memory
let Latency = 3 in {
def : WriteRes<WriteAtomicSTW, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicSTD, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicW, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicD, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicLDW, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicLDD, [SiFiveP600LDST]>;
}

// Floating point
let Latency = 2 in {
def : WriteRes<WriteFAdd16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFAdd32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFAdd64, [SiFiveP600FloatArith]>;
}
let Latency = 3 in {
def : WriteRes<WriteFMul16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMul32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMul64, [SiFiveP600FloatArith]>;
}
let Latency = 4 in {
def : WriteRes<WriteFMA16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMA32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMA64, [SiFiveP600FloatArith]>;
}

let Latency = 2 in {
def : WriteRes<WriteFSGNJ16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFSGNJ32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFSGNJ64, [SiFiveP600FloatArith]>;

def : WriteRes<WriteFMinMax16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMinMax32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMinMax64, [SiFiveP600FloatArith]>;
}

// Half precision.
def : WriteRes<WriteFDiv16, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 4;
  let ResourceCycles = [1, 4];
}
def : WriteRes<WriteFSqrt16, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 18;
  let ResourceCycles = [1, 17];
}

// Single precision.
def : WriteRes<WriteFDiv32, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 6;
  let ResourceCycles = [1, 6];
}
def : WriteRes<WriteFSqrt32, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 18;
  let ResourceCycles = [1, 17];
}

// Double precision
def : WriteRes<WriteFDiv64, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 11;
  let ResourceCycles = [1, 11];
}
def : WriteRes<WriteFSqrt64, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 33;
  let ResourceCycles = [1, 32];
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFiveP600FloatArith]>;

def : WriteRes<WriteFClass16, [SiFiveP600F2I]>;
def : WriteRes<WriteFClass32, [SiFiveP600F2I]>;
def : WriteRes<WriteFClass64, [SiFiveP600F2I]>;
def : WriteRes<WriteFCmp16, [SiFiveP600F2I]>;
def : WriteRes<WriteFCmp32, [SiFiveP600F2I]>;
def : WriteRes<WriteFCmp64, [SiFiveP600F2I]>;
def : WriteRes<WriteFMovI16ToF16, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFMovF16ToI16, [SiFiveP600F2I]>;
def : WriteRes<WriteFMovI32ToF32, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFMovF32ToI32, [SiFiveP600F2I]>;
def : WriteRes<WriteFMovI64ToF64, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFMovF64ToI64, [SiFiveP600F2I]>;
}

// Others
def : WriteRes<WriteCSR, [SiFiveP600SYS]>;
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
def : ReadAdvance<ReadFAdd16, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
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

def : ReadAdvance<ReadCMOV, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedSFB;
}
