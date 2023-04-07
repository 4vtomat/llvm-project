//==- RISCVSchedSiFive6.td - SiFive6 Scheduling Definitions --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// SiFive6 machine model for scheduling and other instruction cost heuristics.
// https://sifive.atlassian.net/wiki/spaces/PMGT/pages/1891696749/Pellet+Core+Series
// https://github.com/sifive/arch-specs/blob/master/assets/images/pellet/Pellet%20Pipeline%20-%20Diagram%201.png
// Above Pellet pipeline diagram is unclear, so implementation is based on sifive-7-series and only change
// issue size and add V into UnsupportedFeatures.
def SiFive6Model : SchedMachineModel {
  let MicroOpBufferSize = 0; // Explicitly set to zero since SiFive6 is in-order.
  let IssueWidth = 1;        // 1 micro-ops are dispatched per cycle.
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let UnsupportedFeatures = [HasVInstructions,
                             HasExtXsfvqmaccqoq, HasExtXsfvqmaccdod,
                             HasExtXsfvfhbfmin];
  let CompleteModel = false;
}

// The SiFive6 microarchitecure has two pipelines: A, B
// Pipe A can handle memory, integer alu and vector operations.
// Pipe B can handle integer alu, control flow, integer multiply and divide,
// and floating point computation.
let SchedModel = SiFive6Model in {
let BufferSize = 0 in {
def SiFive6PipeA       : ProcResource<1>;
def SiFive6PipeB       : ProcResource<1>;
}

let BufferSize = 1 in {
def SiFive6IDiv        : ProcResource<1> { let Super = SiFive6PipeB; } // Int Division
def SiFive6FDiv        : ProcResource<1> { let Super = SiFive6PipeB; } // FP Division/Sqrt
}

def SiFive6PipeAB : ProcResGroup<[SiFive6PipeA, SiFive6PipeB]>;

// Branching
def : WriteRes<WriteJmp, [SiFive6PipeB]>;
def : WriteRes<WriteJal, [SiFive6PipeB]>;
def : WriteRes<WriteJalr, [SiFive6PipeB]>;
def : WriteRes<WriteJmpReg, [SiFive6PipeB]>;

//Short forward branch
def : WriteRes<WriteSFB, [SiFive6PipeA, SiFive6PipeB]> {
  let Latency = 3;
}

// Integer arithmetic and logic
let Latency = 3 in {
def : WriteRes<WriteIALU, [SiFive6PipeAB]>;
def : WriteRes<WriteIALU32, [SiFive6PipeAB]>;
def : WriteRes<WriteShiftImm, [SiFive6PipeAB]>;
def : WriteRes<WriteShiftImm32, [SiFive6PipeAB]>;
def : WriteRes<WriteShiftReg, [SiFive6PipeAB]>;
def : WriteRes<WriteShiftReg32, [SiFive6PipeAB]>;
}

// Integer multiplication
let Latency = 3 in {
def : WriteRes<WriteIMul, [SiFive6PipeB]>;
def : WriteRes<WriteIMul32, [SiFive6PipeB]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFive6PipeB, SiFive6IDiv]> {
  let Latency = 16;
  let ResourceCycles = [1, 15];
}
def : WriteRes<WriteIDiv32,  [SiFive6PipeB, SiFive6IDiv]> {
  let Latency = 16;
  let ResourceCycles = [1, 15];
}

// Bitmanip
let Latency = 3 in {
// Rotates are in the late-B ALU.
def : WriteRes<WriteRotateImm, [SiFive6PipeB]>;
def : WriteRes<WriteRotateImm32, [SiFive6PipeB]>;
def : WriteRes<WriteRotateReg, [SiFive6PipeB]>;
def : WriteRes<WriteRotateReg32, [SiFive6PipeB]>;

// clz[w]/ctz[w] are in the late-B ALU.
def : WriteRes<WriteCLZ, [SiFive6PipeB]>;
def : WriteRes<WriteCLZ32, [SiFive6PipeB]>;
def : WriteRes<WriteCTZ, [SiFive6PipeB]>;
def : WriteRes<WriteCTZ32, [SiFive6PipeB]>;

// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFive6PipeB]>;
def : WriteRes<WriteCPOP32, [SiFive6PipeB]>;

// orc.b is in the late-B ALU.
def : WriteRes<WriteORCB, [SiFive6PipeB]>;

// rev8 is in the late-A and late-B ALUs.
def : WriteRes<WriteREV8, [SiFive6PipeAB]>;

// shNadd[.uw] is on the early-B and late-B ALUs.
def : WriteRes<WriteSHXADD, [SiFive6PipeB]>;
def : WriteRes<WriteSHXADD32, [SiFive6PipeB]>;
}

// Memory
def : WriteRes<WriteSTB, [SiFive6PipeA]>;
def : WriteRes<WriteSTH, [SiFive6PipeA]>;
def : WriteRes<WriteSTW, [SiFive6PipeA]>;
def : WriteRes<WriteSTD, [SiFive6PipeA]>;
def : WriteRes<WriteFST16, [SiFive6PipeA]>;
def : WriteRes<WriteFST32, [SiFive6PipeA]>;
def : WriteRes<WriteFST64, [SiFive6PipeA]>;

let Latency = 3 in {
def : WriteRes<WriteLDB, [SiFive6PipeA]>;
def : WriteRes<WriteLDH, [SiFive6PipeA]>;
def : WriteRes<WriteLDW, [SiFive6PipeA]>;
def : WriteRes<WriteLDD, [SiFive6PipeA]>;
}

let Latency = 2 in {
def : WriteRes<WriteFLD16, [SiFive6PipeA]>;
def : WriteRes<WriteFLD32, [SiFive6PipeA]>;
def : WriteRes<WriteFLD64, [SiFive6PipeA]>;
}

// Atomic memory
def : WriteRes<WriteAtomicSTW, [SiFive6PipeA]>;
def : WriteRes<WriteAtomicSTD, [SiFive6PipeA]>;

let Latency = 3 in {
def : WriteRes<WriteAtomicW, [SiFive6PipeA]>;
def : WriteRes<WriteAtomicD, [SiFive6PipeA]>;
def : WriteRes<WriteAtomicLDW, [SiFive6PipeA]>;
def : WriteRes<WriteAtomicLDD, [SiFive6PipeA]>;
}

// Half precision.
let Latency = 5 in {
def : WriteRes<WriteFAdd16, [SiFive6PipeB]>;
def : WriteRes<WriteFMul16, [SiFive6PipeB]>;
def : WriteRes<WriteFMA16, [SiFive6PipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ16, [SiFive6PipeB]>;
def : WriteRes<WriteFMinMax16, [SiFive6PipeB]>;
}

def : WriteRes<WriteFDiv16, [SiFive6PipeB, SiFive6FDiv]> { let Latency = 14;
                                                         let ResourceCycles = [1, 13]; }
def : WriteRes<WriteFSqrt16, [SiFive6PipeB, SiFive6FDiv]> { let Latency = 14;
                                                          let ResourceCycles = [1, 13]; }

// Single precision.
let Latency = 5 in {
def : WriteRes<WriteFAdd32, [SiFive6PipeB]>;
def : WriteRes<WriteFMul32, [SiFive6PipeB]>;
def : WriteRes<WriteFMA32, [SiFive6PipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ32, [SiFive6PipeB]>;
def : WriteRes<WriteFMinMax32, [SiFive6PipeB]>;
}

def : WriteRes<WriteFDiv32, [SiFive6PipeB, SiFive6FDiv]> { let Latency = 27;
                                                         let ResourceCycles = [1, 26]; }
def : WriteRes<WriteFSqrt32, [SiFive6PipeB, SiFive6FDiv]> { let Latency = 27;
                                                          let ResourceCycles = [1, 26]; }

// Double precision
let Latency = 7 in {
def : WriteRes<WriteFAdd64, [SiFive6PipeB]>;
def : WriteRes<WriteFMul64, [SiFive6PipeB]>;
def : WriteRes<WriteFMA64, [SiFive6PipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ64, [SiFive6PipeB]>;
def : WriteRes<WriteFMinMax64, [SiFive6PipeB]>;
}

def : WriteRes<WriteFDiv64, [SiFive6PipeB, SiFive6FDiv]> { let Latency = 56;
                                                         let ResourceCycles = [1, 55]; }
def : WriteRes<WriteFSqrt64, [SiFive6PipeB, SiFive6FDiv]> { let Latency = 56;
                                                          let ResourceCycles = [1, 55]; }

// Conversions
let Latency = 3 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFive6PipeB]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFive6PipeB]>;

def : WriteRes<WriteFClass16, [SiFive6PipeB]>;
def : WriteRes<WriteFClass32, [SiFive6PipeB]>;
def : WriteRes<WriteFClass64, [SiFive6PipeB]>;
def : WriteRes<WriteFCmp16, [SiFive6PipeB]>;
def : WriteRes<WriteFCmp32, [SiFive6PipeB]>;
def : WriteRes<WriteFCmp64, [SiFive6PipeB]>;
def : WriteRes<WriteFMovI16ToF16, [SiFive6PipeB]>;
def : WriteRes<WriteFMovF16ToI16, [SiFive6PipeB]>;
def : WriteRes<WriteFMovI32ToF32, [SiFive6PipeB]>;
def : WriteRes<WriteFMovF32ToI32, [SiFive6PipeB]>;
def : WriteRes<WriteFMovI64ToF64, [SiFive6PipeB]>;
def : WriteRes<WriteFMovF64ToI64, [SiFive6PipeB]>;
}


// Others
def : WriteRes<WriteCSR, [SiFive6PipeB]>;
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

//Short forward branch
def : ReadAdvance<ReadSFB, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedCMOV; // SIFIVE
}
