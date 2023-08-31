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
                             HasVendorXSfvqmaccqoq, HasVendorXSfvqmaccdod,
                             HasVendorXSfvfhbfmin];
  let CompleteModel = false;
}

// The SiFive6 microarchitecure has one unnamed pipe.
let SchedModel = SiFive6Model in {
let BufferSize = 0 in
def SiFive6Pipe       : ProcResource<1>;

let BufferSize = 1 in {
def SiFive6IDiv        : ProcResource<1> { let Super = SiFive6Pipe; } // Int Division
def SiFive6FDiv        : ProcResource<1> { let Super = SiFive6Pipe; } // FP Division/Sqrt
}

// Branching
let Latency = 3 in {
def : WriteRes<WriteJmp, [SiFive6Pipe]>;
def : WriteRes<WriteJal, [SiFive6Pipe]>;
def : WriteRes<WriteJalr, [SiFive6Pipe]>;
}

//Short forward branch
let Latency = 4, ReleaseAtCycles = [2] in
def : WriteRes<WriteSFB, [SiFive6Pipe]>;

// Integer arithmetic and logic
let Latency = 3 in {
def : WriteRes<WriteIALU, [SiFive6Pipe]>;
def : WriteRes<WriteIALU32, [SiFive6Pipe]>;
def : WriteRes<WriteShiftImm, [SiFive6Pipe]>;
def : WriteRes<WriteShiftImm32, [SiFive6Pipe]>;
def : WriteRes<WriteShiftReg, [SiFive6Pipe]>;
def : WriteRes<WriteShiftReg32, [SiFive6Pipe]>;
}

// Integer multiplication
let Latency = 3 in {
def : WriteRes<WriteIMul, [SiFive6Pipe]>;
def : WriteRes<WriteIMul32, [SiFive6Pipe]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFive6Pipe, SiFive6IDiv]> {
  let Latency = 16;
  let ReleaseAtCycles = [1, 15];
}
def : WriteRes<WriteIDiv32,  [SiFive6Pipe, SiFive6IDiv]> {
  let Latency = 16;
  let ReleaseAtCycles = [1, 15];
}

// Bitmanip
let Latency = 3 in {
// Rotates are in the late-B ALU.
def : WriteRes<WriteRotateImm, [SiFive6Pipe]>;
def : WriteRes<WriteRotateImm32, [SiFive6Pipe]>;
def : WriteRes<WriteRotateReg, [SiFive6Pipe]>;
def : WriteRes<WriteRotateReg32, [SiFive6Pipe]>;

// clz[w]/ctz[w] are in the late-B ALU.
def : WriteRes<WriteCLZ, [SiFive6Pipe]>;
def : WriteRes<WriteCLZ32, [SiFive6Pipe]>;
def : WriteRes<WriteCTZ, [SiFive6Pipe]>;
def : WriteRes<WriteCTZ32, [SiFive6Pipe]>;

// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFive6Pipe]>;
def : WriteRes<WriteCPOP32, [SiFive6Pipe]>;

// orc.b is in the late-B ALU.
def : WriteRes<WriteORCB, [SiFive6Pipe]>;

// rev8 is in the late-A and late-B ALUs.
def : WriteRes<WriteREV8, [SiFive6Pipe]>;

// shNadd[.uw] is on the early-B and late-B ALUs.
def : WriteRes<WriteSHXADD, [SiFive6Pipe]>;
def : WriteRes<WriteSHXADD32, [SiFive6Pipe]>;
}

// Single-bit instructions
let Latency = 3 in {
def : WriteRes<WriteSingleBit, [SiFive6Pipe]>;
def : WriteRes<WriteSingleBitImm, [SiFive6Pipe]>;
def : WriteRes<WriteBEXT, [SiFive6Pipe]>;
def : WriteRes<WriteBEXTI, [SiFive6Pipe]>;
}

// Memory
def : WriteRes<WriteSTB, [SiFive6Pipe]>;
def : WriteRes<WriteSTH, [SiFive6Pipe]>;
def : WriteRes<WriteSTW, [SiFive6Pipe]>;
def : WriteRes<WriteSTD, [SiFive6Pipe]>;
def : WriteRes<WriteFST16, [SiFive6Pipe]>;
def : WriteRes<WriteFST32, [SiFive6Pipe]>;
def : WriteRes<WriteFST64, [SiFive6Pipe]>;

let Latency = 3 in {
def : WriteRes<WriteLDB, [SiFive6Pipe]>;
def : WriteRes<WriteLDH, [SiFive6Pipe]>;
def : WriteRes<WriteLDW, [SiFive6Pipe]>;
def : WriteRes<WriteLDD, [SiFive6Pipe]>;
}

let Latency = 2 in {
def : WriteRes<WriteFLD16, [SiFive6Pipe]>;
def : WriteRes<WriteFLD32, [SiFive6Pipe]>;
def : WriteRes<WriteFLD64, [SiFive6Pipe]>;
}

// Atomic memory
def : WriteRes<WriteAtomicSTW, [SiFive6Pipe]>;
def : WriteRes<WriteAtomicSTD, [SiFive6Pipe]>;

let Latency = 3 in {
def : WriteRes<WriteAtomicW, [SiFive6Pipe]>;
def : WriteRes<WriteAtomicD, [SiFive6Pipe]>;
def : WriteRes<WriteAtomicLDW, [SiFive6Pipe]>;
def : WriteRes<WriteAtomicLDD, [SiFive6Pipe]>;
}

// Half precision.
let Latency = 5 in {
def : WriteRes<WriteFAdd16, [SiFive6Pipe]>;
def : WriteRes<WriteFMul16, [SiFive6Pipe]>;
def : WriteRes<WriteFMA16, [SiFive6Pipe]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ16, [SiFive6Pipe]>;
def : WriteRes<WriteFMinMax16, [SiFive6Pipe]>;
}

def : WriteRes<WriteFDiv16, [SiFive6Pipe, SiFive6FDiv]> { let Latency = 14;
                                                         let ReleaseAtCycles = [1, 13]; }
def : WriteRes<WriteFSqrt16, [SiFive6Pipe, SiFive6FDiv]> { let Latency = 14;
                                                          let ReleaseAtCycles = [1, 13]; }

// Single precision.
let Latency = 5 in {
def : WriteRes<WriteFAdd32, [SiFive6Pipe]>;
def : WriteRes<WriteFMul32, [SiFive6Pipe]>;
def : WriteRes<WriteFMA32, [SiFive6Pipe]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ32, [SiFive6Pipe]>;
def : WriteRes<WriteFMinMax32, [SiFive6Pipe]>;
}

def : WriteRes<WriteFDiv32, [SiFive6Pipe, SiFive6FDiv]> { let Latency = 27;
                                                         let ReleaseAtCycles = [1, 26]; }
def : WriteRes<WriteFSqrt32, [SiFive6Pipe, SiFive6FDiv]> { let Latency = 27;
                                                          let ReleaseAtCycles = [1, 26]; }

// Double precision
let Latency = 7 in {
def : WriteRes<WriteFAdd64, [SiFive6Pipe]>;
def : WriteRes<WriteFMul64, [SiFive6Pipe]>;
def : WriteRes<WriteFMA64, [SiFive6Pipe]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ64, [SiFive6Pipe]>;
def : WriteRes<WriteFMinMax64, [SiFive6Pipe]>;
}

def : WriteRes<WriteFDiv64, [SiFive6Pipe, SiFive6FDiv]> { let Latency = 56;
                                                         let ReleaseAtCycles = [1, 55]; }
def : WriteRes<WriteFSqrt64, [SiFive6Pipe, SiFive6FDiv]> { let Latency = 56;
                                                          let ReleaseAtCycles = [1, 55]; }

// Conversions
let Latency = 3 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFive6Pipe]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFive6Pipe]>;

def : WriteRes<WriteFClass16, [SiFive6Pipe]>;
def : WriteRes<WriteFClass32, [SiFive6Pipe]>;
def : WriteRes<WriteFClass64, [SiFive6Pipe]>;
def : WriteRes<WriteFCmp16, [SiFive6Pipe]>;
def : WriteRes<WriteFCmp32, [SiFive6Pipe]>;
def : WriteRes<WriteFCmp64, [SiFive6Pipe]>;
def : WriteRes<WriteFMovI16ToF16, [SiFive6Pipe]>;
def : WriteRes<WriteFMovF16ToI16, [SiFive6Pipe]>;
def : WriteRes<WriteFMovI32ToF32, [SiFive6Pipe]>;
def : WriteRes<WriteFMovF32ToI32, [SiFive6Pipe]>;
def : WriteRes<WriteFMovI64ToF64, [SiFive6Pipe]>;
def : WriteRes<WriteFMovF64ToI64, [SiFive6Pipe]>;
}

// Others
def : WriteRes<WriteCSR, [SiFive6Pipe]>;
def : WriteRes<WriteNop, []>;

// FIXME: This could be better modeled by looking at the regclasses of the operands.
def : InstRW<[WriteIALU, ReadIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance

def SiFive6ReadMemBaseOffset0 : SchedReadAdvance<1, [WriteLDB, WriteLDH, WriteLDW, WriteLDD]>;

// Bypass the AG stage of a dependendent memory op after a load if the offset is
// zero.
def SiFive6ReadMemBase : SchedReadVariant<[
  SchedVar<MemBaseOffset0Pred, [SiFive6ReadMemBaseOffset0]>,
  SchedVar<NoSchedPred, [NoReadAdvance]>
]>;
def : SchedAlias<ReadMemBase, SiFive6ReadMemBase>;

def : BulletAnyToGPRBypass<ReadJmp>;
def : BulletAnyToGPRBypass<ReadJalr>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : BulletAnyToGPRBypass<ReadIALU>;
def : BulletAnyToGPRBypass<ReadIALU32>;
def : BulletAnyToGPRBypass<ReadShiftImm>;
def : BulletAnyToGPRBypass<ReadShiftImm32>;
def : BulletAnyToGPRBypass<ReadShiftReg>;
def : BulletAnyToGPRBypass<ReadShiftReg32>;
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
def : BulletAnyToFPRBypass<ReadFAdd16>;
def : BulletAnyToFPRBypass<ReadFAdd32>;
def : BulletAnyToFPRBypass<ReadFAdd64>;
def : BulletAnyToFPRBypass<ReadFMul16>;
def : BulletAnyToFPRBypass<ReadFMA16>;
def : BulletAnyToFPRBypass<ReadFMul32>;
def : BulletAnyToFPRBypass<ReadFMA32>;
def : BulletAnyToFPRBypass<ReadFMul64>;
def : BulletAnyToFPRBypass<ReadFMA64>;
def : BulletAnyToFPRBypass<ReadFDiv16>;
def : BulletAnyToFPRBypass<ReadFDiv32>;
def : BulletAnyToFPRBypass<ReadFDiv64>;
def : BulletAnyToFPRBypass<ReadFSqrt16>;
def : BulletAnyToFPRBypass<ReadFSqrt32>;
def : BulletAnyToFPRBypass<ReadFSqrt64>;
def : BulletAnyToFPRBypass<ReadFCmp16>;
def : BulletAnyToFPRBypass<ReadFCmp32>;
def : BulletAnyToFPRBypass<ReadFCmp64>;
def : BulletAnyToFPRBypass<ReadFSGNJ16>;
def : BulletAnyToFPRBypass<ReadFSGNJ32>;
def : BulletAnyToFPRBypass<ReadFSGNJ64>;
def : BulletAnyToFPRBypass<ReadFMinMax16>;
def : BulletAnyToFPRBypass<ReadFMinMax32>;
def : BulletAnyToFPRBypass<ReadFMinMax64>;
def : BulletAnyToFPRBypass<ReadFCvtF16ToI32>;
def : BulletAnyToFPRBypass<ReadFCvtF16ToI64>;
def : BulletAnyToFPRBypass<ReadFCvtF32ToI32>;
def : BulletAnyToFPRBypass<ReadFCvtF32ToI64>;
def : BulletAnyToFPRBypass<ReadFCvtF64ToI32>;
def : BulletAnyToFPRBypass<ReadFCvtF64ToI64>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : BulletAnyToFPRBypass<ReadFCvtF32ToF64>;
def : BulletAnyToFPRBypass<ReadFCvtF64ToF32>;
def : BulletAnyToFPRBypass<ReadFCvtF16ToF32>;
def : BulletAnyToFPRBypass<ReadFCvtF32ToF16>;
def : BulletAnyToFPRBypass<ReadFCvtF16ToF64>;
def : BulletAnyToFPRBypass<ReadFCvtF64ToF16>;
def : BulletAnyToFPRBypass<ReadFMovF16ToI16>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : BulletAnyToFPRBypass<ReadFMovF32ToI32>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : BulletAnyToFPRBypass<ReadFMovF64ToI64>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

// Bitmanip
def : BulletAnyToGPRBypass<ReadRotateImm>;
def : BulletAnyToGPRBypass<ReadRotateImm32>;
def : BulletAnyToGPRBypass<ReadRotateReg>;
def : BulletAnyToGPRBypass<ReadRotateReg32>;
def : BulletAnyToGPRBypass<ReadCLZ>;
def : BulletAnyToGPRBypass<ReadCLZ32>;
def : BulletAnyToGPRBypass<ReadCTZ>;
def : BulletAnyToGPRBypass<ReadCTZ32>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : BulletAnyToGPRBypass<ReadORCB>;
def : BulletAnyToGPRBypass<ReadREV8>;
def : BulletAnyToGPRBypass<ReadSHXADD>;
def : BulletAnyToGPRBypass<ReadSHXADD32>;
// Single-bit instructions
def : BulletAnyToGPRBypass<ReadSingleBit>;
def : BulletAnyToGPRBypass<ReadSingleBitImm>;

//Short forward branch
def : ReadAdvance<ReadSFBJmp, 0>;
def : ReadAdvance<ReadSFBALU, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedCMOV; // SIFIVE
}
