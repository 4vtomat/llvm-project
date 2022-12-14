//==- RISCVSchedSiFive7N.td - SiFive7N Scheduling Definitions --*- tablegen -*-=//
//
// SiFive7 Schedule model for Narwhal Release.
// Scalar FMA latency change: 5/7 -> 4/4 (single/double FMA latency)
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// SiFive7N machine model for scheduling and other instruction cost heuristics.
def SiFive7NModel : SchedMachineModel {
  let MicroOpBufferSize = 0; // Explicitly set to zero since SiFive7N is in-order.
  let IssueWidth = 2;        // 2 micro-ops are dispatched per cycle.
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let CompleteModel = 0;
  let PostRAScheduler = true;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasExtXsfvqmaccqoq, HasExtXsfvqmaccdod, // SIFIVE
                             HasExtXsfvfhbfmin]; // SIFIVE
}

// The SiFive7N microarchitecure has three pipelines: A, B, V.
// Pipe A can handle memory, integer alu and vector operations.
// Pipe B can handle integer alu, control flow, integer multiply and divide,
// and floating point computation.
// Pipe V can handle the V extension.
let SchedModel = SiFive7NModel in {
let BufferSize = 0 in {
def SiFive7NPipeA       : ProcResource<1>;
def SiFive7NPipeB       : ProcResource<1>;
def SiFive7NPipeV       : ProcResource<1>;
}

let BufferSize = 1 in {
def SiFive7NMem         : ProcResource<1> { let Super = SiFive7NPipeA; } // Memory
def SiFive7NIDiv        : ProcResource<1> { let Super = SiFive7NPipeB; } // Int Division
def SiFive7NFDiv        : ProcResource<1> { let Super = SiFive7NPipeB; } // FP Division/Sqrt
def SiFive7NVA          : ProcResource<1> { let Super = SiFive7NPipeV; } // Arithmetic sequencer
def SiFive7NVL          : ProcResource<1> { let Super = SiFive7NPipeV; } // Load sequencer
def SiFive7NVS          : ProcResource<1> { let Super = SiFive7NPipeV; } // Store sequencer
}

def SiFive7NPipeAB : ProcResGroup<[SiFive7NPipeA, SiFive7NPipeB]>;

// Branching
def : WriteRes<WriteJmp, [SiFive7NPipeB]>;
def : WriteRes<WriteJal, [SiFive7NPipeB]>;
def : WriteRes<WriteJalr, [SiFive7NPipeB]>;
def : WriteRes<WriteJmpReg, [SiFive7NPipeB]>;

//Short forward branch
def : WriteRes<WriteSFB, [SiFive7NPipeA, SiFive7NPipeB]> {
  let Latency = 3;
  let NumMicroOps = 2;
}

// Integer arithmetic and logic
let Latency = 3 in {
def : WriteRes<WriteIALU, [SiFive7NPipeAB]>;
def : WriteRes<WriteIALU32, [SiFive7NPipeAB]>;
def : WriteRes<WriteShiftImm, [SiFive7NPipeAB]>;
def : WriteRes<WriteShiftImm32, [SiFive7NPipeAB]>;
def : WriteRes<WriteShiftReg, [SiFive7NPipeAB]>;
def : WriteRes<WriteShiftReg32, [SiFive7NPipeAB]>;
}

// Integer multiplication
let Latency = 3 in {
def : WriteRes<WriteIMul, [SiFive7NPipeB]>;
def : WriteRes<WriteIMul32, [SiFive7NPipeB]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFive7NPipeB, SiFive7NIDiv]> {
  let Latency = 66;
  let ResourceCycles = [1, 65];
}
def : WriteRes<WriteIDiv32,  [SiFive7NPipeB, SiFive7NIDiv]> {
  let Latency = 34;
  let ResourceCycles = [1, 33];
}

// Bitmanip
let Latency = 3 in {
// Rotates are in the late-B ALU.
def : WriteRes<WriteRotateImm, [SiFive7NPipeB]>;
def : WriteRes<WriteRotateImm32, [SiFive7NPipeB]>;
def : WriteRes<WriteRotateReg, [SiFive7NPipeB]>;
def : WriteRes<WriteRotateReg32, [SiFive7NPipeB]>;

// clz[w]/ctz[w] are in the late-B ALU.
def : WriteRes<WriteCLZ, [SiFive7NPipeB]>;
def : WriteRes<WriteCLZ32, [SiFive7NPipeB]>;
def : WriteRes<WriteCTZ, [SiFive7NPipeB]>;
def : WriteRes<WriteCTZ32, [SiFive7NPipeB]>;

// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFive7NPipeB]>;
def : WriteRes<WriteCPOP32, [SiFive7NPipeB]>;

// orc.b is in the late-B ALU.
def : WriteRes<WriteORCB, [SiFive7NPipeB]>;

// rev8 is in the late-A and late-B ALUs.
def : WriteRes<WriteREV8, [SiFive7NPipeAB]>;

// shNadd[.uw] is on the early-B and late-B ALUs.
def : WriteRes<WriteSHXADD, [SiFive7NPipeB]>;
def : WriteRes<WriteSHXADD32, [SiFive7NPipeB]>;
}

// Memory
let ResourceCycles = [0, 3] in {
def : WriteRes<WriteSTB, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteSTH, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteFST16, [SiFive7NPipeA, SiFive7NMem]>;
}

let ResourceCycles = [0, 1] in {
def : WriteRes<WriteSTW, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteSTD, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteFST32, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteFST64, [SiFive7NPipeA, SiFive7NMem]>;
}

let Latency = 3, ResourceCycles = [0, 1] in {
def : WriteRes<WriteLDB, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteLDH, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteLDW, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteLDD, [SiFive7NPipeA, SiFive7NMem]>;
}

let Latency = 2, ResourceCycles = [0, 1] in {
def : WriteRes<WriteFLD16, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteFLD32, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteFLD64, [SiFive7NPipeA, SiFive7NMem]>;
}

// Atomic memory
let ResourceCycles = [0, 1] in {
def : WriteRes<WriteAtomicSTW, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteAtomicSTD, [SiFive7NPipeA, SiFive7NMem]>;
}

let Latency = 3, ResourceCycles = [0, 1] in {
def : WriteRes<WriteAtomicW, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteAtomicD, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteAtomicLDW, [SiFive7NPipeA, SiFive7NMem]>;
def : WriteRes<WriteAtomicLDD, [SiFive7NPipeA, SiFive7NMem]>;
}

// Half precision.
let Latency = 4 in {
def : WriteRes<WriteFAdd16, [SiFive7NPipeB]>;
def : WriteRes<WriteFMul16, [SiFive7NPipeB]>;
def : WriteRes<WriteFMA16, [SiFive7NPipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ16, [SiFive7NPipeB]>;
def : WriteRes<WriteFMinMax16, [SiFive7NPipeB]>;
}

def : WriteRes<WriteFDiv16, [SiFive7NPipeB, SiFive7NFDiv]> { let Latency = 14;
                                                         let ResourceCycles = [1, 13]; }
def : WriteRes<WriteFSqrt16, [SiFive7NPipeB, SiFive7NFDiv]> { let Latency = 14;
                                                          let ResourceCycles = [1, 13]; }

// Single precision.
let Latency = 4 in {
def : WriteRes<WriteFAdd32, [SiFive7NPipeB]>;
def : WriteRes<WriteFMul32, [SiFive7NPipeB]>;
def : WriteRes<WriteFMA32, [SiFive7NPipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ32, [SiFive7NPipeB]>;
def : WriteRes<WriteFMinMax32, [SiFive7NPipeB]>;
}

def : WriteRes<WriteFDiv32, [SiFive7NPipeB, SiFive7NFDiv]> { let Latency = 27;
                                                         let ResourceCycles = [1, 26]; }
def : WriteRes<WriteFSqrt32, [SiFive7NPipeB, SiFive7NFDiv]> { let Latency = 27;
                                                          let ResourceCycles = [1, 26]; }

// Double precision
let Latency = 4 in {
def : WriteRes<WriteFAdd64, [SiFive7NPipeB]>;
def : WriteRes<WriteFMul64, [SiFive7NPipeB]>;
def : WriteRes<WriteFMA64, [SiFive7NPipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ64, [SiFive7NPipeB]>;
def : WriteRes<WriteFMinMax64, [SiFive7NPipeB]>;
}

def : WriteRes<WriteFDiv64, [SiFive7NPipeB, SiFive7NFDiv]> { let Latency = 56;
                                                         let ResourceCycles = [1, 55]; }
def : WriteRes<WriteFSqrt64, [SiFive7NPipeB, SiFive7NFDiv]> { let Latency = 56;
                                                          let ResourceCycles = [1, 55]; }

// Conversions
let Latency = 3 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFive7NPipeB]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFive7NPipeB]>;

def : WriteRes<WriteFClass16, [SiFive7NPipeB]>;
def : WriteRes<WriteFClass32, [SiFive7NPipeB]>;
def : WriteRes<WriteFClass64, [SiFive7NPipeB]>;
def : WriteRes<WriteFCmp16, [SiFive7NPipeB]>;
def : WriteRes<WriteFCmp32, [SiFive7NPipeB]>;
def : WriteRes<WriteFCmp64, [SiFive7NPipeB]>;
def : WriteRes<WriteFMovI16ToF16, [SiFive7NPipeA]>;
def : WriteRes<WriteFMovF16ToI16, [SiFive7NPipeA]>;
def : WriteRes<WriteFMovI32ToF32, [SiFive7NPipeA]>;
def : WriteRes<WriteFMovF32ToI32, [SiFive7NPipeA]>;
def : WriteRes<WriteFMovI64ToF64, [SiFive7NPipeA]>;
def : WriteRes<WriteFMovF64ToI64, [SiFive7NPipeA]>;
}

// 6. Configuration-Setting Instructions
let Latency = 3 in {
def : WriteRes<WriteVSETVLI, [SiFive7NPipeA]>;
def : WriteRes<WriteVSETIVLI, [SiFive7NPipeA]>;
def : WriteRes<WriteVSETVL, [SiFive7NPipeA]>;
}
// 7. Vector Loads and Stores
let Latency = 4 in
defm "" : LMULWriteRes<"WriteVLDE", [SiFive7NVL]>;
let Latency = 1 in
defm "" : LMULWriteRes<"WriteVSTE", [SiFive7NVS]>;
let Latency = 4 in
defm "" : LMULWriteRes<"WriteVLDM", [SiFive7NVL]>;
let Latency = 1 in
defm "" : LMULWriteRes<"WriteVSTM", [SiFive7NVS]>;
let Latency = 4 in {
defm "" : LMULWriteRes<"WriteVLDS8", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDS16", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDS32", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDS64", [SiFive7NVL]>;
}
let Latency = 1 in {
defm "" : LMULWriteRes<"WriteVSTS8", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTS16", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTS32", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTS64", [SiFive7NVS]>;
}
let Latency = 4 in {
defm "" : LMULWriteRes<"WriteVLDUX8", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDUX16", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDUX32", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDUX64", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDOX8", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDOX16", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDOX32", [SiFive7NVL]>;
defm "" : LMULWriteRes<"WriteVLDOX64", [SiFive7NVL]>;
}
let Latency = 1 in {
defm "" : LMULWriteRes<"WriteVSTUX8", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTUX16", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTUX32", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTUX64", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTOX8", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTOX16", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTOX32", [SiFive7NVS]>;
defm "" : LMULWriteRes<"WriteVSTOX64", [SiFive7NVS]>;
}
let Latency = 4 in {
defm "" : LMULWriteRes<"WriteVLDFF", [SiFive7NVL]>;
// LMUL Aware
def : WriteRes<WriteVLD1R, [SiFive7NVL]>;
def : WriteRes<WriteVLD2R, [SiFive7NVL]>;
def : WriteRes<WriteVLD4R, [SiFive7NVL]>;
def : WriteRes<WriteVLD8R, [SiFive7NVL]>;
}
let Latency = 1 in {
// LMUL Aware
def : WriteRes<WriteVST1R, [SiFive7NVS]>;
def : WriteRes<WriteVST2R, [SiFive7NVS]>;
def : WriteRes<WriteVST4R, [SiFive7NVS]>;
def : WriteRes<WriteVST8R, [SiFive7NVS]>;
}

let Latency = 4 in {
foreach nf=2-8 in {
  foreach eew = [8, 16, 32, 64] in {
    defm "" : LMULWriteRes<"WriteVLSEG" # nf # "e" # eew, [SiFive7NVL]>;
    defm "" : LMULWriteRes<"WriteVLSEGFF" # nf # "e" # eew, [SiFive7NVL]>;
    defm "" : LMULWriteRes<"WriteVLSSEG" # nf # "e" # eew, [SiFive7NVL]>;
    defm "" : LMULWriteRes<"WriteVLUXSEG" # nf # "e" # eew, [SiFive7NVL]>;
    defm "" : LMULWriteRes<"WriteVLOXSEG" # nf # "e" # eew, [SiFive7NVL]>;
  }
}
}

let Latency = 1, ResourceCycles = [4] in {
foreach nf=2-8 in {
  foreach eew = [8, 16, 32, 64] in {
    defm "" : LMULWriteRes<"WriteVSSEG" # nf # "e" # eew, [SiFive7NVS]>;
    defm "" : LMULWriteRes<"WriteVSSSEG" # nf # "e" # eew, [SiFive7NVS]>;
    defm "" : LMULWriteRes<"WriteVSUXSEG" # nf # "e" # eew, [SiFive7NVS]>;
    defm "" : LMULWriteRes<"WriteVSOXSEG" # nf # "e" # eew, [SiFive7NVS]>;
  }
}
}

// 12. Vector Integer Arithmetic Instructions
let Latency = 4 in {
defm : LMULWriteRes<"WriteVIALUV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIALUX", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIALUI", [SiFive7NVA]>;
}
let Latency = 8 in {
defm : LMULWriteResW<"WriteVIWALUV", [SiFive7NVA]>;
defm : LMULWriteResW<"WriteVIWALUX", [SiFive7NVA]>;
defm : LMULWriteResW<"WriteVIWALUI", [SiFive7NVA]>;
}
let Latency = 4 in {
defm : LMULWriteRes<"WriteVExtV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVICALUV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVICALUX", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVICALUI", [SiFive7NVA]>;
}
let Latency = 8 in {
defm : LMULWriteRes<"WriteVShiftV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVShiftX", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVShiftI", [SiFive7NVA]>;
defm : LMULWriteResW<"WriteVNShiftV", [SiFive7NVA]>;
defm : LMULWriteResW<"WriteVNShiftX", [SiFive7NVA]>;
defm : LMULWriteResW<"WriteVNShiftI", [SiFive7NVA]>;
}
let Latency = 4 in {
defm : LMULWriteRes<"WriteVICmpV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVICmpX", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVICmpI", [SiFive7NVA]>;
}
let Latency = 8 in {
defm : LMULWriteRes<"WriteVIMulV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIMulX", [SiFive7NVA]>;
}
let Latency = 16 in {
defm : LMULWriteRes<"WriteVIDivV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIDivX", [SiFive7NVA]>;
}
let Latency = 8 in {
defm : LMULWriteResW<"WriteVIWMulV", [SiFive7NVA]>;
defm : LMULWriteResW<"WriteVIWMulX", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIMulAddV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIMulAddX", [SiFive7NVA]>;
defm : LMULWriteResW<"WriteVIWMulAddV", [SiFive7NVA]>;
defm : LMULWriteResW<"WriteVIWMulAddX", [SiFive7NVA]>;
}
let Latency = 4 in {
defm : LMULWriteRes<"WriteVIMergeV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIMergeX", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIMergeI", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIMovV", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIMovX", [SiFive7NVA]>;
defm : LMULWriteRes<"WriteVIMovI", [SiFive7NVA]>;
}

// 13. Vector Fixed-Point Arithmetic Instructions
let Latency = 8 in {
defm "" : LMULWriteRes<"WriteVSALUV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVSALUX", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVSALUI", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVAALUV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVAALUX", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVSMulV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVSMulX", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVSShiftV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVSShiftX", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVSShiftI", [SiFive7NVA]>;
defm "" : LMULWriteResW<"WriteVNClipV", [SiFive7NVA]>;
defm "" : LMULWriteResW<"WriteVNClipX", [SiFive7NVA]>;
defm "" : LMULWriteResW<"WriteVNClipI", [SiFive7NVA]>;

// 14. Vector Floating-Point Instructions
defm "" : LMULWriteRes<"WriteVFALUV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFALUF", [SiFive7NVA]>;
defm "" : LMULWriteResFW<"WriteVFWALUV", [SiFive7NVA]>;
defm "" : LMULWriteResFW<"WriteVFWALUF", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFMulV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFMulF", [SiFive7NVA]>;
}
let Latency = 16 in {
defm "" : LMULWriteRes<"WriteVFDivV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFDivF", [SiFive7NVA]>;
}
let Latency = 8 in {
defm "" : LMULWriteResFW<"WriteVFWMulV", [SiFive7NVA]>;
defm "" : LMULWriteResFW<"WriteVFWMulF", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFMulAddV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFMulAddF", [SiFive7NVA]>;
defm "" : LMULWriteResFW<"WriteVFWMulAddV", [SiFive7NVA]>;
defm "" : LMULWriteResFW<"WriteVFWMulAddF", [SiFive7NVA]>;
}
let Latency = 16 in
defm "" : LMULWriteRes<"WriteVFSqrtV", [SiFive7NVA]>;
let Latency = 8 in {
defm "" : LMULWriteRes<"WriteVFRecpV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFCmpV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFCmpF", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFSgnjV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFSgnjF", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFClassV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFMergeV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFMovV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFCvtIToFV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFCvtFToIV", [SiFive7NVA]>;
defm "" : LMULWriteResW<"WriteVFWCvtIToFV", [SiFive7NVA]>;
defm "" : LMULWriteResFW<"WriteVFWCvtFToIV", [SiFive7NVA]>;
defm "" : LMULWriteResFW<"WriteVFWCvtFToFV", [SiFive7NVA]>;
defm "" : LMULWriteResFW<"WriteVFNCvtIToFV", [SiFive7NVA]>;
defm "" : LMULWriteResW<"WriteVFNCvtFToIV", [SiFive7NVA]>;
defm "" : LMULWriteResFW<"WriteVFNCvtFToFV", [SiFive7NVA]>;
}

// 15. Vector Reduction Operations
let Latency = 32 in {
def : WriteRes<WriteVIRedV, [SiFive7NVA]>;
def : WriteRes<WriteVIWRedV, [SiFive7NVA]>;
def : WriteRes<WriteVFRedV, [SiFive7NVA]>;
def : WriteRes<WriteVFRedOV, [SiFive7NVA]>;
def : WriteRes<WriteVFWRedV, [SiFive7NVA]>;
def : WriteRes<WriteVFWRedOV, [SiFive7NVA]>;
}

// 16. Vector Mask Instructions
let Latency = 4 in {
defm "" : LMULWriteRes<"WriteVMALUV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVMPopV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVMFFSV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVMSFSV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVMIotV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVMIdxV", [SiFive7NVA]>;
}

// 17. Vector Permutation Instructions
let Latency = 10 in
defm "" : LMULWriteRes<"WriteVIMovVX", [SiFive7NVA]>;
let Latency = 4 in
defm "" : LMULWriteRes<"WriteVIMovXV", [SiFive7NVA]>;
let Latency = 10 in
defm "" : LMULWriteRes<"WriteVFMovVF", [SiFive7NVA]>;
let Latency = 4 in
defm "" : LMULWriteRes<"WriteVFMovFV", [SiFive7NVA]>;
let Latency = 4 in {
defm "" : LMULWriteRes<"WriteVISlideX", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVISlideI", [SiFive7NVA]>;
}
let Latency = 8 in {
defm "" : LMULWriteRes<"WriteVISlide1X", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFSlide1F", [SiFive7NVA]>;
}
let Latency = 16 in {
defm "" : LMULWriteRes<"WriteVGatherV", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVGatherX", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVGatherI", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVCompressV", [SiFive7NVA]>;
}
let Latency = 4 in {
// LMUL Aware
def : WriteRes<WriteVMov1V, [SiFive7NVA]>;
def : WriteRes<WriteVMov2V, [SiFive7NVA]>;
def : WriteRes<WriteVMov4V, [SiFive7NVA]>;
def : WriteRes<WriteVMov8V, [SiFive7NVA]>;
}

// Others
def : WriteRes<WriteCSR, [SiFive7NPipeB]>;
def : WriteRes<WriteNop, []>;
let Latency = 3 in {
def : WriteRes<WriteRdVLENB, [SiFive7NPipeB]>;
}

// FIXME: This could be better modeled by looking at the regclasses of the operands.
def : InstRW<[WriteIALU, ReadIALU], (instrs COPY)>;

// Bypass and advance
defm : AnyToGPRBypass<ReadJmp, 2>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
//Load to Load/Store address
def : ReadAdvance<ReadMemBase, 1, [WriteLDB, WriteLDH, WriteLDW,
                                   WriteLDW, WriteLDD]>;
defm : AnyToGPRBypass<ReadIALU, 2>;
defm : AnyToGPRBypass<ReadIALU32, 2>;
defm : AnyToGPRBypass<ReadShiftImm, 2>;
defm : AnyToGPRBypass<ReadShiftImm32, 2>;
defm : AnyToGPRBypass<ReadShiftReg, 2>;
defm : AnyToGPRBypass<ReadShiftReg32, 2>;
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
defm : AnytoFPRBypass<ReadFAdd16, 1>;
defm : AnytoFPRBypass<ReadFAdd32, 1>;
defm : AnytoFPRBypass<ReadFAdd64, 1>;
defm : AnytoFPRBypass<ReadFMul16, 1>;
defm : AnytoFPRBypass<ReadFMA16, 1>;
defm : AnytoFPRBypass<ReadFMul32, 1>;
defm : AnytoFPRBypass<ReadFMA32, 1>;
defm : AnytoFPRBypass<ReadFMul64, 1>;
defm : AnytoFPRBypass<ReadFMA64, 1>;
defm : AnytoFPRBypass<ReadFDiv16, 1>;
defm : AnytoFPRBypass<ReadFDiv32, 1>;
defm : AnytoFPRBypass<ReadFDiv64, 1>;
defm : AnytoFPRBypass<ReadFSqrt16, 1>;
defm : AnytoFPRBypass<ReadFSqrt32, 1>;
defm : AnytoFPRBypass<ReadFSqrt64, 1>;
defm : AnytoFPRBypass<ReadFCmp16, 1>;
defm : AnytoFPRBypass<ReadFCmp32, 1>;
defm : AnytoFPRBypass<ReadFCmp64, 1>;
defm : AnytoFPRBypass<ReadFSGNJ16, 1>;
defm : AnytoFPRBypass<ReadFSGNJ32, 1>;
defm : AnytoFPRBypass<ReadFSGNJ64, 1>;
defm : AnytoFPRBypass<ReadFMinMax16, 1>;
defm : AnytoFPRBypass<ReadFMinMax32, 1>;
defm : AnytoFPRBypass<ReadFMinMax64, 1>;
defm : AnytoFPRBypass<ReadFCvtF16ToI32, 1>;
defm : AnytoFPRBypass<ReadFCvtF16ToI64, 1>;
defm : AnytoFPRBypass<ReadFCvtF32ToI32, 1>;
defm : AnytoFPRBypass<ReadFCvtF32ToI64, 1>;
defm : AnytoFPRBypass<ReadFCvtF64ToI32, 1>;
defm : AnytoFPRBypass<ReadFCvtF64ToI64, 1>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
defm : AnytoFPRBypass<ReadFCvtF32ToF64, 1>;
defm : AnytoFPRBypass<ReadFCvtF64ToF32, 1>;
defm : AnytoFPRBypass<ReadFCvtF16ToF32, 1>;
defm : AnytoFPRBypass<ReadFCvtF32ToF16, 1>;
defm : AnytoFPRBypass<ReadFCvtF16ToF64, 1>;
defm : AnytoFPRBypass<ReadFCvtF64ToF16, 1>;
defm : AnytoFPRBypass<ReadFMovF16ToI16, 1>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
defm : AnytoFPRBypass<ReadFMovF32ToI32, 1>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
defm : AnytoFPRBypass<ReadFMovF64ToI64, 1>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

// 6. Configuration-Setting Instructions
def : ReadAdvance<ReadVSETVLI, 2>;
def : ReadAdvance<ReadVSETVL, 2>;

// 7. Vector Loads and Stores
defm "" : LMULReadAdvance<"ReadVLDX", 0>;
defm "" : LMULReadAdvance<"ReadVSTX", 0>;
defm "" : LMULReadAdvance<"ReadVSTEV", 0>;
defm "" : LMULReadAdvance<"ReadVSTM", 0>;
defm "" : LMULReadAdvance<"ReadVLDSX", 0>;
defm "" : LMULReadAdvance<"ReadVSTSX", 0>;
defm "" : LMULReadAdvance<"ReadVSTS8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS64V", 0>;
defm "" : LMULReadAdvance<"ReadVLDUXV", 0>;
defm "" : LMULReadAdvance<"ReadVLDOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTUXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64V", 0>;
// LMUL Aware
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 12. Vector Integer Arithmetic Instructions
defm : LMULReadAdvance<"ReadVIALUV", 0>;
defm : LMULReadAdvance<"ReadVIALUX", 0>;
defm : LMULReadAdvanceW<"ReadVIWALUV", 0>;
defm : LMULReadAdvanceW<"ReadVIWALUX", 0>;
defm : LMULReadAdvance<"ReadVExtV", 0>;
defm : LMULReadAdvance<"ReadVICALUV", 0>;
defm : LMULReadAdvance<"ReadVICALUX", 0>;
defm : LMULReadAdvance<"ReadVShiftV", 0>;
defm : LMULReadAdvance<"ReadVShiftX", 0>;
defm : LMULReadAdvanceW<"ReadVNShiftV", 0>;
defm : LMULReadAdvanceW<"ReadVNShiftX", 0>;
defm : LMULReadAdvance<"ReadVICmpV", 0>;
defm : LMULReadAdvance<"ReadVICmpX", 0>;
defm : LMULReadAdvance<"ReadVIMulV", 0>;
defm : LMULReadAdvance<"ReadVIMulX", 0>;
defm : LMULReadAdvance<"ReadVIDivV", 0>;
defm : LMULReadAdvance<"ReadVIDivX", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulV", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulX", 0>;
defm : LMULReadAdvance<"ReadVIMulAddV", 0>;
defm : LMULReadAdvance<"ReadVIMulAddX", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulAddV", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulAddX", 0>;
defm : LMULReadAdvance<"ReadVIMergeV", 0>;
defm : LMULReadAdvance<"ReadVIMergeX", 0>;
defm : LMULReadAdvance<"ReadVIMovV", 0>;
defm : LMULReadAdvance<"ReadVIMovX", 0>;

// 13. Vector Fixed-Point Arithmetic Instructions
defm "" : LMULReadAdvance<"ReadVSALUV", 0>;
defm "" : LMULReadAdvance<"ReadVSALUX", 0>;
defm "" : LMULReadAdvance<"ReadVAALUV", 0>;
defm "" : LMULReadAdvance<"ReadVAALUX", 0>;
defm "" : LMULReadAdvance<"ReadVSMulV", 0>;
defm "" : LMULReadAdvance<"ReadVSMulX", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftX", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipV", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipX", 0>;

// 14. Vector Floating-Point Instructions
defm "" : LMULReadAdvance<"ReadVFALUV", 0>;
defm "" : LMULReadAdvance<"ReadVFALUF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulF", 0>;
defm "" : LMULReadAdvance<"ReadVFDivV", 0>;
defm "" : LMULReadAdvance<"ReadVFDivF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddF", 0>;
defm "" : LMULReadAdvance<"ReadVFSqrtV", 0>;
defm "" : LMULReadAdvance<"ReadVFRecpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpF", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjV", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjF", 0>;
defm "" : LMULReadAdvance<"ReadVFClassV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovF", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtIToFV", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtFToIV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFWCvtIToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtIToFV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFNCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtFToFV", 0>;

// 15. Vector Reduction Operations
def : ReadAdvance<ReadVIRedV, 0>;
def : ReadAdvance<ReadVIRedV0, 0>;
def : ReadAdvance<ReadVIWRedV, 0>;
def : ReadAdvance<ReadVIWRedV0, 0>;
def : ReadAdvance<ReadVFRedV, 0>;
def : ReadAdvance<ReadVFRedV0, 0>;
def : ReadAdvance<ReadVFRedOV, 0>;
def : ReadAdvance<ReadVFRedOV0, 0>;
def : ReadAdvance<ReadVFWRedV, 0>;
def : ReadAdvance<ReadVFWRedV0, 0>;
def : ReadAdvance<ReadVFWRedOV, 0>;
def : ReadAdvance<ReadVFWRedOV0, 0>;

// 16. Vector Mask Instructions
defm "" : LMULReadAdvance<"ReadVMALUV", 0>;
defm "" : LMULReadAdvance<"ReadVMPopV", 0>;
defm "" : LMULReadAdvance<"ReadVMFFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMSFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMIotV", 0>;

// 17. Vector Permutation Instructions
defm "" : LMULReadAdvance<"ReadVIMovVX", 0>;
defm "" : LMULReadAdvance<"ReadVIMovXV", 0>;
defm "" : LMULReadAdvance<"ReadVIMovXX", 0>;
defm "" : LMULReadAdvance<"ReadVFMovVF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovFV", 0>;
defm "" : LMULReadAdvance<"ReadVFMovFX", 0>;
defm "" : LMULReadAdvance<"ReadVISlideV", 0>;
defm "" : LMULReadAdvance<"ReadVISlideX", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideV", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideF", 0>;
defm "" : LMULReadAdvance<"ReadVGatherV", 0>;
defm "" : LMULReadAdvance<"ReadVGatherX", 0>;
defm "" : LMULReadAdvance<"ReadVCompressV", 0>;
// LMUL Aware
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;

// Bitmanip
defm : AnyToGPRBypass<ReadRotateImm, 2>;
defm : AnyToGPRBypass<ReadRotateImm32, 2>;
defm : AnyToGPRBypass<ReadRotateReg, 2>;
defm : AnyToGPRBypass<ReadRotateReg32, 2>;
defm : AnyToGPRBypass<ReadCLZ, 2>;
defm : AnyToGPRBypass<ReadCLZ32, 2>;
defm : AnyToGPRBypass<ReadCTZ, 2>;
defm : AnyToGPRBypass<ReadCTZ32, 2>;
defm : AnyToGPRBypass<ReadCPOP, 2>;
defm : AnyToGPRBypass<ReadCPOP32, 2>;
defm : AnyToGPRBypass<ReadORCB, 2>;
defm : AnyToGPRBypass<ReadREV8, 2>;
defm : AnyToGPRBypass<ReadSHXADD, 2>;
defm : AnyToGPRBypass<ReadSHXADD32, 2>;

// Short forward branch
// ALU, ALU_B, SFB to SFB
def : ReadAdvance<ReadSFB, 2, [WriteIALU, WriteIALU32,
                               WriteShiftImm, WriteShiftImm32,
                               WriteShiftReg, WriteShiftReg32,
                               WriteSHXADD, WriteSHXADD32,
                               WriteRotateImm, WriteRotateImm32,
                               WriteRotateReg, WriteRotateReg32,
                               WriteCLZ, WriteCLZ32, WriteCTZ,
                               WriteCTZ32, WriteCPOP, WriteCPOP32,
                               WriteREV8, WriteORCB, WriteSFB]>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedCMOV; // SIFIVE
}
