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
def : WriteRes<WriteLDWU, [SiFive7NPipeA, SiFive7NMem]>;
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
def : WriteRes<WriteVLDE, [SiFive7NVL]>;
let Latency = 1 in
def : WriteRes<WriteVSTE, [SiFive7NVS]>;
let Latency = 4 in
def : WriteRes<WriteVLDM, [SiFive7NVL]>;
let Latency = 1 in
def : WriteRes<WriteVSTM, [SiFive7NVS]>;
let Latency = 4 in {
def : WriteRes<WriteVLDS8, [SiFive7NVL]>;
def : WriteRes<WriteVLDS16, [SiFive7NVL]>;
def : WriteRes<WriteVLDS32, [SiFive7NVL]>;
def : WriteRes<WriteVLDS64, [SiFive7NVL]>;
}
let Latency = 1 in {
def : WriteRes<WriteVSTS8, [SiFive7NVS]>;
def : WriteRes<WriteVSTS16, [SiFive7NVS]>;
def : WriteRes<WriteVSTS32, [SiFive7NVS]>;
def : WriteRes<WriteVSTS64, [SiFive7NVS]>;
}
let Latency = 4 in {
def : WriteRes<WriteVLDUX8, [SiFive7NVL]>;
def : WriteRes<WriteVLDUX16, [SiFive7NVL]>;
def : WriteRes<WriteVLDUX32, [SiFive7NVL]>;
def : WriteRes<WriteVLDUX64, [SiFive7NVL]>;
def : WriteRes<WriteVLDOX8, [SiFive7NVL]>;
def : WriteRes<WriteVLDOX16, [SiFive7NVL]>;
def : WriteRes<WriteVLDOX32, [SiFive7NVL]>;
def : WriteRes<WriteVLDOX64, [SiFive7NVL]>;
}
let Latency = 1 in {
def : WriteRes<WriteVSTUX8, [SiFive7NVS]>;
def : WriteRes<WriteVSTUX16, [SiFive7NVS]>;
def : WriteRes<WriteVSTUX32, [SiFive7NVS]>;
def : WriteRes<WriteVSTUX64, [SiFive7NVS]>;
def : WriteRes<WriteVSTOX8, [SiFive7NVS]>;
def : WriteRes<WriteVSTOX16, [SiFive7NVS]>;
def : WriteRes<WriteVSTOX32, [SiFive7NVS]>;
def : WriteRes<WriteVSTOX64, [SiFive7NVS]>;
}
let Latency = 4 in {
def : WriteRes<WriteVLDFF, [SiFive7NVL]>;
def : WriteRes<WriteVLD1R, [SiFive7NVL]>;
def : WriteRes<WriteVLD2R, [SiFive7NVL]>;
def : WriteRes<WriteVLD4R, [SiFive7NVL]>;
def : WriteRes<WriteVLD8R, [SiFive7NVL]>;
}
let Latency = 1 in {
def : WriteRes<WriteVST1R, [SiFive7NVS]>;
def : WriteRes<WriteVST2R, [SiFive7NVS]>;
def : WriteRes<WriteVST4R, [SiFive7NVS]>;
def : WriteRes<WriteVST8R, [SiFive7NVS]>;
}

let Latency = 4 in {
foreach nf=2-8 in {
  foreach eew = [8, 16, 32, 64] in {
    def : WriteRes <!cast<SchedWrite>("WriteVLSEG" # nf # "e" # eew), [SiFive7NVL]>;
    def : WriteRes <!cast<SchedWrite>("WriteVLSEGFF" # nf # "e" # eew), [SiFive7NVL]>;
    def : WriteRes <!cast<SchedWrite>("WriteVLSSEG" # nf # "e" # eew), [SiFive7NVL]>;
    def : WriteRes <!cast<SchedWrite>("WriteVLUXSEG" # nf # "e" # eew), [SiFive7NVL]>;
    def : WriteRes <!cast<SchedWrite>("WriteVLOXSEG" # nf # "e" # eew), [SiFive7NVL]>;
  }
}
}

let Latency = 1, ResourceCycles = [4] in {
foreach nf=2-8 in {
  foreach eew = [8, 16, 32, 64] in {
    def : WriteRes <!cast<SchedWrite>("WriteVSSEG" # nf # "e" # eew), [SiFive7NVS]>;
    def : WriteRes <!cast<SchedWrite>("WriteVSSSEG" # nf # "e" # eew), [SiFive7NVS]>;
    def : WriteRes <!cast<SchedWrite>("WriteVSUXSEG" # nf # "e" # eew), [SiFive7NVS]>;
    def : WriteRes <!cast<SchedWrite>("WriteVSOXSEG" # nf # "e" # eew), [SiFive7NVS]>;
  }
}
}

// 12. Vector Integer Arithmetic Instructions
let Latency = 4 in {
def : WriteRes<WriteVIALUV, [SiFive7NVA]>;
def : WriteRes<WriteVIALUX, [SiFive7NVA]>;
def : WriteRes<WriteVIALUI, [SiFive7NVA]>;
}
let Latency = 8 in {
def : WriteRes<WriteVIWALUV, [SiFive7NVA]>;
def : WriteRes<WriteVIWALUX, [SiFive7NVA]>;
def : WriteRes<WriteVIWALUI, [SiFive7NVA]>;
}
let Latency = 4 in {
def : WriteRes<WriteVExtV, [SiFive7NVA]>;
def : WriteRes<WriteVICALUV, [SiFive7NVA]>;
def : WriteRes<WriteVICALUX, [SiFive7NVA]>;
def : WriteRes<WriteVICALUI, [SiFive7NVA]>;
}
let Latency = 8 in {
def : WriteRes<WriteVShiftV, [SiFive7NVA]>;
def : WriteRes<WriteVShiftX, [SiFive7NVA]>;
def : WriteRes<WriteVShiftI, [SiFive7NVA]>;
def : WriteRes<WriteVNShiftV, [SiFive7NVA]>;
def : WriteRes<WriteVNShiftX, [SiFive7NVA]>;
def : WriteRes<WriteVNShiftI, [SiFive7NVA]>;
}
let Latency = 4 in {
def : WriteRes<WriteVICmpV, [SiFive7NVA]>;
def : WriteRes<WriteVICmpX, [SiFive7NVA]>;
def : WriteRes<WriteVICmpI, [SiFive7NVA]>;
}
let Latency = 8 in {
def : WriteRes<WriteVIMulV, [SiFive7NVA]>;
def : WriteRes<WriteVIMulX, [SiFive7NVA]>;
}
let Latency = 16 in {
def : WriteRes<WriteVIDivV, [SiFive7NVA]>;
def : WriteRes<WriteVIDivX, [SiFive7NVA]>;
}
let Latency = 8 in {
def : WriteRes<WriteVIWMulV, [SiFive7NVA]>;
def : WriteRes<WriteVIWMulX, [SiFive7NVA]>;
def : WriteRes<WriteVIMulAddV, [SiFive7NVA]>;
def : WriteRes<WriteVIMulAddX, [SiFive7NVA]>;
def : WriteRes<WriteVIWMulAddV, [SiFive7NVA]>;
def : WriteRes<WriteVIWMulAddX, [SiFive7NVA]>;
}
let Latency = 4 in {
def : WriteRes<WriteVIMergeV, [SiFive7NVA]>;
def : WriteRes<WriteVIMergeX, [SiFive7NVA]>;
def : WriteRes<WriteVIMergeI, [SiFive7NVA]>;
def : WriteRes<WriteVIMovV, [SiFive7NVA]>;
def : WriteRes<WriteVIMovX, [SiFive7NVA]>;
def : WriteRes<WriteVIMovI, [SiFive7NVA]>;
}

// 13. Vector Fixed-Point Arithmetic Instructions
let Latency = 8 in {
def : WriteRes<WriteVSALUV, [SiFive7NVA]>;
def : WriteRes<WriteVSALUX, [SiFive7NVA]>;
def : WriteRes<WriteVSALUI, [SiFive7NVA]>;
def : WriteRes<WriteVAALUV, [SiFive7NVA]>;
def : WriteRes<WriteVAALUX, [SiFive7NVA]>;
def : WriteRes<WriteVSMulV, [SiFive7NVA]>;
def : WriteRes<WriteVSMulX, [SiFive7NVA]>;
def : WriteRes<WriteVSShiftV, [SiFive7NVA]>;
def : WriteRes<WriteVSShiftX, [SiFive7NVA]>;
def : WriteRes<WriteVSShiftI, [SiFive7NVA]>;
def : WriteRes<WriteVNClipV, [SiFive7NVA]>;
def : WriteRes<WriteVNClipX, [SiFive7NVA]>;
def : WriteRes<WriteVNClipI, [SiFive7NVA]>;

// 14. Vector Floating-Point Instructions
def : WriteRes<WriteVFALUV, [SiFive7NVA]>;
def : WriteRes<WriteVFALUF, [SiFive7NVA]>;
def : WriteRes<WriteVFWALUV, [SiFive7NVA]>;
def : WriteRes<WriteVFWALUF, [SiFive7NVA]>;
def : WriteRes<WriteVFMulV, [SiFive7NVA]>;
def : WriteRes<WriteVFMulF, [SiFive7NVA]>;
}
let Latency = 16 in {
def : WriteRes<WriteVFDivV, [SiFive7NVA]>;
def : WriteRes<WriteVFDivF, [SiFive7NVA]>;
}
let Latency = 8 in {
def : WriteRes<WriteVFWMulV, [SiFive7NVA]>;
def : WriteRes<WriteVFWMulF, [SiFive7NVA]>;
def : WriteRes<WriteVFMulAddV, [SiFive7NVA]>;
def : WriteRes<WriteVFMulAddF, [SiFive7NVA]>;
def : WriteRes<WriteVFWMulAddV, [SiFive7NVA]>;
def : WriteRes<WriteVFWMulAddF, [SiFive7NVA]>;
}
let Latency = 16 in
def : WriteRes<WriteVFSqrtV, [SiFive7NVA]>;
let Latency = 8 in {
def : WriteRes<WriteVFRecpV, [SiFive7NVA]>;
def : WriteRes<WriteVFCmpV, [SiFive7NVA]>;
def : WriteRes<WriteVFCmpF, [SiFive7NVA]>;
def : WriteRes<WriteVFSgnjV, [SiFive7NVA]>;
def : WriteRes<WriteVFSgnjF, [SiFive7NVA]>;
def : WriteRes<WriteVFClassV, [SiFive7NVA]>;
def : WriteRes<WriteVFMergeV, [SiFive7NVA]>;
def : WriteRes<WriteVFMovV, [SiFive7NVA]>;
def : WriteRes<WriteVFCvtIToFV, [SiFive7NVA]>;
def : WriteRes<WriteVFCvtFToIV, [SiFive7NVA]>;
def : WriteRes<WriteVFWCvtIToFV, [SiFive7NVA]>;
def : WriteRes<WriteVFWCvtFToIV, [SiFive7NVA]>;
def : WriteRes<WriteVFWCvtFToFV, [SiFive7NVA]>;
def : WriteRes<WriteVFNCvtIToFV, [SiFive7NVA]>;
def : WriteRes<WriteVFNCvtFToIV, [SiFive7NVA]>;
def : WriteRes<WriteVFNCvtFToFV, [SiFive7NVA]>;
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
def : WriteRes<WriteVMALUV, [SiFive7NVA]>;
def : WriteRes<WriteVMPopV, [SiFive7NVA]>;
def : WriteRes<WriteVMFFSV, [SiFive7NVA]>;
def : WriteRes<WriteVMSFSV, [SiFive7NVA]>;
def : WriteRes<WriteVMIotV, [SiFive7NVA]>;
def : WriteRes<WriteVMIdxV, [SiFive7NVA]>;
}

// 17. Vector Permutation Instructions
let Latency = 10 in
def : WriteRes<WriteVIMovVX, [SiFive7NVA]>;
let Latency = 4 in
def : WriteRes<WriteVIMovXV, [SiFive7NVA]>;
let Latency = 10 in
def : WriteRes<WriteVFMovVF, [SiFive7NVA]>;
let Latency = 4 in
def : WriteRes<WriteVFMovFV, [SiFive7NVA]>;
let Latency = 4 in {
def : WriteRes<WriteVISlideX, [SiFive7NVA]>;
def : WriteRes<WriteVISlideI, [SiFive7NVA]>;
}
let Latency = 8 in {
def : WriteRes<WriteVISlide1X, [SiFive7NVA]>;
def : WriteRes<WriteVFSlide1F, [SiFive7NVA]>;
}
let Latency = 16 in {
def : WriteRes<WriteVGatherV, [SiFive7NVA]>;
def : WriteRes<WriteVGatherX, [SiFive7NVA]>;
def : WriteRes<WriteVGatherI, [SiFive7NVA]>;
def : WriteRes<WriteVCompressV, [SiFive7NVA]>;
}
let Latency = 4 in {
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
                                   WriteLDWU, WriteLDD]>;
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
def : ReadAdvance<ReadVLDX, 0>;
def : ReadAdvance<ReadVSTX, 0>;
def : ReadAdvance<ReadVSTEV, 0>;
def : ReadAdvance<ReadVSTM, 0>;
def : ReadAdvance<ReadVLDSX, 0>;
def : ReadAdvance<ReadVSTSX, 0>;
def : ReadAdvance<ReadVSTS8V, 0>;
def : ReadAdvance<ReadVSTS16V, 0>;
def : ReadAdvance<ReadVSTS32V, 0>;
def : ReadAdvance<ReadVSTS64V, 0>;
def : ReadAdvance<ReadVLDUXV, 0>;
def : ReadAdvance<ReadVLDOXV, 0>;
def : ReadAdvance<ReadVSTUX8, 0>;
def : ReadAdvance<ReadVSTUX16, 0>;
def : ReadAdvance<ReadVSTUX32, 0>;
def : ReadAdvance<ReadVSTUX64, 0>;
def : ReadAdvance<ReadVSTUXV, 0>;
def : ReadAdvance<ReadVSTUX8V, 0>;
def : ReadAdvance<ReadVSTUX16V, 0>;
def : ReadAdvance<ReadVSTUX32V, 0>;
def : ReadAdvance<ReadVSTUX64V, 0>;
def : ReadAdvance<ReadVSTOX8, 0>;
def : ReadAdvance<ReadVSTOX16, 0>;
def : ReadAdvance<ReadVSTOX32, 0>;
def : ReadAdvance<ReadVSTOX64, 0>;
def : ReadAdvance<ReadVSTOXV, 0>;
def : ReadAdvance<ReadVSTOX8V, 0>;
def : ReadAdvance<ReadVSTOX16V, 0>;
def : ReadAdvance<ReadVSTOX32V, 0>;
def : ReadAdvance<ReadVSTOX64V, 0>;
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 12. Vector Integer Arithmetic Instructions
def : ReadAdvance<ReadVIALUV, 0>;
def : ReadAdvance<ReadVIALUX, 0>;
def : ReadAdvance<ReadVIWALUV, 0>;
def : ReadAdvance<ReadVIWALUX, 0>;
def : ReadAdvance<ReadVExtV, 0>;
def : ReadAdvance<ReadVIALUCV, 0>;
def : ReadAdvance<ReadVIALUCX, 0>;
def : ReadAdvance<ReadVShiftV, 0>;
def : ReadAdvance<ReadVShiftX, 0>;
def : ReadAdvance<ReadVNShiftV, 0>;
def : ReadAdvance<ReadVNShiftX, 0>;
def : ReadAdvance<ReadVICmpV, 0>;
def : ReadAdvance<ReadVICmpX, 0>;
def : ReadAdvance<ReadVIMulV, 0>;
def : ReadAdvance<ReadVIMulX, 0>;
def : ReadAdvance<ReadVIDivV, 0>;
def : ReadAdvance<ReadVIDivX, 0>;
def : ReadAdvance<ReadVIWMulV, 0>;
def : ReadAdvance<ReadVIWMulX, 0>;
def : ReadAdvance<ReadVIMulAddV, 0>;
def : ReadAdvance<ReadVIMulAddX, 0>;
def : ReadAdvance<ReadVIWMulAddV, 0>;
def : ReadAdvance<ReadVIWMulAddX, 0>;
def : ReadAdvance<ReadVIMergeV, 0>;
def : ReadAdvance<ReadVIMergeX, 0>;
def : ReadAdvance<ReadVIMovV, 0>;
def : ReadAdvance<ReadVIMovX, 0>;

// 13. Vector Fixed-Point Arithmetic Instructions
def : ReadAdvance<ReadVSALUV, 0>;
def : ReadAdvance<ReadVSALUX, 0>;
def : ReadAdvance<ReadVAALUV, 0>;
def : ReadAdvance<ReadVAALUX, 0>;
def : ReadAdvance<ReadVSMulV, 0>;
def : ReadAdvance<ReadVSMulX, 0>;
def : ReadAdvance<ReadVSShiftV, 0>;
def : ReadAdvance<ReadVSShiftX, 0>;
def : ReadAdvance<ReadVNClipV, 0>;
def : ReadAdvance<ReadVNClipX, 0>;

// 14. Vector Floating-Point Instructions
def : ReadAdvance<ReadVFALUV, 0>;
def : ReadAdvance<ReadVFALUF, 0>;
def : ReadAdvance<ReadVFWALUV, 0>;
def : ReadAdvance<ReadVFWALUF, 0>;
def : ReadAdvance<ReadVFMulV, 0>;
def : ReadAdvance<ReadVFMulF, 0>;
def : ReadAdvance<ReadVFDivV, 0>;
def : ReadAdvance<ReadVFDivF, 0>;
def : ReadAdvance<ReadVFWMulV, 0>;
def : ReadAdvance<ReadVFWMulF, 0>;
def : ReadAdvance<ReadVFMulAddV, 0>;
def : ReadAdvance<ReadVFMulAddF, 0>;
def : ReadAdvance<ReadVFWMulAddV, 0>;
def : ReadAdvance<ReadVFWMulAddF, 0>;
def : ReadAdvance<ReadVFSqrtV, 0>;
def : ReadAdvance<ReadVFRecpV, 0>;
def : ReadAdvance<ReadVFCmpV, 0>;
def : ReadAdvance<ReadVFCmpF, 0>;
def : ReadAdvance<ReadVFSgnjV, 0>;
def : ReadAdvance<ReadVFSgnjF, 0>;
def : ReadAdvance<ReadVFClassV, 0>;
def : ReadAdvance<ReadVFMergeV, 0>;
def : ReadAdvance<ReadVFMergeF, 0>;
def : ReadAdvance<ReadVFMovF, 0>;
def : ReadAdvance<ReadVFCvtIToFV, 0>;
def : ReadAdvance<ReadVFCvtFToIV, 0>;
def : ReadAdvance<ReadVFWCvtIToFV, 0>;
def : ReadAdvance<ReadVFWCvtFToIV, 0>;
def : ReadAdvance<ReadVFWCvtFToFV, 0>;
def : ReadAdvance<ReadVFNCvtIToFV, 0>;
def : ReadAdvance<ReadVFNCvtFToIV, 0>;
def : ReadAdvance<ReadVFNCvtFToFV, 0>;

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
def : ReadAdvance<ReadVMALUV, 0>;
def : ReadAdvance<ReadVMPopV, 0>;
def : ReadAdvance<ReadVMFFSV, 0>;
def : ReadAdvance<ReadVMSFSV, 0>;
def : ReadAdvance<ReadVMIotV, 0>;

// 17. Vector Permutation Instructions
def : ReadAdvance<ReadVIMovVX, 0>;
def : ReadAdvance<ReadVIMovXV, 0>;
def : ReadAdvance<ReadVIMovXX, 0>;
def : ReadAdvance<ReadVFMovVF, 0>;
def : ReadAdvance<ReadVFMovFV, 0>;
def : ReadAdvance<ReadVFMovFX, 0>;
def : ReadAdvance<ReadVISlideV, 0>;
def : ReadAdvance<ReadVISlideX, 0>;
def : ReadAdvance<ReadVFSlideV, 0>;
def : ReadAdvance<ReadVFSlideF, 0>;
def : ReadAdvance<ReadVGatherV, 0>;
def : ReadAdvance<ReadVGatherX, 0>;
def : ReadAdvance<ReadVCompressV, 0>;
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
