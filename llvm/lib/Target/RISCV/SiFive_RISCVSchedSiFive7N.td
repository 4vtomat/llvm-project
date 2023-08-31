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

/// c is true if mx has the worst case behavior compared to LMULs in MxList.
/// On the SiFive7N, the worst case LMUL is the Largest LMUL
/// and the worst case sew is the smallest SEW for that LMUL.
class SiFive7NIsWorstCaseMX<string mx, list<string> MxList> {
  string LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

/// c is true if mx and sew have the worst case behavior compared to LMULs in
/// MxList. On the SiFive7N, the worst case LMUL is the Largest LMUL
/// and the worst case sew is the smallest SEW for that LMUL.
class SiFive7NIsWorstCaseMXSEW<string mx, int sew, list<string> MxList, bit isF = 0> {
  string LLMUL = LargestLMUL<MxList>.r;
  int SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

/// Number of DLEN parts = (LMUL * VLEN) / DLEN.
/// Since DLEN = VLEN / 2, Num DLEN parts = 2 * LMUL.
class SiFive7NGetCyclesDefault<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 2,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

class SiFive7NGetCyclesNarrowing<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 8,
    !eq(mx, "M4") : 16,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

class SiFive7NGetCyclesVMask<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 1,
    !eq(mx, "M2") : 1,
    !eq(mx, "M4") : 1,
    !eq(mx, "M8") : 2,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

/// VLDM and VSTM can't read/write more than 2 DLENs of data.
/// 2 DLENs when LMUL=8. 1 DLEN for all other DLENs.
/// In other words, ((LMUL*VLEN)/8)/DLEN.
class SiFive7NGetMaskLoadStoreCycles<string mx> {
  int c = !cond(
    !eq(mx, "M8")  : 2,
    true : 1
  );
}

// Cycles for nf=2 segmented loads and stores are calculated using the
// formula (2 * VLEN * LMUL) / DLEN = 4 * LMUL
class SiFive7NGetCyclesSegmentedSeg2<string mx> {
  int c = !cond(
    !eq(mx, "M1") :  4,
    !eq(mx, "M2") :  8,
    !eq(mx, "M4") :  16,
    !eq(mx, "M8") :  32,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

// Cycles for segmented loads and stores are calculated using the
// formula vl * ceil((SEW * nf) / DLEN), where SEW * nf is the segment size.
class SiFive7NGetCyclesSegmented<string mx, int sew, int nf> {
  defvar VLEN = 512;
  defvar DLEN = 256;
  // (VLEN * LMUL) / SEW
  defvar VLUpperBound  = !cond(
    !eq(mx, "M1") : !div(VLEN, sew),
    !eq(mx, "M2") : !div(!mul(VLEN, 2), sew),
    !eq(mx, "M4") : !div(!mul(VLEN, 4), sew),
    !eq(mx, "M8") : !div(!mul(VLEN, 8), sew),
    !eq(mx, "MF2") : !div(!div(VLEN, 2), sew),
    !eq(mx, "MF4") : !div(!div(VLEN, 4), sew),
    !eq(mx, "MF8") : !div(!div(VLEN, 8), sew),
  );
  // We can calculate ceil(a/b) using (a + b - 1) / b.
  defvar a = !mul(sew, nf);
  defvar b = DLEN;
  int c = !mul(VLUpperBound, !div(!sub(!add(a, b), 1), b));
}

class SiFive7NGetCyclesOnePerElement<string mx, int sew> {
  // FIXME: On SiFive7N, VLEN is 512. Although a user can request the compiler
  // to use a different VLEN, this model will not make scheduling decisions
  // based on the user specified VLEN.
  // c = ceil(VLEN / SEW) * LMUL
  // Note: c >= 1 since the smallest VLEN is 512 / 8 = 8, and the
  // largest division performed on  VLEN is in MF8 case with division
  // by 8. Therefore, there is no need to ceil the result.
  int VLEN = !div(512, sew);
  int c = !cond(
    !eq(mx, "M1")  : VLEN,
    !eq(mx, "M2")  : !mul(VLEN, 2),
    !eq(mx, "M4")  : !mul(VLEN, 4),
    !eq(mx, "M8")  : !mul(VLEN, 8),
    !eq(mx, "MF2") : !div(VLEN, 2),
    !eq(mx, "MF4") : !div(VLEN, 4),
    !eq(mx, "MF8") : !div(VLEN, 8)
  );
}

class SiFive7NGetDivOrSqrtFactor<int sew> {
  int c = !cond(
    !eq(sew, 16) : 15,
    !eq(sew, 32) : 28,
    !eq(sew, 64) : 57
  );
}

/// Cycles for reductions take approximately VL*SEW/DLEN + 5(4 + log(DLEN/SEW))
/// cycles
class SiFive7NGetReductionCycles<string mx, int sew> {
  // VLUpperBound*SEW/DLEN is equivalent to 2*LMUL since
  // VLUpperBound=(VLEN*LMUL)/SEW.
  defvar VLEN = 512;
  defvar DLEN = !div(VLEN, 2);
  defvar TwoTimesLMUL = !cond(
    !eq(mx, "M1") : 2,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
  int c = !add(
    TwoTimesLMUL,
    !mul(5, !add(4, !logtwo(!div(DLEN, sew))))
  );
}

/// Cycles for ordered reductions take approximately 6*VL cycles
class SiFive7NGetOrderedReductionCycles<string mx, int sew> {
  defvar VLEN = 512;
  // (VLEN * LMUL) / SEW
  defvar VLUpperBound  = !cond(
    !eq(mx, "M1") : !div(VLEN, sew),
    !eq(mx, "M2") : !div(!mul(VLEN, 2), sew),
    !eq(mx, "M4") : !div(!mul(VLEN, 4), sew),
    !eq(mx, "M8") : !div(!mul(VLEN, 8), sew),
    !eq(mx, "MF2") : !div(!div(VLEN, 2), sew),
    !eq(mx, "MF4") : !div(!div(VLEN, 4), sew),
    !eq(mx, "MF8") : !div(!div(VLEN, 8), sew),
  );
  int c = !mul(6, VLUpperBound);
}

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
                             HasVendorXSfvqmaccqoq, HasVendorXSfvqmaccdod, // SIFIVE
                             HasVendorXSfvfhbfmin]; // SIFIVE
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
let Latency = 3 in {
def : WriteRes<WriteJmp, [SiFive7NPipeB]>;
def : WriteRes<WriteJal, [SiFive7NPipeB]>;
def : WriteRes<WriteJalr, [SiFive7NPipeB]>;
}

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

// Single-bit instructions
// BEXT[I] instruction is available on all ALUs and the other instructions
// are only available on the SiFive7NB pipe.
let Latency = 3 in {
def : WriteRes<WriteSingleBit, [SiFive7NPipeB]>;
def : WriteRes<WriteSingleBitImm, [SiFive7NPipeB]>;
def : WriteRes<WriteBEXT, [SiFive7NPipeAB]>;
def : WriteRes<WriteBEXTI, [SiFive7NPipeAB]>;
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
// Unit-stride loads and stores can operate at the full bandwidth of the memory
// pipe. The memory pipe is DLEN bits wide on x280.
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 4, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVLDE",    [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDFF",   [SiFive7NVL], mx, IsWorstCase>;
  }
  let Latency = 1, ResourceCycles = [Cycles] in
    defm "" : LMULWriteResMX<"WriteVSTE",    [SiFive7NVS], mx, IsWorstCase>;
}

foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetMaskLoadStoreCycles<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 4, ResourceCycles = [Cycles] in
    defm "" : LMULWriteResMX<"WriteVLDM",    [SiFive7NVL], mx, IsWorstCase>;
  let Latency = 1, ResourceCycles = [Cycles] in
    defm "" : LMULWriteResMX<"WriteVSTM",    [SiFive7NVS], mx, IsWorstCase>;
}

// Strided loads and stores operate at one element per cycle and should be
// scheduled accordingly. Indexed loads and stores operate at one element per
// cycle, and they stall the machine until all addresses have been generated,
// so they cannot be scheduled. Indexed and strided loads and stores have LMUL
// specific suffixes, but since SEW is already encoded in the name of the
// resource, we do not need to use LMULSEWXXX constructors. However, we do
// use the SEW from the name to determine the number of Cycles.
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesOnePerElement<mx, 8>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = !add(3, Cycles), ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVLDS8",  [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX8", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX8", [SiFive7NVL], mx, IsWorstCase>;
  }
  let Latency = 1, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVSTS8",  [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX8", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX8", [SiFive7NVS], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesOnePerElement<mx, 16>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = !add(3, Cycles), ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVLDS16",  [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX16", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX16", [SiFive7NVL], mx, IsWorstCase>;
  }
  let Latency = 1, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVSTS16",  [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX16", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX16", [SiFive7NVS], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesOnePerElement<mx, 32>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = !add(3, Cycles), ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVLDS32",  [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX32", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX32", [SiFive7NVL], mx, IsWorstCase>;
  }
  let Latency = 1, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVSTS32",  [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX32", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX32", [SiFive7NVS], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesOnePerElement<mx, 64>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = !add(3, Cycles), ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVLDS64",  [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX64", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX64", [SiFive7NVL], mx, IsWorstCase>;
  }
  let Latency = 1, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVSTS64",  [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX64", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX64", [SiFive7NVS], mx, IsWorstCase>;
  }
}

// VLD*R is LMUL aware
let Latency = 4, ResourceCycles = [2] in
  def : WriteRes<WriteVLD1R,  [SiFive7NVL]>;
let Latency = 4, ResourceCycles = [4] in
  def : WriteRes<WriteVLD2R,  [SiFive7NVL]>;
let Latency = 4, ResourceCycles = [8] in
  def : WriteRes<WriteVLD4R,  [SiFive7NVL]>;
let Latency = 4, ResourceCycles = [16] in
  def : WriteRes<WriteVLD8R,  [SiFive7NVL]>;
// VST*R is LMUL aware
let Latency = 1, ResourceCycles = [2] in
  def : WriteRes<WriteVST1R,   [SiFive7NVS]>;
let Latency = 1, ResourceCycles = [4] in
  def : WriteRes<WriteVST2R,   [SiFive7NVS]>;
let Latency = 1, ResourceCycles = [8] in
  def : WriteRes<WriteVST4R,   [SiFive7NVS]>;
let Latency = 1, ResourceCycles = [16] in
  def : WriteRes<WriteVST8R,   [SiFive7NVS]>;

// Segmented Loads and Stores
// Unit-stride segmented loads and stores are effectively converted into strided
// segment loads and stores. Strided segment loads and stores operate at up to
// one segment per cycle if the segment fits within one aligned memory beat.
// Indexed segment loads and stores operate at the same rate as strided ones,
// but they stall the machine until all addresses have been generated.
foreach mx = SchedMxList in {
  foreach eew = [8, 16, 32, 64] in {
    defvar Cycles = SiFive7NGetCyclesSegmentedSeg2<mx>.c;
    defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
    // Does not chain so set latency high
    let Latency = !add(3, Cycles), ResourceCycles = [Cycles] in {
      defm "" : LMULWriteResMX<"WriteVLSEG2e" # eew,   [SiFive7NVL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLSEGFF2e" # eew, [SiFive7NVL], mx, IsWorstCase>;
    }
    let Latency = 1, ResourceCycles = [Cycles] in
      defm "" : LMULWriteResMX<"WriteVSSEG2e" # eew,   [SiFive7NVS], mx, IsWorstCase>;
    foreach nf=3-8 in {
      defvar Cycles = SiFive7NGetCyclesSegmented<mx, eew, nf>.c;
      defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
      // Does not chain so set latency high
      let Latency = !add(3, Cycles), ResourceCycles = [Cycles] in {
        defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" # eew,   [SiFive7NVL], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" # eew, [SiFive7NVL], mx, IsWorstCase>;
      }
      let Latency = 1, ResourceCycles = [Cycles] in
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" # eew,   [SiFive7NVS], mx, IsWorstCase>;
    }
  }
}
foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar Cycles = SiFive7NGetCyclesSegmented<mx, eew, nf>.c;
      defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
      // Does not chain so set latency high
      let Latency = !add(3, Cycles), ResourceCycles = [Cycles] in {
        defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" # eew,  [SiFive7NVL], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" # eew, [SiFive7NVL], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" # eew, [SiFive7NVL], mx, IsWorstCase>;
      }
      let Latency = 1, ResourceCycles = [Cycles] in {
        defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" # eew,  [SiFive7NVS], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" # eew, [SiFive7NVS], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" # eew, [SiFive7NVS], mx, IsWorstCase>;
      }
    }
  }
}

// 11. Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 4, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVIALUV",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUX",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUI",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUV",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUX",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUI",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftV",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftX",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftI",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxV",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxX",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulV",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulX",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddV",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddX",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeV",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeX",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeI",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovV",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovX",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovI",     [SiFive7NVA], mx, IsWorstCase>;
  }
  // Mask results can't chain.
  let Latency = !add(Cycles, 3), ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVICmpV",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI",     [SiFive7NVA], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 4, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVExtV",      [SiFive7NVA], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    // One bit at a time, but up to four elements can be processed at a time.
    // Add 3 to number of elements to ensure the group formed by remainder
    // elements are accounted for.
    defvar Cycles =
        !mul(sew, !div(!add(3, SiFive7NGetCyclesOnePerElement<mx, sew>.c), 4));
    defvar IsWorstCase = SiFive7NIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = Cycles, ResourceCycles = [Cycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [SiFive7NVA], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [SiFive7NVA], mx, sew, IsWorstCase>;
    }
  }
}

// Widening
foreach mx = SchedMxListW in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVIWALUV",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUX",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulV",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulX",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [SiFive7NVA], mx, IsWorstCase>;
  }
}
// Narrowing
foreach mx = SchedMxListW in {
  defvar Cycles = SiFive7NGetCyclesNarrowing<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVNShiftV",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI",   [SiFive7NVA], mx, IsWorstCase>;
  }
}

// 12. Vector Fixed-Point Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVSALUV",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUX",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUI",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUV",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUX",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulV",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulX",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftV", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftX", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftI", [SiFive7NVA], mx, IsWorstCase>;
  }
}
// Narrowing
foreach mx = SchedMxListW in {
  defvar Cycles = SiFive7NGetCyclesNarrowing<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVNClipV",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipX",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI",  [SiFive7NVA], mx, IsWorstCase>;
  }
}

// 13. Vector Floating-Point Instructions
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVFALUV",      [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFALUF",      [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMulV",      [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMulF",      [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMulAddV",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMulAddF",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFRecpV",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCvtIToFV",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCvtFToIV",  [SiFive7NVA], mx, IsWorstCase>;
  }
  let Latency = 4, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVFSgnjV",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSgnjF",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMinMaxV",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMinMaxF",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFClassV",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMergeV",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovV",      [SiFive7NVA], mx, IsWorstCase>;
  }
  // Mask results can't chain.
  let Latency = !add(Cycles, 3), ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVFCmpV",      [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCmpF",      [SiFive7NVA], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar Cycles = !mul(SiFive7NGetDivOrSqrtFactor<sew>.c,
                         !div(SiFive7NGetCyclesOnePerElement<mx, sew>.c, 4));
    defvar IsWorstCase = SiFive7NIsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    let Latency = Cycles, ResourceCycles = [Cycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [SiFive7NVA], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV",  [SiFive7NVA], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF",  [SiFive7NVA], mx, sew, IsWorstCase>;
    }
  }
}

// Widening
foreach mx = SchedMxListW in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVFWCvtIToFV", [SiFive7NVA], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxListFW in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxListFW>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVFWALUV",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWMulV",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWMulAddV",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWCvtFToFV", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWMulAddF",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWMulF",     [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWALUF",     [SiFive7NVA], mx, IsWorstCase>;
  }
}
// Narrowing
foreach mx = SchedMxListW in {
  defvar Cycles = SiFive7NGetCyclesNarrowing<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [SiFive7NVA], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxListFW in {
  defvar Cycles = SiFive7NGetCyclesNarrowing<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxListFW>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVFNCvtIToFV", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFNCvtFToFV", [SiFive7NVA], mx, IsWorstCase>;
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar Cycles = SiFive7NGetReductionCycles<mx, sew>.c;
    defvar IsWorstCase = SiFive7NIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = Cycles, ResourceCycles = [Cycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From", [SiFive7NVA],
                                     mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [SiFive7NVA],
                                     mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar Cycles = SiFive7NGetReductionCycles<mx, sew>.c;
    defvar IsWorstCase = SiFive7NIsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;
    let Latency = Cycles, ResourceCycles = [Cycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [SiFive7NVA],
                                     mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar RedCycles = SiFive7NGetReductionCycles<mx, sew>.c;
    defvar IsWorstCase = SiFive7NIsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    let Latency = RedCycles, ResourceCycles = [RedCycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [SiFive7NVA],
                                     mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From", [SiFive7VA],
                                     mx, sew, IsWorstCase>;
    }
    defvar OrdRedCycles = SiFive7NGetOrderedReductionCycles<mx, sew>.c;
    let Latency = OrdRedCycles, ResourceCycles = [OrdRedCycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [SiFive7NVA],
                                 mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListFWRed in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar RedCycles = SiFive7NGetReductionCycles<mx, sew>.c;
    defvar IsWorstCase = SiFive7NIsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;
    let Latency = RedCycles, ResourceCycles = [RedCycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From", [SiFive7NVA],
                                     mx, sew, IsWorstCase>;
    }
    defvar OrdRedCycles = SiFive7NGetOrderedReductionCycles<mx, sew>.c;
    let Latency = OrdRedCycles, ResourceCycles = [OrdRedCycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SiFive7NVA],
                                     mx, sew, IsWorstCase>;
    }
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesVMask<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 4, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVMALUV", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMPopV", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMFFSV", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMSFSV", [SiFive7NVA], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 4, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVMIotV", [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMIdxV", [SiFive7NVA], mx, IsWorstCase>;
  }
}

let Latency = 4, ResourceCycles = [1] in {
  def : WriteRes<WriteVIMovVX, [SiFive7NVA]>;
  def : WriteRes<WriteVIMovXV, [SiFive7NVA]>;
  def : WriteRes<WriteVFMovVF, [SiFive7NVA]>;
  def : WriteRes<WriteVFMovFV, [SiFive7NVA]>;
}

// 16. Vector Permutation Instructions
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRGatherVI",    [SiFive7NVA], mx, IsWorstCase>;
  }
}

// The Orca release makes some changes compared to the Narwhal release. Instead
// of splitting SiFive7N to SiFive7N and SiFiveO, use SchedVariants to model
// the minor changes.
def SiFiveOPred : SchedPredicate<[{SchedModel->getSubtargetInfo()->hasFeature(
                      RISCV::TuneOptimizedVRGatherVCompress)}]>;
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar Cycles = SiFive7NGetCyclesOnePerElement<mx, sew>.c;
    defvar IsWorstCase = SiFive7NIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    // Define defaults
    let Latency = !add(Cycles, 3), ResourceCycles = [Cycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFive7NVA], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFive7NVA], mx, sew, IsWorstCase>;
    }

    if !or(!or(!eq(mx, "MF2"), !eq(mx, "MF4")), !eq(mx, "MF8")) then {
      assert !not(IsWorstCase),
        "On SiFive7N MF2, MF4, and MF8 never lead to WorstCase behavior";

      // Define Narwhal resources
      def "WriteVRGatherVV_" # mx # "_E" # sew # "_N" : SchedWriteRes<[SiFive7NVA]>
        { let Latency = !add(Cycles, 3); let ResourceCycles = [Cycles]; }
      def "WriteVCompressV_" # mx # "_E" # sew # "_N" : SchedWriteRes<[SiFive7NVA]>
        { let Latency = !add(Cycles, 3); let ResourceCycles = [Cycles]; }
      // Define Orca resources
      def "WriteVRGatherVV_" # mx # "_E" # sew # "_O" : SchedWriteRes<[SiFive7NVA]>
        { let Latency = 4; let ResourceCycles = [1]; }
      def "WriteVCompressV_" # mx # "_E" # sew # "_O" : SchedWriteRes<[SiFive7NVA]>
        { let Latency = 4; let ResourceCycles = [1]; }

      // Define variants to pick Narwhal or Orca resources
      def "WriteVRGatherVV_" # mx # "_E" # sew # "Variant" : SchedWriteVariant<[
        SchedVar<SiFiveOPred, [!cast<SchedWriteRes>("WriteVRGatherVV_" # mx # "_E"
                                                    # sew # "_O")]>,
        SchedVar<NoSchedPred, [!cast<SchedWriteRes>("WriteVRGatherVV_" # mx # "_E"
                                                    # sew # "_N")]>]>;
      def "WriteVCompressV_" # mx # "_E" # sew # "Variant" : SchedWriteVariant<[
        SchedVar<SiFiveOPred, [!cast<SchedWriteRes>("WriteVCompressV_" # mx # "_E"
                                                    # sew # "_O")]>,
        SchedVar<NoSchedPred, [!cast<SchedWriteRes>("WriteVCompressV_" # mx # "_E"
                                                    # sew # "_N")]>]>;

      // Tie variant choice to the correct instructions. This overrides default
      // behavior for these instructions that was defined above.
      def : SchedAlias<
        !cast<SchedReadWrite>("WriteVRGatherVV_" # mx # "_E" # sew),
        !cast<SchedReadWrite>("WriteVRGatherVV_" # mx # "_E" # sew # "Variant")>;
      def : SchedAlias<
        !cast<SchedReadWrite>("WriteVCompressV_" # mx # "_E" # sew),
        !cast<SchedReadWrite>("WriteVCompressV_" # mx # "_E" # sew # "Variant")>;
    }
  }
}

foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 4, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVISlideX",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVISlideI",   [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVISlide1X",  [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSlide1F",  [SiFive7NVA], mx, IsWorstCase>;
  }
}

// VMov*V is LMUL Aware
let Latency = 4, ResourceCycles = [2] in
  def : WriteRes<WriteVMov1V,     [SiFive7NVA]>;
let Latency = 4, ResourceCycles = [4] in
  def : WriteRes<WriteVMov2V,     [SiFive7NVA]>;
let Latency = 4, ResourceCycles = [8] in
  def : WriteRes<WriteVMov4V,     [SiFive7NVA]>;
let Latency = 4, ResourceCycles = [16] in
  def : WriteRes<WriteVMov8V,     [SiFive7NVA]>;

// Others
def : WriteRes<WriteCSR, [SiFive7NPipeB]>;
def : WriteRes<WriteNop, []>;
let Latency = 3 in {
def : WriteRes<WriteRdVLENB, [SiFive7NPipeB]>;
}

// FIXME: This could be better modeled by looking at the regclasses of the operands.
def : InstRW<[WriteIALU, ReadIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance

def SiFive7NReadMemBaseOffset0: SchedReadAdvance<1, [WriteLDB, WriteLDH, WriteLDW, WriteLDD]>;

// Bypass the AG stage of a dependendent memory op after a load if the offset is
// zero.
def SiFive7NReadMemBase : SchedReadVariant<[
  SchedVar<MemBaseOffset0Pred, [SiFive7NReadMemBaseOffset0]>,
  SchedVar<NoSchedPred, [NoReadAdvance]>
]>;
def : SchedAlias<ReadMemBase, SiFive7NReadMemBase>;

def : BulletAnyToGPRBypass<ReadJmp>;
def : BulletAnyToGPRBypass<ReadJalr>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
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

// 6. Configuration-Setting Instructions
def : ReadAdvance<ReadVSETVLI, 2>;
def : ReadAdvance<ReadVSETVL, 2>;

// 7. Vector Loads and Stores
def : ReadAdvance<ReadVLDX, 0>;
def : ReadAdvance<ReadVSTX, 0>;
defm "" : LMULReadAdvance<"ReadVSTEV", 0>;
defm "" : LMULReadAdvance<"ReadVSTM", 0>;
def : ReadAdvance<ReadVLDSX, 0>;
def : ReadAdvance<ReadVSTSX, 0>;
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
defm : LMULReadAdvance<"ReadVIMinMaxV", 0>;
defm : LMULReadAdvance<"ReadVIMinMaxX", 0>;
defm : LMULReadAdvance<"ReadVIMulV", 0>;
defm : LMULReadAdvance<"ReadVIMulX", 0>;
defm : LMULSEWReadAdvance<"ReadVIDivV", 0>;
defm : LMULSEWReadAdvance<"ReadVIDivX", 0>;
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
defm "" : LMULSEWReadAdvanceF<"ReadVFDivV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFDivF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFSqrtV", 0>;
defm "" : LMULReadAdvance<"ReadVFRecpV", 0>;
defm "" : LMULReadAdvance<"ReadVFMinMaxV", 0>;
defm "" : LMULReadAdvance<"ReadVFMinMaxF", 0>;
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
def : ReadAdvance<ReadVIMovVX, 0>;
def : ReadAdvance<ReadVIMovXV, 0>;
def : ReadAdvance<ReadVIMovXX, 0>;
def : ReadAdvance<ReadVFMovVF, 0>;
def : ReadAdvance<ReadVFMovFV, 0>;
def : ReadAdvance<ReadVFMovFX, 0>;
defm "" : LMULReadAdvance<"ReadVISlideV", 0>;
defm "" : LMULReadAdvance<"ReadVISlideX", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideV", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_data", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVI_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVCompressV", 0>;
// LMUL Aware
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;
def : ReadAdvance<ReadVMergeOp_WorstCase, 0>;
foreach mx = SchedMxList in {
  def : ReadAdvance<!cast<SchedRead>("ReadVMergeOp_" # mx), 0>;
  foreach sew = SchedSEWSet<mx>.val in
    def : ReadAdvance<!cast<SchedRead>("ReadVMergeOp_" # mx  # "_E" # sew), 0>;
}

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

// Short forward branch
// ALU, ALU_B, SFB to SFB
// FIXME: Is this right?
def : ReadAdvance<ReadSFBJmp, 2, [WriteIALU, WriteIALU32,
                                  WriteShiftImm, WriteShiftImm32,
                                  WriteShiftReg, WriteShiftReg32,
                                  WriteSHXADD, WriteSHXADD32,
                                  WriteRotateImm, WriteRotateImm32,
                                  WriteRotateReg, WriteRotateReg32,
                                  WriteCLZ, WriteCLZ32, WriteCTZ,
                                  WriteCTZ32, WriteCPOP, WriteCPOP32,
                                  WriteREV8, WriteORCB, WriteSFB]>;
def : ReadAdvance<ReadSFBALU, 2, [WriteIALU, WriteIALU32,
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
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedCMOV; // SIFIVE
}
