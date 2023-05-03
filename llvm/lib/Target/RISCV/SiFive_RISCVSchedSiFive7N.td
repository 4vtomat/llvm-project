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

class SiFive7NGetCyclesWidening<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 2,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
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

class SiFive7NGetCyclesOutputLMUL<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 1,
    !eq(mx, "M2") : 2,
    !eq(mx, "M4") : 4,
    !eq(mx, "M8") : 8,
    !eq(mx, "MF2") : 1,
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

// Cycles for segmented loads and stores are calculated using the
// formula ceil(2 * nf * lmul).
class SiFive7NGetCyclesSegmented<string mx, int nf> {
  int c = !cond(
    !eq(mx, "M1") : !mul(!mul(2, nf), 1),
    !eq(mx, "M2") : !mul(!mul(2, nf), 2),
    !eq(mx, "M4") : !mul(!mul(2, nf), 4),
    !eq(mx, "M8") : !mul(!mul(2, nf), 8),
    // We can calculate ceil(a/b) using (a + b - 1) / b.
    // Since the multiplication of fractional lmul is the
    // same as division by the denominator the formula we
    // use is ceil(2 * nf / lmul_denominator). We can use
    // ceil(a/b) where a = 2 * nf, b = lmul_denominator.
    !eq(mx, "MF2") : !div(!sub(!add(!mul(2, nf), 2), 1), 2),
    !eq(mx, "MF4") : !div(!sub(!add(!mul(2, nf), 4), 1), 4),
    !eq(mx, "MF8") : !div(!sub(!add(!mul(2, nf), 8), 1), 8)
  );
}

class SiFive7NGetCyclesOnePerElement<string mx, int sew> {
  // FIXME: On SiFive7N, VLEN is 512. Although a user can request the compiler
  // to use a different VLEN, this model will not make scheduling decisions
  // based on the user specified VLEN.
  // c = ceil(VLEN / SEW) * LMUL
  // Note: c >= 1 since the smallest VLUpperBound is 512 / 8 = 8, and the
  // largest division performed on  VLUpperBound is in MF8 case with division
  // by 8. Therefore, there is no need to ceil the result.
  int VLUpperBound = !div(512, sew);
  int c = !cond(
    !eq(mx, "M1") :        VLUpperBound,
    !eq(mx, "M2") :        !mul(VLUpperBound, 2),
    !eq(mx, "M4") :        !mul(VLUpperBound, 4),
    !eq(mx, "M8") :        !mul(VLUpperBound, 8),
    !eq(mx, "MF2") :       !div(VLUpperBound, 2),
    !eq(mx, "MF4") :       !div(VLUpperBound, 4),
    !eq(mx, "MF8") :       !div(VLUpperBound, 8)
  );
}

class SiFive7NGetDivOrSqrtFactor<int sew> {
  int c = !cond(
    !eq(sew, 16) : 15,
    !eq(sew, 32) : 28,
    !eq(sew, 64) : 57
  );
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
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVLDE",    [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTE",    [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDM",    [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTM",    [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS8",   [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS16",  [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS32",  [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS64",  [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS8",   [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS16",  [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS32",  [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS64",  [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX8",  [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX16", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX32", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX64", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX8",  [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX16", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX32", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX64", [SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX8",  [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX16", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX32", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX64", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX8",  [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX16", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX32", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX64", [SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDFF",   [SiFive7NVL], mx, IsWorstCase>;
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

foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar Cycles = SiFive7NGetCyclesSegmented<mx, nf>.c;
      defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
      let Latency = Cycles, ResourceCycles = [Cycles] in {
        defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" # eew,   [SiFive7NVL], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" # eew, [SiFive7NVL], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" # eew,  [SiFive7NVL], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" # eew, [SiFive7NVL], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" # eew, [SiFive7NVL], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" # eew,   [SiFive7NVS], mx, IsWorstCase>;
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
  defvar Cycles = SiFive7NGetCyclesOutputLMUL<mx>.c;
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
  defvar Cycles = SiFive7NGetCyclesWidening<mx>.c;
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
  foreach sew = SchedSEWSetF<mx>.val in {
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
  defvar Cycles = SiFive7NGetCyclesWidening<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVFWCvtIToFV", [SiFive7NVA], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxListFW in {
  defvar Cycles = SiFive7NGetCyclesWidening<mx>.c;
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
let Latency = 32 in {
defm "" : LMULWriteRes<"WriteVIRedV_From", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVIWRedV_From", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFRedV_From", [SiFive7NVA]>;
defm "" : LMULWriteRes<"WriteVFRedOV_From", [SiFive7NVA]>;
defm "" : LMULWriteResFWRed<"WriteVFWRedV_From", [SiFive7NVA]>;
defm "" : LMULWriteResFWRed<"WriteVFWRedOV_From", [SiFive7NVA]>;
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

// 16. Vector Permutation Instructions
foreach mx = SchedMxList in {
  defvar Cycles = SiFive7NGetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7NIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ResourceCycles = [Cycles] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRGatherVI",    [SiFive7NVA], mx, IsWorstCase>;
  }
  // These are independent of LMUL.
  let Latency = 4, ResourceCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVIMovVX",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovXV",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovVF",    [SiFive7NVA], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovFV",    [SiFive7NVA], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar Cycles = SiFive7NGetCyclesOnePerElement<mx, sew>.c;
    defvar IsWorstCase = SiFive7NIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = !add(Cycles, 3), ResourceCycles = [Cycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFive7NVA], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFive7NVA], mx, sew, IsWorstCase>;
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
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedCMOV; // SIFIVE
}
