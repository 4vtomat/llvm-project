//==- RISCVSchedSiFiveP600.td - SiFiveP600 Scheduling Defs ---*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
include "SiFive_RISCVSchedMallard.td"

class SiFiveP600VSM3CCycles<string mx> {
  // c = ceil(LMUL / 2)
  int c = !cond(!eq(mx, "M2") : 1,
                !eq(mx, "M4") : 2,
                !eq(mx, "M8") : 4,
                true : 1);
}

// P600 can sequence up to 4 uops per cycles. So M8 takes two cycles
// to sequence all 8 uops while other LMULs only take a single cycle.
class SiFiveP600SequencerCycles<string mx> {
  int c = !if(!eq(mx, "M8"), 2, 1);
}

// ======================================================================
// The latency and occupancy data in this section are primarily evaluated
// from llvm-exegesis.
// ======================================================================

class SiFiveP600VCryptoLatency<string mx> {
  int c = !cond(
    !eq(mx, "M4"): 4,
    !eq(mx, "M8"): 8,
    true:          2
  );
}

class SiFiveP600VFMinMaxReduction<string mx, int sew> {
  defvar E64Lat = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 6,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 10,
    true:           2
  );

  defvar E64Cycles = !cond(
    !eq(mx, "M1") : 3,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 5,
    !eq(mx, "M8") : 6,
    true:           2
  );

  int latency = !if(!eq(sew, 64), E64Lat, !add(E64Lat, 2));
  int cycles = !if(!eq(sew, 64), E64Cycles, !add(E64Cycles, 1));
}

class SiFiveP600VFUnorderedReduction<string mx, int sew> {
  defvar E64Lat = !cond(
    !eq(mx, "M1") : 6,
    !eq(mx, "M2") : 12,
    !eq(mx, "M4") : 18,
    !eq(mx, "M8") : 24,
    true:           2
  );

  defvar E32Cycles = !cond(
    !eq(mx, "M1") : 10,
    !eq(mx, "M2") : 10,
    !eq(mx, "M4") : 11,
    !eq(mx, "M8") : 11,
    true:           6
  );

  int latency = !if(!eq(sew, 64), E64Lat, !add(E64Lat, 4));
  int cycles = !if(!eq(sew, 64), 6, E32Cycles);
}

class SiFiveP600VFWidenUnorderedReduction<string mx> {
  int latency = !cond(
    !eq(mx, "M1") : 10,
    !eq(mx, "M2") : 18,
    !eq(mx, "M4") : 24,
    !eq(mx, "M8") : 30,
    true:           6
  );
}

class SiFiveP600VFOrderedReduction<string mx, int sew> {
  defvar Base = !if(!eq(sew, 64), 6, 10);
  int c = !cond(
    !eq(mx, "M1") : Base,
    !eq(mx, "M2") : !mul(Base, 2),
    !eq(mx, "M4") : !mul(Base, 4),
    !eq(mx, "M8") : !mul(Base, 8),
    true:           6
  );
}

class SiFiveP600VIReductionLatency<string mx> {
  int c = !cond(
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    // M1 and lower
    true:           2
  );
}

class SiFiveP600VIMinMaxReductionLatency<string mx, int sew> {
  // +-----+-----+-----+-----+----+
  // |     | E64 | E32 | E16 | E8 |
  // +-----+-----+-----+-----+----+
  // | MF8 |  X  |  X  |  X  |  4 |
  // +-----+-----+-----+-----+----+
  // | MF4 |  X  |  X  |  4  |  6 |
  // +-----+-----+-----+-----+----+
  // | MF2 |  X  |  4  |  6  |  8 |
  // +-----+-----+-----+-----+----+
  // | M1  |  4  |  6  |  8  | 10 |
  // +-----+-----+-----+-----+----+
  // | M2  |  6  |  8  |  10 | 12 |
  // +-----+-----+-----+-----+----+
  // | M4  |  8  |  10 |  12 | 14 |
  // +-----+-----+-----+-----+----+
  // | M8  |  10 |  12 |  14 | 16 |
  // +-----+-----+-----+-----+----+
  defvar BaseIndex = !cond(
    !eq(sew, 64): 0,
    !eq(sew, 32): 1,
    !eq(sew, 16): 2,
    !eq(sew, 8):  3
  );

  defvar Latencies = [4, 6, 8, 10, 12, 14, 16];

  int c = !cond(
    !eq(mx, "M1") : Latencies[BaseIndex],
    !eq(mx, "M2") : Latencies[!add(BaseIndex, 1)],
    !eq(mx, "M4") : Latencies[!add(BaseIndex, 2)],
    !eq(mx, "M8") : Latencies[!add(BaseIndex, 3)],
    // Fractional
    !eq(mx, "MF2"): Latencies[!sub(BaseIndex, 1)],
    !eq(mx, "MF4"): Latencies[!sub(BaseIndex, 2)],
    !eq(mx, "MF8"): Latencies[!sub(BaseIndex, 3)],
  );
}

class SiFiveP600VIMinMaxReductionCycles<string mx, int sew> {
  // +-----+-----+-----+-----+----+
  // |     | E64 | E32 | E16 | E8 |
  // +-----+-----+-----+-----+----+
  // | MF8 |  X  |  X  |  X  |  3 |
  // +-----+-----+-----+-----+----+
  // | MF4 |  X  |  X  |  3  |  5 |
  // +-----+-----+-----+-----+----+
  // | MF2 |  X  |  3  |  5  |  6 |
  // +-----+-----+-----+-----+----+
  // | M1  |  3  |  4  |  6  |  8 |
  // +-----+-----+-----+-----+----+
  // | M2  |  4  |  5  |  8  |  9 |
  // +-----+-----+-----+-----+----+
  // | M4  |  5  |  6  |  10 | 11 |
  // +-----+-----+-----+-----+----+
  // | M8  |  7  |  8  |  9  | 11 |
  // +-----+-----+-----+-----+----+
  defvar Index = !cond(
    !eq(sew, 64): 0,
    !eq(sew, 32): 1,
    !eq(sew, 16): 2,
    !eq(sew, 8):  3
  );

  defvar Cycles = [
    [0, 0, 0,  3],
    [0, 0, 3,  5],
    [0, 3, 5,  6],
    [3, 4, 6,  8],
    [4, 5, 8,  9],
    [5, 6, 10, 11],
    [7, 8, 9,  11]
  ];

  int c = !cond(
    !eq(mx, "MF8"): Cycles[0][Index],
    !eq(mx, "MF4"): Cycles[1][Index],
    !eq(mx, "MF2"): Cycles[2][Index],
    !eq(mx, "M1"):  Cycles[3][Index],
    !eq(mx, "M2"):  Cycles[4][Index],
    !eq(mx, "M4"):  Cycles[5][Index],
    !eq(mx, "M8"):  Cycles[6][Index],
  );
}

class SiFiveP600VSlide1<string mx> {
  int c = !cond(
    !eq(mx, "M2") : 3,
    !eq(mx, "M4") : 4,
    !eq(mx, "M8") : 8,
    // M1 and lower
    true:           2
  );
}

class SiFiveP600VSlideI<string mx> {
  int c = !cond(
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 6,
    !eq(mx, "M8") : 8,
    // M1 and lower
    true:           2
  );
}

class SiFiveP600VSlideXComplex<string mx, bit isUp = false> {
  int latency = !cond(
    !eq(mx, "M2") : 11,
    !eq(mx, "M4") : 14,
    !eq(mx, "M8") : 20
  );

  int cycles = !cond(
    !eq(mx, "M2") : !if(isUp, 10, 11),
    !eq(mx, "M4") : !if(isUp, 12, 14),
    !eq(mx, "M8") : !if(isUp, 16, 20)
  );
}

class SiFiveP600VPermutationComplex<string mx> {
  int c = !cond(
    !eq(mx, "M2") : 12,
    !eq(mx, "M4") : 16,
    !eq(mx, "M8") : 24
  );
}

class SiFiveP600VSHA2MSCycles<string mx, int sew> {
  int c = !cond(
    !eq(mx, "M2") : !if(!eq(sew, 32), 2, 3),
    !eq(mx, "M4") : !if(!eq(sew, 32), 4, 6),
    !eq(mx, "M8") : !if(!eq(sew, 32), 8, 12),
    true: 1
  );
}

// SiFiveP600 machine model for scheduling and other instruction cost heuristics.
def SiFiveP600Model : SchedMachineModel {
  let IssueWidth = 4;          // 4 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 192; // Max micro-ops that can be buffered.
  let LoadLatency = 4;         // Cycles for loads to access the cache.
  let MispredictPenalty = 9;   // Extra cycles for a mispredicted branch.
  let PostRAScheduler = true;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasVendorXSfvqmaccqoq, HasVendorXSfvqmaccdod, // SIFIVE
                             HasVendorXSfvfhbfmin]; // SIFIVE
  let CompleteModel = false;
}

let SchedModel = SiFiveP600Model in {
defm SiFiveP600 : MallardProcResources<num_iex = 4, num_fex = 2, num_vex = 2,
                                       shared_ldst_pipes = true>;

def SiFiveP600Branch : ProcResGroup<[SiFiveP600IEXQ2, SiFiveP600IEXQ3]>;
} // let SchedModel = SiFiveP600Model

def SiFiveP600IntProcResources : MallardIntProcResources {
  let Arith = SiFiveP600IntArith;
  let MulI2F = SiFiveP600IEXQ1;
  let Branch = SiFiveP600Branch;
  let SYS = SiFiveP600IEXQ0;
  let CMOV = SiFiveP600IEXQ0;
  let Div = SiFiveP600Div;
}

def SiFiveP600FloatProcResources : MallardFloatProcResources {
  let Arith = SiFiveP600FloatArith;
  let Div = [SiFiveP600FEXQ1, SiFiveP600FloatDiv];
  let F2I = SiFiveP600FEXQ0;
}

def SiFiveP600LSUProcResources : MallardLSUProcResources {
  let Load = SiFiveP600LDST;
  let Store = SiFiveP600LDST;
}

def SiFiveP600VecProcResources : TGVProcResources {
  let Arith = SiFiveP600VectorArith;
  let Load = SiFiveP600VLD;
  let Store = SiFiveP600VST;
  let Div = [SiFiveP600VEXQ1, SiFiveP600VDiv];
  let FloatDiv = [SiFiveP600VEXQ1, SiFiveP600VFloatDiv];
}

defvar SiFiveP600VectorMask = SiFiveP600VEXQ0;
defvar SiFiveP600VectorCrypto = SiFiveP600VEXQ0;

let SchedModel = SiFiveP600Model in {

defm : MallardBaseWriteRes<SiFiveP600IntProcResources,
                           SiFiveP600FloatProcResources,
                           SiFiveP600LSUProcResources>;

// Floating point arithmetics
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

// Floating point divisions
def : WriteRes<WriteFDiv16, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 4;
  let ReleaseAtCycles = [1, 4];
}
def : WriteRes<WriteFDiv32, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 6;
  let ReleaseAtCycles = [1, 6];
}
def : WriteRes<WriteFDiv64, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 11;
  let ReleaseAtCycles = [1, 11];
}

defm : TGVBaseWriteRes<SiFiveP600IntProcResources, SiFiveP600VecProcResources>;

// Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulLat = TGVGetLMulCycles<mx>.c;
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVExtV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpV",   [SiFiveP600VectorMask],  mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX",   [SiFiveP600VectorMask],  mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI",   [SiFiveP600VectorMask],  mx, IsWorstCase>;
  }
  let ReleaseAtCycles = [LMulLat] in {
    let Latency = 6 in {
      defm "" : LMULWriteResMX<"WriteVIMulV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMulX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMulAddV", [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMulAddX", [SiFiveP600VectorArith], mx, IsWorstCase>;
    }

    let Latency = !mul(2, SiFiveP600SequencerCycles<mx>.c) in {
      defm "" : LMULWriteResMX<"WriteVIALUV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIALUX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIALUI",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUV",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUX",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUI",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUMV",  [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUMX",  [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUMI",  [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMergeV",  [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMergeX",  [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMergeI",  [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMovX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMovI",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVShiftI",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVShiftV",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVShiftX",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMinMaxV", [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMinMaxX", [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMovV",    [SiFiveP600VectorArith], mx, IsWorstCase>;

      def P600WriteVICALUM_ # mx : SchedWriteRes<[SiFiveP600VectorMask]>;
    }
  }

  // Special cases for VMADC / VMSBC, which are mask-producing instruction and can only
  // execute on one of the VEX pipes, unlike its sibling instructions like VADC / VSBC.
  defvar P600VICALUMBaseSchedRW = [!cast<SchedWrite>("P600WriteVICALUM_" # mx),
                                   // VMADC / VMSBC always read the passthru operand.
                                   !cast<SchedRead>("ReadVPassthru_" # mx),
                                   !cast<SchedRead>("ReadVICALUV_" # mx)];
  // .vi / .vim
  def : InstRW<P600VICALUMBaseSchedRW,
               (instregex "^PseudoVM(ADC|SBC)_VI_" # mx # "$")>;
  def : InstRW<!listconcat(P600VICALUMBaseSchedRW, [!cast<SchedRead>("ReadVMask")]),
               (instregex "^PseudoVM(ADC|SBC)_VIM_" # mx # "$")>;
  // .vx / .vxm
  def : InstRW<!listconcat(P600VICALUMBaseSchedRW, [!cast<SchedRead>("ReadVICALUX_" # mx)]),
               (instregex "^PseudoVM(ADC|SBC)_VX_" # mx # "$")>;
  def : InstRW<!listconcat(P600VICALUMBaseSchedRW,
                           [!cast<SchedRead>("ReadVICALUX_" # mx), !cast<SchedRead>("ReadVMask")]),
               (instregex "^PseudoVM(ADC|SBC)_VXM_" # mx # "$")>;
  // .vv / .vvm
  def : InstRW<!listconcat(P600VICALUMBaseSchedRW, [!cast<SchedRead>("ReadVICALUV_" # mx)]),
               (instregex "^PseudoVM(ADC|SBC)_VV_" # mx # "$")>;
  def : InstRW<!listconcat(P600VICALUMBaseSchedRW,
                           [!cast<SchedRead>("ReadVICALUV_" # mx), !cast<SchedRead>("ReadVMask")]),
               (instregex "^PseudoVM(ADC|SBC)_VVM_" # mx # "$")>;
}
// Widening
foreach mx = SchedMxListW in {
  defvar LMulLat = TGVGetLMulCycles<mx>.c;
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIWALUV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [SiFiveP600VectorArith], mx, IsWorstCase>;

    // Special case for variants with widen operands.
    let ReleaseAtCycles = [!mul(LMulLat, 2)] in
    def P600WriteVIWALUWidenOp_ # mx : SchedWriteRes<[SiFiveP600VectorArith]>;
  }

  defvar P600VIWALUBaseSchedRW = [!cast<SchedWrite>("P600WriteVIWALUWidenOp_" # mx),
                                  // VWALU_WV_WX always merge read operand.
                                  !cast<SchedRead>("ReadVPassthru_" # mx),
                                  !cast<SchedRead>("ReadVIALUV_" # mx),
                                  !cast<SchedRead>("ReadVIALUV_" # mx)];
  def : InstRW<P600VIWALUBaseSchedRW,
               (instregex "^PseudoVW(ADD|SUB)[U]?_W(V|X)_" # mx # "(_TIED)?$")>;
  def : InstRW<!listconcat(P600VIWALUBaseSchedRW, [!cast<SchedRead>("ReadVMask")]),
               (instregex "^PseudoVW(ADD|SUB)[U]?_W(V|X)_" # mx # "_MASK(_TIED)?$")>;
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulLat = TGVGetLMulCycles<mx>.c;
    defvar IsWorstCase = TGVIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = !mul(2, SiFiveP600SequencerCycles<mx>.c), ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
    }
    let Latency = !if(!eq(mx, "M8"), 4, 3), ReleaseAtCycles = [!if(!eq(LMulLat, 1), 2, LMulLat)] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
  }
}
foreach mx = SchedMxList in {
  defvar LMulLat = TGVGetLMulCycles<mx>.c;
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = !if(!eq(mx, "M8"), 4, 3), ReleaseAtCycles = [!if(!eq(LMulLat, 1), 2, LMulLat)] in
  defm "" : LMULWriteResMX<"WriteVFCvtFToIV", [SiFiveP600VectorArith], mx, IsWorstCase>;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVFCmpV",  [SiFiveP600VectorMask], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCmpF",  [SiFiveP600VectorMask], mx, IsWorstCase>;
  }
  let Latency = !mul(2, SiFiveP600SequencerCycles<mx>.c),
      ReleaseAtCycles = [!if(!eq(LMulLat, 1), 2, LMulLat)] in {
    defm "" : LMULWriteResMX<"WriteVFClassV",  [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMergeV",  [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar LMulLat = TGVGetLMulCycles<mx>.c;
    defvar IsWorstCase = TGVIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;

      // Special case for variants with widen operands.
      let ReleaseAtCycles = [!mul(LMulLat, 2)] in
      def P600WriteVFWALUWidenOp_ # mx # _E # sew : SchedWriteRes<[SiFiveP600VectorArith]>;
    }

    defvar P600VFWALUBaseSchedRW = [!cast<SchedWrite>("P600WriteVFWALUWidenOp_" # mx # "_E" # sew),
                                    // VFWALU_WV_WF always merge read operand.
                                    !cast<SchedRead>("ReadVPassthru_" # mx # "_E" # sew),
                                    !cast<SchedRead>("ReadVFWALUV_" # mx # "_E" # sew)];
    def : InstRW<!listconcat(P600VFWALUBaseSchedRW, [!cast<SchedRead>("ReadVFWALUV_" # mx # "_E" # sew)]),
                 (instregex "^PseudoVFW(ADD|SUB)_WV_" # mx # "_E" # sew # "(_TIED)?$")>;
    def : InstRW<!listconcat(P600VFWALUBaseSchedRW, [!cast<SchedRead>("ReadVFWALUF_" # mx # "_E" # sew)]),
                 (instregex "^PseudoVFW(ADD|SUB)_WFPR" # sew # "_" # mx # "_E" # sew # "(_TIED)?$")>;

    def : InstRW<!listconcat(P600VFWALUBaseSchedRW, [!cast<SchedRead>("ReadVFWALUV_" # mx # "_E" # sew), !cast<SchedRead>("ReadVMask")]),
                 (instregex "^PseudoVFW(ADD|SUB)_WV_" # mx # "_E" # sew # "_MASK(_TIED)?$")>;
    def : InstRW<!listconcat(P600VFWALUBaseSchedRW, [!cast<SchedRead>("ReadVFWALUF_" # mx # "_E" # sew), !cast<SchedRead>("ReadVMask")]),
                 (instregex "^PseudoVFW(ADD|SUB)_WFPR" # sew # "_" # mx # "_E" # sew # "_MASK(_TIED)?$")>;
  }
}
// Narrowing
foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar LMulLat = TGVGetLMulCycles<mx>.c;
    defvar IsWorstCase = TGVIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = 3, ReleaseAtCycles = [!if(!eq(LMulLat, 1), 2, LMulLat)] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
    }
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = TGVGetLMulCycles<mx>.c;
    defvar IsWorstCase = TGVIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    let ReleaseAtCycles = [LMulLat] in {
      let Latency = SiFiveP600VIReductionLatency<mx>.c in
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From", [SiFiveP600VEXQ1],
                                     mx, sew, IsWorstCase>;

      let Latency = SiFiveP600VIMinMaxReductionLatency<mx, sew>.c,
          ReleaseAtCycles = [SiFiveP600VIMinMaxReductionCycles<mx, sew>.c] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [SiFiveP600VEXQ1],
                                     mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar LMulLat = TGVGetLMulCycles<mx>.c;
    defvar IsWorstCase = TGVIsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;
    let Latency = SiFiveP600VIReductionLatency<mx>.c, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [SiFiveP600VEXQ1],
                                     mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar LMulLat = TGVGetLMulCycles<mx>.c;
    defvar IsWorstCase = TGVIsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    let Latency = SiFiveP600VFMinMaxReduction<mx, sew>.latency,
        ReleaseAtCycles = [SiFiveP600VFMinMaxReduction<mx, sew>.cycles] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From",
                                   [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;

    let Latency = SiFiveP600VFUnorderedReduction<mx, sew>.latency,
        ReleaseAtCycles = [SiFiveP600VFUnorderedReduction<mx, sew>.cycles] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [SiFiveP600VEXQ1],
                                   mx, sew, IsWorstCase>;

    let Latency = SiFiveP600VFOrderedReduction<mx, sew>.c,
        ReleaseAtCycles = [SiFiveP600VFOrderedReduction<mx, sew>.c] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [SiFiveP600VEXQ1],
                                   mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListFWRed in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar LMulLat = TGVGetLMulCycles<mx>.c;
    defvar IsWorstCase = TGVIsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;
    let Latency = SiFiveP600VFWidenUnorderedReduction<mx>.latency,
        ReleaseAtCycles = [6] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From",  [SiFiveP600VEXQ1],
                                   mx, sew, IsWorstCase>;

    let Latency = SiFiveP600VFOrderedReduction<mx, sew>.c,
        ReleaseAtCycles = [SiFiveP600VFOrderedReduction<mx, sew>.c] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SiFiveP600VEXQ1],
                                   mx, sew, IsWorstCase>;
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2 in {
    defm "" : LMULWriteResMX<"WriteVMALUV", [SiFiveP600VectorMask], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMSFSV", [SiFiveP600VectorMask], mx, IsWorstCase>;

    let ReleaseAtCycles = [2] in {
      defm "" : LMULWriteResMX<"WriteVMPopV", [SiFiveP600VectorMask], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVMFFSV", [SiFiveP600VectorMask], mx, IsWorstCase>;
    }
  }
  defvar LMulLat = TGVGetLMulCycles<mx>.c;
  let ReleaseAtCycles = [LMulLat] in {
    let Latency = 2 in
    defm "" : LMULWriteResMX<"WriteVIotaV", [SiFiveP600VectorMask], mx, IsWorstCase>;

    // vid.v isn't executed by the mask unit on TGV.
    let Latency = !if(!eq(mx, "M8"), 4, !if(!eq(mx, "M4"), 2, 1)) in
    defm "" : LMULWriteResMX<"WriteVIdxV", [SiFiveP600VectorArith], mx, IsWorstCase>;
  }
}

// 16. Vector Permutation Instructions
// Simple Slide
foreach mx = SchedMxList in {
  defvar LMulLat = TGVGetLMulCycles<mx>.c;
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  let ReleaseAtCycles = [LMulLat] in {
    let Latency = SiFiveP600VSlideI<mx>.c in
    defm "" : LMULWriteResMX<"WriteVSlideI",  [SiFiveP600VEXQ0], mx, IsWorstCase>;

    let Latency = SiFiveP600VSlide1<mx>.c in {
      defm "" : LMULWriteResMX<"WriteVISlide1X", [SiFiveP600VEXQ0], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVFSlide1F", [SiFiveP600VEXQ0], mx, IsWorstCase>;
    }
  }
}
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVSlideUpX",   [SiFiveP600VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SiFiveP600VEXQ0], mx, IsWorstCase>;
  }
}

// Complex Slide
foreach mx = ["M8", "M4", "M2"] in {
  defvar LMulLat = TGVGetLMulCycles<mx>.c;
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = SiFiveP600VSlideXComplex<mx>.latency in {
    let ReleaseAtCycles = [SiFiveP600VSlideXComplex<mx, /*isUp=*/true>.cycles] in
    defm "" : LMULWriteResMX<"WriteVSlideUpX",   [SiFiveP600VEXQ1], mx, IsWorstCase>;
    let ReleaseAtCycles = [SiFiveP600VSlideXComplex<mx, /*isUp=*/false>.cycles] in
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SiFiveP600VEXQ1], mx, IsWorstCase>;
  }
}

let Latency = 2, ReleaseAtCycles = [2] in {
  def : WriteRes<WriteVMovXS, [SiFiveP600VectorArith]>;
  def : WriteRes<WriteVMovSX, [SiFiveP600VectorArith]>;
  def : WriteRes<WriteVMovFS, [SiFiveP600VectorArith]>;
  def : WriteRes<WriteVMovSF, [SiFiveP600VectorArith]>;
}

// Simple Gather and Compress
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP600VEXQ1], mx, IsWorstCase>;
  }
}

foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 3, ReleaseAtCycles = [1] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
    }
  }
}

// Complex Gather and Compress
foreach mx = ["M2", "M4", "M8"] in {
  defvar LMulLat = TGVGetLMulCycles<mx>.c;
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 6, ReleaseAtCycles = [SiFiveP600VPermutationComplex<mx>.c] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP600VEXQ1], mx, IsWorstCase>;
  }
}

foreach mx = ["M2", "M4", "M8"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = TGVGetLMulCycles<mx>.c;
    defvar IsWorstCase = TGVIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = 6 in {
      let ReleaseAtCycles = [SiFiveP600VPermutationComplex<mx>.c] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
      }

      let ReleaseAtCycles = [!add(SiFiveP600VPermutationComplex<mx>.c, 1)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
    }
  }
}

// Simple Vrgather.vi
foreach mx = SchedMxList in {
  defvar LMulLat = TGVGetLMulCycles<mx>.c;
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [SiFiveP600VEXQ1], mx, IsWorstCase>;
  }
}

// Vector Crypto
foreach mx = SchedMxList in {
  defvar LMulLat = TGVGetLMulCycles<mx>.c;
  defvar IsWorstCase = TGVIsWorstCaseMX<mx, SchedMxList>.c;
  // Zvbb
  let ReleaseAtCycles = [LMulLat] in {
    let Latency = 2 in {
      // FIXME: Exegesis was not able to measure the latency of these instructions.
      // We probably should update them at some point.
      defm "" : LMULWriteResMX<"WriteVCPOPV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVWSLLV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVWSLLX",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVWSLLI",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    }

    let Latency = SiFiveP600VCryptoLatency<mx>.c in {
      defm "" : LMULWriteResMX<"WriteVBREVV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVCLZV",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVCTZV",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;

      def P600WriteVANDN_ # mx : SchedWriteRes<[SiFiveP600VectorCrypto]>;
    }
  }

  // Special case for VANDN.
  defvar P600VANDNBaseSchedRW = [!cast<SchedWrite>("P600WriteVANDN_" # mx),
                                 // VANDN always merge read operand.
                                 !cast<SchedRead>("ReadVPassthru_" # mx),
                                 !cast<SchedRead>("ReadVIALUV_" # mx),
                                 !cast<SchedRead>("ReadVIALUV_" # mx)];
  def : InstRW<P600VANDNBaseSchedRW,
               (instregex "^PseudoVANDN_V(V|X)_" # mx # "$")>;
  def : InstRW<!listconcat(P600VANDNBaseSchedRW, [!cast<SchedRead>("ReadVMask")]),
               (instregex "^PseudoVANDN_V(V|X)_" # mx # "_MASK$")>;

  // Zvbc
  let Latency = SiFiveP600VCryptoLatency<mx>.c, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVCLMULV", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCLMULX", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
  // Zvkb
  // VANDN uses WriteVIALU[V|X|I]
  let Latency = SiFiveP600VCryptoLatency<mx>.c, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVBREV8V",  [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVREV8V",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotV",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotX",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotI",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
  // Zvkg
  let Latency = SiFiveP600VCryptoLatency<mx>.c, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVGHSHV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVGMULV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
  // ZvknhaOrZvknhb
  // FIXME: The latency is probably wrong.
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSHA2CHV", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSHA2CLV", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defvar ZvknhSEWs = !listremove(SchedSEWSet<mx>.val, [8, 16]);
    // Largest SEW is the last element, assuming SchedSEWSet is sorted in ascending
    // order.
    defvar LargestZvknhSEW = !foldl(!head(ZvknhSEWs), ZvknhSEWs, last, curr, curr);
    foreach sew = ZvknhSEWs in {
      // The worst case for Zvknh[ab] is designated to the largest SEW and LMUL.
      defvar IsWorstCaseVSHA2MSV = !and(IsWorstCase, !eq(sew, LargestZvknhSEW));
      let ReleaseAtCycles = [SiFiveP600VSHA2MSCycles<mx, sew>.c] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVSHA2MSV", [SiFiveP600VectorCrypto], mx, sew,
                                     IsWorstCaseVSHA2MSV>;
    }
  }
  // Zvkned
  let Latency = 2 in {
    let ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULWriteResMX<"WriteVAESMVV",  [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVAESKF1V", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVAESKF2V", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    }

    let ReleaseAtCycles = [!if(!lt(LMulLat, 2), LMulLat, !div(LMulLat, 2))] in
    defm "" : LMULWriteResMX<"WriteVAESZV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
  // Zvksed
  let Latency = 3, ReleaseAtCycles = [SiFiveP600VSM3CCycles<mx>.c] in
  defm "" : LMULWriteResMX<"WriteVSM3CV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVSM3MEV",  [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSM4KV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSM4RV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
}

//===----------------------------------------------------------------------===//
// Bypass and advance

defm : MallardBaseReadAdvances;

defm : TGVBaseReadAdvances;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedQ;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfaWithQ;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedXsf;
} // let SchedModel = SiFiveP600Model
