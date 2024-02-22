//==- RISCVSchedSiFive7.td - SiFive7 Scheduling Definitions --*- tablegen -*-=//
//
// SiFive7 Schedule model for Mongoose Release.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

multiclass SiFive7SchedResources {
  let BufferSize = 0 in
  defm SiFive7 : BulletProcResources;
  def SiFive7PipeAB : ProcResGroup<[
    !cast<ProcResource>(NAME #SiFive7PipeA),
    !cast<ProcResource>(NAME #SiFive7PipeB)
  ]>;

  // Pull out defs from BulletProcResources so we can refer to them by name.
  defvar SiFive7PipeA = !cast<ProcResource>(NAME # SiFive7PipeA);
  defvar SiFive7PipeB = !cast<ProcResource>(NAME # SiFive7PipeB);
  defvar SiFive7PipeAB = !cast<ProcResGroup>(NAME # SiFive7PipeAB);
  defvar SiFive7Mem = !cast<ProcResource>(NAME # SiFive7Mem);
  defvar SiFive7IDiv = !cast<ProcResource>(NAME # SiFive7IDiv);
  defvar SiFive7FDiv = !cast<ProcResource>(NAME # SiFive7FDiv);
  defvar SiFive7VA = !cast<ProcResource>(NAME # SiFive7VA);
  defvar SiFive7VL = !cast<ProcResource>(NAME # SiFive7VL);
  defvar SiFive7VS = !cast<ProcResource>(NAME # SiFive7VS);
  defvar SiFive7VCQ = !cast<ProcResource>(NAME # SiFive7VCQ);

  // FIXME: On SiFive7, VLEN is 512. Although a user can request the compiler
  // to use a different VLEN, this model will not make scheduling decisions
  // based on the user specified VLEN.
  defvar VLEN = 512;

  // Define WriteRes records that are the same across all Bullet derived
  // SchedModels. WriteRes records that are not included in that multiclass should
  // be defined below.
  defm "SiFive7" : BulletWriteResBase<VLEN, SiFive7PipeA, SiFive7PipeB, SiFive7PipeAB,
                                      SiFive7Mem, SiFive7IDiv, SiFive7FDiv,
                                      SiFive7VA, SiFive7VL, SiFive7VS, SiFive7VCQ>;

  // Half precision.
  let Latency = 5 in {
  def : WriteRes<WriteFAdd16, [SiFive7PipeB]>;
  def : WriteRes<WriteFMul16, [SiFive7PipeB]>;
  def : WriteRes<WriteFMA16, [SiFive7PipeB]>;
  }

  // Single precision.
  let Latency = 5 in {
  def : WriteRes<WriteFAdd32, [SiFive7PipeB]>;
  def : WriteRes<WriteFMul32, [SiFive7PipeB]>;
  def : WriteRes<WriteFMA32, [SiFive7PipeB]>;
  }

  // Double precision
  let Latency = 7 in {
  def : WriteRes<WriteFAdd64, [SiFive7PipeB]>;
  def : WriteRes<WriteFMul64, [SiFive7PipeB]>;
  def : WriteRes<WriteFMA64, [SiFive7PipeB]>;
  }

  // Strided loads and stores operate at one element per cycle and should be
  // scheduled accordingly. Indexed loads and stores operate at one element per
  // cycle, and they stall the machine until all addresses have been generated,
  // so they cannot be scheduled. Indexed and strided loads and stores have LMUL
  // specific suffixes, but since SEW is already encoded in the name of the
  // resource, we do not need to use LMULSEWXXX constructors. However, we do
  // use the SEW from the name to determine the number of Cycles.

  foreach mx = SchedMxList in {
    defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
    defvar Cycles = BulletGetCyclesOnePerElement<mx, 8, VLEN>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    defm SiFive7 : LMULWriteResMXVariant<"WriteVLDS8",  VLDSX0Pred, [SiFive7VCQ, SiFive7VL],
                                         4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                         [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
    let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVLDUX8", [SiFive7VCQ, SiFive7VL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX8", [SiFive7VCQ, SiFive7VL], mx, IsWorstCase>;
    }
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSTS8",  [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTUX8", [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX8", [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
    }
  }
  foreach mx = ["MF4", "MF2", "M1", "M2", "M4", "M8"] in {
    defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
    defvar Cycles = BulletGetCyclesOnePerElement<mx, 16, VLEN>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    defm SiFive7 : LMULWriteResMXVariant<"WriteVLDS16",  VLDSX0Pred, [SiFive7VCQ, SiFive7VL],
                                         4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                         [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
    let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVLDUX16", [SiFive7VCQ, SiFive7VL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX16", [SiFive7VCQ, SiFive7VL], mx, IsWorstCase>;
    }
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSTS16",  [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTUX16", [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX16", [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
    }
  }
  foreach mx = ["MF2", "M1", "M2", "M4", "M8"] in {
    defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
    defvar Cycles = BulletGetCyclesOnePerElement<mx, 32, VLEN>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    defm SiFive7 : LMULWriteResMXVariant<"WriteVLDS32",  VLDSX0Pred, [SiFive7VCQ, SiFive7VL],
                                         4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                         [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
    let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVLDUX32", [SiFive7VCQ, SiFive7VL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX32", [SiFive7VCQ, SiFive7VL], mx, IsWorstCase>;
    }
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSTS32",  [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTUX32", [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX32", [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
    }
  }
  foreach mx = ["M1", "M2", "M4", "M8"] in {
    defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
    defvar Cycles = BulletGetCyclesOnePerElement<mx, 64, VLEN>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    defm SiFive7 : LMULWriteResMXVariant<"WriteVLDS64",  VLDSX0Pred, [SiFive7VCQ, SiFive7VL],
                                         4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                         [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
    let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVLDUX64", [SiFive7VCQ, SiFive7VL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX64", [SiFive7VCQ, SiFive7VL], mx, IsWorstCase>;
    }
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSTS64",  [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTUX64", [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX64", [SiFive7VCQ, SiFive7VS], mx, IsWorstCase>;
    }
  }

  // 16. Vector Permutation Instructions
  foreach mx = SchedMxList in {
    foreach sew = SchedSEWSet<mx>.val in {
      defvar Cycles = BulletGetCyclesOnePerElement<mx, sew, VLEN>.c;
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
      defvar VRGatherVVLat = !add(7, !div(256, sew)); // 7 + DLEN / SEW
      let Latency = VRGatherVVLat, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, !add(4, Cycles))] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFive7VCQ, SiFive7VA], mx, sew, IsWorstCase>;
      let Latency = !add(Cycles, 7), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, !add(7, Cycles))] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFive7VCQ, SiFive7VA], mx, sew, IsWorstCase>;
    }
  }

  //===----------------------------------------------------------------------===//
  // Bypass and advance
  def SiFive7ReadMemBaseOffset0 : SchedReadAdvance<1, [WriteLDB, WriteLDH, WriteLDW, WriteLDD]>;
  defvar SiFive7ReadMemBaseOffset0 =
      !cast<SchedReadAdvance>(NAME #SiFive7ReadMemBaseOffset0);

  def SiFive7MemBaseOffset0PredVar
      : SchedVar<MemBaseOffset0Pred, [SiFive7ReadMemBaseOffset0]>;
  def SiFive7MemBaseOffset0NoPredVar
      : SchedVar<NoSchedPred, [NoReadAdvance]>;
  defvar MemBaseOffset0PredVar = !cast<SchedVar>(NAME #SiFive7MemBaseOffset0PredVar);
  defvar MemBaseOffset0NoPredVar = !cast<SchedVar>(NAME #SiFive7MemBaseOffset0NoPredVar);

  // Bypass the AG stage of a dependendent memory op after a load if the offset is
  // zero.
  def SiFive7ReadMemBase
      : SchedReadVariant<[MemBaseOffset0PredVar, MemBaseOffset0NoPredVar]>;

  defvar SiFive7ReadMemBase = !cast<SchedReadVariant>(NAME # SiFive7ReadMemBase);
  def : SchedAlias<ReadMemBase, SiFive7ReadMemBase>;

  defm "SiFive7" : BulletReadAdvance;
  //===----------------------------------------------------------------------===//
  // Unsupported extensions
  defm : UnsupportedSchedZabha;
  defm : UnsupportedSchedZbc;
  defm : UnsupportedSchedZbkb;
  defm : UnsupportedSchedZbkx;
  defm : UnsupportedSchedZfa;
}

// SiFive7 machine model for scheduling and other instruction cost heuristics.
def SiFive7Model : SchedMachineModel {
  let MicroOpBufferSize = 0; // Explicitly set to zero since SiFive7 is in-order.
  let IssueWidth = 2;        // 2 micro-ops are dispatched per cycle.
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let CompleteModel = 0;
  let PostRAScheduler = true;
  let EnableIntervals = true;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasVendorXSfvqmaccqoq, HasVendorXSfvqmaccdod, // SIFIVE
                             HasVendorXSfvfhbfmin]; // SIFIVE
}

let SchedModel = SiFive7Model in
defm "" : SiFive7SchedResources;

