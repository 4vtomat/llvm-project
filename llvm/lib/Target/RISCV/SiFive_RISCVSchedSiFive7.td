//==- RISCVSchedSiFive7.td - SiFive7 Scheduling Definitions --*- tablegen -*-=//
//
// SiFive7 Schedule model for Mongoose Release.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

multiclass SiFive7SchedResources<int VLEN> {
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
class SiFive7SchedMachineModel<int vlen> : SchedMachineModel {
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
  int VLEN = vlen;
}

// SiFive7Models with different VLEN
def SiFive7VLEN256Model : SiFive7SchedMachineModel<256>;
def SiFive7VLEN512Model : SiFive7SchedMachineModel<512>;
def SiFive7VLEN1024Model : SiFive7SchedMachineModel<1024>;
// Some processors do not have vector but reuse the scheduler model that has vector.
// Create an alias that does not have VLEN in the name.
defvar SiFive7Model = SiFive7VLEN512Model;
// Define sched resources for each model.
foreach model = [SiFive7VLEN256Model, SiFive7VLEN512Model, SiFive7VLEN1024Model] in {
  let SchedModel = model in
  defm "VLEN" # model.VLEN : SiFive7SchedResources<model.VLEN>;
}
