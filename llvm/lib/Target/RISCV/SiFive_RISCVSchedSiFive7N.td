//==- RISCVSchedSiFive7N.td - SiFive7N Scheduling Definitions --*- tablegen -*-=//
//
// SiFive7 Schedule model for Narwhal Release.
// Scalar FMA latency change: 5/7 -> 4/4 (single/double FMA latency)
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

multiclass SiFive7NSchedResources<int VLEN> {
  let BufferSize = 0 in
  defm SiFive7N : BulletProcResources;
  def SiFive7NPipeAB : ProcResGroup<[
    !cast<ProcResource>(NAME #SiFive7NPipeA),
    !cast<ProcResource>(NAME #SiFive7NPipeB)
  ]>;

  // Pull out defs from BulletProcResources so we can refer to them by name.
  defvar SiFive7NPipeA = !cast<ProcResource>(NAME # SiFive7NPipeA);
  defvar SiFive7NPipeB = !cast<ProcResource>(NAME # SiFive7NPipeB);
  defvar SiFive7NPipeAB = !cast<ProcResGroup>(NAME # SiFive7NPipeAB);
  defvar SiFive7NMem = !cast<ProcResource>(NAME # SiFive7NMem);
  defvar SiFive7NIDiv = !cast<ProcResource>(NAME # SiFive7NIDiv);
  defvar SiFive7NFDiv = !cast<ProcResource>(NAME # SiFive7NFDiv);
  defvar SiFive7NVA = !cast<ProcResource>(NAME # SiFive7NVA);
  defvar SiFive7NVL = !cast<ProcResource>(NAME # SiFive7NVL);
  defvar SiFive7NVS = !cast<ProcResource>(NAME # SiFive7NVS);
  defvar SiFive7NVCQ = !cast<ProcResource>(NAME # SiFive7NVCQ);

  // Define WriteRes records that are the same across all Bullet derived
  // SchedModels. WriteRes records that are not included in that multiclass should
  // be defined below.
  defm "SiFive7N" : BulletWriteResBase<VLEN, SiFive7NPipeA, SiFive7NPipeB, SiFive7NPipeAB,
                               SiFive7NMem, SiFive7NIDiv, SiFive7NFDiv,
                               SiFive7NVA, SiFive7NVL, SiFive7NVS, SiFive7NVCQ>;

  // Half precision.
  let Latency = 4 in {
  def : WriteRes<WriteFAdd16, [SiFive7NPipeB]>;
  def : WriteRes<WriteFMul16, [SiFive7NPipeB]>;
  def : WriteRes<WriteFMA16, [SiFive7NPipeB]>;
  }

  // Single precision.
  let Latency = 4 in {
  def : WriteRes<WriteFAdd32, [SiFive7NPipeB]>;
  def : WriteRes<WriteFMul32, [SiFive7NPipeB]>;
  def : WriteRes<WriteFMA32, [SiFive7NPipeB]>;
  }

  // Double precision
  let Latency = 4 in {
  def : WriteRes<WriteFAdd64, [SiFive7NPipeB]>;
  def : WriteRes<WriteFMul64, [SiFive7NPipeB]>;
  def : WriteRes<WriteFMA64, [SiFive7NPipeB]>;
  }

  // 16. Vector Permutation Instructions
  // The Orca release makes some changes compared to the Narwhal release. Instead
  // of splitting SiFive7N to SiFive7N and SiFiveO, use SchedVariants to model
  // the minor changes.
  def SiFiveOPred : SchedPredicate<[{SchedModel->getSubtargetInfo()->hasFeature(
                        RISCV::TuneOptimizedVRGatherVCompress)}]>;
  defvar SiFiveOPred = !cast<SchedPredicate>(NAME # SiFiveOPred);
  foreach mx = SchedMxList in {
    foreach sew = SchedSEWSet<mx>.val in {
      defvar Cycles = BulletGetCyclesOnePerElement<mx, sew, VLEN>.c;
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

      // Define defaults
      let Latency = !add(Cycles, 3), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFive7NVCQ, SiFive7NVA], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFive7NVCQ, SiFive7NVA], mx, sew, IsWorstCase>;
      }

      if !or(!or(!eq(mx, "MF2"), !eq(mx, "MF4")), !eq(mx, "MF8")) then {
        assert !not(IsWorstCase),
          "On SiFive7N MF2, MF4, and MF8 never lead to WorstCase behavior";

        // Define Narwhal resources
        defvar VRGatherVVLat = !add(7, !div(256, sew)); // 7 + DLEN / SEW
        def "WriteVRGatherVV_" # mx # "_E" # sew # "_N" : SchedWriteRes<[SiFive7NVCQ, SiFive7NVA]> {
          let Latency = VRGatherVVLat;
          let AcquireAtCycles = [0, 1];
          let ReleaseAtCycles = [1, !add(1, !add(4, Cycles))];
        }
        def "WriteVCompressV_" # mx # "_E" # sew # "_N" : SchedWriteRes<[SiFive7NVCQ, SiFive7NVA]> {
          let Latency = !add(Cycles, 7);
          let AcquireAtCycles = [0, 1];
          let ReleaseAtCycles = [1, !add(1, !add(7, Cycles))];
        }
        // Define Orca resources
        def "WriteVRGatherVV_" # mx # "_E" # sew # "_O" : SchedWriteRes<[SiFive7NVCQ, SiFive7NVA]> {
          let Latency = 4;
          let AcquireAtCycles = [0, 1];
          let ReleaseAtCycles = [1, !add(1, 1)];
        }
        def "WriteVCompressV_" # mx # "_E" # sew # "_O" : SchedWriteRes<[SiFive7NVCQ, SiFive7NVA]> {
          let Latency = 4;
          let AcquireAtCycles = [0, 1];
          let ReleaseAtCycles = [1, !add(1, 1)];
        }

        def VRGatherPredVar_ # mx # "_E" # sew : SchedVar<SiFiveOPred, [
          !cast<SchedWriteRes>(NAME # "WriteVRGatherVV_" #mx #"_E" #sew #"_O")
        ]>;
        def VRGatherNoPredVar_ # mx # "_E" # sew: SchedVar<NoSchedPred, [
          !cast<SchedWriteRes>(NAME # "WriteVRGatherVV_" #mx #"_E" #sew #"_N")
        ]>;
        defvar VRGatherPredVar = !cast<SchedVar>(NAME # VRGatherPredVar_ # mx # "_E" # sew);
        defvar VRGatherNoPredVar = !cast<SchedVar>(NAME # VRGatherNoPredVar_ # mx # "_E" # sew);

        def VCompressPredVar_ # mx # "_E" # sew : SchedVar<SiFiveOPred, [
          !cast<SchedWriteRes>(NAME #"WriteVCompressV_" #mx #"_E" #sew #"_O")
        ]>;
        def VCompressNoPredVar_ # mx # "_E" # sew : SchedVar<NoSchedPred, [
          !cast<SchedWriteRes>(NAME # "WriteVCompressV_" #mx #"_E" #sew #"_N")
        ]>;
        defvar VCompressPredVar = !cast<SchedVar>(NAME # VCompressPredVar_ # mx # "_E" # sew);
        defvar VCompressNoPredVar = !cast<SchedVar>(NAME # VCompressNoPredVar_ # mx # "_E" # sew);

        // Define variants to pick Narwhal or Orca resources
        def "WriteVRGatherVV_" #mx #"_E" #sew #"Variant"
            : SchedWriteVariant<[VRGatherPredVar, VRGatherNoPredVar]>;
        def "WriteVCompressV_" #mx #"_E" #sew #"Variant"
            : SchedWriteVariant<[VCompressPredVar, VCompressNoPredVar]>;

        // Tie variant choice to the correct instructions. This overrides default
        // behavior for these instructions that was defined above.
        def : SchedAlias<
          !cast<SchedReadWrite>("WriteVRGatherVV_" # mx # "_E" # sew),
          !cast<SchedReadWrite>(NAME # "WriteVRGatherVV_" # mx # "_E" # sew # "Variant")>;
        def : SchedAlias<
          !cast<SchedReadWrite>("WriteVCompressV_" # mx # "_E" # sew),
          !cast<SchedReadWrite>(NAME # "WriteVCompressV_" # mx # "_E" # sew # "Variant")>;
        }
    }
  }

  //===----------------------------------------------------------------------===//
  // Bypass and advance

  defm "SiFive7N" : BulletReadAdvance;
  //===----------------------------------------------------------------------===//
  // Unsupported extensions
  defm : UnsupportedSchedZabha;
  defm : UnsupportedSchedZbc;
  defm : UnsupportedSchedZbkb;
  defm : UnsupportedSchedZbkx;
  defm : UnsupportedSchedZfa;
}

// SiFive7N machine model for scheduling and other instruction cost heuristics.
class SiFive7NSchedMachineModel<int vlen> : SchedMachineModel {
  let MicroOpBufferSize = 0; // Explicitly set to zero since SiFive7N is in-order.
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

def SiFive7NVLEN256Model : SiFive7NSchedMachineModel<256>;
def SiFive7NVLEN512Model : SiFive7NSchedMachineModel<512>;
// Some processors do not have vector but reuse the scheduler model that has vector.
// Create an alias that does not have VLEN in the name.
defvar SiFive7NModel = SiFive7NVLEN512Model;
// Define sched resources for each model.
foreach model = [SiFive7NVLEN256Model, SiFive7NVLEN512Model] in {
  let SchedModel = model in
  defm "VLEN" # model.VLEN : SiFive7NSchedResources<model.VLEN>;
}
