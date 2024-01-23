//==- RISCVSchedSiFive7N.td - SiFive7N Scheduling Definitions --*- tablegen -*-=//
//
// SiFive7 Schedule model for Narwhal Release.
// Scalar FMA latency change: 5/7 -> 4/4 (single/double FMA latency)
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// SiFive7N machine model for scheduling and other instruction cost heuristics.
def SiFive7NModel : SchedMachineModel {
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
}

let SchedModel = SiFive7NModel in {

let BufferSize = 0 in
defm SiFive7N : BulletProcResources;
def SiFive7NPipeAB : ProcResGroup<[SiFive7NPipeA, SiFive7NPipeB]>;

// FIXME: On SiFive7N, VLEN is 512. Although a user can request the compiler
// to use a different VLEN, this model will not make scheduling decisions
// based on the user specified VLEN.
defvar VLEN = 512;

// Define WriteRes records that are the same across all Bullet derived
// SchedModels. WriteRes records that are not included in that multiclass should
// be defined below.
defm "" : BulletWriteResBase<VLEN, SiFive7NPipeA, SiFive7NPipeB, SiFive7NPipeAB,
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
  defm SiFive7N : LMULWriteResMXVariant<"WriteVLDS8",  VLDSX0Pred, [SiFive7NVCQ, SiFive7NVL],
                                        4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                        [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
  let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
    defm "" : LMULWriteResMX<"WriteVLDUX8", [SiFive7NVCQ, SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX8", [SiFive7NVCQ, SiFive7NVL], mx, IsWorstCase>;
  }
  let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
    defm "" : LMULWriteResMX<"WriteVSTS8",  [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX8", [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX8", [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
  }
}
foreach mx = ["MF4", "MF2", "M1", "M2", "M4", "M8"] in {
  defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
  defvar Cycles = BulletGetCyclesOnePerElement<mx, 16, VLEN>.c;
  defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
  defm SiFive7N : LMULWriteResMXVariant<"WriteVLDS16",  VLDSX0Pred, [SiFive7NVCQ, SiFive7NVL],
                                        4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                        [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
  let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
    defm "" : LMULWriteResMX<"WriteVLDUX16", [SiFive7NVCQ, SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX16", [SiFive7NVCQ, SiFive7NVL], mx, IsWorstCase>;
  }
  let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
    defm "" : LMULWriteResMX<"WriteVSTS16",  [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX16", [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX16", [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
  }
}
foreach mx = ["MF2", "M1", "M2", "M4", "M8"] in {
  defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
  defvar Cycles = BulletGetCyclesOnePerElement<mx, 32, VLEN>.c;
  defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
  defm SiFive7N : LMULWriteResMXVariant<"WriteVLDS32",  VLDSX0Pred, [SiFive7NVCQ, SiFive7NVL],
                                        4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                        [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
  let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
    defm "" : LMULWriteResMX<"WriteVLDUX32", [SiFive7NVCQ, SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX32", [SiFive7NVCQ, SiFive7NVL], mx, IsWorstCase>;
  }
  let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
    defm "" : LMULWriteResMX<"WriteVSTS32",  [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX32", [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX32", [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
  }
}
foreach mx = ["M1", "M2", "M4", "M8"] in {
  defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
  defvar Cycles = BulletGetCyclesOnePerElement<mx, 64, VLEN>.c;
  defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
  defm SiFive7N : LMULWriteResMXVariant<"WriteVLDS64",  VLDSX0Pred, [SiFive7NVCQ, SiFive7NVL],
                                        4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                        [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
  let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
    defm "" : LMULWriteResMX<"WriteVLDUX64", [SiFive7NVCQ, SiFive7NVL], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX64", [SiFive7NVCQ, SiFive7NVL], mx, IsWorstCase>;
  }
  let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
    defm "" : LMULWriteResMX<"WriteVSTS64",  [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX64", [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX64", [SiFive7NVCQ, SiFive7NVS], mx, IsWorstCase>;
  }
}

// 16. Vector Permutation Instructions
// The Orca release makes some changes compared to the Narwhal release. Instead
// of splitting SiFive7N to SiFive7N and SiFiveO, use SchedVariants to model
// the minor changes.
def SiFiveOPred : SchedPredicate<[{SchedModel->getSubtargetInfo()->hasFeature(
                      RISCV::TuneOptimizedVRGatherVCompress)}]>;
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

// FIXME: This could be better modeled by looking at the regclasses of the operands.
def : InstRW<[WriteIALU, ReadIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance
def SiFive7NReadMemBaseOffset0 : SchedReadAdvance<1, [WriteLDB, WriteLDH, WriteLDW, WriteLDD]>;

// Bypass the AG stage of a dependendent memory op after a load if the offset is
// zero.
def SiFive7NReadMemBase : SchedReadVariant<[
  SchedVar<MemBaseOffset0Pred, [SiFive7NReadMemBaseOffset0]>,
  SchedVar<NoSchedPred, [NoReadAdvance]>
]>;
def : SchedAlias<ReadMemBase, SiFive7NReadMemBase>;

defm "" : BulletReadAdvance;
//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
}
