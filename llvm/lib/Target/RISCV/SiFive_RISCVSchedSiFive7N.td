//==- RISCVSchedSiFive7N.td - SiFive7N Scheduling Definitions --*- tablegen -*-=//
//
// SiFive7 Schedule model for Narwhal Release.
// Scalar FMA latency change: 5/7 -> 4/4 (single/double FMA latency)
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

multiclass SiFive7NSchedResources<int VLEN, bit ExtraVALU = false,
                                  bit HasSlowFP64 = false,
                                  bit HasFastGather = false,
                                  bit HasFastLargeGather = false> {
  let BufferSize = 0 in
  defm SiFive7N : BulletProcResources<ExtraVALU>;
  // Don't mark these ProcResGroup as BufferSize = 0, since that will make *all
  // of* the subresources busy if the group is used. Instead, we want *one of*
  // the subunits in the group to be used.
  def SiFive7NPipeAB : ProcResGroup<[
    !cast<ProcResource>(NAME #SiFive7NPipeA),
    !cast<ProcResource>(NAME #SiFive7NPipeB)
  ]>;
  if ExtraVALU then
  def SiFive7NVA1OrVA2 : ProcResGroup<[
    !cast<ProcResource>(NAME # SiFive7NVA1),
    !cast<ProcResource>(NAME # SiFive7NVA2)
  ]>;

  // Pull out defs from BulletProcResources so we can refer to them by name.
  defvar SiFive7NPipeA = !cast<ProcResource>(NAME # SiFive7NPipeA);
  defvar SiFive7NPipeB = !cast<ProcResource>(NAME # SiFive7NPipeB);
  defvar SiFive7NPipeAB = !cast<ProcResGroup>(NAME # SiFive7NPipeAB);
  defvar SiFive7NIDiv = !cast<ProcResource>(NAME # SiFive7NIDiv);
  defvar SiFive7NFDiv = !cast<ProcResource>(NAME # SiFive7NFDiv);
  // Pass SiFive7NVA for VA1 and VA1OrVA2 if there is only 1 VALU.
  defvar SiFive7NVA1 = !if (ExtraVALU,
                            !cast<ProcResource>(NAME # SiFive7NVA1),
                            !cast<ProcResource>(NAME # SiFive7NVA));
  defvar SiFive7NVA1OrVA2 = !if (ExtraVALU,
                            !cast<ProcResGroup>(NAME # SiFive7NVA1OrVA2),
                            !cast<ProcResource>(NAME # SiFive7NVA));
  defvar SiFive7NVA = !cast<ProcResource>(NAME # SiFive7NVA);
  defvar SiFive7NVL = !cast<ProcResource>(NAME # SiFive7NVL);
  defvar SiFive7NVS = !cast<ProcResource>(NAME # SiFive7NVS);
  defvar SiFive7NVCQ = !cast<ProcResource>(NAME # SiFive7NVCQ);


  // Define WriteRes records that are the same across all Bullet derived
  // SchedModels. WriteRes records that are not included in that multiclass should
  // be defined below.
  defm "SiFive7N"
      : BulletWriteResBase<VLEN, SiFive7NPipeA, SiFive7NPipeB, SiFive7NPipeAB,
                           SiFive7NIDiv, SiFive7NFDiv, SiFive7NVA1,
                           SiFive7NVA1OrVA2, SiFive7NVL, SiFive7NVS,
                           SiFive7NVCQ, HasSlowFP64,
                           HasFastGather,
                           HasFastLargeGather>;

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

  //===----------------------------------------------------------------------===//
  // Bypass and advance

  defm "SiFive7N" : BulletReadAdvance;
  //===----------------------------------------------------------------------===//
  // Unsupported extensions
  defm : UnsupportedSchedQ;
  defm : UnsupportedSchedZabha;
  defm : UnsupportedSchedZbc;
  defm : UnsupportedSchedZbkb;
  defm : UnsupportedSchedZbkx;
  defm : UnsupportedSchedZfa;
  defm : UnsupportedSchedZvk;
}

// SiFive7N machine model for scheduling and other instruction cost heuristics.
class SiFive7NSchedMachineModel<
    int vlen, bit extraVALU = false, bit hasSlowFP64 = false,
    bit hasFastGather = false,
    bit hasFastLargeGather = false> : SchedMachineModel {
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
  bit ExtraVALU = extraVALU;
  bit HasSlowFP64 = hasSlowFP64;
  bit HasFastGather = hasFastGather;
  bit HasFastLargeGather = hasFastLargeGather;
}

def SiFive7NVLEN128Model
    : SiFive7NSchedMachineModel<128,
                                hasFastGather = true>;
def SiFive7NVLEN256Model : SiFive7NSchedMachineModel<256>;
def SiFive7NVLEN512Model : SiFive7NSchedMachineModel<512>;
def SiFive7NVLEN512FastGatherModel
    : SiFive7NSchedMachineModel<512, hasFastGather = true>;
def SiFive7SVLEN1024DualVALUSlowFP64Model
    : SiFive7NSchedMachineModel<1024, extraVALU = true, hasSlowFP64 = true,
                                hasFastGather=true,
                                hasFastLargeGather=true>;
def SiFive7SVLEN1024DualVALUFastFP64Model
    : SiFive7NSchedMachineModel<1024, extraVALU = true, hasSlowFP64 = false,
                                hasFastGather=true,
                                hasFastLargeGather=true>;
// Some processors do not have vector but reuse the scheduler model that has vector.
// Create an alias that does not have VLEN in the name.
defvar SiFive7NModel = SiFive7NVLEN512Model;
// Define sched resources for each model.
foreach model = [
  SiFive7NVLEN128Model,
  SiFive7NVLEN256Model,
  SiFive7NVLEN512Model,
  SiFive7SVLEN1024DualVALUSlowFP64Model,
  SiFive7SVLEN1024DualVALUFastFP64Model,
  SiFive7NVLEN512FastGatherModel
] in {
  defvar name = "VLEN" #model.VLEN # !if(model.ExtraVALU, "DualVALU","") #
                !if(model.HasSlowFP64, "SlowFP64", "FastFP64") #
                !if(model.HasFastGather, "FastGather", "");
  let SchedModel = model in
  defm name : SiFive7NSchedResources<model.VLEN, model.ExtraVALU, model.HasSlowFP64,
                                     model.HasFastGather,
                                     model.HasFastLargeGather>;
}
