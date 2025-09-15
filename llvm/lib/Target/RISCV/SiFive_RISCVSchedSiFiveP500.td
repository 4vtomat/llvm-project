//==- RISCVSchedSiFiveP500.td - SiFiveP500 Scheduling Defs ---*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
include "SiFive_RISCVSchedMallard.td"

// SiFiveP500 machine model for scheduling and other instruction cost heuristics.
// https://github.com/sifive/federation/blob/master/specs/design/standard/u84.json
def SiFiveP500Model : SchedMachineModel {
  let IssueWidth = 3;         // 3 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 56; // Max micro-ops that can be buffered.
  let LoadLatency = 4;        // Cycles for loads to access the cache.
  let MispredictPenalty = 9;  // Extra cycles for a mispredicted branch.
  let PostRAScheduler = true;
  let UnsupportedFeatures = [HasVInstructions];
  let CompleteModel = false;
}

// The SiFiveP500 microarchitecure has 6 pipelines:
// Three pipelines for integer operations.
// Two pipelines for FPU operations.
// One pipeline for Load/Store operations.
let SchedModel = SiFiveP500Model in
defm SiFiveP500 : MallardProcResources<num_iex = 3, num_fex = 2,
                                       num_vex = 0>;

def SiFiveP500IntProcResources : MallardIntProcResources {
  let Arith = SiFiveP500IntArith;
  let MulI2F = SiFiveP500IEXQ2;
  let Branch = SiFiveP500IEXQ0;
  let SYS = SiFiveP500IEXQ1;
  let CMOV = SiFiveP500IEXQ1;
  let Div = SiFiveP500Div;
}

def SiFiveP500FloatProcResources : MallardFloatProcResources {
  let Arith = SiFiveP500FloatArith;
  let Div = [SiFiveP500FEXQ1, SiFiveP500FloatDiv];
  let F2I = SiFiveP500FEXQ0;
}

def SiFiveP500LSUProcResources : MallardLSUProcResources {
  let Load = SiFiveP500Load;
  let Store = SiFiveP500Store;
}

let SchedModel = SiFiveP500Model in {

defm : MallardBaseWriteRes<SiFiveP500IntProcResources,
                           SiFiveP500FloatProcResources,
                           SiFiveP500LSUProcResources,
                           legacy_imul_latency=true>;

// Floating point arithmetics
let Latency = 4 in {
def : WriteRes<WriteFAdd16, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFAdd32, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFAdd64, [SiFiveP500FloatArith]>;

def : WriteRes<WriteFMul16, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFMul32, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFMul64, [SiFiveP500FloatArith]>;
}

// Floating point divisions
def : WriteRes<WriteFDiv16, [SiFiveP500FEXQ1, SiFiveP500FloatDiv]> {
  let Latency = 19;
  let ReleaseAtCycles = [1, 18];
}
def : WriteRes<WriteFDiv32, [SiFiveP500FEXQ1, SiFiveP500FloatDiv]> {
  let Latency = 19;
  let ReleaseAtCycles = [1, 18];
}
def : WriteRes<WriteFDiv64, [SiFiveP500FEXQ1, SiFiveP500FloatDiv]> {
  let Latency = 33;
  let ReleaseAtCycles = [1, 32];
}

//===----------------------------------------------------------------------===//
// Bypass and advance

defm : MallardBaseReadAdvances;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedQ;
defm : UnsupportedSchedV;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfaWithQ;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedXsf;
}
