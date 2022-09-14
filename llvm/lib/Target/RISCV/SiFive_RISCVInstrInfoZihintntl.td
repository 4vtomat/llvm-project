//RISCVInstrInfoZihintntl.td - RISC-V 'Zihintntl' instructions -*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions from Non-Temporal Locality Hints 
/// extension document (zihintntl).
///
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 4, isCodeGenOnly = 1 in {
  def PseudoNTLALL :  Pseudo<(outs), (ins), [], "ntl.all">, 
                             PseudoInstExpansion<(ADD X0, X0, X5)>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 2, isCodeGenOnly = 1 in {
  def PseudoCNTLALL :  Pseudo<(outs), (ins), [], "c.ntl.all">,
                              PseudoInstExpansion<(C_ADD_HINT X0, X0, X5)>;
}

class PseudoNTLLoadPat<DAGOperand rdty>
  : Pseudo<(outs rdty:$rd), (ins GPR:$rs1, simm12:$imm12), []> {
  let hasSideEffects = 0;
  let mayLoad = 1; 
  let mayStore = 0;
  let Size = 8;
  let isCodeGenOnly = 1;
  let isPseudo = 1;
  let IsNonTemporalLoadStore = 1;
}

def PseudoNonTemporalLB : PseudoNTLLoadPat<GPR>, Sched<[WriteLDB, ReadMemBase]>; // ntl.all + LB
def PseudoNonTemporalLH : PseudoNTLLoadPat<GPR>, Sched<[WriteLDH, ReadMemBase]>; // ntl.all + LH
def PseudoNonTemporalLW : PseudoNTLLoadPat<GPR>, Sched<[WriteLDW, ReadMemBase]>; // ntl.all + LW
def PseudoNonTemporalLD : PseudoNTLLoadPat<GPR>, Sched<[WriteLDD, ReadMemBase]>; // ntl.all + LD
def PseudoNonTemporalLBU : PseudoNTLLoadPat<GPR>, Sched<[WriteLDB, ReadMemBase]>; // ntl.all + LBU
def PseudoNonTemporalLHU : PseudoNTLLoadPat<GPR>, Sched<[WriteLDH, ReadMemBase]>; // ntl.all + LHU
def PseudoNonTemporalLWU : PseudoNTLLoadPat<GPR>, Sched<[WriteLDW, ReadMemBase]>; // ntl.all + LWU
def PseudoNonTemporalFLH : PseudoNTLLoadPat<FPR16>, Sched<[WriteFLD16, ReadFMemBase]>; // ntl.all + FLH
def PseudoNonTemporalFLW : PseudoNTLLoadPat<FPR32>, Sched<[WriteFLD32, ReadFMemBase]>; // ntl.all + FLW
def PseudoNonTemporalFLD : PseudoNTLLoadPat<FPR64>, Sched<[WriteFLD64, ReadFMemBase]>; // ntl.all + FLD

class PseudoNTLCLoadPat<RegisterClass cls, DAGOperand opnd>
  : Pseudo<(outs cls:$rd), (ins GPRC:$rs1, opnd:$imm), []> {
  let hasSideEffects = 0;
  let mayLoad = 1; 
  let mayStore = 0;
  let Size = 4;
  let isCodeGenOnly = 1;
  let isPseudo = 1;
  let IsNonTemporalLoadStore = 1;
}

def PseudoNonTemporalC_LW : PseudoNTLCLoadPat<GPRC, uimm7_lsb00>, Sched<[WriteLDW, ReadMemBase]>; // c.ntl.all + c.LW
def PseudoNonTemporalC_LD : PseudoNTLCLoadPat<GPRC, uimm8_lsb000>, Sched<[WriteLDD, ReadMemBase]>; // c.ntl.all + c.LD
def PseudoNonTemporalC_FLW : PseudoNTLCLoadPat<FPR32C, uimm7_lsb00>, Sched<[WriteFLD32, ReadFMemBase]>; // c.ntl.all + c.FLW
def PseudoNonTemporalC_FLD : PseudoNTLCLoadPat<FPR64C, uimm8_lsb000>, Sched<[WriteFLD64, ReadFMemBase]>; // c.ntl.all + c.FLD

let Predicates = [HasStdExtC, HasRVCHints] in {
def : CompressPat<(PseudoNonTemporalLW GPRC:$rd, GPRC:$rs1, uimm7_lsb00:$imm),
                  (PseudoNonTemporalC_LW GPRC:$rd, GPRC:$rs1, uimm7_lsb00:$imm)>;
} // Predicates = [HasStdExtC, HasRVCHints]

let Predicates = [HasStdExtC, IsRV64, HasRVCHints] in {
def : CompressPat<(PseudoNonTemporalLD GPRC:$rd, GPRC:$rs1, uimm8_lsb000:$imm),
                  (PseudoNonTemporalC_LD GPRC:$rd, GPRC:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtC, IsRV64, HasRVCHints]

let Predicates = [HasStdExtC, HasStdExtF, IsRV32, HasRVCHints] in {
def : CompressPat<(PseudoNonTemporalFLW FPR32C:$rd, GPRC:$rs1, uimm7_lsb00:$imm),
                  (PseudoNonTemporalC_FLW FPR32C:$rd, GPRC:$rs1, uimm7_lsb00:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtF, IsRV32, HasRVCHints]

let Predicates = [HasStdExtC, HasStdExtD, HasRVCHints] in {
def : CompressPat<(PseudoNonTemporalFLD FPR64C:$rd, GPRC:$rs1, uimm8_lsb000:$imm),
                  (PseudoNonTemporalC_FLD FPR64C:$rd, GPRC:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtD, HasRVCHints]


class nontemporalloadPat<PatFrag LoadOp> 
  : PatFrag<(ops node:$ptr), (LoadOp node:$ptr), [{
  return cast<LoadSDNode>(N)->isNonTemporal();
}]>;

def nontemporalsextloadi8 : nontemporalloadPat<sextloadi8>;
def nontemporalextloadi8 : nontemporalloadPat<extloadi8>;
def nontemporalsextloadi16 : nontemporalloadPat<sextloadi16>;
def nontemporalextloadi16 : nontemporalloadPat<extloadi16>;
def nontemporalzextloadi8 : nontemporalloadPat<zextloadi8>;
def nontemporalzextloadi16 : nontemporalloadPat<zextloadi16>;
def nontemporalsextloadi32 : nontemporalloadPat<sextloadi32>;
def nontemporalextloadi32 : nontemporalloadPat<extloadi32>;
def nontemporalzextloadi32 : nontemporalloadPat<zextloadi32>;

// ntl.all + load
let AddedComplexity = 1 in {

let Predicates = [HasStdExtZihintntl] in {
  // sextload
  defm : LdPat<nontemporalsextloadi8, PseudoNonTemporalLB>;
  defm : LdPat<nontemporalsextloadi16, PseudoNonTemporalLH>;
  // extload
  defm : LdPat<nontemporalextloadi8, PseudoNonTemporalLB>;
  defm : LdPat<nontemporalextloadi16, PseudoNonTemporalLH>;
  // zextload
  defm : LdPat<nontemporalzextloadi8, PseudoNonTemporalLBU>;
  defm : LdPat<nontemporalzextloadi16, PseudoNonTemporalLHU>;
} // Predicates = [HasStdExtZihintntl]

let Predicates = [IsRV32, HasStdExtZihintntl] in {
  defm : LdPat<nontemporalload, PseudoNonTemporalLW>;
} // Predicates = [IsRV32, HasStdExtZihintntl]

let Predicates = [IsRV64, HasStdExtZihintntl] in {
  defm : LdPat<nontemporalsextloadi32, PseudoNonTemporalLW>;
  defm : LdPat<nontemporalextloadi32, PseudoNonTemporalLW>;
  defm : LdPat<nontemporalzextloadi32, PseudoNonTemporalLWU>;
  defm : LdPat<nontemporalload, PseudoNonTemporalLD>;
} // Predicates = [IsRV64, HasStdExtZihintntl]

let Predicates = [HasStdExtF, HasStdExtZihintntl] in {
  defm : LdPat<nontemporalload, PseudoNonTemporalFLW, f32>;
} // Predicates = [HasStdExtF, HasStdExtZihintntl]

let Predicates = [HasStdExtD, HasStdExtZihintntl] in {
  defm : LdPat<nontemporalload, PseudoNonTemporalFLD, f64>;
} // Predicates = [HasStdExtD, HasStdExtZihintntl]

let Predicates = [HasStdExtZfhOrZfhmin, HasStdExtZihintntl] in {
  defm : LdPat<nontemporalload, PseudoNonTemporalFLH, f16>;
} // Predicates = [HasStdExtZfhOrZfhmin, HasStdExtZihintntl]

} // AddedComplexity=1

class PseudoNTLStorePat<DAGOperand rdty>
  : Pseudo<(outs), (ins rdty:$rs2, GPR:$rs1, simm12:$imm12), []> {
  let hasSideEffects = 0;
  let mayLoad = 0; 
  let mayStore = 1;
  let Size = 8;
  let isCodeGenOnly = 1;
  let isPseudo = 1;
  let IsNonTemporalLoadStore = 1;
}

def PseudoNonTemporalSB : PseudoNTLStorePat<GPR>, Sched<[WriteSTB, ReadStoreData, ReadMemBase]>; // ntl.all + SB
def PseudoNonTemporalSH : PseudoNTLStorePat<GPR>, Sched<[WriteSTH, ReadStoreData, ReadMemBase]>; // ntl.all + SH
def PseudoNonTemporalSW : PseudoNTLStorePat<GPR>, Sched<[WriteSTW, ReadStoreData, ReadMemBase]>; // ntl.all + SW
def PseudoNonTemporalSD : PseudoNTLStorePat<GPR>, Sched<[WriteSTD, ReadStoreData, ReadMemBase]>; // ntl.all + SD
def PseudoNonTemporalFSH : PseudoNTLStorePat<FPR16>, Sched<[WriteFST16, ReadStoreData, ReadMemBase]>; // ntl.all + FSH
def PseudoNonTemporalFSW : PseudoNTLStorePat<FPR32>, Sched<[WriteFST32, ReadStoreData, ReadMemBase]>; // ntl.all + FSW
def PseudoNonTemporalFSD : PseudoNTLStorePat<FPR64>, Sched<[WriteFST64, ReadStoreData, ReadMemBase]>; // ntl.all + FSD

class PseudoCNTLStorePat<RegisterClass cls, DAGOperand opnd>
  : Pseudo<(outs), (ins cls:$rs2, GPRC:$rs1, opnd:$imm), []> {
  let hasSideEffects = 0;
  let mayLoad = 0; 
  let mayStore = 1;
  let Size = 4;
  let isCodeGenOnly = 1;
  let isPseudo = 1;
  let IsNonTemporalLoadStore = 1;
}

def PseudoNonTemporalC_SW : PseudoCNTLStorePat<GPRC, uimm7_lsb00>, Sched<[WriteSTW, ReadStoreData, ReadMemBase]>; // c.ntl.all + c.SW
def PseudoNonTemporalC_SD : PseudoCNTLStorePat<GPRC, uimm8_lsb000>, Sched<[WriteSTD, ReadStoreData, ReadMemBase]>; // c.ntl.all + c.SD
def PseudoNonTemporalC_FSW : PseudoCNTLStorePat<FPR32C, uimm7_lsb00>, Sched<[WriteFST32, ReadStoreData, ReadMemBase]>; // c.ntl.all + c.FSW
def PseudoNonTemporalC_FSD : PseudoCNTLStorePat<FPR64C, uimm8_lsb000>, Sched<[WriteFST64, ReadStoreData, ReadMemBase]>; // c.ntl.all + c.FSD

let Predicates = [HasStdExtC, HasRVCHints] in {
def : CompressPat<(PseudoNonTemporalSW GPRC:$rs2, GPRC:$rs1, uimm7_lsb00:$imm),
                  (PseudoNonTemporalC_SW GPRC:$rs2, GPRC:$rs1, uimm7_lsb00:$imm)>;
} // Predicates = [HasStdExtC, HasRVCHints]

let Predicates = [HasStdExtC, IsRV64, HasRVCHints] in {
def : CompressPat<(PseudoNonTemporalSD GPRC:$rs2, GPRC:$rs1, uimm8_lsb000:$imm),
                  (PseudoNonTemporalC_SD GPRC:$rs2, GPRC:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtC, IsRV64, HasRVCHints]

let Predicates = [HasStdExtC, HasStdExtF, IsRV32, HasRVCHints] in {
def : CompressPat<(PseudoNonTemporalFSW FPR32C:$rs2, GPRC:$rs1, uimm7_lsb00:$imm),
                  (PseudoNonTemporalC_FSW FPR32C:$rs2, GPRC:$rs1, uimm7_lsb00:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtF, IsRV32, HasRVCHints]

let Predicates = [HasStdExtC, HasStdExtD, HasRVCHints] in {
def : CompressPat<(PseudoNonTemporalFSD FPR64C:$rs2, GPRC:$rs1, uimm8_lsb000:$imm),
                  (PseudoNonTemporalC_FSD FPR64C:$rs2, GPRC:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtC, HasStdExtD, HasRVCHints]

class nontemporalstorePat<PatFrag StoreOp> 
  : PatFrag<(ops node:$val, node:$ptr),
            (StoreOp node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->isNonTemporal();
}]>;

def nontemporaltruncstorei8 : nontemporalstorePat<truncstorei8>;
def nontemporaltruncstorei16 : nontemporalstorePat<truncstorei16>;
def nontemporaltruncstorei32 : nontemporalstorePat<truncstorei32>;

// ntl.all + store
let AddedComplexity = 1 in {

let Predicates = [HasStdExtZihintntl] in {
  defm : StPat<nontemporaltruncstorei8, PseudoNonTemporalSB, GPR, XLenVT>;
  defm : StPat<nontemporaltruncstorei16, PseudoNonTemporalSH, GPR, XLenVT>;
  defm : StPat<nontemporaltruncstorei32, PseudoNonTemporalSW, GPR, XLenVT>;
} // Predicates = [HasStdExtZihintntl]

let Predicates = [IsRV32, HasStdExtZihintntl] in {
  defm : StPat<nontemporalstore, PseudoNonTemporalSW, GPR, XLenVT>;
} // Predicates = [IsRV32, HasStdExtZihintntl]

let Predicates = [IsRV64, HasStdExtZihintntl] in {
  defm : StPat<nontemporalstore, PseudoNonTemporalSD, GPR, XLenVT>;
} // Predicates = [IsRV64, HasStdExtZihintntl]

let Predicates = [HasStdExtF, HasStdExtZihintntl] in {
  defm : StPat<nontemporalstore, PseudoNonTemporalFSW, FPR32, f32>;
} // Predicates = [HasStdExtF, HasStdExtZihintntl]

let Predicates = [HasStdExtD, HasStdExtZihintntl] in {
  defm : StPat<nontemporalstore, PseudoNonTemporalFSD, FPR64, f64>;
} // Predicates = [HasStdExtD, HasStdExtZihintntl]

let Predicates = [HasStdExtZfhOrZfhmin, HasStdExtZihintntl] in {
  defm : StPat<nontemporalstore, PseudoNonTemporalFSH, FPR16, f16>;
} // Predicates = [HasStdExtZfhOrZfhmin, HasStdExtZihintntl]

} // AddedComplexity = 1
