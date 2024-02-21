//===-- SiFive_RISCVInstrInfoZjid.td - RISC-V I/D Cache Sync Instructions -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard I/D Cache
// Synchronization Extension (Zjid)
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class CBO_ZJID<bits<12> optype, string opcodestr>
    : RVInstI<0b010, OPC_MISC_MEM, (outs), (ins GPRMemZeroOffset:$rs1),
              opcodestr, "$rs1">, Sched<[]> {
  let imm12 = optype;
  let rd = 0b00000;
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class FENCE_ZJID<bits<4> fm, string opcodestr>
    : RVInstI<0b000, OPC_MISC_MEM, (outs), (ins), opcodestr, "">, Sched<[]> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = {fm,0b0010,0b0010};
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let DecoderNamespace = "RVZjid", Predicates = [HasStdExtZjid] in {
  def CLEAN_ID : CBO_ZJID<0b000000001001, "clean.id">;
  def INVAL_I : CBO_ZJID<0b000000001000, "inval.i">;
  def FENCE_CFI : FENCE_ZJID<0b0010, "fence.cfi">;
  def FENCE_IIS : FENCE_ZJID<0b0011, "fence.iis">;
  def IMPORT_I : RVInstI<0b001, OPC_MISC_MEM, (outs), (ins), "import.i", "">, Sched<[]> {
    let rs1 = 0;
    let rd = 0;
    let imm12 = 1;
    let hasSideEffects = 1;
    let mayLoad = 0;
    let mayStore = 0;
  }
} // Predicates = [HasStdExtZjid]
