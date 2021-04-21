//= RISCVInstrInfoCompactPseudos.td - RISC-V Compact Pseudos -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the RISC-V compact pseudo instruction in TableGen format.
/// This file is included from RISCVInstrInfo.td
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

/// Compact optional pseudo gp.

def PseudoGPRegister : AsmOperandClass {
  let Name = "PseudoGPRegister";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isGPR";
  let ParserMethod = "parseRegister";
  let IsOptional = 1;
}

def pseudo_gp : RegisterOperand<GPR> {
  let ParserMatchClass = PseudoGPRegister;
}

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

/// Compact Load Address Pseudos.

let hasSideEffects = 0, mayLoad = 0,
    mayStore = 0, isCodeGenOnly = 0, isAsmParserOnly = 1 in {
def PseudoLLA_GPREL : Pseudo<(outs GPR:$rd),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "lla.gprel", "$rd, $symbol, $optional">;
def PseudoLA_GOT_GPREL : Pseudo<(outs GPR:$rd),
                                (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                                "la.got.gprel", "$rd, $symbol, $optional">;
def PseudoLA_TLS_GD_GPREL : Pseudo<(outs GPR:$rd),
                                   (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                                   "la.tls.gd.gprel", "$rd, $symbol, $optional">;
def PseudoLA_TLS_IE_GPREL : Pseudo<(outs GPR:$rd),
                                   (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                                   "la.tls.ie.gprel", "$rd, $symbol, $optional">;
} // mayLoad = 0, mayStore = 0, ...

/// Compact Load Pseudos.

let hasSideEffects = 0, mayLoad = 1,
    mayStore = 0, isCodeGenOnly = 0, isAsmParserOnly = 1 in {
def PseudoLB_GPREL : Pseudo<(outs GPR:$rd),
                            (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                            "lb.gprel", "$rd, $symbol, $optional">;
def PseudoLBU_GPREL : Pseudo<(outs GPR:$rd),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "lbu.gprel", "$rd, $symbol, $optional">;
def PseudoLH_GPREL : Pseudo<(outs GPR:$rd),
                            (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                            "lh.gprel", "$rd, $symbol, $optional">;
def PseudoLHU_GPREL : Pseudo<(outs GPR:$rd),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "lhu.gprel", "$rd, $symbol, $optional">;
def PseudoLW_GPREL : Pseudo<(outs GPR:$rd),
                            (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                            "lw.gprel", "$rd, $symbol, $optional">;

let Predicates = [IsRV64] in
def PseudoLWU_GPREL : Pseudo<(outs GPR:$rd),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "lwu.gprel", "$rd, $symbol, $optional">;
let Predicates = [IsRV64] in
def PseudoLD_GPREL : Pseudo<(outs GPR:$rd),
                            (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                            "ld.gprel", "$rd, $symbol, $optional">;

let Predicates = [HasStdExtZfh] in
def PseudoFLH_GPREL : Pseudo<(outs FPR16:$rs, GPR:$tmp),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "flh.gprel", "$rs, $symbol, $tmp, $optional">;

let Predicates = [HasStdExtF] in
def PseudoFLW_GPREL : Pseudo<(outs FPR32:$rs, GPR:$tmp),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "flw.gprel", "$rs, $symbol, $tmp, $optional">;

let Predicates = [HasStdExtD] in
def PseudoFLD_GPREL : Pseudo<(outs FPR64:$rs, GPR:$tmp),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "fld.gprel", "$rs, $symbol, $tmp, $optional">;
} // mayLoad = 1, mayStore = 0, ...

/// Compact Store Pseudos.

let hasSideEffects = 0, mayLoad = 0,
    mayStore = 1, isCodeGenOnly = 0, isAsmParserOnly = 1 in {
def PseudoSB_GPREL : Pseudo<(outs GPR:$rs, GPR:$tmp),
                            (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                            "sb.gprel", "$rs, $symbol, $tmp, $optional">;
def PseudoSH_GPREL : Pseudo<(outs GPR:$rs, GPR:$tmp),
                            (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                            "sh.gprel", "$rs, $symbol, $tmp, $optional">;
def PseudoSW_GPREL : Pseudo<(outs GPR:$rs, GPR:$tmp),
                            (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                            "sw.gprel", "$rs, $symbol, $tmp, $optional">;

let Predicates = [IsRV64] in
def PseudoSD_GPREL : Pseudo<(outs GPR:$rs, GPR:$tmp),
                            (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                            "sd.gprel", "$rs, $symbol, $tmp, $optional">;

let Predicates = [HasStdExtZfh] in
def PseudoFSH_GPREL : Pseudo<(outs FPR16:$rs, GPR:$tmp),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "fsh.gprel", "$rs, $symbol, $tmp, $optional">;

let Predicates = [HasStdExtF] in
def PseudoFSW_GPREL : Pseudo<(outs FPR32:$rs, GPR:$tmp),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "fsw.gprel", "$rs, $symbol, $tmp, $optional">;

let Predicates = [HasStdExtD] in
def PseudoFSD_GPREL : Pseudo<(outs FPR64:$rs, GPR:$tmp),
                             (ins bare_symbol:$symbol, pseudo_gp:$optional), [],
                             "fsd.gprel", "$rs, $symbol, $tmp, $optional">;
} // mayLoad = 0, mayStore = 1, ...
