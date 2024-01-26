//==----- RISCVMacroFusion.td - Macro Fusion Definitions -----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the macro fusion predicators.

// Fuse LUI followed by ADDI or ADDIW:
//   rd = imm[31:0] which decomposes to
//   lui rd, imm[31:12]
//   addi(w) rd, rd, imm[11:0]
def TuneLUIADDIFusion
  : SimpleFusion<"lui-addi-fusion", "HasLUIADDIFusion",
                 "Enable LUI+ADDI macro fusion",
                 CheckOpcode<[LUI]>,
                 CheckOpcode<[ADDI, ADDIW]>>;

// Fuse AUIPC followed by ADDI:
//   auipc rd, imm20
//   addi rd, rd, imm12
def TuneAUIPCADDIFusion
  : SimpleFusion<"auipc-addi-fusion", "HasAUIPCADDIFusion",
                 "Enable AUIPC+ADDI macrofusion",
                 CheckOpcode<[AUIPC]>,
                 CheckOpcode<[ADDI]>>;

// Fuse zero extension of halfword:
//   slli rd, rs1, 48
//   srli rd, rd, 48
def TuneZExtHFusion
  : SimpleFusion<"zexth-fusion", "HasZExtHFusion",
                 "Enable SLLI+SRLI to be fused to zero extension of halfword",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 48>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 48>
                 ]>>;

// Fuse zero extension of word:
//   slli rd, rs1, 32
//   srli rd, rd, 32
def TuneZExtWFusion
  : SimpleFusion<"zextw-fusion", "HasZExtWFusion",
                 "Enable SLLI+SRLI to be fused to zero extension of word",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 32>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 32>
                 ]>>;

// Fuse shifted zero extension of word:
//   slli rd, rs1, 32
//   srli rd, rd, x
//   where 0 <= x < 32
def TuneShiftedZExtWFusion
  : SimpleFusion<"shifted-zextw-fusion", "HasShiftedZExtWFusion",
                 "Enable SLLI+SRLI to be fused when computing (shifted) word zero extension",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 32>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperandRange<2, 0, 31>
                 ]>>;

// Fuse load with add:
//   add rd, rs1, rs2
//   ld rd, 0(rd)
def TuneLDADDFusion
  : SimpleFusion<"ld-add-fusion", "HasLDADDFusion", "Enable LD+ADD macrofusion",
                 CheckOpcode<[ADD]>,
                 CheckAll<[
                   CheckOpcode<[LD]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 0>
                 ]>>;

#ifdef SIFIVE_CUSTOMIZATION
def TuneFuseLUILoad
  : SimpleFusion<"fuse-lui-load", "HasFuseLUILoad",
                 "Enable fusing LUI/AUIPIC with load",
                 CheckOpcode<[LUI, AUIPC]>,
                 CheckOpcode<[LBU, LB, LHU, LH, LWU, LW, LD]>>;

def TuneFuseIndexedLoad
  : SimpleFusion<"fuse-indexed-load", "HasFuseIndexedLoad",
                 "Enable fusing ADD with load",
                 CheckOpcode<[ADD]>,
                 CheckOpcode<[LBU, LB, LHU, LH, LWU, LW, LD]>>;

def TuneFuseZbaLoad
  : SimpleFusion<"fuse-zba-load", "HasFuseZbaLoad",
                 "Enable fusing Zba instructions with load",
                 CheckOpcode<[ADD, SH1ADD, SH2ADD, SH3ADD,
                              ADD_UW, SH1ADD_UW, SH2ADD_UW, SH3ADD_UW]>,
                 CheckAll<[
                   CheckOpcode<[LBU, LB, LHU, LH, LWU, LW, LD]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 0>
                 ]>>;

def FuseArithEqZPredicate : FusionPredicateWithCode<[{
  unsigned SrcOpIdx;
  switch (SecondMI.getOpcode()) {
  default:
    return false;
  case RISCV::SLTIU:
    if (SecondMI.getOperand(2).getImm() != 1)
      return false;
    SrcOpIdx = 1;
    break;
  case RISCV::SLTU:
    if (SecondMI.getOperand(1).getReg() != RISCV::X0)
      return false;
    SrcOpIdx = 2;
    break;
  }

  // Assume the 1st instr to be a wildcard if it is unspecified.
  if (!FirstMI)
    return true;

  bool PreRA = false;
  switch (FirstMI->getOpcode()) {
  default:
    return false;
  case RISCV::ADDI:
  case RISCV::XOR:
    // Allow these pre-RA because they are the idioms we use for equality
    // comparisons.
    PreRA = true;
    break;
  case RISCV::SUB:
  case RISCV::OR:
    // Do not allow these pre-RA because we can't ensure that SecondMI is the
    // only user pre-RA.
    // FIXME: We probably need some pseudoinstructions and an earlier fusion
    // peephole.
    break;
  }

  Register FirstDest = FirstMI->getOperand(0).getReg();

  // The SecondMI source operand should match the FirstMI destination.
  if (SecondMI.getOperand(SrcOpIdx).getReg() != FirstDest)
    return false;

  // If the input is virtual make sure this is the only user.
  if (FirstDest.isVirtual()) {
    if (!PreRA)
      return false;
    return MRI.hasOneNonDBGUse(FirstDest);
  }

  // If the FirstMI destination is non-virtual, it should match the SecondMI
  // destination.
  return SecondMI.getOperand(0).getReg() == FirstDest;
}]>;

def TuneFuseArithEqZ
  : Fusion<"fuse-arith-eqz", "HasFuseArithEqZ",
           "Enable fusiong arithmetic with seqz/snez",
           [FuseArithEqZPredicate]>;
#endif
