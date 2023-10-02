//===-- RISCVInstrInfoXsf.td - SiFive custom instructions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by SiFive.
//
//===----------------------------------------------------------------------===//

#ifdef SIFIVE_CUSTOMIZATION
defset list<VTypeInfoToWide> AllVQMACCQoQVectors = {
  def : VTypeInfoToWide<VI8MF2,  VI32M1>;
  def : VTypeInfoToWide<VI8M1,   VI32M2>;
  def : VTypeInfoToWide<VI8M2,   VI32M4>;
  def : VTypeInfoToWide<VI8M4,   VI32M8>;
}

defset list<VTypeInfoToWide> AllVQMACCDoDVectors = {
  def : VTypeInfoToWide<VI8M1,   VI32M1>;
  def : VTypeInfoToWide<VI8M2,   VI32M2>;
  def : VTypeInfoToWide<VI8M4,   VI32M4>;
  def : VTypeInfoToWide<VI8M8,   VI32M8>;
}

defset list<VTypeInfoToWide> AllVFNRCLIPVectors = {
  def : VTypeInfoToWide<VI8MF8,  VF32MF2>;
  def : VTypeInfoToWide<VI8MF4,  VF32M1>;
  def : VTypeInfoToWide<VI8MF2,  VF32M2>;
  def : VTypeInfoToWide<VI8M1,   VF32M4>;
  def : VTypeInfoToWide<VI8M2,   VF32M8>;
}

class RVInstVVCustom2<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
                      string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Inst{6-0}= OPC_CUSTOM_2.Value;

  let Uses = [VTYPE, VL];
  let RVVConstraint = NoConstraint;
}

class RVInstVXCustom2<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
                      string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;

  let Uses = [VTYPE, VL];
  let RVVConstraint = NoConstraint;
}

class CustomSiFiveVV<string opcodestr,  RISCVVFormat opv, bits<6> funct6>
  : RVInstVVCustom2<funct6, opv, (outs VR:$vd),
                    (ins VR:$vs1, VR:$vs2), opcodestr, "$vd, $vs1, $vs2"> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

let vm = 1, RVVConstraint = VS2Constraint, Predicates = [HasVendorXSfvqmaccqoq],
    DecoderNamespace = "XSfvqmaccqoq" in {
  def VQMACCU4x8x4_VV : CustomSiFiveVV<"sf.vqmaccu.4x8x4", OPMVV, 0b111100>;
  def VQMACC4x8x4_VV : CustomSiFiveVV<"sf.vqmacc.4x8x4", OPMVV, 0b111101>;
  def VQMACCUS4x8x4_VV : CustomSiFiveVV<"sf.vqmaccus.4x8x4", OPMVV, 0b111110>;
  def VQMACCSU4x8x4_VV : CustomSiFiveVV<"sf.vqmaccsu.4x8x4", OPMVV, 0b111111>;
} // Predicates = [HasVendorXSfvqmaccqoq]

let vm = 1, RVVConstraint = VS2Constraint, Predicates = [HasVendorXSfvqmaccdod],
    DecoderNamespace = "XSfvqmaccdod" in {
  def VQMACCU2x8x2_VV : CustomSiFiveVV<"sf.vqmaccu.2x8x2", OPMVV, 0b101100>;
  def VQMACC2x8x2_VV : CustomSiFiveVV<"sf.vqmacc.2x8x2", OPMVV, 0b101101>;
  def VQMACCUS2x8x2_VV : CustomSiFiveVV<"sf.vqmaccus.2x8x2", OPMVV, 0b101110>;
  def VQMACCSU2x8x2_VV : CustomSiFiveVV<"sf.vqmaccsu.2x8x2", OPMVV, 0b101111>;
} // Predicates = [HasVendorXSfvqmaccdod]

let Predicates = [HasVendorXSfvfhbfmin], DecoderNamespace = "XSfvfhbfmin",
    Constraints = "@earlyclobber $vd", mayRaiseFPException = true in {
let RVVConstraint = WidenCvt in
defm VFWCVT_F_BF16_V : VWCVTF_FV_VS2<"sf.vfwcvt.f.bf16.v", 0b010010, 0b01101>;
defm VFNCVT_BF16_F_W : VNCVTF_FV_VS2<"sf.vfncvt.bf16.f.w", 0b010010, 0b11101>;
} // Predicates = [HasVendorXSfvqmaccdod]

let vm = 1, RVVConstraint = VS2Constraint, Predicates = [HasVendorXSfvfwmaccqqq],
    DecoderNamespace = "XSfvfwmaccqqq", mayRaiseFPException = true in {
  def VFWMACC4X4X4_VV : CustomSiFiveVV<"sf.vfwmacc.4x4x4", OPFVV, 0b111100>, Sched<[]> ;
} // Predicates = [HasVendorXSfvfwmaccqqq]

class CustomSiFiveVF<string opcodestr,  RISCVVFormat opv, bits<6> funct6>
  : RVInstVXCustom2<funct6, opv, (outs VR:$vd),
                    (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm), opcodestr,
                    "$vd, $vs2, $rs1$vm"> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

let Predicates = [HasVendorXSfvfnrclipxfqf], DecoderNamespace = "XSfvfnrclipxfqf" in {
  def VFNRCLIP_XU_F_QF : CustomSiFiveVF<"sf.vfnrclip.xu.f.qf", OPFVF, 0b100010>, Sched<[]> ;
  def VFNRCLIP_X_F_QF : CustomSiFiveVF<"sf.vfnrclip.x.f.qf", OPFVF, 0b100011>, Sched<[]> ;
} // Predicates = [HasVendorXSfvfnrclipxfqf]
#endif // SIFIVE_CUSTOMIZATION

//===----------------------------------------------------------------------===//
// XSFVCP extension instructions.
//===----------------------------------------------------------------------===//

def VCIXVS2    : RISCVVConstraint<VS2Constraint.Value>;
def VCIXVS2VS1 : RISCVVConstraint<!or(VS2Constraint.Value,
                                      VS1Constraint.Value)>;

class VCIXType<bits<4> val> {
  bits<4> Val = val;
}

def VCIX_X   : VCIXType<0b0000>;
def VCIX_XV  : VCIXType<0b0010>;
def VCIX_XVV : VCIXType<0b1010>;
def VCIX_XVW : VCIXType<0b1111>;

// The payload and tsimm5 operands are all marked as ImmArg in the IR
// intrinsic and will be target constant, so use TImmLeaf rather than ImmLeaf.
def payload1 : Operand<XLenVT>, TImmLeaf<XLenVT, [{return isUInt<1>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<1>;
  let DecoderMethod = "decodeUImmOperand<1>";
  let OperandType = "OPERAND_UIMM1";
  let OperandNamespace = "RISCVOp";
}

def payload2 : Operand<XLenVT>, TImmLeaf<XLenVT, [{return isUInt<2>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<2>;
  let DecoderMethod = "decodeUImmOperand<2>";
  let OperandType = "OPERAND_UIMM2";
  let OperandNamespace = "RISCVOp";
}

def payload5 : Operand<XLenVT>, TImmLeaf<XLenVT, [{return isUInt<5>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<5>;
  let DecoderMethod = "decodeUImmOperand<5>";
  let OperandType = "OPERAND_UIMM5";
  let OperandNamespace = "RISCVOp";
}

def tsimm5 : Operand<XLenVT>, TImmLeaf<XLenVT, [{return isInt<5>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<5>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<5>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<5>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

class SwapVCIXIns<dag funct6, dag rd, dag rs2, dag rs1, bit swap> {
  dag Ins = !con(funct6, !if(swap, rs2, rd), !if(swap, rd, rs2), rs1);
}

class RVInstVCCustom2<bits<4> funct6_hi4, bits<3> funct3, dag outs, dag ins,
                      string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;
  bits<2> funct6_lo2;
  bit vm;

  let Inst{31-28} = funct6_hi4;
  let Inst{27-26} = funct6_lo2;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;

  let Uses = [VTYPE, VL];
  let RVVConstraint = NoConstraint;
}

class RVInstVCFCustom2<bits<4> funct6_hi4, bits<3> funct3, dag outs, dag ins,
                       string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;
  bit funct6_lo1;
  bit vm;

  let Inst{31-28} = funct6_hi4;
  let Inst{27} = 1;
  let Inst{26} = funct6_lo1;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;

  let Uses = [VTYPE, VL];
  let RVVConstraint = NoConstraint;
}

class VCIXInfo<string suffix, VCIXType type, DAGOperand TyRd,
               DAGOperand TyRs2, DAGOperand TyRs1, bit HaveOutputDst> {
  string OpcodeStr = !if(HaveOutputDst, "sf.vc.v." # suffix,
                                        "sf.vc." # suffix);
  bits<4> Funct6_hi4 = type.Val;
  bits<3> Funct3 = !cond(!eq(TyRs1, VR):    0b000,
                         !eq(TyRs1, GPR):   0b100,
                         !eq(TyRs1, FPR32): 0b101,
                         !eq(TyRs1, simm5): 0b011);
  dag Outs = !if(!not(HaveOutputDst), (outs),
                 !if(!or(!eq(type, VCIX_XVV), !eq(type, VCIX_XVW)),
                     (outs TyRd:$rd_wb), (outs TyRd:$rd)));
  dag Ins = SwapVCIXIns<!if(!ne(TyRs1, FPR32), (ins uimm2:$funct6_lo2),
                                               (ins uimm1:$funct6_lo1)),
                        !if(!and(HaveOutputDst, !or(!eq(type, VCIX_X),
                                                    !eq(type, VCIX_XV))),
                            (ins), (ins TyRd:$rd)),
                        (ins TyRs2:$rs2),
                        (ins TyRs1:$rs1),
                        !if(!eq(type, VCIX_X), 1, 0)>.Ins;
  string Prototype = !if(!eq(type, VCIX_X), "$funct6_lo2, $rs2, $rd, $rs1",
                         !if(!ne(TyRs1, FPR32), "$funct6_lo2, $rd, $rs2, $rs1",
                                                "$funct6_lo1, $rd, $rs2, $rs1"));
  string Constraints = !if(!not(HaveOutputDst), "",
                           !if(!or(!eq(type, VCIX_XVV),
                                   !eq(type, VCIX_XVW)), "$rd = $rd_wb", ""));
  RISCVVConstraint RVVConstraint = !if(!or(!not(HaveOutputDst),
                                           !ne(type, VCIX_XVW)), NoConstraint,
                                       !if(!eq(TyRs1, VR), VCIXVS2VS1, VCIXVS2));
}

class CustomSiFiveVCIX<VCIXInfo info>
  : RVInstVCCustom2<info.Funct6_hi4, info.Funct3, info.Outs,
                    info.Ins, info.OpcodeStr, info.Prototype> {
  let Constraints = info.Constraints;
  let RVVConstraint = info.RVVConstraint;
}

class CustomSiFiveVCIF<VCIXInfo info>
  : RVInstVCFCustom2<info.Funct6_hi4, info.Funct3, info.Outs,
                     info.Ins, info.OpcodeStr, info.Prototype> {
  let Constraints = info.Constraints;
  let RVVConstraint = info.RVVConstraint;
}

multiclass CustomSiFiveVCIXorVCIF<string suffix, VCIXType type,
                                  DAGOperand TyRd, DAGOperand TyRs2,
                                  DAGOperand TyRs1, bit HaveOutputDst> {
  defvar info = VCIXInfo<suffix, type, TyRd, TyRs2, TyRs1, HaveOutputDst>;
  if !eq(TyRs1, FPR32) then {
    def NAME : CustomSiFiveVCIF<info>;
  } else {
    def NAME : CustomSiFiveVCIX<info>;
  }
}

multiclass CustomSiFiveVCIX<string suffix, VCIXType type,
                            DAGOperand InTyRd, DAGOperand InTyRs2,
                            DAGOperand InTyRs1> {
  let vm = 1 in
  defm VC_ # NAME   : CustomSiFiveVCIXorVCIF<suffix, type, InTyRd, InTyRs2,
                                             InTyRs1, 0>;
  let vm = 0 in
  defm VC_V_ # NAME : CustomSiFiveVCIXorVCIF<suffix, type, VR, InTyRs2,
                                             InTyRs1, 1>;
}

let Predicates = [HasVendorXSfvcp], mayLoad = 0, mayStore = 0,
    hasSideEffects = 1, hasNoSchedulingInfo = 1, DecoderNamespace = "XSfvcp" in {
  defm X   : CustomSiFiveVCIX<"x",   VCIX_X,   uimm5, uimm5, GPR>,   Sched<[]>;
  defm I   : CustomSiFiveVCIX<"i",   VCIX_X,   uimm5, uimm5, simm5>, Sched<[]>;
  defm XV  : CustomSiFiveVCIX<"xv",  VCIX_XV,  uimm5, VR,    GPR>,   Sched<[]>;
  defm IV  : CustomSiFiveVCIX<"iv",  VCIX_XV,  uimm5, VR,    simm5>, Sched<[]>;
  defm VV  : CustomSiFiveVCIX<"vv",  VCIX_XV,  uimm5, VR,    VR>,    Sched<[]>;
  defm FV  : CustomSiFiveVCIX<"fv",  VCIX_XV,  uimm5, VR,    FPR32>, Sched<[]>;
  defm XVV : CustomSiFiveVCIX<"xvv", VCIX_XVV, VR,    VR,    GPR>,   Sched<[]>;
  defm IVV : CustomSiFiveVCIX<"ivv", VCIX_XVV, VR,    VR,    simm5>, Sched<[]>;
  defm VVV : CustomSiFiveVCIX<"vvv", VCIX_XVV, VR,    VR,    VR>,    Sched<[]>;
  defm FVV : CustomSiFiveVCIX<"fvv", VCIX_XVV, VR,    VR,    FPR32>, Sched<[]>;
  defm XVW : CustomSiFiveVCIX<"xvw", VCIX_XVW, VR,    VR,    GPR>,   Sched<[]>;
  defm IVW : CustomSiFiveVCIX<"ivw", VCIX_XVW, VR,    VR,    simm5>, Sched<[]>;
  defm VVW : CustomSiFiveVCIX<"vvw", VCIX_XVW, VR,    VR,    VR>,    Sched<[]>;
  defm FVW : CustomSiFiveVCIX<"fvw", VCIX_XVW, VR,    VR,    FPR32>, Sched<[]>;
}

class VPseudoVC_X<Operand OpClass, DAGOperand RS1Class,
                  bit HasSideEffect = 1> :
      Pseudo<(outs),
             (ins OpClass:$op1, payload5:$rs2, payload5:$rd, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = HasSideEffect;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_XV<Operand OpClass, VReg RS2Class, DAGOperand RS1Class,
                   bit HasSideEffect = 1> :
      Pseudo<(outs),
             (ins OpClass:$op1, payload5:$rd, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = HasSideEffect;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_XVV<Operand OpClass, VReg RDClass, VReg RS2Class,
                    DAGOperand RS1Class, bit HasSideEffect = 1> :
      Pseudo<(outs),
             (ins OpClass:$op1, RDClass:$rd, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = HasSideEffect;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_V_X<Operand OpClass, VReg RDClass, DAGOperand RS1Class,
                    bit HasSideEffect = 1> :
      Pseudo<(outs RDClass:$rd),
             (ins OpClass:$op1, payload5:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = HasSideEffect;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_V_XV<Operand OpClass, VReg RDClass, VReg RS2Class,
                     DAGOperand RS1Class, bit HasSideEffect = 1> :
      Pseudo<(outs RDClass:$rd),
             (ins OpClass:$op1, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = HasSideEffect;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_V_XVV<Operand OpClass, VReg RDClass, VReg RS2Class,
                      DAGOperand RS1Class, bit HasSideEffect = 1> :
      Pseudo<(outs RDClass:$rd),
             (ins OpClass:$op1, RDClass:$rs3, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = HasSideEffect;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

multiclass VPseudoVC_X<LMULInfo m, DAGOperand RS1Class,
                       Operand OpClass = payload2> {
  let VLMul = m.value in {
    def "PseudoVC_" # NAME # "_SE_" # m.MX : VPseudoVC_X<OpClass, RS1Class>;
    def "PseudoVC_V_" # NAME # "_SE_" # m.MX : VPseudoVC_V_X<OpClass, m.vrclass, RS1Class>;
    def "PseudoVC_V_" # NAME # "_" # m.MX : VPseudoVC_V_X<OpClass, m.vrclass, RS1Class, 0>;
  }
}

multiclass VPseudoVC_XV<LMULInfo m, DAGOperand RS1Class,
                        Operand OpClass = payload2> {
  let VLMul = m.value in {
    def "PseudoVC_" # NAME # "_SE_" # m.MX : VPseudoVC_XV<OpClass, m.vrclass, RS1Class>;
    def "PseudoVC_V_" # NAME # "_SE_" # m.MX : VPseudoVC_V_XV<OpClass, m.vrclass, m.vrclass, RS1Class>;
    def "PseudoVC_V_" # NAME # "_" # m.MX : VPseudoVC_V_XV<OpClass, m.vrclass, m.vrclass, RS1Class, 0>;
  }
}

multiclass VPseudoVC_XVV<LMULInfo m, DAGOperand RS1Class,
                         Operand OpClass = payload2> {
  let VLMul = m.value in {
    def "PseudoVC_" # NAME # "_SE_" # m.MX : VPseudoVC_XVV<OpClass, m.vrclass, m.vrclass, RS1Class>;
    def "PseudoVC_V_" # NAME # "_SE_" # m.MX : VPseudoVC_V_XVV<OpClass, m.vrclass, m.vrclass, RS1Class>;
    def "PseudoVC_V_" # NAME # "_" # m.MX : VPseudoVC_V_XVV<OpClass, m.vrclass, m.vrclass, RS1Class, 0>;
  }
}

multiclass VPseudoVC_XVW<LMULInfo m, DAGOperand RS1Class,
                         Operand OpClass = payload2> {
  let VLMul = m.value in {
    def "PseudoVC_" # NAME # "_SE_" # m.MX : VPseudoVC_XVV<OpClass, m.wvrclass, m.vrclass, RS1Class>;
    let Constraints = "@earlyclobber $rd, $rd = $rs3" in {
      def "PseudoVC_V_" # NAME # "_SE_" # m.MX : VPseudoVC_V_XVV<OpClass, m.wvrclass, m.vrclass, RS1Class>;
      def "PseudoVC_V_" # NAME # "_" # m.MX : VPseudoVC_V_XVV<OpClass, m.wvrclass, m.vrclass, RS1Class, 0>;
    }
  }
}

let Predicates = [HasVendorXSfvcp] in {
  foreach m = MxList in {
    defm X : VPseudoVC_X<m, GPR>;
    defm I : VPseudoVC_X<m, tsimm5>;
    defm XV : VPseudoVC_XV<m, GPR>;
    defm IV : VPseudoVC_XV<m, tsimm5>;
    defm VV : VPseudoVC_XV<m, m.vrclass>;
    defm XVV : VPseudoVC_XVV<m, GPR>;
    defm IVV : VPseudoVC_XVV<m, tsimm5>;
    defm VVV : VPseudoVC_XVV<m, m.vrclass>;
  }
  foreach f = FPList in {
    foreach m = f.MxList in {
    defm f.FX # "V" : VPseudoVC_XV<m, f.fprclass, payload1>;
    defm f.FX # "VV" : VPseudoVC_XVV<m, f.fprclass, payload1>;
    }
  }
  foreach m = MxListW in {
    defm XVW : VPseudoVC_XVW<m, GPR>;
    defm IVW : VPseudoVC_XVW<m, tsimm5>;
    defm VVW : VPseudoVC_XVW<m, m.vrclass>;
  }
  foreach f = FPListW in {
    foreach m = f.MxList in
    defm f.FX # "VW" : VPseudoVC_XVW<m, f.fprclass, payload1>;
  }
}

class VPatVC_OP4<string intrinsic_name,
                 string inst,
                 ValueType op2_type,
                 ValueType op3_type,
                 ValueType op4_type,
                 int sew,
                 DAGOperand op2_kind,
                 DAGOperand op3_kind,
                 DAGOperand op4_kind,
                 Operand op1_kind = payload2> :
  Pat<(!cast<Intrinsic>(intrinsic_name)
       (XLenVT   op1_kind:$op1),
       (op2_type op2_kind:$op2),
       (op3_type op3_kind:$op3),
       (op4_type op4_kind:$op4),
       VLOpFrag),
      (!cast<Instruction>(inst)
       (XLenVT   op1_kind:$op1),
       (op2_type op2_kind:$op2),
       (op3_type op3_kind:$op3),
       (op4_type op4_kind:$op4),
       GPR:$vl, sew)>;

class VPatVC_V_OP4<string intrinsic_name,
                   string inst,
                   ValueType result_type,
                   ValueType op2_type,
                   ValueType op3_type,
                   ValueType op4_type,
                   int sew,
                   DAGOperand op2_kind,
                   DAGOperand op3_kind,
                   DAGOperand op4_kind,
                   Operand op1_kind = payload2> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                    (XLenVT   op1_kind:$op1),
                    (op2_type op2_kind:$op2),
                    (op3_type op3_kind:$op3),
                    (op4_type op4_kind:$op4),
                    VLOpFrag)),
                   (!cast<Instruction>(inst)
                    (XLenVT   op1_kind:$op1),
                    (op2_type op2_kind:$op2),
                    (op3_type op3_kind:$op3),
                    (op4_type op4_kind:$op4),
                    GPR:$vl, sew)>;

class VPatVC_V_OP3<string intrinsic_name,
                   string inst,
                   ValueType result_type,
                   ValueType op2_type,
                   ValueType op3_type,
                   int sew,
                   DAGOperand op2_kind,
                   DAGOperand op3_kind,
                   Operand op1_kind = payload2> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                    (XLenVT   op1_kind:$op1),
                    (op2_type op2_kind:$op2),
                    (op3_type op3_kind:$op3),
                    VLOpFrag)),
                   (!cast<Instruction>(inst)
                    (XLenVT   op1_kind:$op1),
                    (op2_type op2_kind:$op2),
                    (op3_type op3_kind:$op3),
                    GPR:$vl, sew)>;

multiclass VPatVC_X<string intrinsic_suffix, string instruction_suffix,
                    VTypeInfo vti, ValueType type, DAGOperand kind> {
  def : VPatVC_OP4<"int_riscv_sf_vc_" # intrinsic_suffix # "_se_e" # vti.SEW # !tolower(vti.LMul.MX),
                   "PseudoVC_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                   XLenVT, XLenVT, type, vti.Log2SEW,
                   payload5, payload5, kind>;
  def : VPatVC_V_OP3<"int_riscv_sf_vc_v_" # intrinsic_suffix # "_se",
                     "PseudoVC_V_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                     vti.Vector, XLenVT, type, vti.Log2SEW,
                     payload5, kind>;
  def : VPatVC_V_OP3<"int_riscv_sf_vc_v_" # intrinsic_suffix,
                     "PseudoVC_V_" # instruction_suffix # "_" # vti.LMul.MX,
                     vti.Vector, XLenVT, type, vti.Log2SEW,
                     payload5, kind>;
}

multiclass VPatVC_XV<string intrinsic_suffix, string instruction_suffix,
                     VTypeInfo vti, ValueType type, DAGOperand kind,
                     Operand op1_kind = payload2> {
  def : VPatVC_OP4<"int_riscv_sf_vc_" # intrinsic_suffix # "_se",
                   "PseudoVC_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                   XLenVT, vti.Vector, type, vti.Log2SEW,
                   payload5, vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP3<"int_riscv_sf_vc_v_" # intrinsic_suffix # "_se",
                     "PseudoVC_V_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                     vti.Vector, vti.Vector, type, vti.Log2SEW,
                     vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP3<"int_riscv_sf_vc_v_" # intrinsic_suffix,
                     "PseudoVC_V_" # instruction_suffix # "_" # vti.LMul.MX,
                     vti.Vector, vti.Vector, type, vti.Log2SEW,
                     vti.RegClass, kind, op1_kind>;
}

multiclass VPatVC_XVV<string intrinsic_suffix, string instruction_suffix,
                      VTypeInfo wti, VTypeInfo vti, ValueType type, DAGOperand kind,
                      Operand op1_kind = payload2> {
  def : VPatVC_OP4<"int_riscv_sf_vc_" # intrinsic_suffix # "_se",
                   "PseudoVC_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                   wti.Vector, vti.Vector, type, vti.Log2SEW,
                   wti.RegClass, vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP4<"int_riscv_sf_vc_v_" # intrinsic_suffix # "_se",
                     "PseudoVC_V_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                     wti.Vector, wti.Vector, vti.Vector, type, vti.Log2SEW,
                     wti.RegClass, vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP4<"int_riscv_sf_vc_v_" # intrinsic_suffix,
                     "PseudoVC_V_" # instruction_suffix # "_" # vti.LMul.MX,
                     wti.Vector, wti.Vector, vti.Vector, type, vti.Log2SEW,
                     wti.RegClass, vti.RegClass, kind, op1_kind>;
}

class GetFTypeInfo<int Sew> {
  ValueType Scalar = !cond(!eq(Sew, 16) : f16,
                           !eq(Sew, 32) : f32,
                           !eq(Sew, 64) : f64);
  RegisterClass ScalarRegClass = !cond(!eq(Sew, 16) : FPR16,
                                       !eq(Sew, 32) : FPR32,
                                       !eq(Sew, 64) : FPR64);

  string ScalarSuffix = !cond(!eq(Scalar, f16) : "FPR16",
                              !eq(Scalar, f32) : "FPR32",
                              !eq(Scalar, f64) : "FPR64");
}

let Predicates = [HasVendorXSfvcp] in {
  foreach vti = AllVectors in {
    defm : VPatVC_X<"x", "X", vti, XLenVT, GPR>;
    defm : VPatVC_X<"i", "I", vti, XLenVT, tsimm5>;
    defm : VPatVC_XV<"xv", "XV", vti, XLenVT, GPR>;
    defm : VPatVC_XV<"iv", "IV", vti, XLenVT, tsimm5>;
    defm : VPatVC_XV<"vv", "VV", vti, vti.Vector, vti.RegClass>;
    defm : VPatVC_XVV<"xvv", "XVV", vti, vti, XLenVT, GPR>;
    defm : VPatVC_XVV<"ivv", "IVV", vti, vti, XLenVT, tsimm5>;
    defm : VPatVC_XVV<"vvv", "VVV", vti, vti, vti.Vector, vti.RegClass>;

    if !ne(vti.SEW, 8) then {
      defvar finfo = GetFTypeInfo<vti.SEW>;
      defm : VPatVC_XV<"fv", finfo.ScalarSuffix # "V", vti, finfo.Scalar,
                       finfo.ScalarRegClass, payload1>;
      defm : VPatVC_XVV<"fvv", finfo.ScalarSuffix # "VV", vti, vti, finfo.Scalar,
                        finfo.ScalarRegClass, payload1>;
    }
  }
  foreach VtiToWti = !listconcat(AllWidenableIntVectors, AllWidenableFloatVectors) in {
    defvar vti = VtiToWti.Vti;
    defvar wti = VtiToWti.Wti;
    defvar iinfo = GetIntVTypeInfo<vti>.Vti;
    defm : VPatVC_XVV<"xvw", "XVW", wti, vti, iinfo.Scalar, iinfo.ScalarRegClass>;
    defm : VPatVC_XVV<"ivw", "IVW", wti, vti, XLenVT, tsimm5>;
    defm : VPatVC_XVV<"vvw", "VVW", wti, vti, vti.Vector, vti.RegClass>;

    if !ne(vti.SEW, 8) then {
      defvar finfo = GetFTypeInfo<vti.SEW>;
      defm : VPatVC_XVV<"fvw", finfo.ScalarSuffix # "VW", wti, vti, finfo.Scalar,
                        finfo.ScalarRegClass, payload1>;
    }
  }
}

#ifdef SIFIVE_CUSTOMIZATION
let IsRVVWideningReduction = 1 in {
foreach m = [V_MF2, V_M1, V_M2, V_M4] in {
  defvar Constraint = "@earlyclobber $rd";
  let VLMul = m.value in {
    def PseudoVQMACCU4x8x4_VV # "_" # m.MX : VPseudoTernaryNoMaskWithPolicy<m.wvrclass, VR, m.vrclass, Constraint>;
    def PseudoVQMACC4x8x4_VV # "_" # m.MX : VPseudoTernaryNoMaskWithPolicy<m.wvrclass, VR, m.vrclass, Constraint>;
    def PseudoVQMACCUS4x8x4_VV # "_" # m.MX : VPseudoTernaryNoMaskWithPolicy<m.wvrclass, VR, m.vrclass, Constraint>;
    def PseudoVQMACCSU4x8x4_VV # "_" # m.MX : VPseudoTernaryNoMaskWithPolicy<m.wvrclass, VR, m.vrclass, Constraint>;
  }
}

foreach m = [V_M1, V_M2, V_M4, V_M8] in {
  defvar Constraint = "@earlyclobber $rd";
  let VLMul = m.value in {
    def PseudoVQMACCU2x8x2_VV # "_" # m.MX : VPseudoTernaryNoMaskWithPolicy<m.vrclass, VR, m.vrclass, Constraint>;
    def PseudoVQMACC2x8x2_VV # "_" # m.MX : VPseudoTernaryNoMaskWithPolicy<m.vrclass, VR, m.vrclass, Constraint>;
    def PseudoVQMACCUS2x8x2_VV # "_" # m.MX : VPseudoTernaryNoMaskWithPolicy<m.vrclass, VR, m.vrclass, Constraint>;
    def PseudoVQMACCSU2x8x2_VV # "_" # m.MX : VPseudoTernaryNoMaskWithPolicy<m.vrclass, VR, m.vrclass, Constraint>;
  }
}
}

foreach m = [V_MF8, V_MF4, V_MF2, V_M1, V_M2] in {
  defvar Constraint = "@earlyclobber $rd";
  let VLMul = m.value in {
    def PseudoVFNRCLIP_XU_F_QF # "_" # m.MX : VPseudoBinaryNoMaskTU<m.vrclass, m.w2vrclass, FPR32, Constraint>;
    def PseudoVFNRCLIP_X_F_QF # "_" # m.MX : VPseudoBinaryNoMaskTU<m.vrclass, m.w2vrclass, FPR32, Constraint>;
    def PseudoVFNRCLIP_XU_F_QF # "_" # m.MX # "_MASK" : VPseudoBinaryMaskPolicy<m.vrclass, m.w2vrclass, FPR32, Constraint>;
    def PseudoVFNRCLIP_X_F_QF # "_" # m.MX # "_MASK" : VPseudoBinaryMaskPolicy<m.vrclass, m.w2vrclass, FPR32, Constraint>;
  }
}

let Predicates = [HasVendorXSfvfhbfmin], mayRaiseFPException = true in {
#ifdef SIFIVE_CUSTOMIZATION
defm PseudoVFWCVT_F_BF16   : VPseudoVWCVTD_V</*TargetConstraintType*/ 3>;
defm PseudoVFNCVT_BF16_F   : VPseudoVNCVTD_W</*TargetConstraintType*/ 2>;
#endif // SIFIVE_CUSTOMIZATION
}

let IsRVVWideningReduction = 1, mayRaiseFPException = true in {
foreach m = [V_MF4, V_MF2, V_M1, V_M2, V_M4, V_M8] in {
  defvar Constraint = "@earlyclobber $rd";
  let VLMul = m.value in {
    def PseudoVFWMACC4X4X4_VV # "_" # m.MX: VPseudoTernaryNoMaskWithPolicy<m.wvrclass, VR, m.vrclass, Constraint>;
  }
}
}

def : Pat<(XLenVT (int_riscv_vgetvxrm)), (ReadVXRM)>;

def : Pat<(int_riscv_vsetvxrm (XLenVT GPR:$rm)),
          (WriteVXRM $rm)>;

def : Pat<(int_riscv_vsetvxrm uimm5:$rm),
          (WriteVXRMImm uimm5:$rm)>;

multiclass XPatSFVQMACC_VV<string intrinsic, string instruction,
                           list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    def : VPatTernaryNoMaskWithPolicy<intrinsic, instruction, "VV",
                                        Wti.Vector, vint8m1_t, Vti.Vector,
                                        Vti.Log2SEW, Vti.LMul,
                                        Wti.RegClass, VR, Vti.RegClass>;
  }
}

multiclass XPatSFVFNRCLIP<string intrinsic, string instruction,
                          list<VTypeInfoToWide> vtilist> {
  foreach VtiToWti = vtilist in {
    defvar Vti = VtiToWti.Vti;
    defvar Wti = VtiToWti.Wti;
    defm : VPatBinary<intrinsic, instruction # "_" # Vti.LMul.MX, Vti.Vector,
                      Wti.Vector, Wti.Scalar, Vti.Mask, Vti.Log2SEW,
                      Vti.RegClass, Wti.RegClass, Wti.ScalarRegClass>;
  }
}

let Predicates = [HasVendorXSfvqmaccqoq] in {
  defm : XPatSFVQMACC_VV<"int_riscv_sf_vqmaccu_4x8x4", "PseudoVQMACCU4x8x4", AllVQMACCQoQVectors>;
  defm : XPatSFVQMACC_VV<"int_riscv_sf_vqmacc_4x8x4", "PseudoVQMACC4x8x4", AllVQMACCQoQVectors>;
  defm : XPatSFVQMACC_VV<"int_riscv_sf_vqmaccus_4x8x4", "PseudoVQMACCUS4x8x4", AllVQMACCQoQVectors>;
  defm : XPatSFVQMACC_VV<"int_riscv_sf_vqmaccsu_4x8x4", "PseudoVQMACCSU4x8x4", AllVQMACCQoQVectors>;
} // Predicates = [HasVendorXSfvqmaccqoq]

let Predicates = [HasVendorXSfvqmaccdod] in {
  defm : XPatSFVQMACC_VV<"int_riscv_sf_vqmaccu_2x8x2", "PseudoVQMACCU2x8x2", AllVQMACCDoDVectors>;
  defm : XPatSFVQMACC_VV<"int_riscv_sf_vqmacc_2x8x2", "PseudoVQMACC2x8x2", AllVQMACCDoDVectors>;
  defm : XPatSFVQMACC_VV<"int_riscv_sf_vqmaccus_2x8x2", "PseudoVQMACCUS2x8x2", AllVQMACCDoDVectors>;
  defm : XPatSFVQMACC_VV<"int_riscv_sf_vqmaccsu_2x8x2", "PseudoVQMACCSU2x8x2", AllVQMACCDoDVectors>;
} // Predicates = [HasVendorXSfvqmaccdod]

let Predicates = [HasVendorXSfvfnrclipxfqf] in {
  defm : XPatSFVFNRCLIP<"int_riscv_sf_vfnrclip_xu", "PseudoVFNRCLIP_XU_F_QF", AllVFNRCLIPVectors>;
  defm : XPatSFVFNRCLIP<"int_riscv_sf_vfnrclip_x",  "PseudoVFNRCLIP_X_F_QF",  AllVFNRCLIPVectors>;
}

let Predicates = [HasVendorXSfvfhbfmin] in {
defm : XPatConversionWF_VF<"int_riscv_sf_vfwcvt_f_bf16_v", "PseudoVFWCVT_F_BF16">;
defm : XPatConversionVF_WF<"int_riscv_sf_vfncvt_bf16_f_w", "PseudoVFNCVT_BF16_F">;
}

let Predicates = [HasVendorXSfvfwmaccqqq] in {
  foreach fvtiToFWti = AllWidenableBFloatToFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    def: VPatTernaryNoMaskWithPolicy<"int_riscv_sf_vfwmacc_4x4x4", "PseudoVFWMACC4X4X4", "VV",
                                       fwti.Vector, vbfloat16m1_t, fvti.Vector,
                                       fvti.Log2SEW, fvti.LMul,
                                       fwti.RegClass, VR, fvti.RegClass>;
  }
}

let isBarrier = 1, isReturn = 1, isTerminator = 1,
    hasNoSchedulingInfo = 1 in {
def MNRET : Priv<"mnret", 0b0111000>, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b00010;
}
} // let isBarrier = 1, ...

let hasNoSchedulingInfo = 1, hasSideEffects = 1,
    mayLoad = 0, mayStore = 0 in {
def CFLUSH_D_L1
    : RVInstR<0b1111110, 0b000, OPC_SYSTEM, (outs), (ins GPR:$rs1),
              "cflush.d.l1", "$rs1">, Sched<[]> {
  let rd = 0;
  let rs2 = 0b00000;
}

def CDISCARD_D_L1
    : RVInstR<0b1111110, 0b000, OPC_SYSTEM, (outs), (ins GPR:$rs1),
              "cdiscard.d.l1", "$rs1">, Sched<[]> {
  let rd = 0;
  let rs2 = 0b00010;
}

def CFLUSH_I_L1
    : RVInstR<0b1111110, 0b000, OPC_SYSTEM, (outs), (ins),
              "cflush.i.l1", "">, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b00001;
}
} // hasNoSchedulingInfo = 1, ...
#endif // SIFIVE_CUSTOMIZATION
let Predicates = [HasVendorXSfcie] in {
let hasSideEffects = 1, mayLoad = 0, mayStore = 0, DecoderNamespace = "XSfcie" in {
def SF_CFLUSH_D_L1 : RVInstI<0b000, OPC_SYSTEM, (outs), (ins GPR:$rs1), "cflush.d.l1","$rs1">,
                             Sched<[]> {
  let rd = 0;
  let imm12 = {0b1111,0b1100,0b0000};
}

def SF_CDISCARD_D_L1 : RVInstI<0b000, OPC_SYSTEM, (outs), (ins GPR:$rs1), "cdiscard.d.l1","$rs1">,
                               Sched<[]> {
  let rd = 0;
  let imm12 = {0b1111,0b1100,0b0010};
}

def SF_CEASE : RVInstI<0b000, OPC_SYSTEM, (outs), (ins), "cease","">,  Sched<[]> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = {0b0011,0b0000,0b0101};
}
}
def : InstAlias<"cflush.d.l1", (SF_CFLUSH_D_L1 X0)>;
def : InstAlias<"cdiscard.d.l1", (SF_CDISCARD_D_L1 X0)>;
} // Predicates = [HasVendorXScie]
