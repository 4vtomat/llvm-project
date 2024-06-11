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

// FIXME: might need to put more constraint
let RVVConstraint = VMConstraint, Predicates = [HasVendorXSfvqdotq],
    DecoderNamespace = "XSfvqdotq" in {
  def VQDOT_VV   : VALUVV<0b101100, OPMVV, "sf.vqdot.vv">;
  def VQDOT_VX   : VALUVX<0b101100, OPMVX, "sf.vqdot.vx">;
  def VQDOTU_VV  : VALUVV<0b101000, OPMVV, "sf.vqdotu.vv">;
  def VQDOTU_VX  : VALUVX<0b101000, OPMVX, "sf.vqdotu.vx">;
  def VQDOTSU_VV : VALUVV<0b101010, OPMVV, "sf.vqdotsu.vv">;
  def VQDOTSU_VX : VALUVX<0b101010, OPMVX, "sf.vqdotsu.vx">;
  def VQDOTUS_VX : VALUVX<0b101110, OPMVX, "sf.vqdotus.vx">;
} // Predicates = [HasVendorXSfvqdotq]

let Predicates = [HasVendorXSfvfhbfmin] in {
def : InstAlias<"sf.vfwcvt.f.bf16.v $vd, $vs2$vm",
                (VFWCVTBF16_F_F_V VR:$vd, VR:$vs2, VMaskOp:$vm)>;
def : InstAlias<"sf.vfwcvt.f.bf16.v $vd, $vs2",
                (VFWCVTBF16_F_F_V VR:$vd, VR:$vs2, zero_reg)>;
def : InstAlias<"sf.vfncvt.bf16.f.w $vd, $vs2$vm",
                (VFNCVTBF16_F_F_W VR:$vd, VR:$vs2, VMaskOp:$vm)>;
def : InstAlias<"sf.vfncvt.bf16.f.w $vd, $vs2",
                (VFNCVTBF16_F_F_W VR:$vd, VR:$vs2, zero_reg)>;
}

class CustomSiFiveVF<string opcodestr,  RISCVVFormat opv, bits<6> funct6>
  : RVInstVXCustom2<funct6, opv, (outs VR:$vd),
                    (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm), opcodestr,
                    "$vd, $vs2, $rs1$vm"> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

defvar VQDOTLMuls = [VI32MF2, VI32M1, VI32M2, VI32M4, VI32M8];

// These VPat definitions are for vqdot because they have a different operand
// order with other ternary instructions (i.e. vop.vx vd, vs2, rs1)
multiclass VPatTernaryV_VX_AAXASwapped<string intrinsic, string instruction,
                                       list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    let Predicates = GetVTypePredicates<vti>.Predicates in
    // we have op1 and op2 swapped here
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "V"#vti.ScalarSuffix,
                                 vti.Vector, vti.Vector, vti.Scalar, vti.Mask,
                                 vti.Log2SEW, vti.LMul, vti.RegClass,
                                 vti.RegClass, vti.ScalarRegClass>;
}

multiclass VPatTernaryV_VV_VX_AAXASwapped<string intrinsic, string instruction,
                              list<VTypeInfo> vtilist>
    // keep the original VV because the order doesn't matter
    : VPatTernaryV_VV_AAXA<intrinsic, instruction, vtilist>,
      VPatTernaryV_VX_AAXASwapped<intrinsic, instruction, vtilist>;
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
class PayloadOp<int bitsNum> : RISCVOp, TImmLeaf<XLenVT, "return isUInt<" # bitsNum # ">(Imm);"> {
  let ParserMatchClass = UImmAsmOperand<bitsNum>;
  let DecoderMethod = "decodeUImmOperand<"# bitsNum # ">";
  let OperandType = "OPERAND_UIMM" # bitsNum;
}

def payload1 : PayloadOp<1>;
def payload2 : PayloadOp<2>;
def payload5 : PayloadOp<5>;

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

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
class CustomSiFiveVMACC<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVCCustom2<funct6{5-2}, opv.Value, (outs VR:$rd), (ins VR:$rs1, VR:$rs2),
                      opcodestr, "$rd, $rs1, $rs2"> {
  let vm = 1;
  let funct6_lo2 = funct6{1-0};
}
}

class CustomSiFiveVFNRCLIP<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : VALUVF<funct6, opv, opcodestr> {
  let Inst{6-0} = OPC_CUSTOM_2.Value;
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

let mayLoad = 0, mayStore = 0, hasSideEffects = 0,
    Predicates = [HasVendorXSfvfexpa], DecoderNamespace = "XSfvfexpa" in {
  def VFEXPA_V : VALUVs2<0b010011, 0b00110, OPFVV, "sf.vfexpa.v">;
}

let Predicates = [HasVendorXSfvqmaccdod], DecoderNamespace = "XSfvqmaccdod" in {
  def VQMACCU_2x8x2  : CustomSiFiveVMACC<0b101100, OPMVV, "sf.vqmaccu.2x8x2">;
  def VQMACC_2x8x2   : CustomSiFiveVMACC<0b101101, OPMVV, "sf.vqmacc.2x8x2">;
  def VQMACCUS_2x8x2 : CustomSiFiveVMACC<0b101110, OPMVV, "sf.vqmaccus.2x8x2">;
  def VQMACCSU_2x8x2 : CustomSiFiveVMACC<0b101111, OPMVV, "sf.vqmaccsu.2x8x2">;
}

let Predicates = [HasVendorXSfvqmaccqoq], DecoderNamespace = "XSfvqmaccqoq" in {
  def VQMACCU_4x8x4  : CustomSiFiveVMACC<0b111100, OPMVV, "sf.vqmaccu.4x8x4">;
  def VQMACC_4x8x4   : CustomSiFiveVMACC<0b111101, OPMVV, "sf.vqmacc.4x8x4">;
  def VQMACCUS_4x8x4 : CustomSiFiveVMACC<0b111110, OPMVV, "sf.vqmaccus.4x8x4">;
  def VQMACCSU_4x8x4 : CustomSiFiveVMACC<0b111111, OPMVV, "sf.vqmaccsu.4x8x4">;
}

let Predicates = [HasVendorXSfvfwmaccqqq], DecoderNamespace = "XSfvfwmaccqqq" in {
  def VFWMACC_4x4x4 : CustomSiFiveVMACC<0b111100, OPFVV, "sf.vfwmacc.4x4x4">;
}

let Predicates = [HasVendorXSfvfnrclipxfqf], DecoderNamespace = "XSfvfnrclipxfqf" in {
  def VFNRCLIP_XU_F_QF : CustomSiFiveVFNRCLIP<0b100010, OPFVF, "sf.vfnrclip.xu.f.qf">;
  def VFNRCLIP_X_F_QF : CustomSiFiveVFNRCLIP<0b100011, OPFVF, "sf.vfnrclip.x.f.qf">;
}

class VPseudoVC_X<Operand OpClass, DAGOperand RS1Class> :
      Pseudo<(outs),
             (ins OpClass:$op1, payload5:$rs2, payload5:$rd, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_XV<Operand OpClass, VReg RS2Class, DAGOperand RS1Class> :
      Pseudo<(outs),
             (ins OpClass:$op1, payload5:$rd, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_XVV<Operand OpClass, VReg RDClass, VReg RS2Class,
                    DAGOperand RS1Class> :
      Pseudo<(outs),
             (ins OpClass:$op1, RDClass:$rd, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_V_X<Operand OpClass, VReg RDClass, DAGOperand RS1Class> :
      Pseudo<(outs RDClass:$rd),
             (ins OpClass:$op1, payload5:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_V_XV<Operand OpClass, VReg RDClass, VReg RS2Class,
                     DAGOperand RS1Class> :
      Pseudo<(outs RDClass:$rd),
             (ins OpClass:$op1, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class VPseudoVC_V_XVV<Operand OpClass, VReg RDClass, VReg RS2Class,
                      DAGOperand RS1Class> :
      Pseudo<(outs RDClass:$rd),
             (ins OpClass:$op1, RDClass:$rs3, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, ixlenimm:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

multiclass VPseudoVC_X<LMULInfo m, DAGOperand RS1Class,
                       Operand OpClass = payload2> {
  let VLMul = m.value in {
    let Defs = [VCIX_STATE], Uses = [VCIX_STATE] in {
      def "PseudoVC_" # NAME # "_SE_" # m.MX
        : VPseudoVC_X<OpClass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_" # NAME # "_" # m.MX)]>;
      def "PseudoVC_V_" # NAME # "_SE_" # m.MX
        : VPseudoVC_V_X<OpClass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
    }
    def "PseudoVC_V_" # NAME # "_" # m.MX
      : VPseudoVC_V_X<OpClass, m.vrclass, RS1Class>,
        Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
  }
}

multiclass VPseudoVC_XV<LMULInfo m, DAGOperand RS1Class,
                        Operand OpClass = payload2> {
  let VLMul = m.value in {
    let Defs = [VCIX_STATE], Uses = [VCIX_STATE] in {
      def "PseudoVC_" # NAME # "_SE_" # m.MX
        : VPseudoVC_XV<OpClass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_" # NAME # "_" # m.MX)]>;
      def "PseudoVC_V_" # NAME # "_SE_" # m.MX
        : VPseudoVC_V_XV<OpClass, m.vrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
    }
    def "PseudoVC_V_" # NAME # "_" # m.MX
      : VPseudoVC_V_XV<OpClass, m.vrclass, m.vrclass, RS1Class>,
        Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
  }
}

multiclass VPseudoVC_XVV<LMULInfo m, DAGOperand RS1Class,
                         Operand OpClass = payload2> {
  let VLMul = m.value in {
    let Defs = [VCIX_STATE], Uses = [VCIX_STATE] in {
      def "PseudoVC_" # NAME # "_SE_" # m.MX
        : VPseudoVC_XVV<OpClass, m.vrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_" # NAME # "_" # m.MX)]>;
      def "PseudoVC_V_" # NAME # "_SE_" # m.MX
        : VPseudoVC_V_XVV<OpClass, m.vrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
    }
    def "PseudoVC_V_" # NAME # "_" # m.MX
      : VPseudoVC_V_XVV<OpClass, m.vrclass, m.vrclass, RS1Class>,
        Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
  }
}

multiclass VPseudoVC_XVW<LMULInfo m, DAGOperand RS1Class,
                         Operand OpClass = payload2> {
  let VLMul = m.value in {
    let Defs = [VCIX_STATE], Uses = [VCIX_STATE] in
    def "PseudoVC_" # NAME # "_SE_" # m.MX
      : VPseudoVC_XVV<OpClass, m.wvrclass, m.vrclass, RS1Class>,
        Sched<[!cast<SchedWrite>("WriteVC_" # NAME # "_" # m.MX)]>;
    let Constraints = "@earlyclobber $rd, $rd = $rs3" in {
      let Defs = [VCIX_STATE], Uses = [VCIX_STATE] in
      def "PseudoVC_V_" # NAME # "_SE_" # m.MX
        : VPseudoVC_V_XVV<OpClass, m.wvrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
      def "PseudoVC_V_" # NAME # "_" # m.MX
        : VPseudoVC_V_XVV<OpClass, m.wvrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
    }
  }
}

multiclass VPseudoSiFiveVMACC<string mx, VReg vd_type, VReg vs2_type,
                              string Constraint = ""> {
  def "Pseudo" # NAME # "_" # mx
      : VPseudoTernaryNoMaskWithPolicy<vd_type, V_M1.vrclass, vs2_type, Constraint>;
}

multiclass VPseudoSiFiveVQMACCDOD<string Constraint = ""> {
  foreach m = MxListVF8 in
    let VLMul = m.value in
    defm NAME : VPseudoSiFiveVMACC<m.MX, m.vrclass, m.vrclass, Constraint>;
}

multiclass VPseudoSiFiveVQMACCQOQ<string Constraint = ""> {
  foreach m = [V_MF2, V_M1, V_M2, V_M4] in
    let VLMul = m.value in
    defm NAME : VPseudoSiFiveVMACC<m.MX, m.wvrclass, m.vrclass, Constraint>;
}

multiclass VPseudoSiFiveVFWMACC<string Constraint = ""> {
  foreach m = MxListVF2 in
    let VLMul = m.value in
    defm NAME : VPseudoSiFiveVMACC<m.MX, m.wvrclass, m.vrclass, Constraint>;
}

multiclass VPseudoSiFiveVFNRCLIP<string Constraint = "@earlyclobber $rd"> {
  foreach i = 0-4 in
    let hasSideEffects = 0 in
      defm "Pseudo" # NAME : VPseudoBinaryRoundingMode<MxListW[i].vrclass,
                                                       MxListVF4[i].vrclass,
                                                       FPR32, MxListW[i],
                                                       Constraint, /*sew*/0,
                                                       UsesVXRM=0>;
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

let Predicates = [HasVendorXSfvqmaccdod] in {
  defm VQMACCU_2x8x2  : VPseudoSiFiveVQMACCDOD;
  defm VQMACC_2x8x2   : VPseudoSiFiveVQMACCDOD;
  defm VQMACCUS_2x8x2 : VPseudoSiFiveVQMACCDOD;
  defm VQMACCSU_2x8x2 : VPseudoSiFiveVQMACCDOD;
}

let Predicates = [HasVendorXSfvqmaccqoq] in {
  defm VQMACCU_4x8x4  : VPseudoSiFiveVQMACCQOQ;
  defm VQMACC_4x8x4   : VPseudoSiFiveVQMACCQOQ;
  defm VQMACCUS_4x8x4 : VPseudoSiFiveVQMACCQOQ;
  defm VQMACCSU_4x8x4 : VPseudoSiFiveVQMACCQOQ;
}

let Predicates = [HasVendorXSfvfwmaccqqq] in {
  defm VFWMACC_4x4x4 : VPseudoSiFiveVFWMACC;
}

let Predicates = [HasVendorXSfvfnrclipxfqf] in {
  defm VFNRCLIP_XU_F_QF : VPseudoSiFiveVFNRCLIP;
  defm VFNRCLIP_X_F_QF : VPseudoSiFiveVFNRCLIP;
}

let Predicates = [HasVendorXSfvfexpa] in {
  defm PseudoVFEXPA : VPseudoVRCP_V;
}

// SDNode
def SDT_SF_VC_V_X : SDTypeProfile<1, 4, [SDTCisVec<0>,
                                         SDTCisVT<1, XLenVT>,
                                         SDTCisSameAs<1, 2>,
                                         SDTCisSameAs<1, 3>,
                                         SDTCisSameAs<1, 4>]>;

def SDT_SF_VC_XV : SDTypeProfile<0, 5, [SDTCisSameAs<0, 1>,
                                        SDTCisVec<2>,
                                        SDTCisSameAs<0, 4>,
                                        SDTCisVT<0, XLenVT>]>;

def SDT_SF_VC_V_XV : SDTypeProfile<1, 4, [SDTCisVec<0>,
                                          SDTCisVT<1, XLenVT>,
                                          SDTCisSameAs<0, 2>,
                                          SDTCisSameAs<1, 4>]>;

def SDT_SF_VC_XVV : SDTypeProfile<0, 5, [SDTCisVT<0, XLenVT>,
                                         SDTCisVec<1>,
                                         SDTCisSameAs<1, 2>,
                                         SDTCisSameAs<0, 4>]>;

def SDT_SF_VC_V_XVV : SDTypeProfile<1, 5, [SDTCisVec<0>,
                                           SDTCisVT<1, XLenVT>,
                                           SDTCisSameAs<0, 2>,
                                           SDTCisSameAs<0, 3>,
                                           SDTCisSameAs<1, 5>]>;

def SDT_SF_VC_XVW : SDTypeProfile<0, 5, [SDTCisVT<0, XLenVT>,
                                         SDTCisVec<1>, SDTCisVec<2>,
                                         SDTCisSameAs<0, 4>]>;

def SDT_SF_VC_V_XVW : SDTypeProfile<1, 5, [SDTCisVec<0>,
                                           SDTCisVT<1, XLenVT>,
                                           SDTCisSameAs<0, 2>,
                                           SDTCisVec<3>,
                                           SDTCisSameAs<1, 5>]>;

def sf_vc_v_x_se : SDNode<"RISCVISD::SF_VC_V_X_SE", SDT_SF_VC_V_X, [SDNPHasChain]>;
def sf_vc_v_i_se : SDNode<"RISCVISD::SF_VC_V_I_SE", SDT_SF_VC_V_X, [SDNPHasChain]>;
def sf_vc_vv_se : SDNode<"RISCVISD::SF_VC_VV_SE", SDT_SF_VC_XV, [SDNPHasChain]>;
def sf_vc_xv_se : SDNode<"RISCVISD::SF_VC_XV_SE", SDT_SF_VC_XV, [SDNPHasChain]>;
def sf_vc_iv_se : SDNode<"RISCVISD::SF_VC_IV_SE", SDT_SF_VC_XV, [SDNPHasChain]>;
def sf_vc_fv_se : SDNode<"RISCVISD::SF_VC_FV_SE", SDT_SF_VC_XV, [SDNPHasChain]>;
def sf_vc_v_vv_se : SDNode<"RISCVISD::SF_VC_V_VV_SE", SDT_SF_VC_V_XV, [SDNPHasChain]>;
def sf_vc_v_xv_se : SDNode<"RISCVISD::SF_VC_V_XV_SE", SDT_SF_VC_V_XV, [SDNPHasChain]>;
def sf_vc_v_iv_se : SDNode<"RISCVISD::SF_VC_V_IV_SE", SDT_SF_VC_V_XV, [SDNPHasChain]>;
def sf_vc_v_fv_se : SDNode<"RISCVISD::SF_VC_V_FV_SE", SDT_SF_VC_V_XV, [SDNPHasChain]>;
def sf_vc_vvv_se : SDNode<"RISCVISD::SF_VC_VVV_SE", SDT_SF_VC_XVV, [SDNPHasChain]>;
def sf_vc_xvv_se : SDNode<"RISCVISD::SF_VC_XVV_SE", SDT_SF_VC_XVV, [SDNPHasChain]>;
def sf_vc_ivv_se : SDNode<"RISCVISD::SF_VC_IVV_SE", SDT_SF_VC_XVV, [SDNPHasChain]>;
def sf_vc_fvv_se : SDNode<"RISCVISD::SF_VC_FVV_SE", SDT_SF_VC_XVV, [SDNPHasChain]>;
def sf_vc_v_vvv_se : SDNode<"RISCVISD::SF_VC_V_VVV_SE", SDT_SF_VC_V_XVV, [SDNPHasChain]>;
def sf_vc_v_xvv_se : SDNode<"RISCVISD::SF_VC_V_XVV_SE", SDT_SF_VC_V_XVV, [SDNPHasChain]>;
def sf_vc_v_ivv_se : SDNode<"RISCVISD::SF_VC_V_IVV_SE", SDT_SF_VC_V_XVV, [SDNPHasChain]>;
def sf_vc_v_fvv_se : SDNode<"RISCVISD::SF_VC_V_FVV_SE", SDT_SF_VC_V_XVV, [SDNPHasChain]>;
def sf_vc_vvw_se : SDNode<"RISCVISD::SF_VC_VVW_SE", SDT_SF_VC_XVW, [SDNPHasChain]>;
def sf_vc_xvw_se : SDNode<"RISCVISD::SF_VC_XVW_SE", SDT_SF_VC_XVW, [SDNPHasChain]>;
def sf_vc_ivw_se : SDNode<"RISCVISD::SF_VC_IVW_SE", SDT_SF_VC_XVW, [SDNPHasChain]>;
def sf_vc_fvw_se : SDNode<"RISCVISD::SF_VC_FVW_SE", SDT_SF_VC_XVW, [SDNPHasChain]>;
def sf_vc_v_vvw_se : SDNode<"RISCVISD::SF_VC_V_VVW_SE", SDT_SF_VC_V_XVW, [SDNPHasChain]>;
def sf_vc_v_xvw_se : SDNode<"RISCVISD::SF_VC_V_XVW_SE", SDT_SF_VC_V_XVW, [SDNPHasChain]>;
def sf_vc_v_ivw_se : SDNode<"RISCVISD::SF_VC_V_IVW_SE", SDT_SF_VC_V_XVW, [SDNPHasChain]>;
def sf_vc_v_fvw_se : SDNode<"RISCVISD::SF_VC_V_FVW_SE", SDT_SF_VC_V_XVW, [SDNPHasChain]>;

class VPatVC_OP4_ISD<SDPatternOperator op,
                     string inst,
                     ValueType op2_type,
                     ValueType op3_type,
                     ValueType op4_type,
                     int sew,
                     DAGOperand op2_kind,
                     DAGOperand op3_kind,
                     DAGOperand op4_kind,
                     Operand op1_kind = payload2> :
  Pat<(op
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

class VPatVC_V_OP4_ISD<SDPatternOperator op,
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
  Pat<(result_type (op
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


class VPatVC_V_OP3_ISD<SDPatternOperator op,
                       string inst,
                       ValueType result_type,
                       ValueType op2_type,
                       ValueType op3_type,
                       int sew,
                       DAGOperand op2_kind,
                       DAGOperand op3_kind,
                       Operand op1_kind = payload2> :
  Pat<(result_type (op
                    (XLenVT   op1_kind:$op1),
                    (op2_type op2_kind:$op2),
                    (op3_type op3_kind:$op3),
                    VLOpFrag)),
                   (!cast<Instruction>(inst)
                    (XLenVT   op1_kind:$op1),
                    (op2_type op2_kind:$op2),
                    (op3_type op3_kind:$op3),
                    GPR:$vl, sew)>;

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
  def : VPatVC_V_OP3_ISD<!cast<SDPatternOperator>("sf_vc_v_" # intrinsic_suffix # "_se"),
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
  def : VPatVC_OP4_ISD<!cast<SDPatternOperator>("sf_vc_" # intrinsic_suffix # "_se"),
                   "PseudoVC_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                   XLenVT, vti.Vector, type, vti.Log2SEW,
                   payload5, vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP3_ISD<!cast<SDPatternOperator>("sf_vc_v_" # intrinsic_suffix # "_se"),
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
  def : VPatVC_OP4_ISD<!cast<SDPatternOperator>("sf_vc_" # intrinsic_suffix # "_se"),
                   "PseudoVC_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                   wti.Vector, vti.Vector, type, vti.Log2SEW,
                   wti.RegClass, vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP4_ISD<!cast<SDPatternOperator>("sf_vc_v_" # intrinsic_suffix # "_se"),
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

multiclass VPatVMACC<string intrinsic, string instruction, string kind,
                     list<VTypeInfoToWide> info_pairs, ValueType vec_m1> {
  foreach pair = info_pairs in {
    defvar VdInfo = pair.Wti;
    defvar Vs2Info = pair.Vti;
    let Predicates = [HasVInstructions] in
    def : VPatTernaryNoMaskWithPolicy<"int_riscv_sf_" # intrinsic,
                                      "Pseudo" # instruction, kind, VdInfo.Vector,
                                      vec_m1, Vs2Info.Vector,
                                      Vs2Info.Log2SEW, Vs2Info.LMul,
                                      VdInfo.RegClass, VR, Vs2Info.RegClass>;
  }
}

defset list<VTypeInfoToWide> VQMACCDODInfoPairs = {
  def : VTypeInfoToWide<VI8M1, VI32M1>;
  def : VTypeInfoToWide<VI8M2, VI32M2>;
  def : VTypeInfoToWide<VI8M4, VI32M4>;
  def : VTypeInfoToWide<VI8M8, VI32M8>;
}

defset list<VTypeInfoToWide> VQMACCQOQInfoPairs = {
  def : VTypeInfoToWide<VI8MF2, VI32M1>;
  def : VTypeInfoToWide<VI8M1, VI32M2>;
  def : VTypeInfoToWide<VI8M2, VI32M4>;
  def : VTypeInfoToWide<VI8M4, VI32M8>;
}

multiclass VPatVQMACCDOD<string intrinsic, string instruction, string kind>
    : VPatVMACC<intrinsic, instruction, kind, VQMACCDODInfoPairs, vint8m1_t>;

multiclass VPatVQMACCQOQ<string intrinsic, string instruction, string kind>
    : VPatVMACC<intrinsic, instruction, kind, VQMACCQOQInfoPairs, vint8m1_t>;

multiclass VPatVFWMACC<string intrinsic, string instruction, string kind>
    : VPatVMACC<intrinsic, instruction, kind, AllWidenableBFloatToFloatVectors,
                vbfloat16m1_t>;

defset list<VTypeInfoToWide> VFNRCLIPInfoPairs = {
  def : VTypeInfoToWide<VI8MF8, VF32MF2>;
  def : VTypeInfoToWide<VI8MF4, VF32M1>;
  def : VTypeInfoToWide<VI8MF2, VF32M2>;
  def : VTypeInfoToWide<VI8M1,  VF32M4>;
  def : VTypeInfoToWide<VI8M2,  VF32M8>;
}

multiclass VPatVFNRCLIP<string intrinsic, string instruction> {
  foreach pair = VFNRCLIPInfoPairs in {
    defvar Vti = pair.Vti;
    defvar Wti = pair.Wti;
    defm : VPatBinaryRoundingMode<"int_riscv_sf_" # intrinsic,
                                  "Pseudo" # instruction # "_" # Vti.LMul.MX,
                                  Vti.Vector, Wti.Vector, Wti.Scalar, Vti.Mask,
                                  Vti.Log2SEW, Vti.RegClass,
                                  Wti.RegClass, Wti.ScalarRegClass>;
  }
}

let Predicates = [HasVendorXSfvcp] in {
  foreach vti = AllIntegerVectors in {
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
  foreach VtiToWti = AllWidenableIntVectors in {
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

let Predicates = [HasVendorXSfvqmaccdod] in {
  defm : VPatVQMACCDOD<"vqmaccu_2x8x2", "VQMACCU", "2x8x2">;
  defm : VPatVQMACCDOD<"vqmacc_2x8x2", "VQMACC", "2x8x2">;
  defm : VPatVQMACCDOD<"vqmaccus_2x8x2", "VQMACCUS", "2x8x2">;
  defm : VPatVQMACCDOD<"vqmaccsu_2x8x2", "VQMACCSU", "2x8x2">;
}

let Predicates = [HasVendorXSfvqmaccqoq] in {
  defm : VPatVQMACCQOQ<"vqmaccu_4x8x4", "VQMACCU", "4x8x4">;
  defm : VPatVQMACCQOQ<"vqmacc_4x8x4", "VQMACC", "4x8x4">;
  defm : VPatVQMACCQOQ<"vqmaccus_4x8x4", "VQMACCUS", "4x8x4">;
  defm : VPatVQMACCQOQ<"vqmaccsu_4x8x4", "VQMACCSU", "4x8x4">;
}

#ifdef SIFIVE_CUSTOMIZATION
multiclass VPseudoSiFiveVQDOT<string Constraint = ""> {
  foreach m = MxListVF4 in {
    defm "Pseudo" # NAME # "_VV" : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, m.vrclass, m, Constraint>;
    defm "Pseudo" # NAME # "_VX" : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, GPR, m, Constraint>;
  }
}

let Predicates = [HasVendorXSfvqdotq] in {
  defm VQDOT  : VPseudoSiFiveVQDOT;
  defm VQDOTU : VPseudoSiFiveVQDOT;
  defm VQDOTSU: VPseudoSiFiveVQDOT;

  // VQDOTUS does not have a VV variant
  foreach m = MxListVF4 in {
    defm "PseudoVQDOTUS_VX" : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, GPR, m>;
  }
}

let Predicates = [HasVendorXSfvqdotq] in {
  defm : VPatTernaryV_VV_VX_AAXASwapped<"int_riscv_sf_vqdot", "PseudoVQDOT", VQDOTLMuls>;
  defm : VPatTernaryV_VV_VX_AAXASwapped<"int_riscv_sf_vqdotu", "PseudoVQDOTU", VQDOTLMuls>;
  defm : VPatTernaryV_VV_VX_AAXASwapped<"int_riscv_sf_vqdotsu", "PseudoVQDOTSU", VQDOTLMuls>;
  defm : VPatTernaryV_VX_AAXASwapped<"int_riscv_sf_vqdotus", "PseudoVQDOTUS", VQDOTLMuls>;
}

def : Pat<(XLenVT (int_riscv_vgetvxrm)), (ReadVXRM)>;

def : Pat<(int_riscv_vsetvxrm (XLenVT GPR:$rm)),
          (WriteVXRM $rm)>;

def : Pat<(int_riscv_vsetvxrm uimm5:$rm),
          (WriteVXRMImm uimm5:$rm)>;

class XPatUnaryNoMask<string intrinsic_name,
                      string inst,
                      string kind,
                      ValueType result_type,
                      ValueType op2_type,
                      int log2sew,
                      LMULInfo vlmul,
                      VReg result_reg_class,
                      VReg op2_reg_class,
                      bit isSEWAware = 0> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   VLOpFrag)),
                   (!cast<Instruction>(!if(isSEWAware,
                                           inst#"_"#kind#"_"#vlmul.MX#"_E"#!shl(1, log2sew),
                                           inst#"_"#kind#"_"#vlmul.MX))
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   FRM_DYN,
                   GPR:$vl, log2sew, TU_MU)>;

class XPatUnaryMask<string intrinsic_name,
                    string inst,
                    string kind,
                    ValueType result_type,
                    ValueType op2_type,
                    ValueType mask_type,
                    int log2sew,
                    LMULInfo vlmul,
                    VReg result_reg_class,
                    VReg op2_reg_class,
                    bit isSEWAware = 0> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name#"_mask")
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   (mask_type V0),
                   VLOpFrag, (XLenVT timm:$policy))),
                   (!cast<Instruction>(!if(isSEWAware,
                                           inst#"_"#kind#"_"#vlmul.MX#"_E"#!shl(1, log2sew)#"_MASK",
                                           inst#"_"#kind#"_"#vlmul.MX#"_MASK"))
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   (mask_type V0), FRM_DYN, GPR:$vl, log2sew, (XLenVT timm:$policy))>;

multiclass XPatConversionTA<string intrinsic,
                            string inst,
                            string kind,
                            ValueType result_type,
                            ValueType op1_type,
                            ValueType mask_type,
                            int log2sew,
                            LMULInfo vlmul,
                            VReg result_reg_class,
                            VReg op1_reg_class,
                            bit isSEWAware = 0> {
  def : XPatUnaryNoMask<intrinsic, inst, kind, result_type, op1_type,
                        log2sew, vlmul, result_reg_class, op1_reg_class, isSEWAware>;
  def : XPatUnaryMask<intrinsic, inst, kind, result_type, op1_type,
                      mask_type, log2sew, vlmul, result_reg_class, op1_reg_class, isSEWAware>;
}
multiclass XPatConversionVF_WF <string intrinsic, string instruction, bit isSEWAware = 0> {
  foreach fvtiToFWti = AllWidenableBFloatToFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;

    defm : XPatConversionTA<intrinsic, instruction, "W",
                            fvti.Vector, fwti.Vector, fvti.Mask, fvti.Log2SEW,
                            fvti.LMul, fvti.RegClass, fwti.RegClass, isSEWAware>;
  }
}

let Predicates = [HasVendorXSfvfhbfmin] in {
defm : VPatConversionWF_VF_BF<"int_riscv_sf_vfwcvt_f_bf16_v", "PseudoVFWCVTBF16_F_F", isSEWAware=1>;
defm : XPatConversionVF_WF<"int_riscv_sf_vfncvt_bf16_f_w", "PseudoVFNCVTBF16_F_F", isSEWAware=1>;
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

let Predicates = [HasVendorXSfpgflushdlone] in
def SF_PGFLUSH_D_L1 : RVInstR<0b1111110, 0b000, OPC_SYSTEM, (outs), (ins),
                              "sf.pgflush.d.l1", "">, Sched<[]> {
  let rd = 0;
  let rs1 = 0;
  let rs2 = 0b00011;
}
} // hasNoSchedulingInfo = 1, ...
#endif // SIFIVE_CUSTOMIZATION

let Predicates = [HasVendorXSfvfwmaccqqq] in {
  defm : VPatVFWMACC<"vfwmacc_4x4x4", "VFWMACC", "4x4x4">;
}

let Predicates = [HasVendorXSfvfnrclipxfqf] in {
  defm : VPatVFNRCLIP<"vfnrclip_xu_f_qf", "VFNRCLIP_XU_F_QF">;
  defm : VPatVFNRCLIP<"vfnrclip_x_f_qf", "VFNRCLIP_X_F_QF">;
}

let Predicates = [HasVendorXSiFivecdiscarddlone] in {
  let hasNoSchedulingInfo = 1, hasSideEffects = 1, mayLoad = 0, mayStore = 0,
      DecoderNamespace = "XSiFivecdiscarddlone" in
  def SF_CDISCARD_D_L1
      : RVInstIUnary<0b111111000010, 0b000, OPC_SYSTEM, (outs), (ins GPR:$rs1),
                     "sf.cdiscard.d.l1", "$rs1">, Sched<[]> {
    let rd = 0;
  }
  def : InstAlias<"sf.cdiscard.d.l1", (SF_CDISCARD_D_L1 X0)>;
} // Predicates = [HasVendorXSifivecdiscarddlone]

let Predicates = [HasVendorXSiFivecflushdlone] in {
  let hasNoSchedulingInfo = 1, hasSideEffects = 1, mayLoad = 0, mayStore = 0,
      DecoderNamespace = "XSiFivecflushdlone" in
  def SF_CFLUSH_D_L1
      : RVInstIUnary<0b111111000000, 0b000, OPC_SYSTEM, (outs), (ins GPR:$rs1),
                     "sf.cflush.d.l1", "$rs1">, Sched<[]> {
    let rd = 0;
  }
  def : InstAlias<"sf.cflush.d.l1", (SF_CFLUSH_D_L1 X0)>;
} // Predicates = [HasVendorXSifivecflushdlone]

let Predicates = [HasVendorXSfcease] in {
  let hasNoSchedulingInfo = 1, hasSideEffects = 1, mayLoad = 0, mayStore = 0,
      DecoderNamespace = "XSfcease" in
  def SF_CEASE : RVInstIUnary<0b001100000101, 0b000, OPC_SYSTEM, (outs), (ins),
                              "sf.cease", "">, Sched<[]> {
    let rs1 = 0b00000;
    let rd = 0b00000;
  }
}

let Predicates = [HasVendorXSfvfexpa] in {
  defm : VPatUnaryV_V<"int_riscv_sf_vfexpa", "PseudoVFEXPA",
                      [VF32MF2, VF32M1, VF32M2, VF32M4, VF32M8], isSEWAware=1>;
}

let Predicates = [HasVendorXSfvfexpa, HasVInstructionsF16] in {
  defm : VPatUnaryV_V<"int_riscv_sf_vfexpa", "PseudoVFEXPA",
                      [VF16MF4, VF16MF2, VF16M1, VF16M2, VF16M4, VF16M8],
                      isSEWAware=1>;
}

let Predicates = [HasVendorXSfvfexpa64e] in {
  defm : VPatUnaryV_V<"int_riscv_sf_vfexpa", "PseudoVFEXPA",
                      [VF64M1, VF64M2, VF64M4, VF64M8], isSEWAware=1>;
}

#ifdef SIFIVE_CUSTOMIZATION
def MammothWWEEAsmOperand : AsmOperandClass {
  let Name = "MammothWWEE";
  let ParserMethod = "parseMammothWWEE";
  let DiagnosticType = "InvalidMammothWWEE";
  let RenderMethod = "addVTypeIOperands";
}

def MammothWWEEOp : RISCVOp {
  let ParserMatchClass = MammothWWEEAsmOperand;
  let PrintMethod = "printMammothWWEE";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    if (!isUInt<32>(Imm))
      return false;
    return RISCVVType::isValidMammothWWEE(Imm);
  }];
}

class RVInstSetSingle<dag outs, dag ins, bits<5> rs2, string opcodestr,
                      string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatI> {
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = 0b1000010;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = OPCFG.Value;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Defs = [VTYPE, VL];
}

class RVInstTileMemOp<dag outs, dag ins, bits<3> nf, bits<7> opcode,
                      string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;

  let Inst{31-29} = nf;
  let Inst{28} = 1;
  let Inst{27-26} = MOPLDUnitStride.Value;
  let Inst{25} = 1;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = 0b00000;
  let Inst{6-0} = opcode;

  let Uses = [VTYPE, VL];
}

class RVInstTileLoad<bits<3> nf, string opcodestr>
    : RVInstTileMemOp<(outs), (ins GPR:$rs2, GPRMemZeroOffset:$rs1), nf, OPC_LOAD_FP.Value,
                      opcodestr, "$rs2, ${rs1}">;

class RVInstTileStore<bits<3> nf, string opcodestr>
    : RVInstTileMemOp<(outs), (ins GPR:$rs2, GPRMemZeroOffset:$rs1), nf, OPC_STORE_FP.Value,
                      opcodestr, "$rs2, ${rs1}">;

class RVInstTileMoveOp<bits<6> funct6, dag outs, dag ins, string opcodestr,
                       string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> vd;

  let Inst{31-26} = funct6;
  let Inst{25} = 1;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = OPMVX.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
}

class RVInstMatmulF<bits<2> rd_lsb2, dag outs, dag ins, string opcodestr,
                    string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> rd;

  let Inst{31-26} = 0b111100;
  let Inst{25} = 1;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = OPFVV.Value;
  let Inst{11-9} = rd{3-1};
  let Inst{8-7} = rd_lsb2;
  let Inst{6-0} = OPC_OP_VE.Value;

  let Uses = [VTYPE, VL];
}

class RVInstMatmulF8<bit funct6_lsb, bit rd_lsb, dag outs, dag ins,
                     string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> rd;

  let Inst{31-27} = 0b11111;
  let Inst{26} = funct6_lsb;
  let Inst{25} = 1;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = OPFVV.Value;
  let Inst{11-10} = rd{3-2};
  let Inst{9-8} = 0b00;
  let Inst{7} = rd_lsb;
  let Inst{6-0} = OPC_OP_VE.Value;

  let Uses = [VTYPE, VL];
}


class F8Encode<bit encode_bit, string encode_name> {
  bit Bit = encode_bit;
  string Name = encode_name;
}

defvar F8Encodes = [F8Encode<0b0, "e5m2">,
                    F8Encode<0b1, "e4m3">];

class RVInstMatmulI8<bit funct6_lsb2, bit funct6_lsb1, bit rd_lsb1, dag outs, dag ins,
                     string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> rd;

  let Inst{31-28} = 0b1111;
  let Inst{27} = funct6_lsb2;
  let Inst{26} = funct6_lsb1;
  let Inst{25} = 1;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = OPIVV.Value;
  let Inst{11-10} = rd{3-2};
  let Inst{9-8} = 0b00;
  let Inst{7} = rd_lsb1;
  let Inst{6-0} = OPC_OP_VE.Value;

  let Uses = [VTYPE, VL];
}

class I8Encode<bit encode_bit, string encode_name> {
  bit Bit = encode_bit;
  string Name = encode_name;
}

defvar I8Encodes = [I8Encode<0, "u">,
                    I8Encode<1, "s">];

class RVInstSetZero<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> rd;

  let Inst{31-26} = 0b010000;
  let Inst{25} = 1;
  let Inst{24-20} = 0b11110;
  let Inst{19-15} = 0b00000;
  let Inst{14-12} = OPMVX.Value;
  let Inst{11-8} = rd{3-0};
  let Inst{7} = 0;
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
}

class RVInstVtDiscard<string opcodestr>
    : RVInst<(outs), (ins), opcodestr, "", [], InstFormatR> {
  let Inst{31-26} = 0b010000;
  let Inst{25} = 1;
  let Inst{24-20} = 0b11100;
  let Inst{19-15} = 0b00000;
  let Inst{14-12} = OPMVX.Value;
  let Inst{11-7} = 0b00000;
  let Inst{6-0} = OPC_OP_V.Value;
}

let Predicates = [HasVendorXSfmm32eaOrXSfmmbase] in
def : InstAlias<"sf.vsettnt $rd, $rs1, $vtypei",
                (VSETVLI GPR:$rd, GPR:$rs1, MammothWWEEOp:$vtypei)>;

let Predicates = [HasVendorXSfmm32eaOrXSfmmbase], DecoderNamespace = "XSfmm32eaOrXSfmmbase" in {
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
  def SF_VSETTN : RVInstSetSingle<(outs GPR:$rd), (ins GPR:$rs1), 0b00000,
                                  "sf.vsettn", "$rd, $rs1">;
  def SF_VSETTM : RVInstSetSingle<(outs GPR:$rd), (ins GPR:$rs1), 0b00001,
                                  "sf.vsettm", "$rd, $rs1">;
  def SF_VSETTK  : RVInstSetSingle<(outs GPR:$rd), (ins GPR:$rs1), 0b00010,
                                   "sf.vsettk", "$rd, $rs1">;
  def SF_VTDISCARD : RVInstVtDiscard<"sf.vtdiscard">;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  let rs2 = 0b11111 in
  def SF_VTMV_V_T : RVInstTileMoveOp<0b010000, (outs VR:$vd), (ins GPR:$rs1),
                                     "sf.vtmv.v.t", "$vd, $rs1">;
  let vd = 0b00000 in
  def SF_VTMV_T_V : RVInstTileMoveOp<0b010111, (outs), (ins GPR:$rs1, VR:$rs2),
                                     "sf.vtmv.t.v", "$rs1, $rs2">;

  def SF_VTZERO_T : RVInstSetZero<(outs), (ins TR:$rd), "sf.vtzero.t", "$rd">;
}
} // Predicates = [HasVendorXSfmmbaseOrXSfmm32ea]

let Predicates = [HasVendorXSfmmbase], DecoderNamespace = "XSfmmbase" in {
let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
  def SF_VLTE8  : RVInstTileLoad<0b000, "sf.vlte8">;
  def SF_VLTE16 : RVInstTileLoad<0b001, "sf.vlte16">;
  def SF_VLTE32 : RVInstTileLoad<0b010, "sf.vlte32">;
  def SF_VLTE64 : RVInstTileLoad<0b011, "sf.vlte64">;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  def SF_VSTE8  : RVInstTileStore<0b000, "sf.vste8">;
  def SF_VSTE16 : RVInstTileStore<0b001, "sf.vste16">;
  def SF_VSTE32 : RVInstTileStore<0b010, "sf.vste32">;
  def SF_VSTE64 : RVInstTileStore<0b011, "sf.vste64">;
}
} // Predicates = [HasVendorXSfmmbase]

let Predicates = [HasVendorXSfmm32eaOrXSfmm32aOrXSfmm64a], DecoderNamespace = "XSfmm32eaOrXSfmm32aOrXSfmm64a" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def SF_MM_F_F   : RVInstMatmulF<0b00, (outs), (ins TRM2:$rd, VR:$vs2, VR:$vs1),
                                  "sf.mm.f.f", "$rd, $vs2, $vs1">;
}
} // Predicates = [HasVendorXSfmm32eaOrXSfmm32aOrXSfmm64a]

let Predicates = [HasVendorXSfmm32a], DecoderNamespace = "XSfmm32a" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def SF_MM_BF_BF : RVInstMatmulF<0b01, (outs), (ins TRM4:$rd, VR:$vs2, VR:$vs1),
                                  "sf.mm.bf.bf", "$rd, $vs2, $vs1">;
  foreach a = I8Encodes in
    foreach b = I8Encodes in
      def SF_MM_#!toupper(a.Name)#_#!toupper(b.Name)
          : RVInstMatmulI8<0, a.Bit, b.Bit,
                           (outs), (ins TRM4:$rd, VR:$vs2, VR:$vs1),
                           "sf.mm."#a.Name#"."#b.Name, "$rd, $vs2, $vs1">;
}
} // Predicates = [HasVendorXSfmm32a]

let Predicates = [HasVendorXSfmm32a8f], DecoderNamespace = "XSfmm32a8f" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  foreach a = F8Encodes in
    foreach b = F8Encodes in
      def SF_MM_#!toupper(a.Name)#_#!toupper(b.Name)
          : RVInstMatmulF8<a.Bit, b.Bit,
                           (outs), (ins TRM4:$rd, VR:$vs2, VR:$vs1),
                           "sf.mm."#a.Name#"."#b.Name, "$rd, $vs2, $vs1">;
}
} // Predicates = [HasVendorXSfmm32a8f]

let Predicates = [HasVendorXSfmm32a4i], DecoderNamespace = "XSfmm32a4i" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  foreach a = I8Encodes in
    foreach b = I8Encodes in
      def SF_P2MM_#!toupper(a.Name)#_#!toupper(b.Name)
          : RVInstMatmulI8<1, a.Bit, b.Bit,
                           (outs), (ins TRM4:$rd, VR:$vs2, VR:$vs1),
                           "sf.p2mm."#a.Name#"."#b.Name, "$rd, $vs2, $vs1">;
}
} // Predicates = [HasVendorXSfmm32a4i]
#endif // SIFIVE_CUSTOMIZATION
