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
// FIXME: might need to put more constraint
let RVVConstraint = VMConstraint, Predicates = [HasVendorXSfvqdotq],
    DecoderNamespace = "XSfvqdotq" in {
  def SF_VQDOT_VV   : VALUVV<0b101100, OPMVV, "sf.vqdot.vv">;
  def SF_VQDOT_VX   : VALUVX<0b101100, OPMVX, "sf.vqdot.vx">;
  def SF_VQDOTU_VV  : VALUVV<0b101000, OPMVV, "sf.vqdotu.vv">;
  def SF_VQDOTU_VX  : VALUVX<0b101000, OPMVX, "sf.vqdotu.vx">;
  def SF_VQDOTSU_VV : VALUVV<0b101010, OPMVV, "sf.vqdotsu.vv">;
  def SF_VQDOTSU_VX : VALUVX<0b101010, OPMVX, "sf.vqdotsu.vx">;
  def SF_VQDOTUS_VX : VALUVX<0b101110, OPMVX, "sf.vqdotus.vx">;
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

defset list<VTypeInfoToWide> VQDOTInfoPairs = {
  def : VTypeInfoToWide<VI8MF2, VI32MF2>;
  def : VTypeInfoToWide<VI8M1, VI32M1>;
  def : VTypeInfoToWide<VI8M2, VI32M2>;
  def : VTypeInfoToWide<VI8M4, VI32M4>;
  def : VTypeInfoToWide<VI8M8, VI32M8>;
}

// These VPat definitions are for vqdot because they have a different operand
// order with other ternary instructions (i.e. vop.vx vd, vs2, rs1)
multiclass VPatTernaryV_VX_AAXASwapped<string intrinsic, string instruction,
                                       list<VTypeInfoToWide> info_pairs> {
  foreach pair = info_pairs in {
    defvar VdInfo = pair.Wti;
    defvar Vs2Info = pair.Vti;
    let Predicates = GetVTypePredicates<VdInfo>.Predicates in
    // we have op1 and op2 swapped here
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "V"#VdInfo.ScalarSuffix,
                                 VdInfo.Vector, Vs2Info.Vector, Vs2Info.Scalar,
                                 VdInfo.Mask, VdInfo.Log2SEW, VdInfo.LMul,
                                 VdInfo.RegClass, Vs2Info.RegClass,
                                 Vs2Info.ScalarRegClass>;
  }
}

multiclass VPatTernaryV_VV_AAXASwapped<string intrinsic, string instruction,
                                       list<VTypeInfoToWide> info_pairs> {
  foreach pair = info_pairs in {
    defvar VdInfo = pair.Wti;
    defvar Vs2Info = pair.Vti;
    let Predicates = GetVTypePredicates<VdInfo>.Predicates in
    // we have op1 and op2 swapped here
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "VV",
                                 VdInfo.Vector, Vs2Info.Vector, Vs2Info.Vector,
                                 VdInfo.Mask, VdInfo.Log2SEW, VdInfo.LMul,
                                 VdInfo.RegClass, Vs2Info.RegClass,
                                 Vs2Info.RegClass>;
  }
}

multiclass VPatTernaryV_VV_VX_AAXASwapped<string intrinsic, string instruction,
                                          list<VTypeInfoToWide> info_pairs>
    // keep the original VV because the order doesn't matter
    : VPatTernaryV_VV_AAXASwapped<intrinsic, instruction, info_pairs>,
      VPatTernaryV_VX_AAXASwapped<intrinsic, instruction, info_pairs>;
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

  let Uses = [VL, VTYPE];
  let RVVConstraint = NoConstraint;
  let ElementsDependOn = EltDepsVLMask;
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

  let Uses = [VL, VTYPE];
  let RVVConstraint = NoConstraint;
  let ElementsDependOn = EltDepsVLMask;
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
  defm SF_VC_ # NAME   : CustomSiFiveVCIXorVCIF<suffix, type, InTyRd, InTyRs2,
                                             InTyRs1, 0>;
  let vm = 0 in
  defm SF_VC_V_ # NAME : CustomSiFiveVCIXorVCIF<suffix, type, VR, InTyRs2,
                                             InTyRs1, 1>;
}

// For XSfvqmaccdod/qoq and XSfvfwmaccqqq
class SiFiveVMACCScheds<string name> {
  defvar n = !tolower(name);
  defvar prefix = !if(!ne(!find(n, "fw"), -1), "FW", "Q");
  defvar suffix = !if(!ne(!find(n, "2x8x2"), -1), "DOD",
                      !if(!eq(prefix, "Q"), "QOQ", "QQQ"));

  string read = "ReadSF_V" # prefix # "MACC_" # suffix;
  string write = "WriteSF_V" # prefix # "MACC_" # suffix;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
class CustomSiFiveVMACC<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVCCustom2<funct6{5-2}, opv.Value, (outs VR:$rd), (ins VR:$rs1, VR:$rs2),
                      opcodestr, "$rd, $rs1, $rs2">,
      SchedTernaryMC<SiFiveVMACCScheds<NAME>.write,
                     SiFiveVMACCScheds<NAME>.read,
                     SiFiveVMACCScheds<NAME>.read,
                     SiFiveVMACCScheds<NAME>.read> {
  let vm = 1;
  let funct6_lo2 = funct6{1-0};
}
}

class CustomSiFiveVFNRCLIP<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : VALUVF<funct6, opv, opcodestr>,
      SchedBinaryMC<"WriteSF_VFNRClipV", "ReadSF_VFNRClipV", "ReadSF_VFNRClipF"> {
  let Inst{6-0} = OPC_CUSTOM_2.Value;
}

let Predicates = [HasVendorXSfvcp], mayLoad = 0, mayStore = 0,
    hasSideEffects = 1, hasNoSchedulingInfo = 1, DecoderNamespace = "XSfvector" in {
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

#ifdef SIFIVE_CUSTOMIZATION
let Predicates = [HasVendorXSfvfexpAny], DecoderNamespace = "XSfvfexp" in {
  def VFEXP_V : VALUVs2<0b010011, 0b00111, OPFVV, "sf.vfexp.v">;
}

let Predicates = [HasVendorXSfvfexpa], DecoderNamespace = "XSfvfexpa" in {
  def VFEXPA_V : VALUVs2<0b010011, 0b00110, OPFVV, "sf.vfexpa.v">;
}
#endif // SIFIVE_CUSTOMIZATION

let Predicates = [HasVendorXSfvqmaccdod], DecoderNamespace = "XSfvector",
    DestEEW = EEWSEWx4, RVVConstraint=VS2Constraint in {
  def SF_VQMACCU_2x8x2  : CustomSiFiveVMACC<0b101100, OPMVV, "sf.vqmaccu.2x8x2">;
  def SF_VQMACC_2x8x2   : CustomSiFiveVMACC<0b101101, OPMVV, "sf.vqmacc.2x8x2">;
  def SF_VQMACCUS_2x8x2 : CustomSiFiveVMACC<0b101110, OPMVV, "sf.vqmaccus.2x8x2">;
  def SF_VQMACCSU_2x8x2 : CustomSiFiveVMACC<0b101111, OPMVV, "sf.vqmaccsu.2x8x2">;
}

let Predicates = [HasVendorXSfvqmaccqoq], DecoderNamespace = "XSfvector",
    DestEEW = EEWSEWx4, RVVConstraint=WidenVNoMask in {
  def SF_VQMACCU_4x8x4  : CustomSiFiveVMACC<0b111100, OPMVV, "sf.vqmaccu.4x8x4">;
  def SF_VQMACC_4x8x4   : CustomSiFiveVMACC<0b111101, OPMVV, "sf.vqmacc.4x8x4">;
  def SF_VQMACCUS_4x8x4 : CustomSiFiveVMACC<0b111110, OPMVV, "sf.vqmaccus.4x8x4">;
  def SF_VQMACCSU_4x8x4 : CustomSiFiveVMACC<0b111111, OPMVV, "sf.vqmaccsu.4x8x4">;
}

let Predicates = [HasVendorXSfvfwmaccqqq], DecoderNamespace = "XSfvector",
    DestEEW = EEWSEWx2, RVVConstraint=WidenVNoMask in {
  def SF_VFWMACC_4x4x4 : CustomSiFiveVMACC<0b111100, OPFVV, "sf.vfwmacc.4x4x4">;
}

let Predicates = [HasVendorXSfvfnrclipxfqf], DecoderNamespace = "XSfvector",
    Uses = [FRM, VL, VTYPE] in {
  def SF_VFNRCLIP_XU_F_QF : CustomSiFiveVFNRCLIP<0b100010, OPFVF, "sf.vfnrclip.xu.f.qf">;
  def SF_VFNRCLIP_X_F_QF : CustomSiFiveVFNRCLIP<0b100011, OPFVF, "sf.vfnrclip.x.f.qf">;
}

class VPseudoVC_X<Operand OpClass, DAGOperand RS1Class> :
      Pseudo<(outs),
             (ins OpClass:$op1, payload5:$rs2, payload5:$rd, RS1Class:$r1,
                  AVL:$vl, sew:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
}

class VPseudoVC_XV<Operand OpClass, VReg RS2Class, DAGOperand RS1Class> :
      Pseudo<(outs),
             (ins OpClass:$op1, payload5:$rd, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, sew:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
}

class VPseudoVC_XVV<Operand OpClass, VReg RDClass, VReg RS2Class,
                    DAGOperand RS1Class> :
      Pseudo<(outs),
             (ins OpClass:$op1, RDClass:$rd, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, sew:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
}

class VPseudoVC_V_X<Operand OpClass, VReg RDClass, DAGOperand RS1Class> :
      Pseudo<(outs RDClass:$rd),
             (ins OpClass:$op1, payload5:$rs2, RS1Class:$r1,
                  AVL:$vl, sew:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
}

class VPseudoVC_V_XV<Operand OpClass, VReg RDClass, VReg RS2Class,
                     DAGOperand RS1Class> :
      Pseudo<(outs RDClass:$rd),
             (ins OpClass:$op1, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, sew:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
}

class VPseudoVC_V_XVV<Operand OpClass, VReg RDClass, VReg RS2Class,
                      DAGOperand RS1Class> :
      Pseudo<(outs RDClass:$rd),
             (ins OpClass:$op1, RDClass:$rs3, RS2Class:$rs2, RS1Class:$r1,
                  AVL:$vl, sew:$sew), []>,
      RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let hasSideEffects = 0;
  let Constraints = "$rd = $rs3";
}

multiclass VPseudoVC_X<LMULInfo m, DAGOperand RS1Class,
                       Operand OpClass = payload2> {
  let VLMul = m.value in {
    let Defs = [SF_VCIX_STATE], Uses = [SF_VCIX_STATE] in {
      def "PseudoSF_VC_" # NAME # "_SE_" # m.MX
        : VPseudoVC_X<OpClass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_" # NAME # "_" # m.MX)]>;
      def "PseudoSF_VC_V_" # NAME # "_SE_" # m.MX
        : VPseudoVC_V_X<OpClass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
    }
    def "PseudoSF_VC_V_" # NAME # "_" # m.MX
      : VPseudoVC_V_X<OpClass, m.vrclass, RS1Class>,
        Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
  }
}

multiclass VPseudoVC_XV<LMULInfo m, DAGOperand RS1Class,
                        Operand OpClass = payload2> {
  let VLMul = m.value in {
    let Defs = [SF_VCIX_STATE], Uses = [SF_VCIX_STATE] in {
      def "PseudoSF_VC_" # NAME # "_SE_" # m.MX
        : VPseudoVC_XV<OpClass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_" # NAME # "_" # m.MX)]>;
      def "PseudoSF_VC_V_" # NAME # "_SE_" # m.MX
        : VPseudoVC_V_XV<OpClass, m.vrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
    }
    def "PseudoSF_VC_V_" # NAME # "_" # m.MX
      : VPseudoVC_V_XV<OpClass, m.vrclass, m.vrclass, RS1Class>,
        Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
  }
}

multiclass VPseudoVC_XVV<LMULInfo m, DAGOperand RS1Class,
                         Operand OpClass = payload2> {
  let VLMul = m.value in {
    let Defs = [SF_VCIX_STATE], Uses = [SF_VCIX_STATE] in {
      def "PseudoSF_VC_" # NAME # "_SE_" # m.MX
        : VPseudoVC_XVV<OpClass, m.vrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_" # NAME # "_" # m.MX)]>;
      def "PseudoSF_VC_V_" # NAME # "_SE_" # m.MX
        : VPseudoVC_V_XVV<OpClass, m.vrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
    }
    def "PseudoSF_VC_V_" # NAME # "_" # m.MX
      : VPseudoVC_V_XVV<OpClass, m.vrclass, m.vrclass, RS1Class>,
        Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
  }
}

multiclass VPseudoVC_XVW<LMULInfo m, DAGOperand RS1Class,
                         Operand OpClass = payload2> {
  let VLMul = m.value in {
    let Defs = [SF_VCIX_STATE], Uses = [SF_VCIX_STATE] in
    def "PseudoSF_VC_" # NAME # "_SE_" # m.MX
      : VPseudoVC_XVV<OpClass, m.wvrclass, m.vrclass, RS1Class>,
        Sched<[!cast<SchedWrite>("WriteVC_" # NAME # "_" # m.MX)]>;
    let Constraints = "@earlyclobber $rd, $rd = $rs3" in {
      let Defs = [SF_VCIX_STATE], Uses = [SF_VCIX_STATE] in
      def "PseudoSF_VC_V_" # NAME # "_SE_" # m.MX
        : VPseudoVC_V_XVV<OpClass, m.wvrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
      def "PseudoSF_VC_V_" # NAME # "_" # m.MX
        : VPseudoVC_V_XVV<OpClass, m.wvrclass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
    }
  }
}

multiclass VPseudoSiFiveVMACC<string mx, VReg vd_type, VReg vs2_type> {
  defvar SchedWriteName = SiFiveVMACCScheds<NAME>.write;
  defvar SchedReadName = SiFiveVMACCScheds<NAME>.read;
  def "Pseudo" # NAME # "_" # mx
      : VPseudoTernaryNoMaskWithPolicy<vd_type, V_M1.vrclass, vs2_type,
                                       "@earlyclobber $rd">,
        SchedTernary<SchedWriteName, SchedReadName, SchedReadName,
                     SchedReadName, mx>;
}

multiclass VPseudoSiFiveVQMACCDOD {
  foreach m = MxListVF8 in
    let VLMul = m.value in
    defm NAME : VPseudoSiFiveVMACC<m.MX, m.vrclass, m.vrclass>;
}

multiclass VPseudoSiFiveVQMACCQOQ {
  foreach m = [V_MF2, V_M1, V_M2, V_M4] in
    let VLMul = m.value in
    defm NAME : VPseudoSiFiveVMACC<m.MX, m.wvrclass, m.vrclass>;
}

multiclass VPseudoSiFiveVFWMACC {
  foreach m = MxListFW in
    let VLMul = m.value in
    defm NAME : VPseudoSiFiveVMACC<m.MX, m.wvrclass, m.vrclass>;
}

multiclass VPseudoSiFiveVFNRCLIP<string Constraint = "@earlyclobber $rd"> {
  foreach i = 0-4 in
    let hasSideEffects = 0, hasPostISelHook = 1 in
      defm "Pseudo" # NAME : VPseudoBinaryRoundingMode<MxListW[i].vrclass,
                                                       MxListVF4[i].vrclass,
                                                       FPR32, MxListW[i],
                                                       Constraint, /*sew*/0,
                                                       UsesVXRM=0>,
                             SchedBinary<"WriteSF_VFNRClipV", "ReadSF_VFNRClipV",
                                         "ReadSF_VFNRClipF",
                                         MxListW[i].MX>;
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
#ifdef SIFIVE_CUSTOMIZATION
      let AltFmtType = IS_NOT_ALTFMT in {
        defm f.FX # "V" : VPseudoVC_XV<m, f.fprclass, payload1>;
        defm f.FX # "VV" : VPseudoVC_XVV<m, f.fprclass, payload1>;
      }
#endif // SIFIVE_CUSTOMIZATION
    }
  }
  foreach m = MxListW in {
    defm XVW : VPseudoVC_XVW<m, GPR>;
    defm IVW : VPseudoVC_XVW<m, tsimm5>;
    defm VVW : VPseudoVC_XVW<m, m.vrclass>;
  }
  foreach f = FPListW in {
    foreach m = f.MxList in
#ifdef SIFIVE_CUSTOMIZATION
      let AltFmtType = IS_NOT_ALTFMT in
        defm f.FX # "VW" : VPseudoVC_XVW<m, f.fprclass, payload1>;
#endif // SIFIVE_CUSTOMIZATION
  }
}

let Predicates = [HasVendorXSfvqmaccdod] in {
  defm SF_VQMACCU_2x8x2  : VPseudoSiFiveVQMACCDOD;
  defm SF_VQMACC_2x8x2   : VPseudoSiFiveVQMACCDOD;
  defm SF_VQMACCUS_2x8x2 : VPseudoSiFiveVQMACCDOD;
  defm SF_VQMACCSU_2x8x2 : VPseudoSiFiveVQMACCDOD;
}

let Predicates = [HasVendorXSfvqmaccqoq] in {
  defm SF_VQMACCU_4x8x4  : VPseudoSiFiveVQMACCQOQ;
  defm SF_VQMACC_4x8x4   : VPseudoSiFiveVQMACCQOQ;
  defm SF_VQMACCUS_4x8x4 : VPseudoSiFiveVQMACCQOQ;
  defm SF_VQMACCSU_4x8x4 : VPseudoSiFiveVQMACCQOQ;
}

let Predicates = [HasVendorXSfvfwmaccqqq] in {
  defm SF_VFWMACC_4x4x4 : VPseudoSiFiveVFWMACC;
}

let Predicates = [HasVendorXSfvfnrclipxfqf] in {
  defm SF_VFNRCLIP_XU_F_QF : VPseudoSiFiveVFNRCLIP;
  defm SF_VFNRCLIP_X_F_QF : VPseudoSiFiveVFNRCLIP;
}

#ifdef SIFIVE_CUSTOMIZATION
let Predicates = [HasVendorXSfvfbfexp16e], hasSideEffects = 0 in {
  let AltFmtType = IS_ALTFMT in{
    defm PseudoVFEXP_ALT : VPseudoVRCP_V<IsBF16=1>;
  }
}

let Predicates = [HasVendorXSfvfexpAnyFloat], hasSideEffects = 0 in {
  let AltFmtType = IS_NOT_ALTFMT in{
    defm PseudoVFEXP : VPseudoVRCP_V;
  }
}

let Predicates = [HasVendorXSfvfexpa] in {
  defm PseudoVFEXPA : VPseudoVRCP_V;
}
#endif // SIFIVE_CUSTOMIZATION

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

def sf_vc_v_x_se : RVSDNode<"SF_VC_V_X_SE", SDT_SF_VC_V_X, [SDNPHasChain]>;
def sf_vc_v_i_se : RVSDNode<"SF_VC_V_I_SE", SDT_SF_VC_V_X, [SDNPHasChain]>;
def sf_vc_vv_se : RVSDNode<"SF_VC_VV_SE", SDT_SF_VC_XV, [SDNPHasChain]>;
def sf_vc_xv_se : RVSDNode<"SF_VC_XV_SE", SDT_SF_VC_XV, [SDNPHasChain]>;
def sf_vc_iv_se : RVSDNode<"SF_VC_IV_SE", SDT_SF_VC_XV, [SDNPHasChain]>;
def sf_vc_fv_se : RVSDNode<"SF_VC_FV_SE", SDT_SF_VC_XV, [SDNPHasChain]>;
def sf_vc_v_vv_se : RVSDNode<"SF_VC_V_VV_SE", SDT_SF_VC_V_XV, [SDNPHasChain]>;
def sf_vc_v_xv_se : RVSDNode<"SF_VC_V_XV_SE", SDT_SF_VC_V_XV, [SDNPHasChain]>;
def sf_vc_v_iv_se : RVSDNode<"SF_VC_V_IV_SE", SDT_SF_VC_V_XV, [SDNPHasChain]>;
def sf_vc_v_fv_se : RVSDNode<"SF_VC_V_FV_SE", SDT_SF_VC_V_XV, [SDNPHasChain]>;
def sf_vc_vvv_se : RVSDNode<"SF_VC_VVV_SE", SDT_SF_VC_XVV, [SDNPHasChain]>;
def sf_vc_xvv_se : RVSDNode<"SF_VC_XVV_SE", SDT_SF_VC_XVV, [SDNPHasChain]>;
def sf_vc_ivv_se : RVSDNode<"SF_VC_IVV_SE", SDT_SF_VC_XVV, [SDNPHasChain]>;
def sf_vc_fvv_se : RVSDNode<"SF_VC_FVV_SE", SDT_SF_VC_XVV, [SDNPHasChain]>;
def sf_vc_v_vvv_se : RVSDNode<"SF_VC_V_VVV_SE", SDT_SF_VC_V_XVV, [SDNPHasChain]>;
def sf_vc_v_xvv_se : RVSDNode<"SF_VC_V_XVV_SE", SDT_SF_VC_V_XVV, [SDNPHasChain]>;
def sf_vc_v_ivv_se : RVSDNode<"SF_VC_V_IVV_SE", SDT_SF_VC_V_XVV, [SDNPHasChain]>;
def sf_vc_v_fvv_se : RVSDNode<"SF_VC_V_FVV_SE", SDT_SF_VC_V_XVV, [SDNPHasChain]>;
def sf_vc_vvw_se : RVSDNode<"SF_VC_VVW_SE", SDT_SF_VC_XVW, [SDNPHasChain]>;
def sf_vc_xvw_se : RVSDNode<"SF_VC_XVW_SE", SDT_SF_VC_XVW, [SDNPHasChain]>;
def sf_vc_ivw_se : RVSDNode<"SF_VC_IVW_SE", SDT_SF_VC_XVW, [SDNPHasChain]>;
def sf_vc_fvw_se : RVSDNode<"SF_VC_FVW_SE", SDT_SF_VC_XVW, [SDNPHasChain]>;
def sf_vc_v_vvw_se : RVSDNode<"SF_VC_V_VVW_SE", SDT_SF_VC_V_XVW, [SDNPHasChain]>;
def sf_vc_v_xvw_se : RVSDNode<"SF_VC_V_XVW_SE", SDT_SF_VC_V_XVW, [SDNPHasChain]>;
def sf_vc_v_ivw_se : RVSDNode<"SF_VC_V_IVW_SE", SDT_SF_VC_V_XVW, [SDNPHasChain]>;
def sf_vc_v_fvw_se : RVSDNode<"SF_VC_V_FVW_SE", SDT_SF_VC_V_XVW, [SDNPHasChain]>;

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
                         "PseudoSF_VC_V_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                         vti.Vector, XLenVT, type, vti.Log2SEW,
                         payload5, kind>;
  def : VPatVC_V_OP3<"int_riscv_sf_vc_v_" # intrinsic_suffix,
                     "PseudoSF_VC_V_" # instruction_suffix # "_" # vti.LMul.MX,
                     vti.Vector, XLenVT, type, vti.Log2SEW,
                     payload5, kind>;
}

multiclass VPatVC_XV<string intrinsic_suffix, string instruction_suffix,
                     VTypeInfo vti, ValueType type, DAGOperand kind,
                     Operand op1_kind = payload2> {
  def : VPatVC_OP4_ISD<!cast<SDPatternOperator>("sf_vc_" # intrinsic_suffix # "_se"),
                   "PseudoSF_VC_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                   XLenVT, vti.Vector, type, vti.Log2SEW,
                   payload5, vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP3_ISD<!cast<SDPatternOperator>("sf_vc_v_" # intrinsic_suffix # "_se"),
                         "PseudoSF_VC_V_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                         vti.Vector, vti.Vector, type, vti.Log2SEW,
                         vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP3<"int_riscv_sf_vc_v_" # intrinsic_suffix,
                     "PseudoSF_VC_V_" # instruction_suffix # "_" # vti.LMul.MX,
                     vti.Vector, vti.Vector, type, vti.Log2SEW,
                     vti.RegClass, kind, op1_kind>;
}

multiclass VPatVC_XVV<string intrinsic_suffix, string instruction_suffix,
                      VTypeInfo wti, VTypeInfo vti, ValueType type, DAGOperand kind,
                      Operand op1_kind = payload2> {
  def : VPatVC_OP4_ISD<!cast<SDPatternOperator>("sf_vc_" # intrinsic_suffix # "_se"),
                   "PseudoSF_VC_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                   wti.Vector, vti.Vector, type, vti.Log2SEW,
                   wti.RegClass, vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP4_ISD<!cast<SDPatternOperator>("sf_vc_v_" # intrinsic_suffix # "_se"),
                     "PseudoSF_VC_V_" # instruction_suffix # "_SE_" # vti.LMul.MX,
                     wti.Vector, wti.Vector, vti.Vector, type, vti.Log2SEW,
                     wti.RegClass, vti.RegClass, kind, op1_kind>;
  def : VPatVC_V_OP4<"int_riscv_sf_vc_v_" # intrinsic_suffix,
                     "PseudoSF_VC_V_" # instruction_suffix # "_" # vti.LMul.MX,
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
  defm : VPatVQMACCDOD<"vqmaccu_2x8x2", "SF_VQMACCU", "2x8x2">;
  defm : VPatVQMACCDOD<"vqmacc_2x8x2", "SF_VQMACC", "2x8x2">;
  defm : VPatVQMACCDOD<"vqmaccus_2x8x2", "SF_VQMACCUS", "2x8x2">;
  defm : VPatVQMACCDOD<"vqmaccsu_2x8x2", "SF_VQMACCSU", "2x8x2">;
}

let Predicates = [HasVendorXSfvqmaccqoq] in {
  defm : VPatVQMACCQOQ<"vqmaccu_4x8x4", "SF_VQMACCU", "4x8x4">;
  defm : VPatVQMACCQOQ<"vqmacc_4x8x4", "SF_VQMACC", "4x8x4">;
  defm : VPatVQMACCQOQ<"vqmaccus_4x8x4", "SF_VQMACCUS", "4x8x4">;
  defm : VPatVQMACCQOQ<"vqmaccsu_4x8x4", "SF_VQMACCSU", "4x8x4">;
}

#ifdef SIFIVE_CUSTOMIZATION
multiclass VPseudoSiFiveVQDOT<string Constraint = ""> {
  foreach m = MxListVF4 in {
    defm "Pseudo" # NAME # "_VV" : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, m.vrclass, m, Constraint>;
    defm "Pseudo" # NAME # "_VX" : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, GPR, m, Constraint>;
  }
}

let Predicates = [HasVendorXSfvqdotq] in {
  defm SF_VQDOT  : VPseudoSiFiveVQDOT;
  defm SF_VQDOTU : VPseudoSiFiveVQDOT;
  defm SF_VQDOTSU: VPseudoSiFiveVQDOT;

  // VQDOTUS does not have a VV variant
  foreach m = MxListVF4 in {
    defm "PseudoSF_VQDOTUS_VX" : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, GPR, m>;
  }
}

let Predicates = [HasVendorXSfvqdotq] in {
  defm : VPatTernaryV_VV_VX_AAXASwapped<"int_riscv_sf_vqdot", "PseudoSF_VQDOT", VQDOTInfoPairs>;
  defm : VPatTernaryV_VV_VX_AAXASwapped<"int_riscv_sf_vqdotu", "PseudoSF_VQDOTU", VQDOTInfoPairs>;
  defm : VPatTernaryV_VV_VX_AAXASwapped<"int_riscv_sf_vqdotsu", "PseudoSF_VQDOTSU", VQDOTInfoPairs>;
  defm : VPatTernaryV_VX_AAXASwapped<"int_riscv_sf_vqdotus", "PseudoSF_VQDOTUS", VQDOTInfoPairs>;
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
                   (mask_type VMV0:$vm),
                   VLOpFrag, (XLenVT timm:$policy))),
                   (!cast<Instruction>(!if(isSEWAware,
                                           inst#"_"#kind#"_"#vlmul.MX#"_E"#!shl(1, log2sew)#"_MASK",
                                           inst#"_"#kind#"_"#vlmul.MX#"_MASK"))
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   (mask_type VMV0:$vm), FRM_DYN, GPR:$vl, log2sew, (XLenVT timm:$policy))>;

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
  defm : VPatVFWMACC<"vfwmacc_4x4x4", "SF_VFWMACC", "4x4x4">;
}

let Predicates = [HasVendorXSfvfnrclipxfqf] in {
  defm : VPatVFNRCLIP<"vfnrclip_xu_f_qf", "SF_VFNRCLIP_XU_F_QF">;
  defm : VPatVFNRCLIP<"vfnrclip_x_f_qf", "SF_VFNRCLIP_X_F_QF">;
}

let Predicates = [HasVendorXSiFivecdiscarddlone] in {
  let hasNoSchedulingInfo = 1, hasSideEffects = 1, mayLoad = 0, mayStore = 0,
      DecoderNamespace = "XSfsystem" in
  def SF_CDISCARD_D_L1
      : RVInstIUnary<0b111111000010, 0b000, OPC_SYSTEM, (outs), (ins GPR:$rs1),
                     "sf.cdiscard.d.l1", "$rs1">, Sched<[]> {
    let rd = 0;
  }
  def : InstAlias<"sf.cdiscard.d.l1", (SF_CDISCARD_D_L1 X0)>;
} // Predicates = [HasVendorXSifivecdiscarddlone]

let Predicates = [HasVendorXSiFivecflushdlone] in {
  let hasNoSchedulingInfo = 1, hasSideEffects = 1, mayLoad = 0, mayStore = 0,
      DecoderNamespace = "XSfsystem" in
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
  def SF_CEASE : Priv<"sf.cease", 0b0011000>, Sched<[]> {
    let rd = 0b00000;
    let rs1 = 0b00000;
    let rs2 = 0b00101;
  }
}

#ifdef SIFIVE_CUSTOMIZATION
let Predicates = [HasVendorXSfvfbfexp16e] in {
  defm : VPatUnaryV_V<"int_riscv_sf_vfexp", "PseudoVFEXP_ALT",
                      AllBFloatVectors,
                      isSEWAware=1>;
}

let Predicates = [HasVendorXSfvfexp16e] in {
  defm : VPatUnaryV_V<"int_riscv_sf_vfexp", "PseudoVFEXP",
                      [VF16MF4, VF16MF2, VF16M1, VF16M2, VF16M4, VF16M8],
                      isSEWAware=1>;
}

let Predicates = [HasVendorXSfvfexp32e] in {
  defm : VPatUnaryV_V<"int_riscv_sf_vfexp", "PseudoVFEXP",
                      [VF32MF2, VF32M1, VF32M2, VF32M4, VF32M8], isSEWAware=1>;
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
#endif // SIFIVE_CUSTOMIZATION

#ifdef SIFIVE_CUSTOMIZATION
//===----------------------------------------------------------------------===//
// Xsfvfbfa instructions
//===----------------------------------------------------------------------===//
// TODO: Currently we mark all floating type other than bf16 as IS_NOT_ALTFMT,
//       however only f16 should be marked as IS_NOT_ALTFMT. However if we want
//       to get it correct and more precisely, we need to add another set of
//       PseudoInst to represent those that are f16 as we did for bf16, it might
//       add much complexity for refactoring.

let Predicates = [HasVendorXSfvfbfa] in {
let AltFmtType = IS_ALTFMT in {
//===----------------------------------------------------------------------===//
// 13.2. Vector Single-Width Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true, hasPostISelHook = 1 in {
defm PseudoVFADD_ALT : VPseudoVALU_VV_VF_RM<IsBF16=1>;
defm PseudoVFSUB_ALT  : VPseudoVALU_VV_VF_RM<IsBF16=1>;
defm PseudoVFRSUB_ALT : VPseudoVALU_VF_RM<IsBF16=1>;
}

//===----------------------------------------------------------------------===//
// 13.3. Vector Widening Floating-Point Add/Subtract Instructions
//===----------------------------------------------------------------------===//
let IsWiden = 1 in {
let mayRaiseFPException = true, hasSideEffects = 0, hasPostISelHook = 1 in {
defm PseudoVFWADD_ALT : VPseudoVFWALU_VV_VF_RM<IsBF16=1>;
defm PseudoVFWSUB_ALT : VPseudoVFWALU_VV_VF_RM<IsBF16=1>;
defm PseudoVFWADD_ALT : VPseudoVFWALU_WV_WF_RM<IsBF16=1>;
defm PseudoVFWSUB_ALT : VPseudoVFWALU_WV_WF_RM<IsBF16=1>;
}
}

//===----------------------------------------------------------------------===//
// 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true, hasSideEffects = 0, hasPostISelHook = 1 in
defm PseudoVFMUL_ALT : VPseudoVFMUL_VV_VF_RM<IsBF16=1>;

//===----------------------------------------------------------------------===//
// 13.5. Vector Widening Floating-Point Multiply
//===----------------------------------------------------------------------===//
let IsWiden = 1 in
let mayRaiseFPException = true, hasSideEffects = 0 in
defm PseudoVFWMUL_ALT : VPseudoVWMUL_VV_VF_RM<IsBF16=1>;

//===----------------------------------------------------------------------===//
// 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true, hasSideEffects = 0, hasPostISelHook = 1 in {
defm PseudoVFMACC_ALT  : VPseudoVMAC_VV_VF_AAXA_RM<IsBF16=1>;
defm PseudoVFNMACC_ALT : VPseudoVMAC_VV_VF_AAXA_RM<IsBF16=1>;
defm PseudoVFMSAC_ALT  : VPseudoVMAC_VV_VF_AAXA_RM<IsBF16=1>;
defm PseudoVFNMSAC_ALT : VPseudoVMAC_VV_VF_AAXA_RM<IsBF16=1>;
defm PseudoVFMADD_ALT  : VPseudoVMAC_VV_VF_AAXA_RM<IsBF16=1>;
defm PseudoVFNMADD_ALT : VPseudoVMAC_VV_VF_AAXA_RM<IsBF16=1>;
defm PseudoVFMSUB_ALT  : VPseudoVMAC_VV_VF_AAXA_RM<IsBF16=1>;
defm PseudoVFNMSUB_ALT : VPseudoVMAC_VV_VF_AAXA_RM<IsBF16=1>;
}

//===----------------------------------------------------------------------===//
// 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
//===----------------------------------------------------------------------===//
let IsWiden = 1 in {
let mayRaiseFPException = true, hasSideEffects = 0, hasPostISelHook = 1 in {
defm PseudoVFWMACC_ALT  : VPseudoVWMAC_VV_VF_RM<IsBF16=1>;
defm PseudoVFWNMACC_ALT : VPseudoVWMAC_VV_VF_RM<IsBF16=1>;
defm PseudoVFWMSAC_ALT  : VPseudoVWMAC_VV_VF_RM<IsBF16=1>;
defm PseudoVFWNMSAC_ALT : VPseudoVWMAC_VV_VF_RM<IsBF16=1>;
}
}

//===----------------------------------------------------------------------===//
// 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in
defm PseudoVFRSQRT7_ALT : VPseudoVRCP_V<IsBF16=1>;

//===----------------------------------------------------------------------===//
// 13.10. Vector Floating-Point Reciprocal Estimate Instruction
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true, hasSideEffects = 0 in
defm PseudoVFREC7_ALT : VPseudoVRCP_V_RM<IsBF16=1>;

//===----------------------------------------------------------------------===//
// 13.11. Vector Floating-Point Min/Max Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in {
defm PseudoVFMIN_ALT : VPseudoVMAX_VV_VF<IsBF16=1>;
defm PseudoVFMAX_ALT : VPseudoVMAX_VV_VF<IsBF16=1>;
}

//===----------------------------------------------------------------------===//
// 13.12. Vector Floating-Point Sign-Injection Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFSGNJ_ALT  : VPseudoVSGNJ_VV_VF<IsBF16=1>;
defm PseudoVFSGNJN_ALT : VPseudoVSGNJ_VV_VF<IsBF16=1>;
defm PseudoVFSGNJX_ALT : VPseudoVSGNJ_VV_VF<IsBF16=1>;

//===----------------------------------------------------------------------===//
// 13.13. Vector Floating-Point Compare Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in {
defm PseudoVMFEQ_ALT : VPseudoVCMPM_VV_VF;
defm PseudoVMFNE_ALT : VPseudoVCMPM_VV_VF;
defm PseudoVMFLT_ALT : VPseudoVCMPM_VV_VF;
defm PseudoVMFLE_ALT : VPseudoVCMPM_VV_VF;
defm PseudoVMFGT_ALT : VPseudoVCMPM_VF;
defm PseudoVMFGE_ALT : VPseudoVCMPM_VF;
}

//===----------------------------------------------------------------------===//
// 13.14. Vector Floating-Point Classify Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFCLASS_ALT : VPseudoVCLS_V;

//===----------------------------------------------------------------------===//
// 13.15. Vector Floating-Point Merge Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFMERGE_ALT : VPseudoVMRG_FM;

//===----------------------------------------------------------------------===//
// 13.16. Vector Floating-Point Move Instruction
//===----------------------------------------------------------------------===//
defm PseudoVFMV_V_ALT : VPseudoVMV_F;

//===----------------------------------------------------------------------===//
// 13.18. Widening Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
let IsWiden = 1 in {
let mayRaiseFPException = true in {
defm PseudoVFWCVT_F_XU_ALT : VPseudoVWCVTF_V<IsBF16=1>;
defm PseudoVFWCVT_F_X_ALT  : VPseudoVWCVTF_V<IsBF16=1>;

defm PseudoVFWCVT_F_F_ALT  : VPseudoVWCVTD_V<IsBF16=1>;
} // mayRaiseFPException = true
}

//===----------------------------------------------------------------------===//
// 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions
//===----------------------------------------------------------------------===//
let mayRaiseFPException = true in {
let hasSideEffects = 0, hasPostISelHook = 1 in {
defm PseudoVFNCVT_XU_F_ALT : VPseudoVNCVTI_W_RM;
defm PseudoVFNCVT_X_F_ALT  : VPseudoVNCVTI_W_RM;
}

defm PseudoVFNCVT_RTZ_XU_F_ALT : VPseudoVNCVTI_W;
defm PseudoVFNCVT_RTZ_X_F_ALT  : VPseudoVNCVTI_W;

defm PseudoVFNCVT_F_F_ALT  : VPseudoVNCVTD_W_RM<IsBF16=1>;

defm PseudoVFNCVT_ROD_F_F_ALT : VPseudoVNCVTD_W<IsBF16=1>;
} // mayRaiseFPException = true

//===----------------------------------------------------------------------===//
// 16.3. Vector Slide Instructions
//===----------------------------------------------------------------------===//
let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in {
  foreach f = FPList in {
    foreach m = f.MxList in {
      defvar mx = m.MX;
      let VLMul = m.value in {
        let HasSEWOp = 1, BaseInstr = VFMV_F_S in
        def "PseudoVFMV_" # f.FX # "_S_ALT_" # mx :
          Pseudo<(outs f.fprclass:$rd),
                 (ins m.vrclass:$rs2, ixlenimm:$sew), []>,
          Sched<[WriteVMovFS, ReadVMovFS]>,
          RISCVVPseudo;
        let HasVLOp = 1, HasSEWOp = 1, BaseInstr = VFMV_S_F,
            Constraints = "$rd = $rs1" in
        def "PseudoVFMV_S_" # f.FX # "_ALT_" # mx :
                                          Pseudo<(outs m.vrclass:$rd),
                                                 (ins m.vrclass:$rs1, f.fprclass:$rs2,
                                                      AVL:$vl, ixlenimm:$sew),
                                                 []>,
          Sched<[WriteVMovSF, ReadVMovSF_V, ReadVMovSF_F]>,
          RISCVVPseudo;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// 16.3. Vector Slide Instructions
//===----------------------------------------------------------------------===//
defm PseudoVFSLIDE1UP_ALT   : VPseudoVSLD1_VF<"@earlyclobber $rd">;
defm PseudoVFSLIDE1DOWN_ALT : VPseudoVSLD1_VF;
}

defm : VPatBinaryV_VV_VX_RM<"int_riscv_vfadd", "PseudoVFADD_ALT",
                            AllBFloatVectors, isSEWAware = 1>;
defm : VPatBinaryV_VV_VX_RM<"int_riscv_vfsub", "PseudoVFSUB_ALT",
                            AllBFloatVectors, isSEWAware = 1>;
defm : VPatBinaryV_VX_RM<"int_riscv_vfrsub", "PseudoVFRSUB_ALT",
                         AllBFloatVectors, isSEWAware = 1>;
defm : VPatBinaryW_VV_VX_RM<"int_riscv_vfwadd", "PseudoVFWADD_ALT",
                            AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatBinaryW_VV_VX_RM<"int_riscv_vfwsub", "PseudoVFWSUB_ALT",
                            AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatBinaryW_WV_WX_RM<"int_riscv_vfwadd_w", "PseudoVFWADD_ALT",
                            AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatBinaryW_WV_WX_RM<"int_riscv_vfwsub_w", "PseudoVFWSUB_ALT",
                            AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX_RM<"int_riscv_vfmul", "PseudoVFMUL_ALT",
                            AllBFloatVectors, isSEWAware=1>;
defm : VPatBinaryW_VV_VX_RM<"int_riscv_vfwmul", "PseudoVFWMUL_ALT",
                            AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfmacc", "PseudoVFMACC_ALT",
                                  AllBFloatVectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfnmacc", "PseudoVFNMACC_ALT",
                                  AllBFloatVectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfmsac", "PseudoVFMSAC_ALT",
                                  AllBFloatVectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfnmsac", "PseudoVFNMSAC_ALT",
                                  AllBFloatVectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfmadd", "PseudoVFMADD_ALT",
                                  AllBFloatVectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfnmadd", "PseudoVFNMADD_ALT",
                                  AllBFloatVectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfmsub", "PseudoVFMSUB_ALT",
                                  AllBFloatVectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfnmsub", "PseudoVFNMSUB_ALT",
                                  AllBFloatVectors, isSEWAware=1>;
defm : VPatTernaryW_VV_VX_RM<"int_riscv_vfwmacc", "PseudoVFWMACC_ALT",
                             AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatTernaryW_VV_VX_RM<"int_riscv_vfwnmacc", "PseudoVFWNMACC_ALT",
                             AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatTernaryW_VV_VX_RM<"int_riscv_vfwmsac", "PseudoVFWMSAC_ALT",
                             AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatTernaryW_VV_VX_RM<"int_riscv_vfwnmsac", "PseudoVFWNMSAC_ALT",
                             AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatUnaryV_V<"int_riscv_vfrsqrt7", "PseudoVFRSQRT7_ALT",
                    AllBFloatVectors, isSEWAware=1>;
defm : VPatUnaryV_V_RM<"int_riscv_vfrec7", "PseudoVFREC7_ALT",
                       AllBFloatVectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfmin", "PseudoVFMIN_ALT",
                         AllBFloatVectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfmax", "PseudoVFMAX_ALT",
                         AllBFloatVectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfsgnj", "PseudoVFSGNJ_ALT",
                         AllBFloatVectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfsgnjn", "PseudoVFSGNJN_ALT",
                         AllBFloatVectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfsgnjx", "PseudoVFSGNJX_ALT",
                         AllBFloatVectors, isSEWAware=1>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmfeq", "PseudoVMFEQ_ALT", AllBFloatVectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmfle", "PseudoVMFLE_ALT", AllBFloatVectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmflt", "PseudoVMFLT_ALT", AllBFloatVectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmfne", "PseudoVMFNE_ALT", AllBFloatVectors>;
defm : VPatBinaryM_VX<"int_riscv_vmfgt", "PseudoVMFGT_ALT", AllBFloatVectors>;
defm : VPatBinaryM_VX<"int_riscv_vmfge", "PseudoVMFGE_ALT", AllBFloatVectors>;
defm : VPatBinarySwappedM_VV<"int_riscv_vmfgt", "PseudoVMFLT_ALT", AllBFloatVectors>;
defm : VPatBinarySwappedM_VV<"int_riscv_vmfge", "PseudoVMFLE_ALT", AllBFloatVectors>;
defm : VPatConversionVI_VF<"int_riscv_vfclass", "PseudoVFCLASS_ALT", IsBF16=1>;
foreach vti = AllBFloatVectors in {
  let Predicates = GetVTypePredicates<vti>.Predicates in
    defm : VPatBinaryCarryInTAIL<"int_riscv_vfmerge", "PseudoVFMERGE_ALT",
                                 "V"#vti.ScalarSuffix#"M",
                                 vti.Vector,
                                 vti.Vector, vti.Scalar, vti.Mask,
                                 vti.Log2SEW, vti.LMul, vti.RegClass,
                                 vti.RegClass, vti.ScalarRegClass>;
}
defm : VPatConversionWF_VI<"int_riscv_vfwcvt_f_xu_v", "PseudoVFWCVT_F_XU_ALT",
                           isSEWAware=1, IsBF16=1>;
defm : VPatConversionWF_VI<"int_riscv_vfwcvt_f_x_v", "PseudoVFWCVT_F_X_ALT",
                           isSEWAware=1, IsBF16=1>;
defm : VPatConversionWF_VF<"int_riscv_vfwcvt_f_f_v", "PseudoVFWCVT_F_F_ALT",
                           isSEWAware=1, IsBF16=1>;
defm : VPatConversionVI_WF_RM<"int_riscv_vfncvt_xu_f_w", "PseudoVFNCVT_XU_F_ALT",
                              IsBF16=1>;
defm : VPatConversionVI_WF_RM<"int_riscv_vfncvt_x_f_w", "PseudoVFNCVT_X_F_ALT",
                              IsBF16=1>;
defm : VPatConversionVI_WF<"int_riscv_vfncvt_rtz_xu_f_w", "PseudoVFNCVT_RTZ_XU_F_ALT",
                           IsBF16=1>;
defm : VPatConversionVI_WF<"int_riscv_vfncvt_rtz_x_f_w", "PseudoVFNCVT_RTZ_X_F_ALT",
                           IsBF16=1>;
defm : VPatConversionVF_WF_RM<"int_riscv_vfncvt_f_f_w", "PseudoVFNCVT_F_F_ALT",
                              AllWidenableBFloatToFloatVectors, isSEWAware=1>;
defm : VPatConversionVF_WF<"int_riscv_vfncvt_rod_f_f_w", "PseudoVFNCVT_ROD_F_F_ALT",
                           isSEWAware=1, IsBF16=1>;
defm : VPatBinaryV_VX<"int_riscv_vfslide1up", "PseudoVFSLIDE1UP_ALT", AllBFloatVectors>;
defm : VPatBinaryV_VX<"int_riscv_vfslide1down", "PseudoVFSLIDE1DOWN_ALT", AllBFloatVectors>;

foreach fvti = AllBFloatVectors in {
  let Predicates = GetVTypePredicates<fvti>.Predicates in {
    def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                           fvti.Vector:$passthru, (fvti.Scalar fvti.ScalarRegClass:$rs2), VLOpFrag)),
              (!cast<Instruction>("PseudoVFMV_V_ALT_" # fvti.ScalarSuffix # "_" #
                                  fvti.LMul.MX)
               $passthru, (fvti.Scalar fvti.ScalarRegClass:$rs2),
               GPR:$vl, fvti.Log2SEW, TU_MU)>;
  }
}

foreach vti = AllBFloatVectors in {
  let Predicates = GetVTypePredicates<vti>.Predicates in {
    def : Pat<(vti.Vector (riscv_vfmv_s_f_vl (vti.Vector vti.RegClass:$merge),
                                             vti.ScalarRegClass:$rs1,
                                             VLOpFrag)),
              (!cast<Instruction>("PseudoVFMV_S_"#vti.ScalarSuffix#"_ALT_"#vti.LMul.MX)
                  vti.RegClass:$merge,
                  (vti.Scalar vti.ScalarRegClass:$rs1), GPR:$vl, vti.Log2SEW)>;
  }
}

foreach vti = AllBFloatVectors in {
  defvar vmv_f_s_inst = !cast<Instruction>(!strconcat("PseudoVFMV_",
                                                       vti.ScalarSuffix,
                                                       "_S_ALT_", vti.LMul.MX));
  // Only pattern-match extract-element operations where the index is 0. Any
  // other index will have been custom-lowered to slide the vector correctly
  // into place.
  let Predicates = GetVTypePredicates<vti>.Predicates in
  def : Pat<(vti.Scalar (extractelt (vti.Vector vti.RegClass:$rs2), 0)),
            (vmv_f_s_inst vti.RegClass:$rs2, vti.Log2SEW)>;
}

} // Predicates = [HasVendorXSfvfbfa]

// SSCI

def uimm3_timm : RISCVOp<i32>,
                 TImmLeaf<i32, [{return isUInt<3>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<3>;
  let DecoderMethod = "decodeUImmOperand<3>";
  let OperandType = "OPERAND_UIMM3";
}
def uimm7_timm : RISCVOp<i32>,
                 TImmLeaf<i32, [{return isUInt<7>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<7>;
  let DecoderMethod = "decodeUImmOperand<7>";
  let OperandType = "OPERAND_UIMM7";
}
def uimm12_timm : RISCVOp<i32>,
                  TImmLeaf<i32, [{return isUInt<12>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<12>;
  let DecoderMethod = "decodeUImmOperand<12>";
  let OperandType = "OPERAND_UIMM12";
}
def uimm20_timm : RISCVOp<i32>,
                  TImmLeaf<i32, [{return isUInt<20>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<20>;
  let DecoderMethod = "decodeUImmOperand<20>";
  let OperandType = "OPERAND_UIMM20";
}
def uimm25_timm : RISCVOp<i32>,
                  TImmLeaf<i32, [{return isUInt<25>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<25>;
  let DecoderMethod = "decodeUImmOperand<25>";
  let OperandType = "OPERAND_UIMM25";
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class SiFiveSCI_R<RISCVOpcode opcode, string opcodestr>
    : RVInst<(outs GPR:$rd),
             (ins uimm3_timm:$funct3, uimm7_timm:$funct7, GPR:$rs1, GPR:$rs2),
             opcodestr, "$funct3, $funct7, $rd, $rs1, $rs2", [],
             InstFormatR>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]> {
  bits<7> funct7;
  bits<3> funct3;

  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class SiFiveSCI_I<RISCVOpcode opcode, string opcodestr>
    : RVInst<(outs GPR:$rd),
             (ins uimm3_timm:$funct3, GPR:$rs1, uimm12_timm:$imm12),
             opcodestr, "$funct3, $rd, $rs1, $imm12", [],
             InstFormatI>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]> {
  bits<3> funct3;

  bits<12> imm12;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-20} = imm12;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class SiFiveSCI_U<RISCVOpcode opcode, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins uimm20_timm:$imm20),
             opcodestr, "$rd, $imm20", [],
             InstFormatU>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]> {
  bits<20> imm20;
  bits<5> rd;

  let Inst{31-12} = imm20;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class SiFiveSCI_X<RISCVOpcode opcode, string opcodestr>
    : RVInst<(outs), (ins uimm25_timm:$imm25),
             opcodestr, "$imm25", [],
             InstFormatOther>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]> {
  bits<25> imm25;

  let Inst{31-7} = imm25;
  let Inst{6-0} = opcode.Value;
}

let Predicates = [HasVendorXSfsci], DecoderNamespace = "XSfsci" in {
def SF_SCI_0_R : SiFiveSCI_R<OPC_CUSTOM_0, "sf.sci.0.r">;
def SF_SCI_1_R : SiFiveSCI_R<OPC_CUSTOM_1, "sf.sci.1.r">;
def SF_SCI_2_R : SiFiveSCI_R<OPC_CUSTOM_2, "sf.sci.2.r">;
def SF_SCI_3_R : SiFiveSCI_R<OPC_CUSTOM_3, "sf.sci.3.r">;
}

// FIXME: This is the old name. We should remove once everyone is migrated.
def : MnemonicAlias<"sf.sci", "sf.sci.3.r">;

let Predicates = [HasVendorXSfsci], DecoderNamespace = "XSfsciI" in {
def SF_SCI_0_I : SiFiveSCI_I<OPC_CUSTOM_0, "sf.sci.0.i">;
def SF_SCI_1_I : SiFiveSCI_I<OPC_CUSTOM_1, "sf.sci.1.i">;
def SF_SCI_2_I : SiFiveSCI_I<OPC_CUSTOM_2, "sf.sci.2.i">;
def SF_SCI_3_I : SiFiveSCI_I<OPC_CUSTOM_3, "sf.sci.3.i">;
}

let Predicates = [HasVendorXSfsci], DecoderNamespace = "XSfsciU" in {
def SF_SCI_0_U : SiFiveSCI_U<OPC_CUSTOM_0, "sf.sci.0.u">;
def SF_SCI_1_U : SiFiveSCI_U<OPC_CUSTOM_1, "sf.sci.1.u">;
def SF_SCI_2_U : SiFiveSCI_U<OPC_CUSTOM_2, "sf.sci.2.u">;
def SF_SCI_3_U : SiFiveSCI_U<OPC_CUSTOM_3, "sf.sci.3.u">;
}

let Predicates = [HasVendorXSfsci], DecoderNamespace = "XSfsciX" in {
def SF_SCI_0_X : SiFiveSCI_X<OPC_CUSTOM_0, "sf.sci.0.x">;
def SF_SCI_1_X : SiFiveSCI_X<OPC_CUSTOM_1, "sf.sci.1.x">;
def SF_SCI_2_X : SiFiveSCI_X<OPC_CUSTOM_2, "sf.sci.2.x">;
def SF_SCI_3_X : SiFiveSCI_X<OPC_CUSTOM_3, "sf.sci.3.x">;
}

// Pseudo instructions with side effects. Will be converted to the generic
// version when we convert MachineInstr to MCInst.
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
multiclass SiFiveSCI_R_Pseudos<Instruction Instr> {
  def _R_X_XX_SE
    : Pseudo<(outs GPR:$rd),
             (ins uimm3_timm:$funct3, uimm7_timm:$funct7, GPR:$rs1, GPR:$rs2), []>,
             PseudoInstExpansion<(Instr GPR:$rd, uimm3_timm:$funct3,
                                        uimm7_timm:$funct7, GPR:$rs1, GPR:$rs2)>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]>;
  def _R_XX_SE
    : Pseudo<(outs), (ins uimm3_timm:$funct3, uimm7_timm:$funct7, GPR:$rs1, GPR:$rs2), []>,
             PseudoInstExpansion<(Instr X0, uimm3_timm:$funct3, uimm7_timm:$funct7,
                                        GPR:$rs1, GPR:$rs2)>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]>;
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
multiclass SiFiveSCI_I_Pseudos<Instruction Instr> {
  def _I_X_XI_SE
    : Pseudo<(outs GPR:$rd),
             (ins uimm3_timm:$funct3, GPR:$rs1, uimm12_timm:$imm12), []>,
             PseudoInstExpansion<(Instr GPR:$rd, uimm3_timm:$funct3,
                                        GPR:$rs1, uimm12_timm:$imm12)>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]>;
  def _I_XI_SE
    : Pseudo<(outs), (ins uimm3_timm:$funct3, GPR:$rs1, uimm12_timm:$imm12), []>,
             PseudoInstExpansion<(Instr X0, uimm3_timm:$funct3,
                                        GPR:$rs1, uimm12_timm:$imm12)>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]>;
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
multiclass SiFiveSCI_U_Pseudos<Instruction Instr> {
  def _U_X_I_SE
    : Pseudo<(outs GPR:$rd), (ins uimm20_timm:$imm20), []>,
             PseudoInstExpansion<(Instr GPR:$rd, uimm20_timm:$imm20)>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]>;
  def _U_I_SE
    : Pseudo<(outs), (ins uimm20_timm:$imm20), []>,
             PseudoInstExpansion<(Instr X0, uimm20_timm:$imm20)>,
             Sched<[WriteSCI, ReadSCI, ReadSCI]>;
}

defm SF_SCI_0 : SiFiveSCI_R_Pseudos<SF_SCI_0_R>;
defm SF_SCI_1 : SiFiveSCI_R_Pseudos<SF_SCI_1_R>;
defm SF_SCI_2 : SiFiveSCI_R_Pseudos<SF_SCI_2_R>;
defm SF_SCI_3 : SiFiveSCI_R_Pseudos<SF_SCI_3_R>;

defm SF_SCI_0 : SiFiveSCI_I_Pseudos<SF_SCI_0_I>;
defm SF_SCI_1 : SiFiveSCI_I_Pseudos<SF_SCI_1_I>;
defm SF_SCI_2 : SiFiveSCI_I_Pseudos<SF_SCI_2_I>;
defm SF_SCI_3 : SiFiveSCI_I_Pseudos<SF_SCI_3_I>;

defm SF_SCI_0 : SiFiveSCI_U_Pseudos<SF_SCI_0_U>;
defm SF_SCI_1 : SiFiveSCI_U_Pseudos<SF_SCI_1_U>;
defm SF_SCI_2 : SiFiveSCI_U_Pseudos<SF_SCI_2_U>;
defm SF_SCI_3 : SiFiveSCI_U_Pseudos<SF_SCI_3_U>;

multiclass SiFiveSCI_R_Pats<Intrinsic int_x_xx, Instruction X_XX,
                            Intrinsic int_x_xx_se, Instruction X_XX_SE,
                            Intrinsic int_xx_se, Instruction XX_SE> {
  def : Pat<(XLenVT (int_x_xx uimm3_timm:$funct3, uimm7_timm:$funct7,
                              (XLenVT GPR:$rs1), (XLenVT GPR:$rs2))),
            (X_XX uimm3_timm:$funct3, uimm7_timm:$funct7, GPR:$rs1, GPR:$rs2)>;

  def : Pat<(XLenVT (int_x_xx_se uimm3_timm:$funct3, uimm7_timm:$funct7,
                                 (XLenVT GPR:$rs1), (XLenVT GPR:$rs2))),
            (X_XX_SE uimm3_timm:$funct3, uimm7_timm:$funct7, GPR:$rs1, GPR:$rs2)>;

  def : Pat<(int_xx_se uimm3_timm:$funct3, uimm7_timm:$funct7,
                       (XLenVT GPR:$rs1), (XLenVT GPR:$rs2)),
            (XX_SE uimm3_timm:$funct3, uimm7_timm:$funct7, GPR:$rs1, GPR:$rs2)>;
}

multiclass SiFiveSCI_I_Pats<Intrinsic int_x_xi, Instruction X_XI,
                            Intrinsic int_x_xi_se, Instruction X_XI_SE,
                            Intrinsic int_xi_se, Instruction XI_SE> {
  def : Pat<(XLenVT (int_x_xi uimm3_timm:$funct3, (XLenVT GPR:$rs1),
                              uimm12_timm:$imm12)),
            (X_XI uimm3_timm:$funct3, GPR:$rs1, uimm12_timm:$imm12)>;

  def : Pat<(XLenVT (int_x_xi_se uimm3_timm:$funct3, (XLenVT GPR:$rs1),
                                 uimm12_timm:$imm12)),
            (X_XI_SE uimm3_timm:$funct3, GPR:$rs1, uimm12_timm:$imm12)>;

  def : Pat<(int_xi_se uimm3_timm:$funct3, (XLenVT GPR:$rs1), uimm12_timm:$imm12),
            (XI_SE uimm3_timm:$funct3, GPR:$rs1, uimm12_timm:$imm12)>;
}

multiclass SiFiveSCI_U_Pats<Intrinsic int_x_i, Instruction X_I,
                            Intrinsic int_x_i_se, Instruction X_I_SE,
                            Intrinsic int_i_se, Instruction I_SE> {
  def : Pat<(XLenVT (int_x_i uimm20_timm:$imm20)),
            (X_I uimm20_timm:$imm20)>;

  def : Pat<(XLenVT (int_x_i_se uimm20_timm:$imm20)),
            (X_I_SE uimm20_timm:$imm20)>;

  def : Pat<(int_i_se uimm20_timm:$imm20),
            (I_SE uimm20_timm:$imm20)>;
}

multiclass SiFiveSCI_X_Pats<Intrinsic int_i_se, Instruction I> {
  def : Pat<(int_i_se uimm25_timm:$imm25),
            (I uimm25_timm:$imm25)>;
}

defm SF_SCI_0 : SiFiveSCI_R_Pats<int_riscv_sf_sci_0_r_x_xx, SF_SCI_0_R,
                                 int_riscv_sf_sci_0_r_x_xx_se, SF_SCI_0_R_X_XX_SE,
                                 int_riscv_sf_sci_0_r_xx_se, SF_SCI_0_R_XX_SE>;
defm SF_SCI_1 : SiFiveSCI_R_Pats<int_riscv_sf_sci_1_r_x_xx, SF_SCI_1_R,
                                 int_riscv_sf_sci_1_r_x_xx_se, SF_SCI_1_R_X_XX_SE,
                                 int_riscv_sf_sci_1_r_xx_se, SF_SCI_1_R_XX_SE>;
defm SF_SCI_2 : SiFiveSCI_R_Pats<int_riscv_sf_sci_2_r_x_xx, SF_SCI_2_R,
                                 int_riscv_sf_sci_2_r_x_xx_se, SF_SCI_2_R_X_XX_SE,
                                 int_riscv_sf_sci_2_r_xx_se, SF_SCI_2_R_XX_SE>;
defm SF_SCI_3 : SiFiveSCI_R_Pats<int_riscv_sf_sci_3_r_x_xx, SF_SCI_3_R,
                                 int_riscv_sf_sci_3_r_x_xx_se, SF_SCI_3_R_X_XX_SE,
                                 int_riscv_sf_sci_3_r_xx_se, SF_SCI_3_R_XX_SE>;

defm SF_SCI_0 : SiFiveSCI_I_Pats<int_riscv_sf_sci_0_i_x_xi, SF_SCI_0_I,
                                 int_riscv_sf_sci_0_i_x_xi_se, SF_SCI_0_I_X_XI_SE,
                                 int_riscv_sf_sci_0_i_xi_se, SF_SCI_0_I_XI_SE>;
defm SF_SCI_1 : SiFiveSCI_I_Pats<int_riscv_sf_sci_1_i_x_xi, SF_SCI_1_I,
                                 int_riscv_sf_sci_1_i_x_xi_se, SF_SCI_1_I_X_XI_SE,
                                 int_riscv_sf_sci_1_i_xi_se, SF_SCI_1_I_XI_SE>;
defm SF_SCI_2 : SiFiveSCI_I_Pats<int_riscv_sf_sci_2_i_x_xi, SF_SCI_2_I,
                                 int_riscv_sf_sci_2_i_x_xi_se, SF_SCI_2_I_X_XI_SE,
                                 int_riscv_sf_sci_2_i_xi_se, SF_SCI_2_I_XI_SE>;
defm SF_SCI_3 : SiFiveSCI_I_Pats<int_riscv_sf_sci_3_i_x_xi, SF_SCI_3_I,
                                 int_riscv_sf_sci_3_i_x_xi_se, SF_SCI_3_I_X_XI_SE,
                                 int_riscv_sf_sci_3_i_xi_se, SF_SCI_3_I_XI_SE>;

defm SF_SCI_0 : SiFiveSCI_U_Pats<int_riscv_sf_sci_0_u_x_i, SF_SCI_0_U,
                                 int_riscv_sf_sci_0_u_x_i_se, SF_SCI_0_U_X_I_SE,
                                 int_riscv_sf_sci_0_u_i_se, SF_SCI_0_U_I_SE>;
defm SF_SCI_1 : SiFiveSCI_U_Pats<int_riscv_sf_sci_1_u_x_i, SF_SCI_1_U,
                                 int_riscv_sf_sci_1_u_x_i_se, SF_SCI_1_U_X_I_SE,
                                 int_riscv_sf_sci_1_u_i_se, SF_SCI_1_U_I_SE>;
defm SF_SCI_2 : SiFiveSCI_U_Pats<int_riscv_sf_sci_2_u_x_i, SF_SCI_2_U,
                                 int_riscv_sf_sci_2_u_x_i_se, SF_SCI_2_U_X_I_SE,
                                 int_riscv_sf_sci_2_u_i_se, SF_SCI_2_U_I_SE>;
defm SF_SCI_3 : SiFiveSCI_U_Pats<int_riscv_sf_sci_3_u_x_i, SF_SCI_3_U,
                                 int_riscv_sf_sci_3_u_x_i_se, SF_SCI_3_U_X_I_SE,
                                 int_riscv_sf_sci_3_u_i_se, SF_SCI_3_U_I_SE>;

defm SF_SCI_0 : SiFiveSCI_X_Pats<int_riscv_sf_sci_0_x_i_se, SF_SCI_0_X>;
defm SF_SCI_1 : SiFiveSCI_X_Pats<int_riscv_sf_sci_1_x_i_se, SF_SCI_1_X>;
defm SF_SCI_2 : SiFiveSCI_X_Pats<int_riscv_sf_sci_2_x_i_se, SF_SCI_2_X>;
defm SF_SCI_3 : SiFiveSCI_X_Pats<int_riscv_sf_sci_3_x_i_se, SF_SCI_3_X>;
#endif // SIFIVE_CUSTOMIZATION
