//===-- RISCVInstrInfoZvk.td - RISC-V 'Zvk' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvk',
// Vector Cryptography Instructions extension, version 1.0.0-rc1.
//
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
multiclass VCLMUL_MV_V_X<string opcodestr, bits<6> funct6> {
  def V  : VALUVV<funct6, OPMVV, opcodestr # "." # "vv">,
           Sched<[WriteVIALUV_WorstCase, ReadVIALUV_WorstCase,
                  ReadVIALUV_WorstCase, ReadVMask]>;
  def X  : VALUVX<funct6, OPMVX, opcodestr # "." # "vx">,
           Sched<[WriteVIALUX_WorstCase, ReadVIALUV_WorstCase,
                  ReadVIALUX_WorstCase, ReadVMask]>;
}

class RVInstIVI_VROR<bits<6> funct6, dag outs, dag ins, string opcodestr,
                     string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<6> imm;
  bits<5> vd;
  bit vm;

  let Inst{31-27} = funct6{5-1};
  let Inst{26} = imm{5};
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = imm{4-0};
  let Inst{14-12} = OPIVI.Value;
  let Inst{11-7} = vd;
  let Opcode = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
  let RVVConstraint = VMConstraint;
}

multiclass VROR_IV_V_X_I<string opcodestr, bits<6> funct6,
                              Operand optype = uimm6, string vw = "v">
    : VALU_IV_V_X<opcodestr, funct6> {
  def I : RVInstIVI_VROR<funct6, (outs VR:$vd),
              (ins VR:$vs2, optype:$imm, VMaskOp:$vm),
              opcodestr # "." # vw # "i", "$vd, $vs2, $imm$vm">,
         Sched<[WriteVIALUI_WorstCase, ReadVIALUV_WorstCase,
                ReadVMask]>;
}

// op vd, vs2, vs1
class PALUVVNoVm<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : VALUVVNoVm<funct6, opv, opcodestr> {
  let Opcode = OPC_OP_P.Value;
}

// op vd, vs2, imm, vm
class PALUVINoVm<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : VALUVINoVm<funct6, opcodestr, optype> {
  let Opcode = OPC_OP_P.Value;
  let Inst{14-12} = OPMVV.Value;
}

// op vd, vs2 (use vs1 as instruction encoding)
class PALUVs2NoVm<bits<6> funct6, bits<5> vs1, RISCVVFormat opv, string opcodestr>
    : VALUVs2NoVm<funct6, vs1, opv, opcodestr> {
  let Opcode = OPC_OP_P.Value;
}

multiclass VAES_MV_V_S<bits<6> funct6_vv, bits<6> funct6_vs, bits<5> vs1,
                         RISCVVFormat opv, string opcodestr> {
  def NAME # _VV : PALUVs2NoVm<funct6_vv, vs1, opv, opcodestr # ".vv">;
  def NAME # _VS : PALUVs2NoVm<funct6_vs, vs1, opv, opcodestr # ".vs">;
}

// vaeskf1.vi and vaeskf2.vi uses different opcode and format, we need
// to customize one for them.
class VAESKF_MV_I<bits<6> funct6, string opcodestr, Operand optype>
    : VALUVINoVm<funct6, opcodestr, optype> {
  let Opcode = OPC_OP_P.Value;
  let Inst{14-12} = OPMVV.Value;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Predicates = [HasStdExtZvbbOrZvkb0p1] in {
  let Predicates = [HasStdExtZvkb0p1], DecoderNamespace = "Zvk0p1" in
    defm VANDN_0P1_V : VALU_IV_V_X_I<"vandn", 0b000001>;
  def VBREV8_V : VALUVs2<0b010010, 0b01000, OPMVV, "vbrev8.v">;
  def VREV8_V : VALUVs2<0b010010, 0b01001, OPMVV, "vrev8.v">;
  defm VROL_V : VALU_IV_V_X<"vrol", 0b010101>;
  defm VROR_V : VROR_IV_V_X_I<"vror", 0b010100>;
  defm VANDN_V : VALU_IV_V_X_I<"vandn", 0b000001>;
  def  VBREV_V  : VALUVs2<0b010010, 0b01010, OPMVV, "vbrev.v">;
  def  VCLZ_V   : VALUVs2<0b010010, 0b01100, OPMVV, "vclz.v">;
  def  VCPOP_V  : VALUVs2<0b010010, 0b01110, OPMVV, "vcpop.v">;
  def  VCTZ_V   : VALUVs2<0b010010, 0b01101, OPMVV, "vctz.v">;
  let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in
  defm VWSLL_V  : VSHT_IV_V_X_I<"vwsll", 0b110101>;
} // Predicates = [HasStdExtZvbbOrZvkb0p1]

let Predicates = [HasStdExtZvbcOrZvkb0p1] in {
  defm VCLMUL_V  : VCLMUL_MV_V_X<"vclmul", 0b001100>;
  defm VCLMULH_V : VCLMUL_MV_V_X<"vclmulh", 0b001101>;
} // Predicates = [HasStdExtZvbcOrZvkb0p1]

let Predicates = [HasStdExtZvkg], RVVConstraint = NoConstraint in {
  def VGHSH_VV : PALUVVNoVm<0b101100, OPMVV, "vghsh.vv">;
  def VGMUL_VV : PALUVs2NoVm<0b101000, 0b10001, OPMVV, "vgmul.vv">;
} // Predicates = [HasStdExtZvkg]

let Predicates = [HasStdExtZvknha], RVVConstraint = NoConstraint in {
  def VSHA2CH_VV : PALUVVNoVm<0b101110, OPMVV, "vsha2ch.vv">;
  def VSHA2CL_VV : PALUVVNoVm<0b101111, OPMVV, "vsha2cl.vv">;
  def VSHA2MS_VV : PALUVVNoVm<0b101101, OPMVV, "vsha2ms.vv">;
} // Predicates = [HasStdExtZvknha]

let Predicates = [HasStdExtZvknsOrZvkned], RVVConstraint = NoConstraint in {
  defm VAESDF : VAES_MV_V_S<0b101000, 0b101001, 0b00001, OPMVV, "vaesdf">;
  defm VAESDM : VAES_MV_V_S<0b101000, 0b101001, 0b00000, OPMVV, "vaesdm">;
  defm VAESEF : VAES_MV_V_S<0b101000, 0b101001, 0b00011, OPMVV, "vaesef">;
  defm VAESEM : VAES_MV_V_S<0b101000, 0b101001, 0b00010, OPMVV, "vaesem">;
#ifdef SIFIVE_CUSTOMIZATION
  def VAESKF1_VI : VAESKF_MV_I<0b100010, "vaeskf1.vi", uimm5>;
  def VAESKF2_VI : VAESKF_MV_I<0b101010, "vaeskf2.vi", uimm5>;
#endif // SIFIVE_CUSTOMIZATION
  def VAESZ_VS : PALUVs2NoVm<0b101001, 0b00111, OPMVV, "vaesz.vs">;
} // Predicates = [HasStdExtZvknsOrZvkned]

let Predicates = [HasStdExtZvksed], RVVConstraint = NoConstraint in {
#ifdef SIFIVE_CUSTOMIZATION
  def  VSM4K_VI : PALUVINoVm<0b100001, "vsm4k.vi", uimm5>;
#endif // SIFIVE_CUSTOMIZATION
  defm VSM4R    : VAES_MV_V_S<0b101000, 0b101001, 0b10000, OPMVV, "vsm4r">;
} // Predicates = [HasStdExtZvksed]

let Predicates = [HasStdExtZvksh], RVVConstraint = NoConstraint in {
  def VSM3C_VI  : PALUVINoVm<0b101011, "vsm3c.vi", uimm5>;
  def VSM3ME_VV : PALUVVNoVm<0b100000, OPMVV, "vsm3me.vv">;
} // Predicates = [HasStdExtZvksh]

#ifdef SIFIVE_CUSTOMIZATION
let Predicates = [HasStdExtZvkg0p1],
    RVVConstraint = NoConstraint,
    DecoderNamespace = "Zvk0p1" in {
  def VGHMAC_VV : PALUVVNoVm<0b101100, OPMVV, "vghmac.vv">;
} // Predicates = [HasStdExtZvkg0p1]
#endif // SIFIVE_CUSTOMIZATION

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

#ifdef SIFIVE_CUSTOMIZATION
multiclass VPseudoUnaryV_V<LMULInfo m, string Constraint = ""> {
  let VLMul = m.value in {
    def "_V_" # m.MX : VPseudoUnaryNoMask<m.vrclass, m.vrclass, Constraint>;

    def "_V_" # m.MX # "_MASK" : VPseudoUnaryMaskTA<m.vrclass, m.vrclass, Constraint>,
                                 RISCVMaskedPseudo</*MaskOpIdx*/ 2>;
  }
}

class PPseudoUnaryNoMask<DAGOperand RetClass, VReg OpClass, string Constraint = ""> :
      Pseudo<(outs RetClass:$rd),
        (ins RetClass:$merge, OpClass:$rs2, AVL:$vl, ixlenimm:$sew, ixlenimm:$policy), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = !interleave([Constraint, "$rd = $merge"], ",");
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasVecPolicyOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

class PPseudoBinaryNoMask<VReg RetClass,
                          VReg Op1Class,
                          DAGOperand Op2Class,
                          string Constraint> :
        Pseudo<(outs RetClass:$rd),
               (ins RetClass:$merge, Op1Class:$rs2, Op2Class:$rs1,
                    AVL:$vl, ixlenimm:$sew, ixlenimm:$policy), []>,
        RISCVVPseudo {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = !interleave([Constraint, "$rd = $merge"], ",");
  let HasVLOp = 1;
  let HasSEWOp = 1;
  let HasVecPolicyOp = 1;
  let BaseInstr = !cast<Instruction>(PseudoToVInst<NAME>.VInst);
}

multiclass PPseudoBinaryNoMask<VReg RetClass,
                               VReg Op1Class,
                               DAGOperand Op2Class,
                               LMULInfo MInfo,
                               string Constraint = ""> {
  let VLMul = MInfo.value in
    def "_" # MInfo.MX : PPseudoBinaryNoMask<RetClass, Op1Class, Op2Class,
                                             Constraint>;
}

multiclass PPseudoBinaryV_VV_NoMask<LMULInfo m, string Constraint = ""> {
  defm _VV : PPseudoBinaryNoMask<m.vrclass, m.vrclass, m.vrclass, m, Constraint>;
}


multiclass PPseudoUnaryV_V_NoMask<LMULInfo m, string Constraint = ""> {
  let VLMul = m.value in {
    def "_VV_" # m.MX : PPseudoUnaryNoMask<m.vrclass, m.vrclass, Constraint>;
  }
}

multiclass PPseudoUnaryV_S_NoMask<LMULInfo m, string suffix, string Constraint = ""> {
  let VLMul = m.value in {
    def "_" # suffix # "_" # m.MX : PPseudoUnaryNoMask<m.vrclass, V_M1.vrclass, Constraint>;
  }
}

multiclass PPseudoBinaryV_VI<LMULInfo m, Operand ImmType = simm5, string Constraint = ""> {
  defm _VI : VPseudoBinary<m.vrclass, m.vrclass, ImmType, m, Constraint>;
}

multiclass PPseudoBinaryV_VI_NoMask<LMULInfo m, Operand ImmType = simm5, string Constraint = ""> {
  defm _VI : PPseudoBinaryNoMask<m.vrclass, m.vrclass, ImmType, m, Constraint>;
}

multiclass VPseudoVALU_V<string Constraint = ""> {
  foreach m = MxList in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);

    defm "" : VPseudoUnaryV_V<m, Constraint>,
              Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

multiclass PPseudoVALU_V_NoMask<string Constraint = ""> {
  foreach m = MxListVF4 in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);

    defm "" : PPseudoUnaryV_V_NoMask<m, Constraint>,
              Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

multiclass PPseudoVALU_S_NoMask<string suffix = "VS", string Constraint = ""> {
  foreach m = MxListVF4 in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);

    defm "" : PPseudoUnaryV_S_NoMask<m, suffix, Constraint>,
              Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

multiclass PPseudoVALU_V_S_NoMask<string Constraint = ""> {
  defm "" : PPseudoVALU_V_NoMask<Constraint>;
  defm "" : PPseudoVALU_S_NoMask<"VS", Constraint>;
}

multiclass PPseudoVALU_VV_NoMask<string Constraint = ""> {
  foreach m = MxListVF4 in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);

    defm "" : PPseudoBinaryV_VV_NoMask<m, Constraint>,
              Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

multiclass PPseudoVALU_VI<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxListVF4 in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);

    defm "" : PPseudoBinaryV_VI<m, ImmType, Constraint>,
              Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

multiclass PPseudoVALU_VI_NoMask<Operand ImmType = simm5, string Constraint = ""> {
  foreach m = MxListVF4 in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);

    defm "" : PPseudoBinaryV_VI_NoMask<m, ImmType, Constraint>,
              Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

multiclass PPseudoVALU_VI_NoMaskTU<Operand ImmType = uimm5, string Constraint = ""> {
  foreach m = MxListVF4 in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);

    def "_VI_" # m.MX : VPseudoBinaryNoMaskTU<m.vrclass, m.vrclass, ImmType, Constraint>,
                                Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

multiclass PPseudoVALU_VV_NoMaskTU<string Constraint = ""> {
  foreach m = MxListVF4 in {
    defvar mx = m.MX;
    defvar WriteVIALUV_MX = !cast<SchedWrite>("WriteVIALUV_" # mx);
    defvar ReadVIALUV_MX = !cast<SchedRead>("ReadVIALUV_" # mx);

    def "_VV_" # m.MX : VPseudoBinaryNoMaskTU<m.vrclass, m.vrclass, m.vrclass, Constraint>,
                                Sched<[WriteVIALUV_MX, ReadVIALUV_MX, ReadVIALUV_MX, ReadVMask]>;
  }
}

#endif // SIFIVE_CUSTOMIZATION

#ifdef SIFIVE_CUSTOMIZATION
let Predicates = [HasStdExtZvknsOrZvkned] in {
  defm PseudoVAESDF : PPseudoVALU_V_S_NoMask;
  defm PseudoVAESDM : PPseudoVALU_V_S_NoMask;
  defm PseudoVAESEF : PPseudoVALU_V_S_NoMask;
  defm PseudoVAESEM : PPseudoVALU_V_S_NoMask;
  defm PseudoVAESKF1 : PPseudoVALU_VI_NoMaskTU;
  defm PseudoVAESKF2 : PPseudoVALU_VI_NoMask<uimm5>;
  defm PseudoVAESZ : PPseudoVALU_S_NoMask;
} // Predicates = [HasStdExtZvkns]

let Predicates = [HasStdExtZvbbOrZvkb0p1] in {
  defm PseudoVANDN : VPseudoVALU_VV_VX;
  defm PseudoVBREV8 : VPseudoVALU_V;
  defm PseudoVROL : VPseudoVALU_VV_VX;
  defm PseudoVROR : VPseudoVALU_VV_VX_VI;
} // Predicates = [HasStdExtZvbbOrZvkb0p1]

let Predicates = [HasStdExtZvbcOrZvkb0p1] in {
  defm PseudoVCLMUL : VPseudoVALU_VCLMUL;
  defm PseudoVCLMULH : VPseudoVALU_VCLMUL;
} // Predicates = [HasStdExtZvbcOrZvkb0p1]

let Predicates = [HasStdExtZvkg] in {
  defm PseudoVGHSH : PPseudoVALU_VV_NoMask;
  defm PseudoVGMUL : PPseudoVALU_S_NoMask<"VV">;
} // Predicates = [HasStdExtZvkg]

let Predicates = [HasStdExtZvkg0p1] in {
  defm PseudoVGHMAC : PPseudoVALU_VV_NoMask;
} // Predicates = [HasStdExtZvkg0p1]

let Predicates = [HasStdExtZvksed] in {
  defm PseudoVSM4K : PPseudoVALU_VI_NoMaskTU;
  defm PseudoVSM4R : PPseudoVALU_V_S_NoMask;
} // Predicates = [HasStdExtZvksed]

let Predicates = [HasStdExtZvksh] in {
  defm PseudoVSM3C : PPseudoVALU_VI_NoMask<uimm5>;
  defm PseudoVSM3ME : PPseudoVALU_VV_NoMaskTU;
} // Predicates = [HasStdExtZvksh]

let Predicates = [HasStdExtZvknha] in {
  defm PseudoVSHA2CH : PPseudoVALU_VV_NoMask;
  defm PseudoVSHA2CL : PPseudoVALU_VV_NoMask;
  defm PseudoVSHA2MS : PPseudoVALU_VV_NoMask;
} // Predicates = [HasStdExtZvknha]

multiclass VPatBinaryTANoMask<string intrinsic,
                              string inst,
                              ValueType result_type,
                              ValueType op1_type,
                              ValueType op2_type,
                              int sew,
                              VReg result_reg_class,
                              VReg op1_reg_class,
                              DAGOperand op2_kind>
{
  def : VPatBinaryNoMaskTU<intrinsic, inst, result_type, op1_type, op2_type,
                           sew, result_reg_class, op1_reg_class, op2_kind>;
}

multiclass VPatBinaryV_VI_NoMaskTA<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist, Operand imm_type = simm5> {
  foreach vti = vtilist in
    defm : VPatBinaryTANoMask<intrinsic, instruction # "_VI_" # vti.LMul.MX,
                        vti.Vector, vti.Vector, XLenVT, vti.Log2SEW,
                        vti.RegClass, vti.RegClass, imm_type>;
}

multiclass VPatBinaryV_VV_NoMaskTA<string intrinsic, string instruction,
                          list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    defm : VPatBinaryTANoMask<intrinsic, instruction # "_VV_" # vti.LMul.MX,
                        vti.Vector, vti.Vector, vti.Vector, vti.Log2SEW,
                        vti.RegClass, vti.RegClass, vti.RegClass>;
}

class PPatUnaryNoMask<string intrinsic_name,
                      string inst,
                      string kind,
                      ValueType result_type,
                      ValueType op2_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg result_reg_class,
                      VReg op2_reg_class> :
  Pat<(result_type (!cast<Intrinsic>(intrinsic_name)
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   VLOpFrag, (XLenVT timm:$policy))),
                   (!cast<Instruction>(inst#"_"#kind#"_"#vlmul.MX)
                   (result_type result_reg_class:$merge),
                   (op2_type op2_reg_class:$rs2),
                   GPR:$vl, sew, (XLenVT timm:$policy))>;

multiclass PPatUnaryV_V_NoMask<string intrinsic, string instruction,
                               list<VTypeInfo> vtilist> {
  foreach vti = vtilist in
    def : PPatUnaryNoMask<intrinsic # "_vv", instruction, "VV",
                          vti.Vector, vti.Vector, vti.Log2SEW,
                          vti.LMul, vti.RegClass, vti.RegClass>;
}

multiclass PPatUnaryV_S_NoMask<string intrinsic, string instruction,
                        list<VTypeInfo> vtilist, string suffix = "vs"> {
  foreach vti = vtilist in {
    defvar vectorM1 = !cast<VTypeInfo>("VI" # vti.SEW # "M1");
    def : PPatUnaryNoMask<intrinsic # "_" # suffix, instruction, !toupper(suffix),
                          vti.Vector, vectorM1.Vector, vti.Log2SEW,
                          vti.LMul, vti.RegClass, VR>;
  }
}

multiclass PPatUnaryV_V_S_NoMask<string intrinsic, string instruction,
                                 list<VTypeInfo> vtilist> {
  defm : PPatUnaryV_V_NoMask<intrinsic, instruction, vtilist>;
  defm : PPatUnaryV_S_NoMask<intrinsic, instruction, vtilist>;
}


defm PseudoVBREV : VPseudoVALU_V;
defm PseudoVREV8 : VPseudoVALU_V;
defm PseudoVCLZ : VPseudoVALU_V;
defm PseudoVCTZ : VPseudoVALU_V;
defm PseudoVCPOP : VPseudoVALU_V;

let Predicates = [HasStdExtZvknsOrZvkned] in {
  defm : PPatUnaryV_V_S_NoMask<"int_riscv_vaesdf", "PseudoVAESDF", I32IntegerVectors>;
  defm : PPatUnaryV_V_S_NoMask<"int_riscv_vaesdm", "PseudoVAESDM", I32IntegerVectors>;
  defm : PPatUnaryV_V_S_NoMask<"int_riscv_vaesef", "PseudoVAESEF", I32IntegerVectors>;
  defm : PPatUnaryV_V_S_NoMask<"int_riscv_vaesem", "PseudoVAESEM", I32IntegerVectors>;
  defm : VPatBinaryV_VI_NoMaskTA<"int_riscv_vaeskf1", "PseudoVAESKF1", I32IntegerVectors, uimm5>;
  defm : VPatBinaryV_VI_NoMask<"int_riscv_vaeskf2", "PseudoVAESKF2", I32IntegerVectors, uimm5>;
  defm : PPatUnaryV_S_NoMask<"int_riscv_vaesz", "PseudoVAESZ", I32IntegerVectors>;
} // Predicates = [HasStdExtZvknsOrZvkned]

let Predicates = [HasStdExtZvbbOrZvkb0p1] in {
  defm : VPatBinaryV_VV_VX<"int_riscv_vandn", "PseudoVANDN", AllIntegerVectors>;
  defm : VPatUnaryV_V<"int_riscv_vbrev8", "PseudoVBREV8", AllIntegerVectors>;
  defm : VPatUnaryV_V<"int_riscv_vrev8", "PseudoVREV8", AllIntegerVectors>;
  defm : VPatBinaryV_VV_VX<"int_riscv_vrol", "PseudoVROL", AllIntegerVectors>;
  defm : VPatBinaryV_VV_VX_VI<"int_riscv_vror", "PseudoVROR", AllIntegerVectors>;
} // Predicates = [HasStdExtZvbbOrZvkb0p1]

let Predicates = [HasStdExtZvbcOrZvkb0p1] in {
  defm : VPatBinaryV_VV_VX<"int_riscv_vclmul", "PseudoVCLMUL", I64IntegerVectors>;
  defm : VPatBinaryV_VV_VX<"int_riscv_vclmulh", "PseudoVCLMULH", I64IntegerVectors>;
} // Predicates = [HasStdExtZvbcOrZvkb0p1]

let Predicates = [HasStdExtZvkg] in {
  defm : VPatBinaryV_VV_NoMask<"int_riscv_vghsh", "PseudoVGHSH", I32IntegerVectors>;
  defm : PPatUnaryV_S_NoMask<"int_riscv_vgmul", "PseudoVGMUL", I32IntegerVectors, "vv">;
} // Predicates = [HasStdExtZvkg]

let Predicates = [HasStdExtZvkg0p1] in {
  defm : VPatBinaryV_VV_NoMask<"int_riscv_vghmac", "PseudoVGHMAC", I32IntegerVectors>;
} // Predicates = [HasStdExtZvkg0p1]

let Predicates = [HasStdExtZvksed] in {
  defm : VPatBinaryV_VI_NoMaskTA<"int_riscv_vsm4k", "PseudoVSM4K", I32IntegerVectors, uimm5>;
  defm : PPatUnaryV_V_S_NoMask<"int_riscv_vsm4r", "PseudoVSM4R", I32IntegerVectors>;
} // Predicates = [HasStdExtZvksed]

let Predicates = [HasStdExtZvksh] in {
  defm : VPatBinaryV_VI_NoMask<"int_riscv_vsm3c", "PseudoVSM3C", I32IntegerVectors, uimm5>;
  defm : VPatBinaryV_VV_NoMaskTA<"int_riscv_vsm3me", "PseudoVSM3ME", I32IntegerVectors>;
} // Predicates = [HasStdExtZvksh]

let Predicates = [HasStdExtZvknha] in {
  defm : VPatBinaryV_VV_NoMask<"int_riscv_vsha2ch", "PseudoVSHA2CH", I32I64IntegerVectors>;
  defm : VPatBinaryV_VV_NoMask<"int_riscv_vsha2cl", "PseudoVSHA2CH", I32I64IntegerVectors>;
  defm : VPatBinaryV_VV_NoMask<"int_riscv_vsha2ms", "PseudoVSHA2MS", I32I64IntegerVectors>;
} // Predicates = [HasStdExtZvknha]
#endif // SIFIVE_CUSTOMIZATION

//===----------------------------------------------------------------------===//
// SDNode patterns
//===----------------------------------------------------------------------===//

multiclass VPatUnarySDNode_V<SDPatternOperator op, string instruction_name> {
  foreach vti = AllIntegerVectors in {
    let Predicates = GetVTypePredicates<vti>.Predicates in {
      def : Pat<(vti.Vector (op (vti.Vector vti.RegClass:$rs1))),
                (!cast<Instruction>(instruction_name#"_V_"#vti.LMul.MX)
                   (vti.Vector (IMPLICIT_DEF)),
                   vti.RegClass:$rs1,
                   vti.AVL, vti.Log2SEW, TA_MA)>;
    }
  }
}

let Predicates = [HasStdExtZvbb] in {
  defm : VPatUnarySDNode_V<bitreverse, "PseudoVBREV">;
  defm : VPatUnarySDNode_V<bswap, "PseudoVREV8">;
  defm : VPatUnarySDNode_V<ctlz, "PseudoVCLZ">;
  defm : VPatUnarySDNode_V<cttz, "PseudoVCTZ">;
  defm : VPatUnarySDNode_V<ctpop, "PseudoVCPOP">;
}
