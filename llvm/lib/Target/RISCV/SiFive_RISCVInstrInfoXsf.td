//===-- RISCVInstrInfoXsf.td - SiFive custom instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the SiFive custom instructions.
///
//===----------------------------------------------------------------------===//

class CustomSiFiveVV<string opcodestr,  RISCVVFormat opv, bits<6> funct6>
  : RVInstVVCustom2<funct6, opv, (outs VR:$vd),
                    (ins VR:$vs1, VR:$vs2), opcodestr, "$vd, $vs1, $vs2"> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

let vm = 1, RVVConstraint = VS2Constraint, Predicates = [HasExtXsfvqmaccqoq] in {
  def VQMACCU4x8x4_VV : CustomSiFiveVV<"sf.vqmaccu.4x8x4", OPMVV, 0b111100>;
  def VQMACC4x8x4_VV : CustomSiFiveVV<"sf.vqmacc.4x8x4", OPMVV, 0b111101>;
  def VQMACCUS4x8x4_VV : CustomSiFiveVV<"sf.vqmaccus.4x8x4", OPMVV, 0b111110>;
  def VQMACCSU4x8x4_VV : CustomSiFiveVV<"sf.vqmaccsu.4x8x4", OPMVV, 0b111111>;
} // Predicates = [HasExtXsfvqmaccqoq]

let vm = 1, RVVConstraint = VS2Constraint, Predicates = [HasExtXsfvqmaccdod] in {
  def VQMACCU2x8x2_VV : CustomSiFiveVV<"sf.vqmaccu.2x8x2", OPMVV, 0b101100>;
  def VQMACC2x8x2_VV : CustomSiFiveVV<"sf.vqmacc.2x8x2", OPMVV, 0b101101>;
  def VQMACCUS2x8x2_VV : CustomSiFiveVV<"sf.vqmaccus.2x8x2", OPMVV, 0b101110>;
  def VQMACCSU2x8x2_VV : CustomSiFiveVV<"sf.vqmaccsu.2x8x2", OPMVV, 0b101111>;
} // Predicates = [HasExtXsfvqmaccdod]

let Predicates = [HasExtXsfvfhbfmin] in {
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenCvt in {
defm VFWCVT_F_BF16_V : VWCVTF_FV_VS2<"sf.vfwcvt.f.bf16.v", 0b010010, 0b01101>;
}
let Constraints = "@earlyclobber $vd" in {
defm VFNCVT_BF16_F_W : VNCVTF_FV_VS2<"sf.vfncvt.bf16.f.w", 0b010010, 0b11101>;
}
} // Predicates = [HasExtXsfvqmaccdod]

let vm = 1, RVVConstraint = VS2Constraint, Predicates = [HasExtXsfvfwmaccqqq] in {
  def VFWMACC4X4X4_VV : CustomSiFiveVV<"sf.vfwmacc.4x4x4", OPFVV, 0b111100>, Sched<[]> ;
} // Predicates = [HasExtXsfvfwmaccqqq]

class CustomSiFiveVF<string opcodestr,  RISCVVFormat opv, bits<6> funct6>
  : RVInstVXCustom2<funct6, opv, (outs VR:$vd),
                    (ins VR:$vs2, FPR32:$rs1), opcodestr, "$vd, $vs2, $rs1"> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

let vm = 1, Predicates = [HasExtXsfvfnrclipxfqf] in {
  def VFNRCLIP_XU_F_QF : CustomSiFiveVF<"sf.vfnrclip.xu.f.qf", OPFVF, 0b100010>, Sched<[]> ;
  def VFNRCLIP_X_F_QF : CustomSiFiveVF<"sf.vfnrclip.x.f.qf", OPFVF, 0b100011>, Sched<[]> ;
} // Predicates = [HasExtXsfvfnrclipxfqf]

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

class SwapVCIXIns<dag funct6, dag rd, dag rs2, dag rs1, bit swap> {
   dag Ins = !con(funct6, !if(swap, rs2, rd), !if(swap, rd, rs2), rs1);
}

class VCIXInfo<string suffix, VCIXType type, DAGOperand TyRd,
               DAGOperand TyRs2, DAGOperand TyRs1, bit HaveOutputDst> {
  string OpcodeStr = !if(HaveOutputDst, "sf.vc.v." # suffix,
                                        "sf.vc." # suffix);
  bits<4> Funct6_hi4 = type.Val;
  bits<3> Funct3 = !cond(!eq(TyRs1, VR):    0b000,
                         !eq(TyRs1, GPR):   0b100,
                         !eq(TyRs1, FPR32): 0b101,
                         !eq(TyRs1, simm5_i32): 0b011);
  dag Outs = !if(!not(HaveOutputDst), (outs),
                 !if(!or(!eq(type, VCIX_XVV), !eq(type, VCIX_XVW)),
                     (outs TyRd:$rd_wb), (outs TyRd:$rd)));
  dag Ins = SwapVCIXIns<!if(!ne(TyRs1, FPR32), (ins payload2:$funct6_lo2),
                                               (ins payload1:$funct6_lo1)),
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

let Predicates = [HasExtXsfvcp], mayLoad = 0, mayStore = 0,
    hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  defm X   : CustomSiFiveVCIX<"x",   VCIX_X,   payload5, payload5, GPR>,       Sched<[]>;
  defm I   : CustomSiFiveVCIX<"i",   VCIX_X,   payload5, payload5, simm5_i32>, Sched<[]>;
  defm XV  : CustomSiFiveVCIX<"xv",  VCIX_XV,  payload5, VR,       GPR>,       Sched<[]>;
  defm IV  : CustomSiFiveVCIX<"iv",  VCIX_XV,  payload5, VR,       simm5_i32>, Sched<[]>;
  defm VV  : CustomSiFiveVCIX<"vv",  VCIX_XV,  payload5, VR,       VR>,        Sched<[]>;
  defm FV  : CustomSiFiveVCIX<"fv",  VCIX_XV,  payload5, VR,       FPR32>,     Sched<[]>;
  defm XVV : CustomSiFiveVCIX<"xvv", VCIX_XVV, VR,       VR,       GPR>,       Sched<[]>;
  defm IVV : CustomSiFiveVCIX<"ivv", VCIX_XVV, VR,       VR,       simm5_i32>, Sched<[]>;
  defm VVV : CustomSiFiveVCIX<"vvv", VCIX_XVV, VR,       VR,       VR>,        Sched<[]>;
  defm FVV : CustomSiFiveVCIX<"fvv", VCIX_XVV, VR,       VR,       FPR32>,     Sched<[]>;
  defm XVW : CustomSiFiveVCIX<"xvw", VCIX_XVW, VR,       VR,       GPR>,       Sched<[]>;
  defm IVW : CustomSiFiveVCIX<"ivw", VCIX_XVW, VR,       VR,       simm5_i32>, Sched<[]>;
  defm VVW : CustomSiFiveVCIX<"vvw", VCIX_XVW, VR,       VR,       VR>,        Sched<[]>;
  defm FVW : CustomSiFiveVCIX<"fvw", VCIX_XVW, VR,       VR,       FPR32>,     Sched<[]>;
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
