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

let vm = 1, Predicates = [HasExtXsfvqmaccqoq] in {
  def VQMACCU4x8x4_VV : CustomSiFiveVV<"sf.vqmaccu.4x8x4", OPMVV, 0b111100>;
  def VQMACC4x8x4_VV : CustomSiFiveVV<"sf.vqmacc.4x8x4", OPMVV, 0b111101>;
  def VQMACCUS4x8x4_VV : CustomSiFiveVV<"sf.vqmaccus.4x8x4", OPMVV, 0b111110>;
  def VQMACCSU4x8x4_VV : CustomSiFiveVV<"sf.vqmaccsu.4x8x4", OPMVV, 0b111111>;
} // Predicates = [HasExtXsfvqmaccqoq]

let vm = 1, Predicates = [HasExtXsfvqmaccdod] in {
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
