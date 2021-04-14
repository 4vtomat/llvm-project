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

class CustomSiFiveV<string opcodestr,  RISCVVFormat opv, bits<6> funct6>
  : RVInstVVCustom2<funct6, opv, (outs VR:$vd),
                    (ins VR:$vs1, VR:$vs2), opcodestr, "$vd, $vs1, $vs2"> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

let vm = 1, Predicates = [HasExtXsfvqmaccqoq] in {
  def VQMACCU4x8x4_VV : CustomSiFiveV<"sf.vqmaccu.4x8x4", OPMVV, 0b111100>;
  def VQMACC4x8x4_VV : CustomSiFiveV<"sf.vqmacc.4x8x4", OPMVV, 0b111101>;
  def VQMACCUS4x8x4_VV : CustomSiFiveV<"sf.vqmaccus.4x8x4", OPMVV, 0b111110>;
  def VQMACCSU4x8x4_VV : CustomSiFiveV<"sf.vqmaccsu.4x8x4", OPMVV, 0b111111>;
} // Predicates = [HasExtXsfvqmaccqoq]

let vm = 1, Predicates = [HasExtXsfvqmaccdod] in {
  def VQMACCU2x8x2_VV : CustomSiFiveV<"sf.vqmaccu.2x8x2", OPMVV, 0b101100>;
  def VQMACC2x8x2_VV : CustomSiFiveV<"sf.vqmacc.2x8x2", OPMVV, 0b101101>;
  def VQMACCUS2x8x2_VV : CustomSiFiveV<"sf.vqmaccus.2x8x2", OPMVV, 0b101110>;
  def VQMACCSU2x8x2_VV : CustomSiFiveV<"sf.vqmaccsu.2x8x2", OPMVV, 0b101111>;
} // Predicates = [HasExtXsfvqmaccdod]

let Predicates = [HasExtXsfvfhbfmin] in {
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenCvt in {
defm VFWCVT_F_BF16_V : VWCVTF_FV_VS2<"sf.vfwcvt.f.bf16.v", 0b010010, 0b01101>;
}
let Constraints = "@earlyclobber $vd" in {
defm VFNCVT_BF16_F_W : VNCVTF_FV_VS2<"sf.vfncvt.bf16.f.w", 0b010010, 0b11101>;
}
} // Predicates = [HasExtXsfvqmaccdod]
