//===-- RISCVSubtarget.cpp - RISCV Subtarget Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISCV specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "RISCVSubtarget.h"
#include "RISCV.h"
#include "RISCVFrameLowering.h"
#include "RISCVMacroFusion.h"
#include "RISCVTargetMachine.h"
#include "GISel/RISCVCallLowering.h"
#include "GISel/RISCVLegalizerInfo.h"
#include "GISel/RISCVRegisterBankInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "RISCVGenSubtargetInfo.inc"

static cl::opt<bool> EnableSubRegLiveness("riscv-enable-subreg-liveness",
                                          cl::init(false), cl::Hidden);

static cl::opt<unsigned> RVVVectorLMULMax(
    "riscv-v-fixed-length-vector-lmul-max",
    cl::desc("The maximum LMUL value to use for fixed length vectors. "
             "Fractional LMUL values are not supported."),
    cl::init(8), cl::Hidden);

static cl::opt<bool> RISCVDisableUsingConstantPoolForLargeInts(
    "riscv-disable-using-constant-pool-for-large-ints",
    cl::desc("Disable using constant pool for large integers."),
    cl::init(false), cl::Hidden);

static cl::opt<unsigned> RISCVMaxBuildIntsCost(
    "riscv-max-build-ints-cost",
    cl::desc("The maximum cost used for building integers."), cl::init(0),
    cl::Hidden);

#if SIFIVE_CUSTOMIZATION
static cl::opt<bool> UseAA("riscv-use-aa", cl::init(false),
                           cl::desc("Enable the use of AA during codegen."));
static cl::opt<bool> UseAltGPROrder("riscv-use-alt-gpr-order", cl::init(false),
                                    cl::desc("Enable alternate GPR order."),
                                    cl::ReallyHidden);
static cl::opt<bool> UseAltFPROrder("riscv-use-alt-fpr-order", cl::init(false),
                                    cl::desc("Enable alternate FPR order."),
                                    cl::ReallyHidden);
static cl::opt<bool> UseAltVROrder("riscv-use-alt-vr-order", cl::init(false),
                                   cl::desc("Enable alternate VR order."),
                                   cl::ReallyHidden);
#endif // SIFIVE_CUSTOMIZATION

void RISCVSubtarget::anchor() {}

RISCVSubtarget &
RISCVSubtarget::initializeSubtargetDependencies(const Triple &TT, StringRef CPU,
                                                StringRef TuneCPU, StringRef FS,
                                                StringRef ABIName) {
  // Determine default and user-specified characteristics
  bool Is64Bit = TT.isArch64Bit();
  if (CPU.empty() || CPU == "generic")
    CPU = Is64Bit ? "generic-rv64" : "generic-rv32";

  if (TuneCPU.empty())
    TuneCPU = CPU;

  ParseSubtargetFeatures(CPU, TuneCPU, FS);
  if (Is64Bit) {
    XLenVT = MVT::i64;
    XLen = 64;
  }

  TargetABI = RISCVABI::computeTargetABI(TT, getFeatureBits(), ABIName);
  RISCVFeatures::validate(TT, getFeatureBits());
  return *this;
}

RISCVSubtarget::RISCVSubtarget(const Triple &TT, StringRef CPU,
                               StringRef TuneCPU, StringRef FS,
                               StringRef ABIName, unsigned RVVVectorBitsMin,
                               unsigned RVVVectorBitsMax,
                               const TargetMachine &TM)
    : RISCVGenSubtargetInfo(TT, CPU, TuneCPU, FS),
      RVVVectorBitsMin(RVVVectorBitsMin), RVVVectorBitsMax(RVVVectorBitsMax),
      FrameLowering(
          initializeSubtargetDependencies(TT, CPU, TuneCPU, FS, ABIName)),
      InstrInfo(*this), RegInfo(getHwMode()), TLInfo(TM, *this) {
  CallLoweringInfo.reset(new RISCVCallLowering(*getTargetLowering()));
  Legalizer.reset(new RISCVLegalizerInfo(*this));

  auto *RBI = new RISCVRegisterBankInfo(*getRegisterInfo());
  RegBankInfo.reset(RBI);
  InstSelector.reset(createRISCVInstructionSelector(
      *static_cast<const RISCVTargetMachine *>(&TM), *this, *RBI));
}

const CallLowering *RISCVSubtarget::getCallLowering() const {
  return CallLoweringInfo.get();
}

InstructionSelector *RISCVSubtarget::getInstructionSelector() const {
  return InstSelector.get();
}

const LegalizerInfo *RISCVSubtarget::getLegalizerInfo() const {
  return Legalizer.get();
}

const RegisterBankInfo *RISCVSubtarget::getRegBankInfo() const {
  return RegBankInfo.get();
}

bool RISCVSubtarget::useConstantPoolForLargeInts() const {
  return !RISCVDisableUsingConstantPoolForLargeInts;
}

unsigned RISCVSubtarget::getMaxBuildIntsCost() const {
  // Loading integer from constant pool needs two instructions (the reason why
  // the minimum cost is 2): an address calculation instruction and a load
  // instruction. Usually, address calculation and instructions used for
  // building integers (addi, slli, etc.) can be done in one cycle, so here we
  // set the default cost to (LoadLatency + 1) if no threshold is provided.
  return RISCVMaxBuildIntsCost == 0
             ? getSchedModel().LoadLatency + 1
             : std::max<unsigned>(2, RISCVMaxBuildIntsCost);
}

unsigned RISCVSubtarget::getMaxRVVVectorSizeInBits() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");

  // ZvlLen specifies the minimum required vlen. The upper bound provided by
  // riscv-v-vector-bits-max should be no less than it.
  if (RVVVectorBitsMax != 0 && RVVVectorBitsMax < ZvlLen)
    report_fatal_error("riscv-v-vector-bits-max specified is lower "
                       "than the Zvl*b limitation");

  return RVVVectorBitsMax;
}

unsigned RISCVSubtarget::getMinRVVVectorSizeInBits() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");

  if (RVVVectorBitsMin == -1U)
    return ZvlLen;

  // ZvlLen specifies the minimum required vlen. The lower bound provided by
  // riscv-v-vector-bits-min should be no less than it.
  if (RVVVectorBitsMin != 0 && RVVVectorBitsMin < ZvlLen)
    report_fatal_error("riscv-v-vector-bits-min specified is lower "
                       "than the Zvl*b limitation");

  return RVVVectorBitsMin;
}

unsigned RISCVSubtarget::getMaxLMULForFixedLengthVectors() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");
  assert(RVVVectorLMULMax <= 8 && isPowerOf2_32(RVVVectorLMULMax) &&
         "V extension requires a LMUL to be at most 8 and a power of 2!");
  return PowerOf2Floor(
      std::max<unsigned>(std::min<unsigned>(RVVVectorLMULMax, 8), 1));
}

bool RISCVSubtarget::useRVVForFixedLengthVectors() const {
  return hasVInstructions() && getMinRVVVectorSizeInBits() != 0;
}

bool RISCVSubtarget::enableSubRegLiveness() const {
  // FIXME: Enable subregister liveness by default for RVV to better handle
  // LMUL>1 and segment load/store.
  return EnableSubRegLiveness;
}

void RISCVSubtarget::getPostRAMutations(
    std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(createRISCVMacroFusionDAGMutation());
}

#if SIFIVE_CUSTOMIZATION
static unsigned factorLMul(unsigned Lat, RISCVII::VLMUL LMul,
                           unsigned DLenFactor) {
  // Every DLEN chunk is processed every VLEN / DLEN cycles.
  return Lat + (RISCVII::getLMULGroups(LMul) - 1) * DLenFactor;
}

static unsigned factorSegmentLMul(unsigned Lat, RISCVII::VLMUL LMul,
                                  unsigned NF, unsigned DLenFactor) {
  switch (LMul) {
  default:
    return Lat + (RISCVII::getLMULGroups(LMul) * NF - 1) * DLenFactor;
  case RISCVII::LMUL_F8:
    return Lat + (divideCeil(NF, 8) - 1) * DLenFactor;
  case RISCVII::LMUL_F4:
    return Lat + (divideCeil(NF, 4) - 1) * DLenFactor;
  case RISCVII::LMUL_F2:
    return Lat + (divideCeil(NF, 2) - 1) * DLenFactor;
  }
}

static unsigned
calculateLatency(const RISCVSubtarget *ST, const MachineInstr *MI, unsigned Lat,
                 RISCVSubtarget::RISCVProcFamilyEnum ProcModel) {
  switch(ProcModel) {
  default:
    return Lat;

  case RISCVSubtarget::SiFive7:
    {
      const MCInstrDesc &Desc = MI->getDesc();

      // Use the latency information from the base instruction for vector
      // pseudos.
      const RISCVVPseudosTable::PseudoInfo *RVV =
          RISCVVPseudosTable::getPseudoInfo(MI->getOpcode());
      if (RVV == nullptr)
        return Lat;

      // Use the base opcode.
      unsigned Opcode = RVV->BaseInstr;

      RISCVII::VLMUL LMul = RISCVII::getLMul(Desc.TSFlags);

      // Assume VLEN same as DLEN, when we can't get DLEN information.
      unsigned DLenFactor = 1;
      if (ST->hasKnownDLen())
        DLenFactor = divideCeil(ST->getRealMinVLen(), ST->getDLen());

      // Instructions without SEW, if any.
      if (!RISCVII::hasSEWOp(Desc.TSFlags))
        return factorLMul(Lat, LMul, DLenFactor);

      unsigned SEW =
          1 << MI->getOperand(RISCVII::getSEWOpNum(MI->getDesc())).getImm();

      switch(Opcode) {
      default:
        return factorLMul(Lat, LMul, DLenFactor);
      // VRGATHER latency is proportional to the number of elements.
      case RISCV::VRGATHER_VV:
      case RISCV::VRGATHER_VI:
      case RISCV::VRGATHER_VX:
      // VRGATHEREI16 always uses an EEW of 16 bits.
      case RISCV::VRGATHEREI16_VV:
      // VCOMPRESS latency is proportional to the number of elements.
      case RISCV::VCOMPRESS_VM:
        return RISCVII::getLMULGroups(LMul) * ST->getRealMinVLen() / SEW;
      // Reduction latency is complex.
      case RISCV::VREDAND_VS:
      case RISCV::VREDMAX_VS:
      case RISCV::VREDMAXU_VS:
      case RISCV::VREDMIN_VS:
      case RISCV::VREDMINU_VS:
      case RISCV::VREDOR_VS:
      case RISCV::VREDSUM_VS:
      case RISCV::VREDXOR_VS:
      case RISCV::VFREDMAX_VS:
      case RISCV::VFREDMIN_VS:
      case RISCV::VFREDUSUM_VS:
      // TODO: Assuming that the SEW is based on the input operands.
      case RISCV::VFWREDUSUM_VS:
      case RISCV::VWREDSUM_VS:
      case RISCV::VWREDSUMU_VS:
        return RISCVII::getLMULGroups(LMul) *
            7 * (4 + Log2_32(ST->getRealMinVLen()) - 1 - Log2_32(SEW));
      case RISCV::VFREDOSUM_VS:
      case RISCV::VFWREDOSUM_VS:
        return RISCVII::getLMULGroups(LMul) * 5 * ST->getRealMinVLen() / SEW ;
      // Narrowing latency:
      case RISCV::VNCLIP_WV:
      case RISCV::VNCLIP_WX:
      case RISCV::VNCLIP_WI:
      case RISCV::VNCLIPU_WV:
      case RISCV::VNCLIPU_WX:
      case RISCV::VNCLIPU_WI:
      case RISCV::VFNCVT_F_F_W:
      case RISCV::VFNCVT_F_X_W:
      case RISCV::VFNCVT_F_XU_W:
      case RISCV::VFNCVT_ROD_F_F_W:
      case RISCV::VFNCVT_RTZ_X_F_W:
      case RISCV::VFNCVT_RTZ_XU_F_W:
      case RISCV::VFNCVT_X_F_W:
      case RISCV::VFNCVT_XU_F_W:
      case RISCV::VNSRA_WV:
      case RISCV::VNSRA_WX:
      case RISCV::VNSRA_WI:
      case RISCV::VNSRL_WV:
      case RISCV::VNSRL_WX:
      case RISCV::VNSRL_WI:
        // FIXME: It may be more complex than this.
        return factorLMul(Lat, LMul, DLenFactor);
      // Widening latency.
      case RISCV::VFWADD_VV:
      case RISCV::VFWADD_VF:
      case RISCV::VFWCVT_F_F_V:
      case RISCV::VFWCVT_F_X_V:
      case RISCV::VFWCVT_F_XU_V:
      case RISCV::VFWCVT_RTZ_X_F_V:
      case RISCV::VFWCVT_RTZ_XU_F_V:
      case RISCV::VFWCVT_X_F_V:
      case RISCV::VFWCVT_XU_F_V:
      case RISCV::VFWMACC_VV:
      case RISCV::VFWMACC_VF:
      case RISCV::VFWMUL_VV:
      case RISCV::VFWMUL_VF:
      case RISCV::VFWNMACC_VV:
      case RISCV::VFWNMACC_VF:
      case RISCV::VFWNMSAC_VV:
      case RISCV::VFWNMSAC_VF:
      case RISCV::VFWMSAC_VV:
      case RISCV::VFWMSAC_VF:
      case RISCV::VFWSUB_VV:
      case RISCV::VFWSUB_VF:
      case RISCV::VWADD_VV:
      case RISCV::VWADD_VX:
      case RISCV::VWADD_WV:
      case RISCV::VWADD_WX:
      case RISCV::VWADDU_VV:
      case RISCV::VWADDU_VX:
      case RISCV::VWADDU_WV:
      case RISCV::VWADDU_WX:
      case RISCV::VWMACC_VV:
      case RISCV::VWMACC_VX:
      case RISCV::VWMACCSU_VV:
      case RISCV::VWMACCSU_VX:
      case RISCV::VWMACCU_VV:
      case RISCV::VWMACCU_VX:
      case RISCV::VWMACCUS_VX:
      case RISCV::VWMUL_VV:
      case RISCV::VWMUL_VX:
      case RISCV::VWMULSU_VV:
      case RISCV::VWMULSU_VX:
      case RISCV::VWMULU_VV:
      case RISCV::VWMULU_VX:
      case RISCV::VWSUB_VV:
      case RISCV::VWSUB_VX:
      case RISCV::VWSUB_WV:
      case RISCV::VWSUB_WX:
      case RISCV::VWSUBU_VV:
      case RISCV::VWSUBU_VX:
      case RISCV::VWSUBU_WV:
      case RISCV::VWSUBU_WX:
        // FIXME: It may be more complex than this.
        return factorLMul(Lat, LMul, DLenFactor);
      case RISCV::VLSEG2E8_V:
      case RISCV::VLSEG2E16_V:
      case RISCV::VLSEG2E32_V:
      case RISCV::VLSEG2E64_V:
      case RISCV::VLSEG2E8FF_V:
      case RISCV::VLSEG2E16FF_V:
      case RISCV::VLSEG2E32FF_V:
      case RISCV::VLSEG2E64FF_V:
      case RISCV::VLSSEG2E8_V:
      case RISCV::VLSSEG2E16_V:
      case RISCV::VLSSEG2E32_V:
      case RISCV::VLSSEG2E64_V:
      case RISCV::VLOXSEG2EI8_V:
      case RISCV::VLOXSEG2EI16_V:
      case RISCV::VLOXSEG2EI32_V:
      case RISCV::VLOXSEG2EI64_V:
      case RISCV::VLUXSEG2EI8_V:
      case RISCV::VLUXSEG2EI16_V:
      case RISCV::VLUXSEG2EI32_V:
      case RISCV::VLUXSEG2EI64_V:
        return factorSegmentLMul(Lat, LMul, 2, DLenFactor);
      case RISCV::VLSEG3E8_V:
      case RISCV::VLSEG3E16_V:
      case RISCV::VLSEG3E32_V:
      case RISCV::VLSEG3E64_V:
      case RISCV::VLSEG3E8FF_V:
      case RISCV::VLSEG3E16FF_V:
      case RISCV::VLSEG3E32FF_V:
      case RISCV::VLSEG3E64FF_V:
      case RISCV::VLSSEG3E8_V:
      case RISCV::VLSSEG3E16_V:
      case RISCV::VLSSEG3E32_V:
      case RISCV::VLSSEG3E64_V:
      case RISCV::VLOXSEG3EI8_V:
      case RISCV::VLOXSEG3EI16_V:
      case RISCV::VLOXSEG3EI32_V:
      case RISCV::VLOXSEG3EI64_V:
      case RISCV::VLUXSEG3EI8_V:
      case RISCV::VLUXSEG3EI16_V:
      case RISCV::VLUXSEG3EI32_V:
      case RISCV::VLUXSEG3EI64_V:
        return factorSegmentLMul(Lat, LMul, 3, DLenFactor);
      case RISCV::VLSEG4E8_V:
      case RISCV::VLSEG4E16_V:
      case RISCV::VLSEG4E32_V:
      case RISCV::VLSEG4E64_V:
      case RISCV::VLSEG4E8FF_V:
      case RISCV::VLSEG4E16FF_V:
      case RISCV::VLSEG4E32FF_V:
      case RISCV::VLSEG4E64FF_V:
      case RISCV::VLSSEG4E8_V:
      case RISCV::VLSSEG4E16_V:
      case RISCV::VLSSEG4E32_V:
      case RISCV::VLSSEG4E64_V:
      case RISCV::VLOXSEG4EI8_V:
      case RISCV::VLOXSEG4EI16_V:
      case RISCV::VLOXSEG4EI32_V:
      case RISCV::VLOXSEG4EI64_V:
      case RISCV::VLUXSEG4EI8_V:
      case RISCV::VLUXSEG4EI16_V:
      case RISCV::VLUXSEG4EI32_V:
      case RISCV::VLUXSEG4EI64_V:
        return factorSegmentLMul(Lat, LMul, 4, DLenFactor);
      case RISCV::VLSEG5E8_V:
      case RISCV::VLSEG5E16_V:
      case RISCV::VLSEG5E32_V:
      case RISCV::VLSEG5E64_V:
      case RISCV::VLSEG5E8FF_V:
      case RISCV::VLSEG5E16FF_V:
      case RISCV::VLSEG5E32FF_V:
      case RISCV::VLSEG5E64FF_V:
      case RISCV::VLSSEG5E8_V:
      case RISCV::VLSSEG5E16_V:
      case RISCV::VLSSEG5E32_V:
      case RISCV::VLSSEG5E64_V:
      case RISCV::VLOXSEG5EI8_V:
      case RISCV::VLOXSEG5EI16_V:
      case RISCV::VLOXSEG5EI32_V:
      case RISCV::VLOXSEG5EI64_V:
      case RISCV::VLUXSEG5EI8_V:
      case RISCV::VLUXSEG5EI16_V:
      case RISCV::VLUXSEG5EI32_V:
      case RISCV::VLUXSEG5EI64_V:
        return factorSegmentLMul(Lat, LMul, 5, DLenFactor);
      case RISCV::VLSEG6E8_V:
      case RISCV::VLSEG6E16_V:
      case RISCV::VLSEG6E32_V:
      case RISCV::VLSEG6E64_V:
      case RISCV::VLSEG6E8FF_V:
      case RISCV::VLSEG6E16FF_V:
      case RISCV::VLSEG6E32FF_V:
      case RISCV::VLSEG6E64FF_V:
      case RISCV::VLSSEG6E8_V:
      case RISCV::VLSSEG6E16_V:
      case RISCV::VLSSEG6E32_V:
      case RISCV::VLSSEG6E64_V:
      case RISCV::VLOXSEG6EI8_V:
      case RISCV::VLOXSEG6EI16_V:
      case RISCV::VLOXSEG6EI32_V:
      case RISCV::VLOXSEG6EI64_V:
      case RISCV::VLUXSEG6EI8_V:
      case RISCV::VLUXSEG6EI16_V:
      case RISCV::VLUXSEG6EI32_V:
      case RISCV::VLUXSEG6EI64_V:
        return factorSegmentLMul(Lat, LMul, 6, DLenFactor);
      case RISCV::VLSEG7E8_V:
      case RISCV::VLSEG7E16_V:
      case RISCV::VLSEG7E32_V:
      case RISCV::VLSEG7E64_V:
      case RISCV::VLSEG7E8FF_V:
      case RISCV::VLSEG7E16FF_V:
      case RISCV::VLSEG7E32FF_V:
      case RISCV::VLSEG7E64FF_V:
      case RISCV::VLSSEG7E8_V:
      case RISCV::VLSSEG7E16_V:
      case RISCV::VLSSEG7E32_V:
      case RISCV::VLSSEG7E64_V:
      case RISCV::VLOXSEG7EI8_V:
      case RISCV::VLOXSEG7EI16_V:
      case RISCV::VLOXSEG7EI32_V:
      case RISCV::VLOXSEG7EI64_V:
      case RISCV::VLUXSEG7EI8_V:
      case RISCV::VLUXSEG7EI16_V:
      case RISCV::VLUXSEG7EI32_V:
      case RISCV::VLUXSEG7EI64_V:
        return factorSegmentLMul(Lat, LMul, 7, DLenFactor);
      case RISCV::VLSEG8E8_V:
      case RISCV::VLSEG8E16_V:
      case RISCV::VLSEG8E32_V:
      case RISCV::VLSEG8E64_V:
      case RISCV::VLSEG8E8FF_V:
      case RISCV::VLSEG8E16FF_V:
      case RISCV::VLSEG8E32FF_V:
      case RISCV::VLSEG8E64FF_V:
      case RISCV::VLSSEG8E8_V:
      case RISCV::VLSSEG8E16_V:
      case RISCV::VLSSEG8E32_V:
      case RISCV::VLSSEG8E64_V:
      case RISCV::VLOXSEG8EI8_V:
      case RISCV::VLOXSEG8EI16_V:
      case RISCV::VLOXSEG8EI32_V:
      case RISCV::VLOXSEG8EI64_V:
      case RISCV::VLUXSEG8EI8_V:
      case RISCV::VLUXSEG8EI16_V:
      case RISCV::VLUXSEG8EI32_V:
      case RISCV::VLUXSEG8EI64_V:
        return factorSegmentLMul(Lat, LMul, 8, DLenFactor);
      case RISCV::VSSEG2E8_V:
      case RISCV::VSSEG2E16_V:
      case RISCV::VSSEG2E32_V:
      case RISCV::VSSEG2E64_V:
      case RISCV::VSSSEG2E8_V:
      case RISCV::VSSSEG2E16_V:
      case RISCV::VSSSEG2E32_V:
      case RISCV::VSSSEG2E64_V:
      case RISCV::VSOXSEG2EI8_V:
      case RISCV::VSOXSEG2EI16_V:
      case RISCV::VSOXSEG2EI32_V:
      case RISCV::VSOXSEG2EI64_V:
      case RISCV::VSUXSEG2EI8_V:
      case RISCV::VSUXSEG2EI16_V:
      case RISCV::VSUXSEG2EI32_V:
      case RISCV::VSUXSEG2EI64_V:
      case RISCV::VSSEG3E8_V:
      case RISCV::VSSEG3E16_V:
      case RISCV::VSSEG3E32_V:
      case RISCV::VSSEG3E64_V:
      case RISCV::VSSSEG3E8_V:
      case RISCV::VSSSEG3E16_V:
      case RISCV::VSSSEG3E32_V:
      case RISCV::VSSSEG3E64_V:
      case RISCV::VSOXSEG3EI8_V:
      case RISCV::VSOXSEG3EI16_V:
      case RISCV::VSOXSEG3EI32_V:
      case RISCV::VSOXSEG3EI64_V:
      case RISCV::VSUXSEG3EI8_V:
      case RISCV::VSUXSEG3EI16_V:
      case RISCV::VSUXSEG3EI32_V:
      case RISCV::VSUXSEG3EI64_V:
      case RISCV::VSSEG4E8_V:
      case RISCV::VSSEG4E16_V:
      case RISCV::VSSEG4E32_V:
      case RISCV::VSSEG4E64_V:
      case RISCV::VSSSEG4E8_V:
      case RISCV::VSSSEG4E16_V:
      case RISCV::VSSSEG4E32_V:
      case RISCV::VSSSEG4E64_V:
      case RISCV::VSOXSEG4EI8_V:
      case RISCV::VSOXSEG4EI16_V:
      case RISCV::VSOXSEG4EI32_V:
      case RISCV::VSOXSEG4EI64_V:
      case RISCV::VSUXSEG4EI8_V:
      case RISCV::VSUXSEG4EI16_V:
      case RISCV::VSUXSEG4EI32_V:
      case RISCV::VSUXSEG4EI64_V:
      case RISCV::VSSEG5E8_V:
      case RISCV::VSSEG5E16_V:
      case RISCV::VSSEG5E32_V:
      case RISCV::VSSEG5E64_V:
      case RISCV::VSSSEG5E8_V:
      case RISCV::VSSSEG5E16_V:
      case RISCV::VSSSEG5E32_V:
      case RISCV::VSSSEG5E64_V:
      case RISCV::VSOXSEG5EI8_V:
      case RISCV::VSOXSEG5EI16_V:
      case RISCV::VSOXSEG5EI32_V:
      case RISCV::VSOXSEG5EI64_V:
      case RISCV::VSUXSEG5EI8_V:
      case RISCV::VSUXSEG5EI16_V:
      case RISCV::VSUXSEG5EI32_V:
      case RISCV::VSUXSEG5EI64_V:
      case RISCV::VSSEG6E8_V:
      case RISCV::VSSEG6E16_V:
      case RISCV::VSSEG6E32_V:
      case RISCV::VSSEG6E64_V:
      case RISCV::VSSSEG6E8_V:
      case RISCV::VSSSEG6E16_V:
      case RISCV::VSSSEG6E32_V:
      case RISCV::VSSSEG6E64_V:
      case RISCV::VSOXSEG6EI8_V:
      case RISCV::VSOXSEG6EI16_V:
      case RISCV::VSOXSEG6EI32_V:
      case RISCV::VSOXSEG6EI64_V:
      case RISCV::VSUXSEG6EI8_V:
      case RISCV::VSUXSEG6EI16_V:
      case RISCV::VSUXSEG6EI32_V:
      case RISCV::VSUXSEG6EI64_V:
      case RISCV::VSSEG7E8_V:
      case RISCV::VSSEG7E16_V:
      case RISCV::VSSEG7E32_V:
      case RISCV::VSSEG7E64_V:
      case RISCV::VSSSEG7E8_V:
      case RISCV::VSSSEG7E16_V:
      case RISCV::VSSSEG7E32_V:
      case RISCV::VSSSEG7E64_V:
      case RISCV::VSOXSEG7EI8_V:
      case RISCV::VSOXSEG7EI16_V:
      case RISCV::VSOXSEG7EI32_V:
      case RISCV::VSOXSEG7EI64_V:
      case RISCV::VSUXSEG7EI8_V:
      case RISCV::VSUXSEG7EI16_V:
      case RISCV::VSUXSEG7EI32_V:
      case RISCV::VSUXSEG7EI64_V:
      case RISCV::VSSEG8E8_V:
      case RISCV::VSSEG8E16_V:
      case RISCV::VSSEG8E32_V:
      case RISCV::VSSEG8E64_V:
      case RISCV::VSSSEG8E8_V:
      case RISCV::VSSSEG8E16_V:
      case RISCV::VSSSEG8E32_V:
      case RISCV::VSSSEG8E64_V:
      case RISCV::VSOXSEG8EI8_V:
      case RISCV::VSOXSEG8EI16_V:
      case RISCV::VSOXSEG8EI32_V:
      case RISCV::VSOXSEG8EI64_V:
      case RISCV::VSUXSEG8EI8_V:
      case RISCV::VSUXSEG8EI16_V:
      case RISCV::VSUXSEG8EI32_V:
      case RISCV::VSUXSEG8EI64_V:
        return Lat;
      }
    }
  }

  llvm_unreachable("Unexpected processor model!");
}

// Perform adjustments to the latency of a schedule dependency.
// TODO: Consider the impact on the throughput.
void RISCVSubtarget::adjustSchedDependency(SUnit *SrcSU, int SrcOpIdx,
                                           SUnit *DstSU, int DstOpIdx,
                                           SDep &Dep) const {
  // At the moment, only RVV uses the adjustment of the latency.
  if (!hasVInstructions())
    return;

  if (SrcSU->isInstr()) {
    MachineInstr *SrcMI = SrcSU->getInstr();
    Dep.setLatency(
        calculateLatency(this, SrcMI, Dep.getLatency(), getProcFamily()));
  }

  if (DstSU->isInstr()) {
    MachineInstr *DstMI = DstSU->getInstr();
    // Stores don't have dependents, but occupy units.
    if (DstMI->mayStore()) {
      Dep.setLatency(
          calculateLatency(this, DstMI, Dep.getLatency(), getProcFamily()));
    }
  }
}

void RISCVSubtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                         unsigned NumRegionInstrs) const {
  if (getProcFamily() == RISCVSubtarget::SiFive7)
    Policy.OnlyBottomUp = false;
}

bool RISCVSubtarget::useAA() const { return UseAA; }

bool RISCVSubtarget::useAltGPROrder() const { return UseAltGPROrder; }
bool RISCVSubtarget::useAltFPROrder() const { return UseAltFPROrder; }
bool RISCVSubtarget::useAltVROrder() const { return UseAltVROrder; }
#endif // SIFIVE_CUSTOMIZATION
