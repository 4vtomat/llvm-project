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
#include "RISCVCallLowering.h"
#include "RISCVFrameLowering.h"
#include "RISCVLegalizerInfo.h"
<<<<<<< HEAD
#include "SiFive_RISCVMacroFusion.h"
=======
#include "RISCVMacroFusion.h"
>>>>>>> upstream/main
#include "RISCVRegisterBankInfo.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "RISCVGenSubtargetInfo.inc"

static cl::opt<bool> EnableSubRegLiveness("riscv-enable-subreg-liveness",
                                          cl::init(false), cl::Hidden);

static cl::opt<int> RVVVectorBitsMax(
    "riscv-v-vector-bits-max",
    cl::desc("Assume V extension vector registers are at most this big, "
             "with zero meaning no maximum size is assumed."),
    cl::init(0), cl::Hidden);

static cl::opt<int> RVVVectorBitsMin(
    "riscv-v-vector-bits-min",
    cl::desc("Assume V extension vector registers are at least this big, "
             "with zero meaning no minimum size is assumed. A value of -1 "
             "means use Zvl*b extension. This is primarily used to enable "
             "autovectorization with fixed width vectors."),
    cl::init(-1), cl::Hidden);

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
                               StringRef ABIName, const TargetMachine &TM)
    : RISCVGenSubtargetInfo(TT, CPU, TuneCPU, FS),
      UserReservedRegister(RISCV::NUM_TARGET_REGS),
      FrameLowering(initializeSubtargetDependencies(TT, CPU, TuneCPU, FS, ABIName)),
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
  if (RVVVectorBitsMax == 0)
    return 0;

  // ZvlLen specifies the minimum required vlen. The upper bound provided by
  // riscv-v-vector-bits-max should be no less than it.
  if (RVVVectorBitsMax < (int)ZvlLen)
    report_fatal_error("riscv-v-vector-bits-max specified is lower "
                       "than the Zvl*b limitation");

  // FIXME: Change to >= 32 when VLEN = 32 is supported
  assert(
      RVVVectorBitsMax >= 64 && RVVVectorBitsMax <= 65536 &&
      isPowerOf2_32(RVVVectorBitsMax) &&
      "V or Zve* extension requires vector length to be in the range of 64 to "
      "65536 and a power of 2!");
  assert(RVVVectorBitsMax >= RVVVectorBitsMin &&
         "Minimum V extension vector length should not be larger than its "
         "maximum!");
  unsigned Max = std::max(RVVVectorBitsMin, RVVVectorBitsMax);
  return PowerOf2Floor((Max < 64 || Max > 65536) ? 0 : Max);
}

unsigned RISCVSubtarget::getMinRVVVectorSizeInBits() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");

  if (RVVVectorBitsMin == -1)
    return ZvlLen;

  // ZvlLen specifies the minimum required vlen. The lower bound provided by
  // riscv-v-vector-bits-min should be no less than it.
  if (RVVVectorBitsMin != 0 && RVVVectorBitsMin < (int)ZvlLen)
    report_fatal_error("riscv-v-vector-bits-min specified is lower "
                       "than the Zvl*b limitation");

  // FIXME: Change to >= 32 when VLEN = 32 is supported
  assert(
      (RVVVectorBitsMin == 0 ||
       (RVVVectorBitsMin >= 64 && RVVVectorBitsMin <= 65536 &&
        isPowerOf2_32(RVVVectorBitsMin))) &&
      "V or Zve* extension requires vector length to be in the range of 64 to "
      "65536 and a power of 2!");
  assert((RVVVectorBitsMax >= RVVVectorBitsMin || RVVVectorBitsMax == 0) &&
         "Minimum V extension vector length should not be larger than its "
         "maximum!");
  unsigned Min = RVVVectorBitsMin;
  if (RVVVectorBitsMax != 0)
    Min = std::min(RVVVectorBitsMin, RVVVectorBitsMax);
  return PowerOf2Floor((Min < 64 || Min > 65536) ? 0 : Min);
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
  if (EnableSubRegLiveness.getNumOccurrences())
    return EnableSubRegLiveness;
  // Enable subregister liveness for RVV to better handle LMUL>1 and segment
  // load/store.
  return hasVInstructions();
}

<<<<<<< HEAD
#if SIFIVE_CUSTOMIZATION
static unsigned factorLMul(unsigned Lat, RISCVII::VLMUL LMul) {
  // Every DLEN chunk is processed every VLEN / DLEN cycles, or, virtually
  // always, every 2 cycles,
  return Lat + (RISCVII::getLMULGroups(LMul) - 1) * 2;
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

      // Instructions without SEW, if any.
      if (!RISCVII::hasSEWOp(Desc.TSFlags))
        return factorLMul(Lat, LMul);

      unsigned SEW =
          1 << MI->getOperand(MI->getNumExplicitOperands() - 1).getImm();

      switch(Opcode) {
      default:
        return factorLMul(Lat, LMul);
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
        return factorLMul(Lat, LMul);
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
        return factorLMul(Lat, LMul);
      }
    }
  }

  llvm_unreachable("Unexpected processor model!");
}

=======
>>>>>>> upstream/main
void RISCVSubtarget::getPostRAMutations(
    std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(createRISCVMacroFusionDAGMutation());
}
<<<<<<< HEAD

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
#endif // SIFIVE_CUSTOMIZATION
=======
>>>>>>> upstream/main
