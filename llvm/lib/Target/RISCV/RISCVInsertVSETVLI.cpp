//===- RISCVInsertVSETVLI.cpp - Insert VSETVLI instructions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts VSETVLI instructions where
// needed and expands the vl outputs of VLEFF/VLSEGFF to PseudoReadVL
// instructions.
//
// This pass consists of 3 phases:
//
// Phase 1 collects how each basic block affects VL/VTYPE.
//
// Phase 2 uses the information from phase 1 to do a data flow analysis to
// propagate the VL/VTYPE changes through the function. This gives us the
// VL/VTYPE at the start of each basic block.
//
// Phase 3 inserts VSETVLI instructions in each basic block. Information from
// phase 2 is used to prevent inserting a VSETVLI before the first vector
// instruction in the block if possible.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <queue>
using namespace llvm;

#define DEBUG_TYPE "riscv-insert-vsetvli"
#define RISCV_INSERT_VSETVLI_NAME "RISC-V Insert VSETVLI pass"

STATISTIC(NumInsertedVSETVL, "Number of VSETVL inst inserted");
STATISTIC(NumCoalescedVSETVL, "Number of VSETVL inst coalesced");

#if SIFIVE_CUSTOMIZATION
cl::opt<bool> ForceTailUndisturbed(
    "riscv-force-tail-undisturbed", cl::init(false), cl::Hidden,
    cl::desc("Force to use tail undisturbed for all vector intrinsics."));
cl::opt<bool> ForceMaskUndisturbed(
    "riscv-force-mask-undisturbed", cl::init(false), cl::Hidden,
    cl::desc("Force to use mask undisturbed for all vector intrinsics."));
#endif // SIFIVE_CUSTOMIZATION
static cl::opt<bool> EnsureWholeVectorRegisterMoveValidVTYPE(
    DEBUG_TYPE "-whole-vector-register-move-valid-vtype", cl::Hidden,
    cl::desc("Insert vsetvlis before vmvNr.vs to ensure vtype is valid and "
             "vill is cleared"),
    cl::init(true));

namespace {

/// Given a virtual register \p Reg, return the corresponding VNInfo for it.
/// This will return nullptr if the virtual register is an implicit_def or
/// if LiveIntervals is not available.
static VNInfo *getVNInfoFromReg(Register Reg, const MachineInstr &MI,
                                const LiveIntervals *LIS) {
  assert(Reg.isVirtual());
  if (!LIS)
    return nullptr;
  auto &LI = LIS->getInterval(Reg);
  SlotIndex SI = LIS->getSlotIndexes()->getInstructionIndex(MI);
  return LI.getVNInfoBefore(SI);
}

static unsigned getVLOpNum(const MachineInstr &MI) {
  return RISCVII::getVLOpNum(MI.getDesc());
}

static unsigned getSEWOpNum(const MachineInstr &MI) {
  return RISCVII::getSEWOpNum(MI.getDesc());
}

#if SIFIVE_CUSTOMIZATION
// Returns true if the instruction is a Mammoth instruction that is also an
// alias of VSETVLI instruction, i.e. SF_VSETTNT, SF_VSETTNTX0
static bool isMammothVectorTNConfigInstr(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::PseudoSF_VSETTNT ||
         MI.getOpcode() == RISCV::PseudoSF_VSETTNTX0;
}

static bool isMammothVectorConfigTMTKInstr(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::PseudoSF_VSETTM ||
         MI.getOpcode() == RISCV::PseudoSF_VSETTK;
}

static bool isMammothVectorConfigInstr(const MachineInstr &MI) {
  return isMammothVectorTNConfigInstr(MI) || isMammothVectorConfigTMTKInstr(MI);
}
#endif // SIFIVE_CUSTOMIZATION

/// Get the EEW for a load or store instruction.  Return std::nullopt if MI is
/// not a load or store which ignores SEW.
static std::optional<unsigned> getEEWForLoadStore(const MachineInstr &MI) {
  switch (RISCV::getRVVMCOpcode(MI.getOpcode())) {
  default:
    return std::nullopt;
  case RISCV::VLE8_V:
  case RISCV::VLSE8_V:
  case RISCV::VSE8_V:
  case RISCV::VSSE8_V:
    return 8;
  case RISCV::VLE16_V:
  case RISCV::VLSE16_V:
  case RISCV::VSE16_V:
  case RISCV::VSSE16_V:
    return 16;
  case RISCV::VLE32_V:
  case RISCV::VLSE32_V:
  case RISCV::VSE32_V:
  case RISCV::VSSE32_V:
    return 32;
  case RISCV::VLE64_V:
  case RISCV::VLSE64_V:
  case RISCV::VSE64_V:
  case RISCV::VSSE64_V:
    return 64;
  }
}

/// Return true if this is an operation on mask registers.  Note that
/// this includes both arithmetic/logical ops and load/store (vlm/vsm).
static bool isMaskRegOp(const MachineInstr &MI) {
  if (!RISCVII::hasSEWOp(MI.getDesc().TSFlags))
    return false;
  const unsigned Log2SEW = MI.getOperand(getSEWOpNum(MI)).getImm();
  // A Log2SEW of 0 is an operation on mask registers only.
  return Log2SEW == 0;
}

/// Return true if the inactive elements in the result are entirely undefined.
/// Note that this is different from "agnostic" as defined by the vector
/// specification.  Agnostic requires each lane to either be undisturbed, or
/// take the value -1; no other value is allowed.
static bool hasUndefinedPassthru(const MachineInstr &MI) {

  unsigned UseOpIdx;
  if (!MI.isRegTiedToUseOperand(0, &UseOpIdx))
    // If there is no passthrough operand, then the pass through
    // lanes are undefined.
    return true;

  // All undefined passthrus should be $noreg: see
  // RISCVDAGToDAGISel::doPeepholeNoRegPassThru
  const MachineOperand &UseMO = MI.getOperand(UseOpIdx);
  return UseMO.getReg() == RISCV::NoRegister || UseMO.isUndef();
}

/// Return true if \p MI is a copy that will be lowered to one or more vmvNr.vs.
static bool isVectorCopy(const TargetRegisterInfo *TRI,
                         const MachineInstr &MI) {
  return MI.isCopy() && MI.getOperand(0).getReg().isPhysical() &&
         RISCVRegisterInfo::isRVVRegClass(
             TRI->getMinimalPhysRegClass(MI.getOperand(0).getReg()));
}

/// Which subfields of VL or VTYPE have values we need to preserve?
struct DemandedFields {
  // Some unknown property of VL is used.  If demanded, must preserve entire
  // value.
  bool VLAny = false;
  // Only zero vs non-zero is used. If demanded, can change non-zero values.
  bool VLZeroness = false;
  // What properties of SEW we need to preserve.
  enum : uint8_t {
    SEWEqual = 3, // The exact value of SEW needs to be preserved.
    SEWGreaterThanOrEqualAndLessThan64 =
        2, // SEW can be changed as long as it's greater
           // than or equal to the original value, but must be less
           // than 64.
    SEWGreaterThanOrEqual = 1, // SEW can be changed as long as it's greater
                               // than or equal to the original value.
    SEWNone = 0                // We don't need to preserve SEW at all.
  } SEW = SEWNone;
  enum : uint8_t {
    LMULEqual = 2, // The exact value of LMUL needs to be preserved.
    LMULLessThanOrEqualToM1 = 1, // We can use any LMUL <= M1.
    LMULNone = 0                 // We don't need to preserve LMUL at all.
  } LMUL = LMULNone;
  bool SEWLMULRatio = false;
  bool TailPolicy = false;
  bool MaskPolicy = false;
  // If this is true, we demand that VTYPE is set to some legal state, i.e. that
  // vill is unset.
  bool VILL = false;
#ifdef SIFIVE_CUSTOMIZATION
  bool UseTWiden = false;
  bool UseAltFmt = false;
#endif // SIFIVE_CUSTOMIZATION

  // Return true if any part of VTYPE was used
  bool usedVTYPE() const {
#ifdef SIFIVE_CUSTOMIZATION
    return SEW || LMUL || SEWLMULRatio || TailPolicy || MaskPolicy || VILL ||
           UseTWiden || UseAltFmt;
#endif // SIFIVE_CUSTOMIZATION
  }

  // Return true if any property of VL was used
  bool usedVL() {
    return VLAny || VLZeroness;
  }

  // Mark all VTYPE subfields and properties as demanded
  void demandVTYPE() {
    SEW = SEWEqual;
    LMUL = LMULEqual;
    SEWLMULRatio = true;
    TailPolicy = true;
    MaskPolicy = true;
    VILL = true;
#ifdef SIFIVE_CUSTOMIZATION
    UseTWiden = true;
    UseAltFmt = true;
#endif // SIFIVE_CUSTOMIZATION
  }

  // Mark all VL properties as demanded
  void demandVL() {
    VLAny = true;
    VLZeroness = true;
  }

  static DemandedFields all() {
    DemandedFields DF;
    DF.demandVTYPE();
    DF.demandVL();
    return DF;
  }

  // Make this the result of demanding both the fields in this and B.
  void doUnion(const DemandedFields &B) {
    VLAny |= B.VLAny;
    VLZeroness |= B.VLZeroness;
    SEW = std::max(SEW, B.SEW);
    LMUL = std::max(LMUL, B.LMUL);
    SEWLMULRatio |= B.SEWLMULRatio;
    TailPolicy |= B.TailPolicy;
    MaskPolicy |= B.MaskPolicy;
    VILL |= B.VILL;
#ifdef SIFIVE_CUSTOMIZATION
    UseAltFmt |= B.UseAltFmt;
    UseTWiden |= B.UseTWiden;
#endif // SIFIVE_CUSTOMIZATION
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  /// Implement operator<<.
  void print(raw_ostream &OS) const {
    OS << "{";
    OS << "VLAny=" << VLAny << ", ";
    OS << "VLZeroness=" << VLZeroness << ", ";
    OS << "SEW=";
    switch (SEW) {
    case SEWEqual:
      OS << "SEWEqual";
      break;
    case SEWGreaterThanOrEqual:
      OS << "SEWGreaterThanOrEqual";
      break;
    case SEWGreaterThanOrEqualAndLessThan64:
      OS << "SEWGreaterThanOrEqualAndLessThan64";
      break;
    case SEWNone:
      OS << "SEWNone";
      break;
    };
    OS << ", ";
    OS << "LMUL=";
    switch (LMUL) {
    case LMULEqual:
      OS << "LMULEqual";
      break;
    case LMULLessThanOrEqualToM1:
      OS << "LMULLessThanOrEqualToM1";
      break;
    case LMULNone:
      OS << "LMULNone";
      break;
    };
    OS << ", ";
    OS << "SEWLMULRatio=" << SEWLMULRatio << ", ";
    OS << "TailPolicy=" << TailPolicy << ", ";
    OS << "MaskPolicy=" << MaskPolicy << ", ";
    OS << "VILL=" << VILL;
#ifdef SIFIVE_CUSTOMIZATION
    OS << "MaskPolicy=" << MaskPolicy << ", ";
    OS << "UseAltFmt=" << UseAltFmt << ", ";
    OS << "UseTWiden=" << UseTWiden;
#endif // SIFIVE_CUSTOMIZATION
    OS << "}";
  }
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const DemandedFields &DF) {
  DF.print(OS);
  return OS;
}
#endif

static bool isLMUL1OrSmaller(RISCVVType::VLMUL LMUL) {
  auto [LMul, Fractional] = RISCVVType::decodeVLMUL(LMUL);
  return Fractional || LMul == 1;
}

/// Return true if moving from CurVType to NewVType is
/// indistinguishable from the perspective of an instruction (or set
/// of instructions) which use only the Used subfields and properties.
static bool areCompatibleVTYPEs(uint64_t CurVType, uint64_t NewVType,
                                const DemandedFields &Used) {
  switch (Used.SEW) {
  case DemandedFields::SEWNone:
    break;
  case DemandedFields::SEWEqual:
    if (RISCVVType::getSEW(CurVType) != RISCVVType::getSEW(NewVType))
      return false;
    break;
  case DemandedFields::SEWGreaterThanOrEqual:
    if (RISCVVType::getSEW(NewVType) < RISCVVType::getSEW(CurVType))
      return false;
    break;
  case DemandedFields::SEWGreaterThanOrEqualAndLessThan64:
    if (RISCVVType::getSEW(NewVType) < RISCVVType::getSEW(CurVType) ||
        RISCVVType::getSEW(NewVType) >= 64)
      return false;
    break;
  }

  switch (Used.LMUL) {
  case DemandedFields::LMULNone:
    break;
  case DemandedFields::LMULEqual:
    if (RISCVVType::getVLMUL(CurVType) != RISCVVType::getVLMUL(NewVType))
      return false;
    break;
  case DemandedFields::LMULLessThanOrEqualToM1:
    if (!isLMUL1OrSmaller(RISCVVType::getVLMUL(NewVType)))
      return false;
    break;
  }

  if (Used.SEWLMULRatio) {
    auto Ratio1 = RISCVVType::getSEWLMULRatio(RISCVVType::getSEW(CurVType),
                                              RISCVVType::getVLMUL(CurVType));
    auto Ratio2 = RISCVVType::getSEWLMULRatio(RISCVVType::getSEW(NewVType),
                                              RISCVVType::getVLMUL(NewVType));
    if (Ratio1 != Ratio2)
      return false;
  }

  if (Used.TailPolicy && RISCVVType::isTailAgnostic(CurVType) !=
                             RISCVVType::isTailAgnostic(NewVType))
    return false;
  if (Used.MaskPolicy && RISCVVType::isMaskAgnostic(CurVType) !=
                             RISCVVType::isMaskAgnostic(NewVType))
    return false;
#ifdef SIFIVE_CUSTOMIZATION
  // TODO: Since normal RVV instructions don't care about twiden, so we might be
  // able to reuse it if the previous twiden is same as we need for this
  // instruction.
  if (Used.UseTWiden && (RISCVVType::hasXSfmmWiden(CurVType) !=
                             RISCVVType::hasXSfmmWiden(NewVType) ||
                         (RISCVVType::hasXSfmmWiden(CurVType) &&
                          RISCVVType::getXSfmmWiden(CurVType) !=
                              RISCVVType::getXSfmmWiden(NewVType))))
    return false;
  if (Used.UseAltFmt == true &&
      RISCVVType::isAltFmt(CurVType) != RISCVVType::isAltFmt(NewVType))
    return false;
#endif // SIFIVE_CUSTOMIZATION
  return true;
}

/// Return the fields and properties demanded by the provided instruction.
DemandedFields getDemanded(const MachineInstr &MI, const RISCVSubtarget *ST) {
  // This function works in coalesceVSETVLI too. We can still use the value of a
  // SEW, VL, or Policy operand even though it might not be the exact value in
  // the VL or VTYPE, since we only care about what the instruction originally
  // demanded.

  // Most instructions don't use any of these subfeilds.
  DemandedFields Res;
  // Start conservative if registers are used
  if (MI.isCall() || MI.isInlineAsm() ||
      MI.readsRegister(RISCV::VL, /*TRI=*/nullptr))
    Res.demandVL();
  if (MI.isCall() || MI.isInlineAsm() ||
      MI.readsRegister(RISCV::VTYPE, /*TRI=*/nullptr))
    Res.demandVTYPE();
  // Start conservative on the unlowered form too
  uint64_t TSFlags = MI.getDesc().TSFlags;
  if (RISCVII::hasSEWOp(TSFlags)) {
    Res.demandVTYPE();
    if (RISCVII::hasVLOp(TSFlags))
      if (const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
          !VLOp.isReg() || !VLOp.isUndef())
        Res.demandVL();

    // Behavior is independent of mask policy.
    if (!RISCVII::usesMaskPolicy(TSFlags))
      Res.MaskPolicy = false;
  }

  // Loads and stores with implicit EEW do not demand SEW or LMUL directly.
  // They instead demand the ratio of the two which is used in computing
  // EMUL, but which allows us the flexibility to change SEW and LMUL
  // provided we don't change the ratio.
  // Note: We assume that the instructions initial SEW is the EEW encoded
  // in the opcode.  This is asserted when constructing the VSETVLIInfo.
  if (getEEWForLoadStore(MI)) {
    Res.SEW = DemandedFields::SEWNone;
    Res.LMUL = DemandedFields::LMULNone;
  }

  // Store instructions don't use the policy fields.
  if (RISCVII::hasSEWOp(TSFlags) && MI.getNumExplicitDefs() == 0) {
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

  // If this is a mask reg operation, it only cares about VLMAX.
  // TODO: Possible extensions to this logic
  // * Probably ok if available VLMax is larger than demanded
  // * The policy bits can probably be ignored..
  if (isMaskRegOp(MI)) {
    Res.SEW = DemandedFields::SEWNone;
    Res.LMUL = DemandedFields::LMULNone;
  }

  // For vmv.s.x and vfmv.s.f, there are only two behaviors, VL = 0 and VL > 0.
  if (RISCVInstrInfo::isScalarInsertInstr(MI)) {
    Res.LMUL = DemandedFields::LMULNone;
    Res.SEWLMULRatio = false;
    Res.VLAny = false;
    // For vmv.s.x and vfmv.s.f, if the passthru is *undefined*, we don't
    // need to preserve any other bits and are thus compatible with any larger,
    // etype and can disregard policy bits.  Warning: It's tempting to try doing
    // this for any tail agnostic operation, but we can't as TA requires
    // tail lanes to either be the original value or -1.  We are writing
    // unknown bits to the lanes here.
    if (hasUndefinedPassthru(MI)) {
      if (RISCVInstrInfo::isFloatScalarMoveOrScalarSplatInstr(MI) &&
          !ST->hasVInstructionsF64())
        Res.SEW = DemandedFields::SEWGreaterThanOrEqualAndLessThan64;
      else
        Res.SEW = DemandedFields::SEWGreaterThanOrEqual;
      Res.TailPolicy = false;
    }
  }

  // vmv.x.s, and vfmv.f.s are unconditional and ignore everything except SEW.
  if (RISCVInstrInfo::isScalarExtractInstr(MI)) {
    assert(!RISCVII::hasVLOp(TSFlags));
    Res.LMUL = DemandedFields::LMULNone;
    Res.SEWLMULRatio = false;
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

  if (RISCVII::hasVLOp(MI.getDesc().TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
    // A slidedown/slideup with an *undefined* passthru can freely clobber
    // elements not copied from the source vector (e.g. masked off, tail, or
    // slideup's prefix). Notes:
    // * We can't modify SEW here since the slide amount is in units of SEW.
    // * VL=1 is special only because we have existing support for zero vs
    //   non-zero VL.  We could generalize this if we had a VL > C predicate.
    // * The LMUL1 restriction is for machines whose latency may depend on VL.
    // * As above, this is only legal for tail "undefined" not "agnostic".
    if (RISCVInstrInfo::isVSlideInstr(MI) && VLOp.isImm() &&
        VLOp.getImm() == 1 && hasUndefinedPassthru(MI)) {
      Res.VLAny = false;
      Res.VLZeroness = true;
      Res.LMUL = DemandedFields::LMULLessThanOrEqualToM1;
      Res.TailPolicy = false;
    }

    // A tail undefined vmv.v.i/x or vfmv.v.f with VL=1 can be treated in the
    // same semantically as vmv.s.x.  This is particularly useful since we don't
    // have an immediate form of vmv.s.x, and thus frequently use vmv.v.i in
    // it's place. Since a splat is non-constant time in LMUL, we do need to be
    // careful to not increase the number of active vector registers (unlike for
    // vmv.s.x.)
    if (RISCVInstrInfo::isScalarSplatInstr(MI) && VLOp.isImm() &&
        VLOp.getImm() == 1 && hasUndefinedPassthru(MI)) {
      Res.LMUL = DemandedFields::LMULLessThanOrEqualToM1;
      Res.SEWLMULRatio = false;
      Res.VLAny = false;
      if (RISCVInstrInfo::isFloatScalarMoveOrScalarSplatInstr(MI) &&
          !ST->hasVInstructionsF64())
        Res.SEW = DemandedFields::SEWGreaterThanOrEqualAndLessThan64;
      else
        Res.SEW = DemandedFields::SEWGreaterThanOrEqual;
      Res.TailPolicy = false;
    }
  }

  // In §32.16.6, whole vector register moves have a dependency on SEW. At the
  // MIR level though we don't encode the element type, and it gives the same
  // result whatever the SEW may be.
  //
  // However it does need valid SEW, i.e. vill must be cleared. The entry to a
  // function, calls and inline assembly may all set it, so make sure we clear
  // it for whole register copies. Do this by leaving VILL demanded.
  if (isVectorCopy(ST->getRegisterInfo(), MI)) {
    Res.LMUL = DemandedFields::LMULNone;
    Res.SEW = DemandedFields::SEWNone;
    Res.SEWLMULRatio = false;
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

#ifdef SIFIVE_CUSTOMIZATION
  Res.UseAltFmt = RISCVII::getAltFmtType(MI.getDesc().TSFlags) !=
                  RISCVII::AltFmtType::DontCare;
  Res.UseTWiden = RISCVII::hasTWidenOp(MI.getDesc().TSFlags) ||
                  isMammothVectorConfigInstr(MI);
#endif // SIFIVE_CUSTOMIZATION
  if (RISCVInstrInfo::isVExtractInstr(MI)) {
    assert(!RISCVII::hasVLOp(TSFlags));
    // TODO: LMUL can be any larger value (without cost)
    Res.TailPolicy = false;
  }

  return Res;
}

/// Defines the abstract state with which the forward dataflow models the
/// values of the VL and VTYPE registers after insertion.
class VSETVLIInfo {
  struct AVLDef {
    // Every AVLDef should have a VNInfo, unless we're running without
    // LiveIntervals in which case this will be nullptr.
    const VNInfo *ValNo;
    Register DefReg;
  };
  union {
    AVLDef AVLRegDef;
    unsigned AVLImm;
  };

  enum : uint8_t {
    Uninitialized,
    AVLIsReg,
    AVLIsImm,
    AVLIsVLMAX,
    Unknown, // AVL and VTYPE are fully unknown
  } State = Uninitialized;

  // Fields from VTYPE.
  RISCVVType::VLMUL VLMul = RISCVVType::LMUL_1;
  uint8_t SEW = 0;
  uint8_t TailAgnostic : 1;
  uint8_t MaskAgnostic : 1;
  uint8_t SEWLMULRatioOnly : 1;
#ifdef SIFIVE_CUSTOMIZATION
  uint8_t AltFmt : 1;
  uint8_t TWiden = 0;
#endif // SIFIVE_CUSTOMIZATION

public:
  VSETVLIInfo()
      : AVLImm(0), TailAgnostic(false), MaskAgnostic(false),
        SEWLMULRatioOnly(false) {}

  static VSETVLIInfo getUnknown() {
    VSETVLIInfo Info;
    Info.setUnknown();
    return Info;
  }

  bool isValid() const { return State != Uninitialized; }
  void setUnknown() { State = Unknown; }
  bool isUnknown() const { return State == Unknown; }

  void setAVLRegDef(const VNInfo *VNInfo, Register AVLReg) {
    assert(AVLReg.isVirtual());
    AVLRegDef.ValNo = VNInfo;
    AVLRegDef.DefReg = AVLReg;
    State = AVLIsReg;
  }

  void setAVLImm(unsigned Imm) {
    AVLImm = Imm;
    State = AVLIsImm;
  }

  void setAVLVLMAX() { State = AVLIsVLMAX; }

  bool hasAVLImm() const { return State == AVLIsImm; }
  bool hasAVLReg() const { return State == AVLIsReg; }
  bool hasAVLVLMAX() const { return State == AVLIsVLMAX; }
  Register getAVLReg() const {
    assert(hasAVLReg() && AVLRegDef.DefReg.isVirtual());
    return AVLRegDef.DefReg;
  }
  unsigned getAVLImm() const {
    assert(hasAVLImm());
    return AVLImm;
  }
  const VNInfo *getAVLVNInfo() const {
    assert(hasAVLReg());
    return AVLRegDef.ValNo;
  }
  // Most AVLIsReg infos will have a single defining MachineInstr, unless it was
  // a PHI node. In that case getAVLVNInfo()->def will point to the block
  // boundary slot and this will return nullptr.  If LiveIntervals isn't
  // available, nullptr is also returned.
  const MachineInstr *getAVLDefMI(const LiveIntervals *LIS) const {
    assert(hasAVLReg());
    if (!LIS || getAVLVNInfo()->isPHIDef())
      return nullptr;
    auto *MI = LIS->getInstructionFromIndex(getAVLVNInfo()->def);
    assert(MI);
    return MI;
  }

  void setAVL(const VSETVLIInfo &Info) {
    assert(Info.isValid());
    if (Info.isUnknown())
      setUnknown();
    else if (Info.hasAVLReg())
      setAVLRegDef(Info.getAVLVNInfo(), Info.getAVLReg());
    else if (Info.hasAVLVLMAX())
      setAVLVLMAX();
    else {
      assert(Info.hasAVLImm());
      setAVLImm(Info.getAVLImm());
    }
  }

  unsigned getSEW() const { return SEW; }
  RISCVVType::VLMUL getVLMUL() const { return VLMul; }
  bool getTailAgnostic() const { return TailAgnostic; }
  bool getMaskAgnostic() const { return MaskAgnostic; }
#ifdef SIFIVE_CUSTOMIZATION
  bool getAltFmt() const { return AltFmt; }
  unsigned getTWiden() const { return TWiden; }
#endif // SIFIVE_CUSTOMIZATION

  bool hasNonZeroAVL(const LiveIntervals *LIS) const {
    if (hasAVLImm())
      return getAVLImm() > 0;
    if (hasAVLReg()) {
      if (auto *DefMI = getAVLDefMI(LIS))
        return RISCVInstrInfo::isNonZeroLoadImmediate(*DefMI);
    }
    if (hasAVLVLMAX())
      return true;
    return false;
  }

  bool hasEquallyZeroAVL(const VSETVLIInfo &Other,
                         const LiveIntervals *LIS) const {
    if (hasSameAVL(Other))
      return true;
    return (hasNonZeroAVL(LIS) && Other.hasNonZeroAVL(LIS));
  }

  bool hasSameAVLLatticeValue(const VSETVLIInfo &Other) const {
    if (hasAVLReg() && Other.hasAVLReg()) {
      assert(!getAVLVNInfo() == !Other.getAVLVNInfo() &&
             "we either have intervals or we don't");
      if (!getAVLVNInfo())
        return getAVLReg() == Other.getAVLReg();
      return getAVLVNInfo()->id == Other.getAVLVNInfo()->id &&
             getAVLReg() == Other.getAVLReg();
    }

    if (hasAVLImm() && Other.hasAVLImm())
      return getAVLImm() == Other.getAVLImm();

    if (hasAVLVLMAX())
      return Other.hasAVLVLMAX() && hasSameVLMAX(Other);

    return false;
  }

  // Return true if the two lattice values are guaranteed to have
  // the same AVL value at runtime.
  bool hasSameAVL(const VSETVLIInfo &Other) const {
    // Without LiveIntervals, we don't know which instruction defines a
    // register.  Since a register may be redefined, this means all AVLIsReg
    // states must be treated as possibly distinct.
    if (hasAVLReg() && Other.hasAVLReg()) {
      assert(!getAVLVNInfo() == !Other.getAVLVNInfo() &&
             "we either have intervals or we don't");
      if (!getAVLVNInfo())
        return false;
    }
    return hasSameAVLLatticeValue(Other);
  }

  void setVTYPE(unsigned VType) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = RISCVVType::getVLMUL(VType);
    SEW = RISCVVType::getSEW(VType);
    TailAgnostic = RISCVVType::isTailAgnostic(VType);
    MaskAgnostic = RISCVVType::isMaskAgnostic(VType);
#if SIFIVE_CUSTOMIZATION
    AltFmt = RISCVVType::isAltFmt(VType);
    TWiden =
        RISCVVType::hasXSfmmWiden(VType) ? RISCVVType::getXSfmmWiden(VType) : 0;
#endif // SIFIVE_CUSTOMIZATION
  }
#if SIFIVE_CUSTOMIZATION
  void setVTYPE(RISCVVType::VLMUL L, unsigned S, bool TA, bool MA, bool AF,
                unsigned W) {
    assert((isValid() && !isUnknown()) &&
           "Can't set VTYPE for uninitialized or unknown");
#endif // SIFIVE_CUSTOMIZATION
    VLMul = L;
    SEW = S;
    TailAgnostic = TA;
    MaskAgnostic = MA;
#if SIFIVE_CUSTOMIZATION
    AltFmt = AF;
    TWiden = W;
#endif // SIFIVE_CUSTOMIZATION
  }

#if SIFIVE_CUSTOMIZATION
  void setAltFmt(bool AF) { AltFmt = AF; }

  void setTWiden(unsigned W) { TWiden = W; }
#endif // SIFIVE_CUSTOMIZATION

  void setVLMul(RISCVVType::VLMUL VLMul) { this->VLMul = VLMul; }

  unsigned encodeVTYPE() const {
    assert(isValid() && !isUnknown() && !SEWLMULRatioOnly &&
           "Can't encode VTYPE for uninitialized or unknown");
#if SIFIVE_CUSTOMIZATION
    if (TWiden != 0)
      return RISCVVType::encodeXSfmmVType(SEW, TWiden, AltFmt);
    return RISCVVType::encodeVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic,
                                   AltFmt);
#endif // SIFIVE_CUSTOMIZATION
  }

  bool hasSEWLMULRatioOnly() const { return SEWLMULRatioOnly; }

  bool hasSameVTYPE(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    assert(!SEWLMULRatioOnly && !Other.SEWLMULRatioOnly &&
           "Can't compare when only LMUL/SEW ratio is valid.");
#if SIFIVE_CUSTOMIZATION
    return std::tie(VLMul, SEW, TailAgnostic, MaskAgnostic, AltFmt, TWiden) ==
           std::tie(Other.VLMul, Other.SEW, Other.TailAgnostic,
                    Other.MaskAgnostic, Other.AltFmt, Other.TWiden);
#endif // SIFIVE_CUSTOMIZATION
  }

  unsigned getSEWLMULRatio() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return RISCVVType::getSEWLMULRatio(SEW, VLMul);
  }

  // Check if the VTYPE for these two VSETVLIInfos produce the same VLMAX.
  // Note that having the same VLMAX ensures that both share the same
  // function from AVL to VL; that is, they must produce the same VL value
  // for any given AVL value.
  bool hasSameVLMAX(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
#ifdef SIFIVE_CUSTOMIZATION
    if (getTWiden() != Other.getTWiden())
      return false;
    if (getTWiden() == 0)
      return getSEWLMULRatio() == Other.getSEWLMULRatio();
    return getSEW() == Other.getSEW();
#else
    return getSEWLMULRatio() == Other.getSEWLMULRatio();
#endif // SIFIVE_CUSTOMIZATION
  }

  bool hasCompatibleVTYPE(const DemandedFields &Used,
                          const VSETVLIInfo &Require) const {
    return areCompatibleVTYPEs(Require.encodeVTYPE(), encodeVTYPE(), Used);
  }

  // Determine whether the vector instructions requirements represented by
  // Require are compatible with the previous vsetvli instruction represented
  // by this.  MI is the instruction whose requirements we're considering.
  bool isCompatible(const DemandedFields &Used, const VSETVLIInfo &Require,
                    const LiveIntervals *LIS) const {
    assert(isValid() && Require.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    // Nothing is compatible with Unknown.
    if (isUnknown() || Require.isUnknown())
      return false;

    // If only our VLMAX ratio is valid, then this isn't compatible.
    if (SEWLMULRatioOnly || Require.SEWLMULRatioOnly)
      return false;

    if (Used.VLAny && !(hasSameAVL(Require) && hasSameVLMAX(Require)))
      return false;

    if (Used.VLZeroness && !hasEquallyZeroAVL(Require, LIS))
      return false;

    return hasCompatibleVTYPE(Used, Require);
  }

  bool operator==(const VSETVLIInfo &Other) const {
    // Uninitialized is only equal to another Uninitialized.
    if (!isValid())
      return !Other.isValid();
    if (!Other.isValid())
      return !isValid();

    // Unknown is only equal to another Unknown.
    if (isUnknown())
      return Other.isUnknown();
    if (Other.isUnknown())
      return isUnknown();

    if (!hasSameAVLLatticeValue(Other))
      return false;

    // If the SEWLMULRatioOnly bits are different, then they aren't equal.
    if (SEWLMULRatioOnly != Other.SEWLMULRatioOnly)
      return false;

    // If only the VLMAX is valid, check that it is the same.
    if (SEWLMULRatioOnly)
      return hasSameVLMAX(Other);

    // If the full VTYPE is valid, check that it is the same.
    return hasSameVTYPE(Other);
  }

  bool operator!=(const VSETVLIInfo &Other) const {
    return !(*this == Other);
  }

  // Calculate the VSETVLIInfo visible to a block assuming this and Other are
  // both predecessors.
  VSETVLIInfo intersect(const VSETVLIInfo &Other) const {
    // If the new value isn't valid, ignore it.
    if (!Other.isValid())
      return *this;

    // If this value isn't valid, this must be the first predecessor, use it.
    if (!isValid())
      return Other;

    // If either is unknown, the result is unknown.
    if (isUnknown() || Other.isUnknown())
      return VSETVLIInfo::getUnknown();

    // If we have an exact, match return this.
    if (*this == Other)
      return *this;

    // Not an exact match, but maybe the AVL and VLMAX are the same. If so,
    // return an SEW/LMUL ratio only value.
    if (hasSameAVL(Other) && hasSameVLMAX(Other)) {
      VSETVLIInfo MergeInfo = *this;
      MergeInfo.SEWLMULRatioOnly = true;
      return MergeInfo;
    }

    // Otherwise the result is unknown.
    return VSETVLIInfo::getUnknown();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  /// Implement operator<<.
  /// @{
  void print(raw_ostream &OS) const {
    OS << "{";
    if (!isValid())
      OS << "Uninitialized";
    if (isUnknown())
      OS << "unknown";
    if (hasAVLReg())
      OS << "AVLReg=" << llvm::printReg(getAVLReg());
    if (hasAVLImm())
      OS << "AVLImm=" << (unsigned)AVLImm;
    if (hasAVLVLMAX())
      OS << "AVLVLMAX";
    OS << ", ";

    unsigned LMul;
    bool Fractional;
    std::tie(LMul, Fractional) = decodeVLMUL(VLMul);

    OS << "VLMul=";
    if (Fractional)
      OS << "mf";
    else
      OS << "m";
    OS << LMul << ", "
       << "SEW=e" << (unsigned)SEW << ", "
       << "TailAgnostic=" << (bool)TailAgnostic << ", "
       << "MaskAgnostic=" << (bool)MaskAgnostic << ", "
       << "SEWLMULRatioOnly=" << (bool)SEWLMULRatioOnly << "}";
#if SIFIVE_CUSTOMIZATION
    OS << "TWiden=" << (bool)TWiden << ", ";
    OS << "AltFmt=" << (bool)AltFmt << "}";
#endif // SIFIVE_CUSTOMIZATION
  }
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const VSETVLIInfo &V) {
  V.print(OS);
  return OS;
}
#endif

struct BlockData {
  // The VSETVLIInfo that represents the VL/VTYPE settings on exit from this
  // block. Calculated in Phase 2.
  VSETVLIInfo Exit;

  // The VSETVLIInfo that represents the VL/VTYPE settings from all predecessor
  // blocks. Calculated in Phase 2, and used by Phase 3.
  VSETVLIInfo Pred;

  // Keeps track of whether the block is already in the queue.
  bool InQueue = false;

  BlockData() = default;
};

#ifdef SIFIVE_CUSTOMIZATION
enum TKTMMode {
  VSETTK = 0,
  VSETTM = 1,
};
#endif // SIFIVE_CUSTOMIZATION

class RISCVInsertVSETVLI : public MachineFunctionPass {
  const RISCVSubtarget *ST;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  // Possibly null!
  LiveIntervals *LIS;

  std::vector<BlockData> BlockInfo;
  std::queue<const MachineBasicBlock *> WorkList;

public:
  static char ID;

  RISCVInsertVSETVLI() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();

    AU.addUsedIfAvailable<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addPreserved<LiveDebugVariablesWrapperLegacy>();
    AU.addPreserved<LiveStacksWrapperLegacy>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INSERT_VSETVLI_NAME; }

private:
  bool needVSETVLI(const DemandedFields &Used, const VSETVLIInfo &Require,
                   const VSETVLIInfo &CurInfo) const;
  bool needVSETVLIPHI(const VSETVLIInfo &Require,
                      const MachineBasicBlock &MBB) const;
  void insertVSETVLI(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc DL,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo);

  void transferBefore(VSETVLIInfo &Info, const MachineInstr &MI) const;
  void transferAfter(VSETVLIInfo &Info, const MachineInstr &MI) const;
  bool computeVLVTYPEChanges(const MachineBasicBlock &MBB,
                             VSETVLIInfo &Info) const;
  void computeIncomingVLVTYPE(const MachineBasicBlock &MBB);
  void emitVSETVLIs(MachineBasicBlock &MBB);
  void doPRE(MachineBasicBlock &MBB);
  void insertReadVL(MachineBasicBlock &MBB);

  bool canMutatePriorConfig(const MachineInstr &PrevMI, const MachineInstr &MI,
                            const DemandedFields &Used) const;

  void coalesceVSETVLIs(MachineBasicBlock &MBB) const;
#ifdef SIFIVE_CUSTOMIZATION
  bool canMutatePriorConfigWithTWiden(const MachineInstr &PrevMI,
                                      const MachineInstr &MI) const;
  void coalesceVSETVLIsForTWiden(MachineBasicBlock &MBB) const;
#endif // SIFIVE_CUSTOMIZATION

  VSETVLIInfo getInfoForVSETVLI(const MachineInstr &MI) const;
  VSETVLIInfo computeInfoForInstr(const MachineInstr &MI) const;
  void forwardVSETVLIAVL(VSETVLIInfo &Info) const;
#ifdef SIFIVE_CUSTOMIZATION
  bool insertVSETMTK(MachineBasicBlock &MBB, TKTMMode Mode) const;
#endif // SIFIVE_CUSTOMIZATION
};

} // end anonymous namespace

char RISCVInsertVSETVLI::ID = 0;
char &llvm::RISCVInsertVSETVLIID = RISCVInsertVSETVLI::ID;

INITIALIZE_PASS(RISCVInsertVSETVLI, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

// If the AVL is defined by a vsetvli's output vl with the same VLMAX, we can
// replace the AVL operand with the AVL of the defining vsetvli. E.g.
//
// %vl = PseudoVSETVLI %avl:gpr, SEW=32, LMUL=M1
// $x0 = PseudoVSETVLI %vl:gpr, SEW=32, LMUL=M1
// ->
// %vl = PseudoVSETVLI %avl:gpr, SEW=32, LMUL=M1
// $x0 = PseudoVSETVLI %avl:gpr, SEW=32, LMUL=M1
void RISCVInsertVSETVLI::forwardVSETVLIAVL(VSETVLIInfo &Info) const {
  if (!Info.hasAVLReg())
    return;
  const MachineInstr *DefMI = Info.getAVLDefMI(LIS);
  if (!DefMI || !RISCVInstrInfo::isVectorConfigInstr(*DefMI))
    return;
  VSETVLIInfo DefInstrInfo = getInfoForVSETVLI(*DefMI);
  if (!DefInstrInfo.hasSameVLMAX(Info))
    return;
  Info.setAVL(DefInstrInfo);
}

// Return a VSETVLIInfo representing the changes made by this VSETVLI or
// VSETIVLI instruction.
VSETVLIInfo
RISCVInsertVSETVLI::getInfoForVSETVLI(const MachineInstr &MI) const {
  VSETVLIInfo NewInfo;
  if (MI.getOpcode() == RISCV::PseudoVSETIVLI) {
    NewInfo.setAVLImm(MI.getOperand(1).getImm());
#ifdef SIFIVE_CUSTOMIZATION
  } else if (isMammothVectorTNConfigInstr(MI)) {
    Register ATReg = MI.getOperand(1).getReg();
    switch (MI.getOpcode()) {
    case RISCV::PseudoSF_VSETTNTX0:
      NewInfo.setAVLVLMAX();
      break;
    case RISCV::PseudoSF_VSETTNT:
      NewInfo.setAVLRegDef(getVNInfoFromReg(ATReg, MI, LIS), ATReg);
      break;
    }
#endif // SIFIVE_CUSTOMIZATION
  } else {
    assert(MI.getOpcode() == RISCV::PseudoVSETVLI ||
           MI.getOpcode() == RISCV::PseudoVSETVLIX0);
    if (MI.getOpcode() == RISCV::PseudoVSETVLIX0)
      NewInfo.setAVLVLMAX();
    else if (MI.getOperand(1).isUndef())
      // Otherwise use an AVL of 1 to avoid depending on previous vl.
      NewInfo.setAVLImm(1);
    else {
      Register AVLReg = MI.getOperand(1).getReg();
      VNInfo *VNI = getVNInfoFromReg(AVLReg, MI, LIS);
      NewInfo.setAVLRegDef(VNI, AVLReg);
    }
  }
  NewInfo.setVTYPE(MI.getOperand(2).getImm());

  forwardVSETVLIAVL(NewInfo);

  return NewInfo;
}

static unsigned computeVLMAX(unsigned VLEN, unsigned SEW,
                             RISCVVType::VLMUL VLMul) {
  auto [LMul, Fractional] = RISCVVType::decodeVLMUL(VLMul);
  if (Fractional)
    VLEN = VLEN / LMul;
  else
    VLEN = VLEN * LMul;
  return VLEN/SEW;
}

VSETVLIInfo
RISCVInsertVSETVLI::computeInfoForInstr(const MachineInstr &MI) const {
  VSETVLIInfo InstrInfo;
  const uint64_t TSFlags = MI.getDesc().TSFlags;

  bool TailAgnostic = true;
  bool MaskAgnostic = true;
  if (!hasUndefinedPassthru(MI)) {
    // Start with undisturbed.
    TailAgnostic = false;
    MaskAgnostic = false;

    // If there is a policy operand, use it.
    if (RISCVII::hasVecPolicyOp(TSFlags)) {
      const MachineOperand &Op = MI.getOperand(MI.getNumExplicitOperands() - 1);
      uint64_t Policy = Op.getImm();
      assert(Policy <=
                 (RISCVVType::TAIL_AGNOSTIC | RISCVVType::MASK_AGNOSTIC) &&
             "Invalid Policy Value");
      TailAgnostic = Policy & RISCVVType::TAIL_AGNOSTIC;
      MaskAgnostic = Policy & RISCVVType::MASK_AGNOSTIC;
    }

    if (!RISCVII::usesMaskPolicy(TSFlags))
      MaskAgnostic = true;
  }

#if SIFIVE_CUSTOMIZATION
  // Override to always use tail undisturbed. Useful to reduce vsetvli on
  // CPUs that treat them the same.
  if (ForceTailUndisturbed)
    TailAgnostic = false;
  if (ForceMaskUndisturbed)
    MaskAgnostic = false;
#endif // SIFIVE_CUSTOMIZATION

  RISCVVType::VLMUL VLMul = RISCVII::getLMul(TSFlags);

#if SIFIVE_CUSTOMIZATION
  bool AltFmt = RISCVII::getAltFmtType(TSFlags) == RISCVII::AltFmtType::AltFmt;
  InstrInfo.setAltFmt(AltFmt);

#endif // SIFIVE_CUSTOMIZATION
  unsigned Log2SEW = MI.getOperand(getSEWOpNum(MI)).getImm();
  // A Log2SEW of 0 is an operation on mask registers only.
  unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
  assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");

#if SIFIVE_CUSTOMIZATION
  if (RISCVII::hasTWidenOp(TSFlags)) {
    assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");

    // FIXME: When TWiden is enable, VLMul is a run-time variable.
    // The VLMul should be unknown. And it could only be compare between
    // TWiden status.
    VLMul = RISCVVType::LMUL_1;

    const MachineOperand &TWidenOp =
        MI.getOperand(MI.getNumExplicitOperands() - 1);
    unsigned TWiden = TWidenOp.getImm();

    InstrInfo.setAVLVLMAX();
    if (RISCVII::hasVLOp(TSFlags)) {
      const MachineOperand &TnOp =
          MI.getOperand(RISCVII::getTNOpNum(MI.getDesc()));

      if (TnOp.getReg().isVirtual())
        InstrInfo.setAVLRegDef(getVNInfoFromReg(TnOp.getReg(), MI, LIS),
                               TnOp.getReg());
    }

    InstrInfo.setVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic, AltFmt, TWiden);

    return InstrInfo;
  }
#endif // SIFIVE_CUSTOMIZATION

  if (RISCVII::hasVLOp(TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
    if (VLOp.isImm()) {
      int64_t Imm = VLOp.getImm();
      // Convert the VLMax sentintel to X0 register.
      if (Imm == RISCV::VLMaxSentinel) {
        // If we know the exact VLEN, see if we can use the constant encoding
        // for the VLMAX instead.  This reduces register pressure slightly.
        const unsigned VLMAX = computeVLMAX(ST->getRealMaxVLen(), SEW, VLMul);
        if (ST->getRealMinVLen() == ST->getRealMaxVLen() && VLMAX <= 31)
          InstrInfo.setAVLImm(VLMAX);
        else
          InstrInfo.setAVLVLMAX();
      }
      else
        InstrInfo.setAVLImm(Imm);
    } else if (VLOp.isUndef()) {
      // Otherwise use an AVL of 1 to avoid depending on previous vl.
      InstrInfo.setAVLImm(1);
    } else {
      VNInfo *VNI = getVNInfoFromReg(VLOp.getReg(), MI, LIS);
      InstrInfo.setAVLRegDef(VNI, VLOp.getReg());
    }
  } else {
    assert(RISCVInstrInfo::isScalarExtractInstr(MI) ||
           RISCVInstrInfo::isVExtractInstr(MI));
    // Pick a random value for state tracking purposes, will be ignored via
    // the demanded fields mechanism
    InstrInfo.setAVLImm(1);
  }
#ifndef NDEBUG
  if (std::optional<unsigned> EEW = getEEWForLoadStore(MI)) {
    assert(SEW == EEW && "Initial SEW doesn't match expected EEW");
  }
#endif
#if SIFIVE_CUSTOMIZATION
  // TODO: Propagate the twiden from previous vtype for potential reuse.
  InstrInfo.setVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic, AltFmt,
                     /*TWiden*/ 0);
#endif // SIFIVE_CUSTOMIZATION

  forwardVSETVLIAVL(InstrInfo);

  return InstrInfo;
}

void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator InsertPt,
                                       DebugLoc DL, const VSETVLIInfo &Info,
                                       const VSETVLIInfo &PrevInfo) {
  ++NumInsertedVSETVL;

  if (PrevInfo.isValid() && !PrevInfo.isUnknown()) {
    // Use X0, X0 form if the AVL is the same and the SEW+LMUL gives the same
    // VLMAX.
    if (Info.hasSameAVL(PrevInfo) && Info.hasSameVLMAX(PrevInfo)) {
#ifdef SIFIVE_CUSTOMIZATION
      auto MI = BuildMI(MBB, InsertPt, DL,
                        TII->get(Info.getTWiden() ? RISCV::PseudoSF_VSETTNTX0X0
                                                  : RISCV::PseudoVSETVLIX0X0))
#else
      auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0X0))
#endif // SIFIVE_CUSTOMIZATION
                    .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                    .addReg(RISCV::X0, RegState::Kill)
                    .addImm(Info.encodeVTYPE())
                    .addReg(RISCV::VL, RegState::Implicit);
      if (LIS)
        LIS->InsertMachineInstrInMaps(*MI);
      return;
    }

    // If our AVL is a virtual register, it might be defined by a VSET(I)VLI. If
    // it has the same VLMAX we want and the last VL/VTYPE we observed is the
    // same, we can use the X0, X0 form.
    if (Info.hasSameVLMAX(PrevInfo) && Info.hasAVLReg()) {
      if (const MachineInstr *DefMI = Info.getAVLDefMI(LIS);
          DefMI && RISCVInstrInfo::isVectorConfigInstr(*DefMI)) {
        VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
        if (DefInfo.hasSameAVL(PrevInfo) && DefInfo.hasSameVLMAX(PrevInfo)) {
#ifdef SIFIVE_CUSTOMIZATION
          auto MI =
              BuildMI(MBB, InsertPt, DL,
                      TII->get(Info.getTWiden() ? RISCV::PseudoSF_VSETTNTX0X0
                                                : RISCV::PseudoVSETVLIX0X0))
                  .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                  .addReg(RISCV::X0, RegState::Kill)
                  .addImm(Info.encodeVTYPE())
                  .addReg(RISCV::VL, RegState::Implicit);
#else
          auto MI =
              BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0X0))
                  .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                  .addReg(RISCV::X0, RegState::Kill)
                  .addImm(Info.encodeVTYPE())
                  .addReg(RISCV::VL, RegState::Implicit);
#endif // SIFIVE_CUSTOMIZATION
          if (LIS)
            LIS->InsertMachineInstrInMaps(*MI);
          return;
        }
      }
    }
  }

  if (Info.hasAVLImm()) {
    auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETIVLI))
                  .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                  .addImm(Info.getAVLImm())
                  .addImm(Info.encodeVTYPE());
    if (LIS)
      LIS->InsertMachineInstrInMaps(*MI);
    return;
  }

  if (Info.hasAVLVLMAX()) {
    Register DestReg = MRI->createVirtualRegister(&RISCV::GPRNoX0RegClass);
#ifdef SIFIVE_CUSTOMIZATION
    auto MI = BuildMI(MBB, InsertPt, DL,
                      TII->get(Info.getTWiden() ? RISCV::PseudoSF_VSETTNTX0
                                                : RISCV::PseudoVSETVLIX0))
#else
    auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
#endif // SIFIVE_CUSTOMIZATION
                  .addReg(DestReg, RegState::Define | RegState::Dead)
                  .addReg(RISCV::X0, RegState::Kill)
                  .addImm(Info.encodeVTYPE());
    if (LIS) {
      LIS->InsertMachineInstrInMaps(*MI);
      LIS->createAndComputeVirtRegInterval(DestReg);
    }
    return;
  }

  Register AVLReg = Info.getAVLReg();
  MRI->constrainRegClass(AVLReg, &RISCV::GPRNoX0RegClass);
#ifdef SIFIVE_CUSTOMIZATION
  auto MI = BuildMI(MBB, InsertPt, DL,
                    TII->get(Info.getTWiden() ? RISCV::PseudoSF_VSETTNT
                                              : RISCV::PseudoVSETVLI))
#else
  auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLI))
#endif // SIFIVE_CUSTOMIZATION
                .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                .addReg(AVLReg)
                .addImm(Info.encodeVTYPE());
  if (LIS) {
    LIS->InsertMachineInstrInMaps(*MI);
    LiveInterval &LI = LIS->getInterval(AVLReg);
    SlotIndex SI = LIS->getInstructionIndex(*MI).getRegSlot();
    const VNInfo *CurVNI = Info.getAVLVNInfo();
    // If the AVL value isn't live at MI, do a quick check to see if it's easily
    // extendable. Otherwise, we need to copy it.
    if (LI.getVNInfoBefore(SI) != CurVNI) {
      if (!LI.liveAt(SI) && LI.containsOneValue())
        LIS->extendToIndices(LI, SI);
      else {
        Register AVLCopyReg =
            MRI->createVirtualRegister(&RISCV::GPRNoX0RegClass);
        MachineBasicBlock *MBB = LIS->getMBBFromIndex(CurVNI->def);
        MachineBasicBlock::iterator II;
        if (CurVNI->isPHIDef())
          II = MBB->getFirstNonPHI();
        else {
          II = LIS->getInstructionFromIndex(CurVNI->def);
          II = std::next(II);
        }
        assert(II.isValid());
        auto AVLCopy = BuildMI(*MBB, II, DL, TII->get(RISCV::COPY), AVLCopyReg)
                           .addReg(AVLReg);
        LIS->InsertMachineInstrInMaps(*AVLCopy);
        MI->getOperand(1).setReg(AVLCopyReg);
        LIS->createAndComputeVirtRegInterval(AVLCopyReg);
      }
    }
  }
}

/// Return true if a VSETVLI is required to transition from CurInfo to Require
/// given a set of DemandedFields \p Used.
bool RISCVInsertVSETVLI::needVSETVLI(const DemandedFields &Used,
                                     const VSETVLIInfo &Require,
                                     const VSETVLIInfo &CurInfo) const {
  if (!CurInfo.isValid() || CurInfo.isUnknown() || CurInfo.hasSEWLMULRatioOnly())
    return true;

  if (CurInfo.isCompatible(Used, Require, LIS))
    return false;

  return true;
}

// If we don't use LMUL or the SEW/LMUL ratio, then adjust LMUL so that we
// maintain the SEW/LMUL ratio. This allows us to eliminate VL toggles in more
// places.
static VSETVLIInfo adjustIncoming(const VSETVLIInfo &PrevInfo,
                                  const VSETVLIInfo &NewInfo,
                                  DemandedFields &Demanded) {
  VSETVLIInfo Info = NewInfo;

  if (!Demanded.LMUL && !Demanded.SEWLMULRatio && PrevInfo.isValid() &&
      !PrevInfo.isUnknown()) {
#ifdef SIFIVE_CUSTOMIZATION
    if (PrevInfo.getTWiden() == NewInfo.getTWiden() && PrevInfo.getTWiden() == 0)
      if (auto NewVLMul = RISCVVType::getSameRatioLMUL(
              PrevInfo.getSEW(), PrevInfo.getVLMUL(), Info.getSEW()))
        Info.setVLMul(*NewVLMul);
#else
    if (auto NewVLMul = RISCVVType::getSameRatioLMUL(
            PrevInfo.getSEW(), PrevInfo.getVLMUL(), Info.getSEW()))
      Info.setVLMul(*NewVLMul);
#endif // SIFIVE_CUSTOMIZATION
    Demanded.LMUL = DemandedFields::LMULEqual;
  }

  return Info;
}

// Given an incoming state reaching MI, minimally modifies that state so that it
// is compatible with MI. The resulting state is guaranteed to be semantically
// legal for MI, but may not be the state requested by MI.
void RISCVInsertVSETVLI::transferBefore(VSETVLIInfo &Info,
                                        const MachineInstr &MI) const {
  if (isVectorCopy(ST->getRegisterInfo(), MI) &&
      (Info.isUnknown() || !Info.isValid() || Info.hasSEWLMULRatioOnly())) {
    // Use an arbitrary but valid AVL and VTYPE so vill will be cleared. It may
    // be coalesced into another vsetvli since we won't demand any fields.
    VSETVLIInfo NewInfo; // Need a new VSETVLIInfo to clear SEWLMULRatioOnly
    NewInfo.setAVLImm(1);
#if SIFIVE_CUSTOMIZATION
    NewInfo.setVTYPE(RISCVVType::LMUL_1, /*sew*/ 8, /*ta*/ true,
                     /*ma*/ true, /*AltFmt*/ false, /*W*/ 0);
#endif // SIFIVE_CUSTOMIZATION
    Info = NewInfo;
    return;
  }

  if (!RISCVII::hasSEWOp(MI.getDesc().TSFlags))
    return;

  DemandedFields Demanded = getDemanded(MI, ST);

  const VSETVLIInfo NewInfo = computeInfoForInstr(MI);
  assert(NewInfo.isValid() && !NewInfo.isUnknown());
  if (Info.isValid() && !needVSETVLI(Demanded, NewInfo, Info))
    return;

  const VSETVLIInfo PrevInfo = Info;
  if (!Info.isValid() || Info.isUnknown())
    Info = NewInfo;

  const VSETVLIInfo IncomingInfo = adjustIncoming(PrevInfo, NewInfo, Demanded);

  // If MI only demands that VL has the same zeroness, we only need to set the
  // AVL if the zeroness differs.  This removes a vsetvli entirely if the types
  // match or allows use of cheaper avl preserving variant if VLMAX doesn't
  // change. If VLMAX might change, we couldn't use the 'vsetvli x0, x0, vtype"
  // variant, so we avoid the transform to prevent extending live range of an
  // avl register operand.
  // TODO: We can probably relax this for immediates.
  bool EquallyZero = IncomingInfo.hasEquallyZeroAVL(PrevInfo, LIS) &&
                     IncomingInfo.hasSameVLMAX(PrevInfo);
  if (Demanded.VLAny || (Demanded.VLZeroness && !EquallyZero))
    Info.setAVL(IncomingInfo);

  Info.setVTYPE(
      ((Demanded.LMUL || Demanded.SEWLMULRatio) ? IncomingInfo : Info)
          .getVLMUL(),
      ((Demanded.SEW || Demanded.SEWLMULRatio) ? IncomingInfo : Info).getSEW(),
      // Prefer tail/mask agnostic since it can be relaxed to undisturbed later
      // if needed.
      (Demanded.TailPolicy ? IncomingInfo : Info).getTailAgnostic() ||
          IncomingInfo.getTailAgnostic(),
      (Demanded.MaskPolicy ? IncomingInfo : Info).getMaskAgnostic() ||
#if SIFIVE_CUSTOMIZATION
          IncomingInfo.getMaskAgnostic(),
      (Demanded.UseAltFmt ? IncomingInfo : Info).getAltFmt(),
      Demanded.UseTWiden ? IncomingInfo.getTWiden() : 0);
#endif // SIFIVE_CUSTOMIZATION

  // If we only knew the sew/lmul ratio previously, replace the VTYPE but keep
  // the AVL.
  if (Info.hasSEWLMULRatioOnly()) {
    VSETVLIInfo RatiolessInfo = IncomingInfo;
    RatiolessInfo.setAVL(Info);
    Info = RatiolessInfo;
  }
}

// Given a state with which we evaluated MI (see transferBefore above for why
// this might be different that the state MI requested), modify the state to
// reflect the changes MI might make.
void RISCVInsertVSETVLI::transferAfter(VSETVLIInfo &Info,
                                       const MachineInstr &MI) const {
  if (RISCVInstrInfo::isVectorConfigInstr(MI)) {
    Info = getInfoForVSETVLI(MI);
    return;
  }
#ifdef SIFIVE_CUSTOMIZATION
  // SETTM/TK will modify VTYPE, but it only affects the TM/TK bits.
  // It is safe for other RVV operations.
  // The TM/TK value will be maintained in insertVSETMTK.
  if (isMammothVectorConfigTMTKInstr(MI))
    return;
#endif // SIFIVE_CUSTOMIZATION

  if (RISCVInstrInfo::isFaultOnlyFirstLoad(MI)) {
    // Update AVL to vl-output of the fault first load.
    assert(MI.getOperand(1).getReg().isVirtual());
    if (LIS) {
      auto &LI = LIS->getInterval(MI.getOperand(1).getReg());
      SlotIndex SI =
          LIS->getSlotIndexes()->getInstructionIndex(MI).getRegSlot();
      VNInfo *VNI = LI.getVNInfoAt(SI);
      Info.setAVLRegDef(VNI, MI.getOperand(1).getReg());
    } else
      Info.setAVLRegDef(nullptr, MI.getOperand(1).getReg());
    return;
  }

  // If this is something that updates VL/VTYPE that we don't know about, set
  // the state to unknown.
  if (MI.isCall() || MI.isInlineAsm() ||
      MI.modifiesRegister(RISCV::VL, /*TRI=*/nullptr) ||
      MI.modifiesRegister(RISCV::VTYPE, /*TRI=*/nullptr))
    Info = VSETVLIInfo::getUnknown();
}

bool RISCVInsertVSETVLI::computeVLVTYPEChanges(const MachineBasicBlock &MBB,
                                               VSETVLIInfo &Info) const {
  bool HadVectorOp = false;

  Info = BlockInfo[MBB.getNumber()].Pred;
  for (const MachineInstr &MI : MBB) {
    transferBefore(Info, MI);

    if (RISCVInstrInfo::isVectorConfigInstr(MI) ||
        RISCVII::hasSEWOp(MI.getDesc().TSFlags) ||
#if SIFIVE_CUSTOMIZATION
        isVectorCopy(ST->getRegisterInfo(), MI) ||
        isMammothVectorConfigInstr(MI))
#endif // SIFIVE_CUSTOMIZATION
      HadVectorOp = true;

    transferAfter(Info, MI);
  }

  return HadVectorOp;
}

void RISCVInsertVSETVLI::computeIncomingVLVTYPE(const MachineBasicBlock &MBB) {

  BlockData &BBInfo = BlockInfo[MBB.getNumber()];

  BBInfo.InQueue = false;

  // Start with the previous entry so that we keep the most conservative state
  // we have ever found.
  VSETVLIInfo InInfo = BBInfo.Pred;
  if (MBB.pred_empty()) {
    // There are no predecessors, so use the default starting status.
    InInfo.setUnknown();
  } else {
    for (MachineBasicBlock *P : MBB.predecessors())
      InInfo = InInfo.intersect(BlockInfo[P->getNumber()].Exit);
  }

  // If we don't have any valid predecessor value, wait until we do.
  if (!InInfo.isValid())
    return;

  // If no change, no need to rerun block
  if (InInfo == BBInfo.Pred)
    return;

  BBInfo.Pred = InInfo;
  LLVM_DEBUG(dbgs() << "Entry state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Pred << "\n");

  // Note: It's tempting to cache the state changes here, but due to the
  // compatibility checks performed a blocks output state can change based on
  // the input state.  To cache, we'd have to add logic for finding
  // never-compatible state changes.
  VSETVLIInfo TmpStatus;
  computeVLVTYPEChanges(MBB, TmpStatus);

  // If the new exit value matches the old exit value, we don't need to revisit
  // any blocks.
  if (BBInfo.Exit == TmpStatus)
    return;

  BBInfo.Exit = TmpStatus;
  LLVM_DEBUG(dbgs() << "Exit state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Exit << "\n");

  // Add the successors to the work list so we can propagate the changed exit
  // status.
  for (MachineBasicBlock *S : MBB.successors())
    if (!BlockInfo[S->getNumber()].InQueue) {
      BlockInfo[S->getNumber()].InQueue = true;
      WorkList.push(S);
    }
}

// If we weren't able to prove a vsetvli was directly unneeded, it might still
// be unneeded if the AVL was a phi node where all incoming values are VL
// outputs from the last VSETVLI in their respective basic blocks.
bool RISCVInsertVSETVLI::needVSETVLIPHI(const VSETVLIInfo &Require,
                                        const MachineBasicBlock &MBB) const {
  if (!Require.hasAVLReg())
    return true;

  if (!LIS)
    return true;

  // We need the AVL to have been produced by a PHI node in this basic block.
  const VNInfo *Valno = Require.getAVLVNInfo();
  if (!Valno->isPHIDef() || LIS->getMBBFromIndex(Valno->def) != &MBB)
    return true;

  const LiveRange &LR = LIS->getInterval(Require.getAVLReg());

  for (auto *PBB : MBB.predecessors()) {
    const VSETVLIInfo &PBBExit = BlockInfo[PBB->getNumber()].Exit;

    // We need the PHI input to the be the output of a VSET(I)VLI.
    const VNInfo *Value = LR.getVNInfoBefore(LIS->getMBBEndIdx(PBB));
    if (!Value)
      return true;
    MachineInstr *DefMI = LIS->getInstructionFromIndex(Value->def);
    if (!DefMI || !RISCVInstrInfo::isVectorConfigInstr(*DefMI))
      return true;

    // We found a VSET(I)VLI make sure it matches the output of the
    // predecessor block.
    VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
    if (DefInfo != PBBExit)
      return true;

    // Require has the same VL as PBBExit, so if the exit from the
    // predecessor has the VTYPE we are looking for we might be able
    // to avoid a VSETVLI.
    if (PBBExit.isUnknown() || !PBBExit.hasSameVTYPE(Require))
      return true;
  }

  // If all the incoming values to the PHI checked out, we don't need
  // to insert a VSETVLI.
  return false;
}

void RISCVInsertVSETVLI::emitVSETVLIs(MachineBasicBlock &MBB) {
  VSETVLIInfo CurInfo = BlockInfo[MBB.getNumber()].Pred;
  // Track whether the prefix of the block we've scanned is transparent
  // (meaning has not yet changed the abstract state).
  bool PrefixTransparent = true;
  for (MachineInstr &MI : MBB) {
    const VSETVLIInfo PrevInfo = CurInfo;
    transferBefore(CurInfo, MI);
    // If this is an explicit VSETVLI or VSETIVLI, update our state.
    if (RISCVInstrInfo::isVectorConfigInstr(MI)) {
      // Conservatively, mark the VL and VTYPE as live.
      assert(MI.getOperand(3).getReg() == RISCV::VL &&
             MI.getOperand(4).getReg() == RISCV::VTYPE &&
             "Unexpected operands where VL and VTYPE should be");
      MI.getOperand(3).setIsDead(false);
      MI.getOperand(4).setIsDead(false);
      PrefixTransparent = false;
    }

    if (EnsureWholeVectorRegisterMoveValidVTYPE &&
        isVectorCopy(ST->getRegisterInfo(), MI)) {
      if (!PrevInfo.isCompatible(DemandedFields::all(), CurInfo, LIS)) {
        insertVSETVLI(MBB, MI, MI.getDebugLoc(), CurInfo, PrevInfo);
        PrefixTransparent = false;
      }
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ false,
                                              /*isImp*/ true));
    }

    uint64_t TSFlags = MI.getDesc().TSFlags;
    if (RISCVII::hasSEWOp(TSFlags)) {
      if (!PrevInfo.isCompatible(DemandedFields::all(), CurInfo, LIS)) {
        // If this is the first implicit state change, and the state change
        // requested can be proven to produce the same register contents, we
        // can skip emitting the actual state change and continue as if we
        // had since we know the GPR result of the implicit state change
        // wouldn't be used and VL/VTYPE registers are correct.  Note that
        // we *do* need to model the state as if it changed as while the
        // register contents are unchanged, the abstract model can change.
        if (!PrefixTransparent || needVSETVLIPHI(CurInfo, MBB))
          insertVSETVLI(MBB, MI, MI.getDebugLoc(), CurInfo, PrevInfo);
        PrefixTransparent = false;
      }

      if (RISCVII::hasVLOp(TSFlags)) {
        MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
        if (VLOp.isReg()) {
          Register Reg = VLOp.getReg();

          // Erase the AVL operand from the instruction.
          VLOp.setReg(RISCV::NoRegister);
          VLOp.setIsKill(false);
          if (LIS) {
            LiveInterval &LI = LIS->getInterval(Reg);
            SmallVector<MachineInstr *> DeadMIs;
            LIS->shrinkToUses(&LI, &DeadMIs);
            // We might have separate components that need split due to
            // needVSETVLIPHI causing us to skip inserting a new VL def.
            SmallVector<LiveInterval *> SplitLIs;
            LIS->splitSeparateComponents(LI, SplitLIs);

            // If the AVL was an immediate > 31, then it would have been emitted
            // as an ADDI. However, the ADDI might not have been used in the
            // vsetvli, or a vsetvli might not have been emitted, so it may be
            // dead now.
            for (MachineInstr *DeadMI : DeadMIs) {
              if (!TII->isAddImmediate(*DeadMI, Reg))
                continue;
              LIS->RemoveMachineInstrFromMaps(*DeadMI);
              DeadMI->eraseFromParent();
            }
          }
        }
        MI.addOperand(MachineOperand::CreateReg(RISCV::VL, /*isDef*/ false,
                                                /*isImp*/ true));
      }
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ false,
                                              /*isImp*/ true));
    }

    if (MI.isInlineAsm()) {
      MI.addOperand(MachineOperand::CreateReg(RISCV::VL, /*isDef*/ true,
                                              /*isImp*/ true));
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ true,
                                              /*isImp*/ true));
    }

    if (MI.isCall() || MI.isInlineAsm() ||
        MI.modifiesRegister(RISCV::VL, /*TRI=*/nullptr) ||
        MI.modifiesRegister(RISCV::VTYPE, /*TRI=*/nullptr))
      PrefixTransparent = false;

    transferAfter(CurInfo, MI);
  }

  const auto &Info = BlockInfo[MBB.getNumber()];
  if (CurInfo != Info.Exit) {
    LLVM_DEBUG(dbgs() << "in block " << printMBBReference(MBB) << "\n");
    LLVM_DEBUG(dbgs() << "  begin        state: " << Info.Pred << "\n");
    LLVM_DEBUG(dbgs() << "  expected end state: " << Info.Exit << "\n");
    LLVM_DEBUG(dbgs() << "  actual   end state: " << CurInfo << "\n");
  }
  assert(CurInfo == Info.Exit && "InsertVSETVLI dataflow invariant violated");
}

/// Perform simple partial redundancy elimination of the VSETVLI instructions
/// we're about to insert by looking for cases where we can PRE from the
/// beginning of one block to the end of one of its predecessors.  Specifically,
/// this is geared to catch the common case of a fixed length vsetvl in a single
/// block loop when it could execute once in the preheader instead.
void RISCVInsertVSETVLI::doPRE(MachineBasicBlock &MBB) {
  if (!BlockInfo[MBB.getNumber()].Pred.isUnknown())
    return;

  MachineBasicBlock *UnavailablePred = nullptr;
  VSETVLIInfo AvailableInfo;
  for (MachineBasicBlock *P : MBB.predecessors()) {
    const VSETVLIInfo &PredInfo = BlockInfo[P->getNumber()].Exit;
    if (PredInfo.isUnknown()) {
      if (UnavailablePred)
        return;
      UnavailablePred = P;
    } else if (!AvailableInfo.isValid()) {
      AvailableInfo = PredInfo;
    } else if (AvailableInfo != PredInfo) {
      return;
    }
  }

  // Unreachable, single pred, or full redundancy. Note that FRE is handled by
  // phase 3.
  if (!UnavailablePred || !AvailableInfo.isValid())
    return;

  if (!LIS)
    return;

  // If we don't know the exact VTYPE, we can't copy the vsetvli to the exit of
  // the unavailable pred.
  if (AvailableInfo.hasSEWLMULRatioOnly())
    return;

  // Critical edge - TODO: consider splitting?
  if (UnavailablePred->succ_size() != 1)
    return;

  // If the AVL value is a register (other than our VLMAX sentinel),
  // we need to prove the value is available at the point we're going
  // to insert the vsetvli at.
  if (AvailableInfo.hasAVLReg()) {
    SlotIndex SI = AvailableInfo.getAVLVNInfo()->def;
    // This is an inline dominance check which covers the case of
    // UnavailablePred being the preheader of a loop.
    if (LIS->getMBBFromIndex(SI) != UnavailablePred)
      return;
    if (!UnavailablePred->terminators().empty() &&
        SI >= LIS->getInstructionIndex(*UnavailablePred->getFirstTerminator()))
      return;
  }

  // Model the effect of changing the input state of the block MBB to
  // AvailableInfo.  We're looking for two issues here; one legality,
  // one profitability.
  // 1) If the block doesn't use some of the fields from VL or VTYPE, we
  //    may hit the end of the block with a different end state.  We can
  //    not make this change without reflowing later blocks as well.
  // 2) If we don't actually remove a transition, inserting a vsetvli
  //    into the predecessor block would be correct, but unprofitable.
  VSETVLIInfo OldInfo = BlockInfo[MBB.getNumber()].Pred;
  VSETVLIInfo CurInfo = AvailableInfo;
  int TransitionsRemoved = 0;
  for (const MachineInstr &MI : MBB) {
    const VSETVLIInfo LastInfo = CurInfo;
    const VSETVLIInfo LastOldInfo = OldInfo;
    transferBefore(CurInfo, MI);
    transferBefore(OldInfo, MI);
    if (CurInfo == LastInfo)
      TransitionsRemoved++;
    if (LastOldInfo == OldInfo)
      TransitionsRemoved--;
    transferAfter(CurInfo, MI);
    transferAfter(OldInfo, MI);
    if (CurInfo == OldInfo)
      // Convergence.  All transitions after this must match by construction.
      break;
  }
  if (CurInfo != OldInfo || TransitionsRemoved <= 0)
    // Issues 1 and 2 above
    return;

  // Finally, update both data flow state and insert the actual vsetvli.
  // Doing both keeps the code in sync with the dataflow results, which
  // is critical for correctness of phase 3.
  auto OldExit = BlockInfo[UnavailablePred->getNumber()].Exit;
  LLVM_DEBUG(dbgs() << "PRE VSETVLI from " << MBB.getName() << " to "
                    << UnavailablePred->getName() << " with state "
                    << AvailableInfo << "\n");
  BlockInfo[UnavailablePred->getNumber()].Exit = AvailableInfo;
  BlockInfo[MBB.getNumber()].Pred = AvailableInfo;

  // Note there's an implicit assumption here that terminators never use
  // or modify VL or VTYPE.  Also, fallthrough will return end().
  auto InsertPt = UnavailablePred->getFirstInstrTerminator();
  insertVSETVLI(*UnavailablePred, InsertPt,
                UnavailablePred->findDebugLoc(InsertPt),
                AvailableInfo, OldExit);
}

// Return true if we can mutate PrevMI to match MI without changing any the
// fields which would be observed.
bool RISCVInsertVSETVLI::canMutatePriorConfig(
    const MachineInstr &PrevMI, const MachineInstr &MI,
    const DemandedFields &Used) const {
  // If the VL values aren't equal, return false if either a) the former is
  // demanded, or b) we can't rewrite the former to be the later for
  // implementation reasons.
  if (!RISCVInstrInfo::isVLPreservingConfig(MI)) {
    if (Used.VLAny)
      return false;

    if (Used.VLZeroness) {
      if (RISCVInstrInfo::isVLPreservingConfig(PrevMI))
        return false;
      if (!getInfoForVSETVLI(PrevMI).hasEquallyZeroAVL(getInfoForVSETVLI(MI),
                                                       LIS))
        return false;
    }

    auto &AVL = MI.getOperand(1);

    // If the AVL is a register, we need to make sure its definition is the same
    // at PrevMI as it was at MI.
    if (AVL.isReg() && AVL.getReg() != RISCV::X0) {
      VNInfo *VNI = getVNInfoFromReg(AVL.getReg(), MI, LIS);
      VNInfo *PrevVNI = getVNInfoFromReg(AVL.getReg(), PrevMI, LIS);
      if (!VNI || !PrevVNI || VNI != PrevVNI)
        return false;
    }
  }

  assert(PrevMI.getOperand(2).isImm() && MI.getOperand(2).isImm());
  auto PriorVType = PrevMI.getOperand(2).getImm();
  auto VType = MI.getOperand(2).getImm();
  return areCompatibleVTYPEs(PriorVType, VType, Used);
}

#ifdef SIFIVE_CUSTOMIZATION
// When twiden != 0, LMUL, tail policy, and mask policy from the user are
// ignored. The tail policy and mask policy are always treated as agnostic. The
// normal RVV instruction will ignore the twiden parameter. This observation
// could allow the RVV instruction and mommath instruction to share the same
// configuration instruction.
//
// We need to make sure the AVL, SEW, and AltFmt is same between VSETVL and
// VSETVLTN.
//
// For example:
//
// %avl = SETTM or SETTK
// ...
// VSETVL %avl, type1
// VSETVLTN %avl, type2
//
// ->
//
// %avl = SETTM or SETTK
// ...
// VSETVLTN %avl, type2
//
bool RISCVInsertVSETVLI::canMutatePriorConfigWithTWiden(
    const MachineInstr &PrevMI, const MachineInstr &MI) const {

  if (PrevMI.getOpcode() != RISCV::PseudoVSETVLI)
    return false;

  if (MI.getOpcode() != RISCV::PseudoSF_VSETTNT)
    return false;

  auto PrevInfo = getInfoForVSETVLI(PrevMI);
  auto CurrInfo = getInfoForVSETVLI(MI);

  auto AVLReg = CurrInfo.getAVLReg();

  auto *AVLRegDefMI = MRI->getUniqueVRegDef(AVLReg);

  if (!AVLRegDefMI)
    return false;

  if (!isMammothVectorConfigTMTKInstr(*AVLRegDefMI))
    return false;

  auto AVLRegDefMIInfo = computeInfoForInstr(*AVLRegDefMI);
  if (AVLRegDefMIInfo.getTWiden() != CurrInfo.getTWiden())
    return false;

  if (AVLRegDefMIInfo.getSEW() != PrevInfo.getSEW())
    return false;

  // CurrInfo twiden != 0, so TailAgnostic and MaskAgnostic bit default to 1
  if (!PrevInfo.getTailAgnostic() || !PrevInfo.getMaskAgnostic())
    return false;

  if (!PrevInfo.hasSameAVL(CurrInfo))
    return false;

  if (PrevInfo.getSEW() != CurrInfo.getSEW())
    return false;

  if (PrevInfo.getAltFmt() != CurrInfo.getAltFmt())
    return false;

  return true;
}

void RISCVInsertVSETVLI::coalesceVSETVLIsForTWiden(
    MachineBasicBlock &MBB) const {
  MachineInstr *NextMI = nullptr;

  for (MachineInstr &MI : make_early_inc_range(reverse(MBB))) {

    if (!RISCVInstrInfo::isVectorConfigInstr(MI))
      continue;

    if (NextMI) {
      // If only TWiden different. Update the MI and drop the NextMI.
      if (canMutatePriorConfigWithTWiden(MI, *NextMI)) {

        auto NextInfo = getInfoForVSETVLI(*NextMI);
        MI.getOperand(2).setImm(NextInfo.encodeVTYPE());

        if (LIS)
          LIS->RemoveMachineInstrFromMaps(*NextMI);
        NextMI->eraseFromParent();
      }
    }
    NextMI = &MI;
  }
}
#endif // SIFIVE_CUSTOMIZATION

void RISCVInsertVSETVLI::coalesceVSETVLIs(MachineBasicBlock &MBB) const {
  MachineInstr *NextMI = nullptr;
  // We can have arbitrary code in successors, so VL and VTYPE
  // must be considered demanded.
  DemandedFields Used;
  Used.demandVL();
  Used.demandVTYPE();
  SmallVector<MachineInstr*> ToDelete;

  auto dropAVLUse = [&](MachineOperand &MO) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      return;
    Register OldVLReg = MO.getReg();
    MO.setReg(RISCV::NoRegister);

    if (LIS)
      LIS->shrinkToUses(&LIS->getInterval(OldVLReg));

    MachineInstr *VLOpDef = MRI->getUniqueVRegDef(OldVLReg);
    if (VLOpDef && TII->isAddImmediate(*VLOpDef, OldVLReg) &&
        MRI->use_nodbg_empty(OldVLReg))
      ToDelete.push_back(VLOpDef);
  };

  for (MachineInstr &MI : make_early_inc_range(reverse(MBB))) {

    if (!RISCVInstrInfo::isVectorConfigInstr(MI)) {
      Used.doUnion(getDemanded(MI, ST));
      if (MI.isCall() || MI.isInlineAsm() ||
          MI.modifiesRegister(RISCV::VL, /*TRI=*/nullptr) ||
          MI.modifiesRegister(RISCV::VTYPE, /*TRI=*/nullptr))
        NextMI = nullptr;
      continue;
    }

    if (!MI.getOperand(0).isDead())
      Used.demandVL();

    if (NextMI) {
      if (!Used.usedVL() && !Used.usedVTYPE()) {
        dropAVLUse(MI.getOperand(1));
        if (LIS)
          LIS->RemoveMachineInstrFromMaps(MI);
        MI.eraseFromParent();
        NumCoalescedVSETVL++;
        // Leave NextMI unchanged
        continue;
      }

      if (canMutatePriorConfig(MI, *NextMI, Used)) {
        if (!RISCVInstrInfo::isVLPreservingConfig(*NextMI)) {
          Register DefReg = NextMI->getOperand(0).getReg();

          MI.getOperand(0).setReg(DefReg);
          MI.getOperand(0).setIsDead(false);

          // Move the AVL from NextMI to MI
          dropAVLUse(MI.getOperand(1));
          if (NextMI->getOperand(1).isImm())
            MI.getOperand(1).ChangeToImmediate(NextMI->getOperand(1).getImm());
          else
            MI.getOperand(1).ChangeToRegister(NextMI->getOperand(1).getReg(),
                                              false);
          dropAVLUse(NextMI->getOperand(1));

          // The def of DefReg moved to MI, so extend the LiveInterval up to
          // it.
          if (DefReg.isVirtual() && LIS) {
            LiveInterval &DefLI = LIS->getInterval(DefReg);
            SlotIndex MISlot = LIS->getInstructionIndex(MI).getRegSlot();
            SlotIndex NextMISlot =
                LIS->getInstructionIndex(*NextMI).getRegSlot();
            VNInfo *DefVNI = DefLI.getVNInfoAt(NextMISlot);
            LiveInterval::Segment S(MISlot, NextMISlot, DefVNI);
            DefLI.addSegment(S);
            DefVNI->def = MISlot;
            // Mark DefLI as spillable if it was previously unspillable
            DefLI.setWeight(0);

            // DefReg may have had no uses, in which case we need to shrink
            // the LiveInterval up to MI.
            LIS->shrinkToUses(&DefLI);
          }

          MI.setDesc(NextMI->getDesc());
        }
        MI.getOperand(2).setImm(NextMI->getOperand(2).getImm());

        dropAVLUse(NextMI->getOperand(1));
        if (LIS)
          LIS->RemoveMachineInstrFromMaps(*NextMI);
        NextMI->eraseFromParent();
        NumCoalescedVSETVL++;
        // fallthrough
      }
    }
    NextMI = &MI;
    Used = getDemanded(MI, ST);
  }

  // Loop over the dead AVL values, and delete them now.  This has
  // to be outside the above loop to avoid invalidating iterators.
  for (auto *MI : ToDelete) {
    if (LIS) {
      LIS->removeInterval(MI->getOperand(0).getReg());
      LIS->RemoveMachineInstrFromMaps(*MI);
    }
    MI->eraseFromParent();
  }
}

void RISCVInsertVSETVLI::insertReadVL(MachineBasicBlock &MBB) {
  for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr &MI = *I++;
    if (RISCVInstrInfo::isFaultOnlyFirstLoad(MI)) {
      Register VLOutput = MI.getOperand(1).getReg();
      assert(VLOutput.isVirtual());
      if (!MI.getOperand(1).isDead()) {
        auto ReadVLMI = BuildMI(MBB, I, MI.getDebugLoc(),
                                TII->get(RISCV::PseudoReadVL), VLOutput);
        // Move the LiveInterval's definition down to PseudoReadVL.
        if (LIS) {
          SlotIndex NewDefSI =
              LIS->InsertMachineInstrInMaps(*ReadVLMI).getRegSlot();
          LiveInterval &DefLI = LIS->getInterval(VLOutput);
          VNInfo *DefVNI = DefLI.getVNInfoAt(DefLI.beginIndex());
          DefLI.removeSegment(DefLI.beginIndex(), NewDefSI);
          DefVNI->def = NewDefSI;
        }
      }
      // We don't use the vl output of the VLEFF/VLSEGFF anymore.
      MI.getOperand(1).setReg(RISCV::X0);
      MI.addRegisterDefined(RISCV::VL, MRI->getTargetRegisterInfo());
    }
  }
}

#ifdef SIFIVE_CUSTOMIZATION
static void shrinkIntervalAndRemoveDeadMI(MachineOperand &MO,
                                          LiveIntervals *LIS) {
  Register Reg = MO.getReg();
  MO.setReg(RISCV::NoRegister);
  MO.setIsKill(false);

  if (!LIS)
    return;

  LiveInterval &LI = LIS->getInterval(Reg);

  // Erase the AVL operand from the instruction.
  SmallVector<MachineInstr *> DeadMIs;
  LIS->shrinkToUses(&LI, &DeadMIs);
  // TODO: Enable this once needVSETVLIPHI is supported.
  // SmallVector<LiveInterval *> SplitLIs;
  // LIS->splitSeparateComponents(LI, SplitLIs);

  for (MachineInstr *DeadMI : DeadMIs) {
    LIS->RemoveMachineInstrFromMaps(*DeadMI);
    DeadMI->eraseFromParent();
  }
}

bool RISCVInsertVSETVLI::insertVSETMTK(MachineBasicBlock &MBB,
                                       TKTMMode Mode) const {

  bool Changed = false;
  VSETVLIInfo PreInfo = VSETVLIInfo::getUnknown();
  for (auto &MI : MBB) {

    if (RISCVInstrInfo::isVectorConfigInstr(MI)) {
      PreInfo = VSETVLIInfo::getUnknown();
      continue;
    }

    uint64_t TSFlags = MI.getDesc().TSFlags;
    if (isMammothVectorConfigTMTKInstr(MI) || !RISCVII::hasSEWOp(TSFlags) ||
        !RISCVII::hasTWidenOp(TSFlags))
      continue;

    if (Mode == VSETTK && !RISCVII::hasTKOp(TSFlags))
      continue;

    if (Mode == VSETTM && !RISCVII::hasTMOp(TSFlags))
      continue;

    unsigned OpNum = 0;
    unsigned Opcode = 0;
    switch (Mode) {
    case VSETTK:
      OpNum = RISCVII::getTKOpNum(MI.getDesc());
      Opcode = RISCV::PseudoSF_VSETTK;
      break;
    case VSETTM:
      OpNum = RISCVII::getTMOpNum(MI.getDesc());
      Opcode = RISCV::PseudoSF_VSETTM;
      break;
    }

    assert(OpNum && Opcode && "Invalid OpNum or Opcode");

    const MachineOperand &Op = MI.getOperand(OpNum);

    VSETVLIInfo CurrInfo = computeInfoForInstr(MI);

    CurrInfo.setAVLRegDef(getVNInfoFromReg(Op.getReg(), MI, LIS), Op.getReg());

    if (!PreInfo.isCompatible(DemandedFields::all(), CurrInfo, LIS)) {
      auto TmpMI = BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(Opcode))
                       .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                       .addReg(Op.getReg())
                       .addImm(Log2_32(CurrInfo.getSEW()))
                       .addImm((CurrInfo.getTWiden() >> 1) + 1);

      Changed = true;
      if (LIS)
        LIS->InsertMachineInstrInMaps(*TmpMI);
    }

    shrinkIntervalAndRemoveDeadMI(MI.getOperand(OpNum), LIS);

    PreInfo = CurrInfo;
  }
  return Changed;
}
#endif // SIFIVE_CUSTOMIZATION

bool RISCVInsertVSETVLI::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  ST = &MF.getSubtarget<RISCVSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  LLVM_DEBUG(dbgs() << "Entering InsertVSETVLI for " << MF.getName() << "\n");

  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  auto *LISWrapper = getAnalysisIfAvailable<LiveIntervalsWrapperPass>();
  LIS = LISWrapper ? &LISWrapper->getLIS() : nullptr;

  assert(BlockInfo.empty() && "Expect empty block infos");
  BlockInfo.resize(MF.getNumBlockIDs());

  bool HaveVectorOp = false;

  // Phase 1 - determine how VL/VTYPE are affected by the each block.
  for (const MachineBasicBlock &MBB : MF) {
    VSETVLIInfo TmpStatus;
    HaveVectorOp |= computeVLVTYPEChanges(MBB, TmpStatus);
    // Initial exit state is whatever change we found in the block.
    BlockData &BBInfo = BlockInfo[MBB.getNumber()];
    BBInfo.Exit = TmpStatus;
    LLVM_DEBUG(dbgs() << "Initial exit state of " << printMBBReference(MBB)
                      << " is " << BBInfo.Exit << "\n");

  }

  // If we didn't find any instructions that need VSETVLI, we're done.
  if (!HaveVectorOp) {
    BlockInfo.clear();
    return false;
  }

  // Phase 2 - determine the exit VL/VTYPE from each block. We add all
  // blocks to the list here, but will also add any that need to be revisited
  // during Phase 2 processing.
  for (const MachineBasicBlock &MBB : MF) {
    WorkList.push(&MBB);
    BlockInfo[MBB.getNumber()].InQueue = true;
  }
  while (!WorkList.empty()) {
    const MachineBasicBlock &MBB = *WorkList.front();
    WorkList.pop();
    computeIncomingVLVTYPE(MBB);
  }

  // Perform partial redundancy elimination of vsetvli transitions.
  for (MachineBasicBlock &MBB : MF)
    doPRE(MBB);

  // Phase 3 - add any vsetvli instructions needed in the block. Use the
  // Phase 2 information to avoid adding vsetvlis before the first vector
  // instruction in the block if the VL/VTYPE is satisfied by its
  // predecessors.
  for (MachineBasicBlock &MBB : MF)
    emitVSETVLIs(MBB);

  // Now that all vsetvlis are explicit, go through and do block local
  // DSE and peephole based demanded fields based transforms.  Note that
  // this *must* be done outside the main dataflow so long as we allow
  // any cross block analysis within the dataflow.  We can't have both
  // demanded fields based mutation and non-local analysis in the
  // dataflow at the same time without introducing inconsistencies.
  // We're visiting blocks from the bottom up because a VSETVLI in the
  // earlier block might become dead when its uses in later blocks are
  // optimized away.
  for (MachineBasicBlock *MBB : post_order(&MF))
    coalesceVSETVLIs(*MBB);

#ifdef SIFIVE_CUSTOMIZATION
  for (MachineBasicBlock &MBB : MF)
    coalesceVSETVLIsForTWiden(MBB);
#endif // SIFIVE_CUSTOMIZATION

  // Insert PseudoReadVL after VLEFF/VLSEGFF and replace it with the vl output
  // of VLEFF/VLSEGFF.
  for (MachineBasicBlock &MBB : MF)
    insertReadVL(MBB);

  for (MachineBasicBlock &MBB : MF)
    insertVSETMTK(MBB, VSETTM);

  for (MachineBasicBlock &MBB : MF)
    insertVSETMTK(MBB, VSETTK);

  BlockInfo.clear();
  return HaveVectorOp;
}

/// Returns an instance of the Insert VSETVLI pass.
FunctionPass *llvm::createRISCVInsertVSETVLIPass() {
  return new RISCVInsertVSETVLI();
}
