//=- RISCVMachineConstProp.cpp - Machine constant propagation for RISCV ----=//
//
// This pass tries:
// 1. Fold the register with immediate value produced by ADDI X0, imm into
//    its user's immediate version instruction, e.g. XOR -> XORI. For instance:
//  BB:
//    %a0 = ADDI %x0, 1
//    %a1 = XOR %a1, %a0
//  this can be reduced to:
//  BB:
//    %a1 = XORI %a1, 1
//
// 2. Fold the register with immediate value produced by ADDI X0, imm into
//    user with another immediate operation. For instance:
//  BB:
//    %a0 = ADDI %x0, 1
//    %a1 = XORI %a0, 2
//  this can be reduced to:
//  BB:
//    %a1 = ADDI %x0, 3
//
// 3. Fold the register move instruction produced by ADD X0, %reg or
//    ADDI %reg, 0 into its user. For instance:
//  BB:
//    %a0 = ADD %x0, %x3
//    %a1 = XOR %a1, %a0
//  this can be reduced to:
//  BB:
//    %a1 = XOR %a1, %x3
//
// Its usage includes the machine IR after tail duplication or block placement
// due to the duplication and gluing some instructions to some blocks.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-mc-constprop"

namespace {
class RISCVMachineConstPropagation : public MachineFunctionPass {
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;

public:
  static char ID;
  RISCVMachineConstPropagation() : MachineFunctionPass(ID) {
    initializeRISCVMachineConstPropagationPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  StringRef getPassName() const override {
    return "RISCV Machine Constant Propagation";
  }

private:
  bool optimizeBlock(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVMachineConstPropagation::ID = 0;

INITIALIZE_PASS(RISCVMachineConstPropagation, DEBUG_TYPE,
                "RISCV Machine Constant Propagation", false, false)

// This function recognizes the %x = LI imm pattern which is composed by
// %x = ADDI x0, imm
static bool hasLiPattern(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::ADDI && MI.getOperand(1).isReg() &&
         MI.getOperand(2).isImm() && MI.getOperand(1).getReg() == RISCV::X0;
}

// This function recognizes the %x = MOV %reg pattern which is composed by
// %x = ADD x0, %reg or %x = ADDI %reg, 0
static bool hasMovPattern(const MachineInstr &MI) {
  return (MI.getOpcode() == RISCV::ADD && MI.getOperand(1).isReg() &&
          MI.getOperand(2).isReg() &&
          (MI.getOperand(1).getReg() == RISCV::X0 ||
           MI.getOperand(2).getReg() == RISCV::X0)) ||
         (MI.getOpcode() == RISCV::ADDI && MI.getOperand(1).isReg() &&
          MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == 0);
}

static bool tryFoldBinOp(const TargetInstrInfo *TII, MachineInstr &Root) {
  if (Root.isCall())
    return false;

  MachineBasicBlock &MBB = *Root.getParent();

  auto MatchBinOp = [&](int &Pos, MachineBasicBlock::iterator &Inst) -> bool {
    if (Root.getNumOperands() != 3)
      return false;

    for (int i = 1; i < 3; ++i) {
      MachineOperand &MO = Root.getOperand(i);
      if (MO.isReg() && MO.isKill() &&
          Register::isPhysicalRegister(MO.getReg())) {
        MachineBasicBlock::reverse_iterator RevInst = Root.getReverseIterator();
        DenseSet<Register> DefRegs;
        for (MachineInstr &MI : make_range(std::next(RevInst), MBB.rend())) {
          if (!MI.getNumOperands())
            continue;

          if (MI.isCall())
            break;

          if (MI.getOperand(0).isReg() &&
              MI.getOperand(0).getReg() == MO.getReg()) {
            if (!hasLiPattern(MI) && !hasMovPattern(MI))
              break;

            if (any_of(MI.operands(), [&DefRegs](const MachineOperand &Op) {
                  return Op.isReg() && Op.getReg() != RISCV::X0 &&
                         DefRegs.contains(Op.getReg());
                }))
              break;

            Inst = MI;
            Pos = i;
            return true;
          }

          // Skip the operand if there is any use or def between these
          // two instructions to be combined that has the same register
          // as operand's, e.g.
          // BB:
          //   %a0 = ADDI %x0, 1    <- The %a0 we want
          //   %a3 = ADDI %a0, 1    <- %a0 is first used here
          //   %a1 = XOR %a1, %a0   <- Although we found %a0 which is of
          //                           the pattern we want, we can't do
          //                           the folding.
          if (any_of(MI.operands(), [&MO](const MachineOperand &Op) {
                return Op.isReg() && Op.getReg() == MO.getReg();
              }))
            break;

          if (MI.getOperand(0).isReg())
            DefRegs.insert(MI.getOperand(0).getReg());
        }
        DefRegs.clear();
      }
    }
    return false;
  };

  int Pos;
  MachineBasicBlock::iterator Inst;

  if (!MatchBinOp(Pos, Inst))
    return false;

  int OtherPos = Pos == 1 ? 2 : 1;
  if (hasLiPattern(*Inst)) {
    if (Root.getOperand(2).isImm()) {
      // Second case
      int64_t ImmValue;
      int64_t V1 = Root.getOperand(2).getImm();
      int64_t V2 = Inst->getOperand(2).getImm();
      switch (Root.getOpcode()) {
      default:
        return false;
      case RISCV::ANDI:
        ImmValue = V1 & V2;
        break;
      case RISCV::ORI:
        ImmValue = V1 | V2;
        break;
      case RISCV::XORI:
        ImmValue = V1 ^ V2;
        break;
      case RISCV::ADDI:
        ImmValue = (uint64_t)V1 + (uint64_t)V2;
        break;
      }

      if (!isInt<12>(ImmValue))
        return false;

      BuildMI(MBB, Root, Root.getDebugLoc(), TII->get(RISCV::ADDI))
          .add(Root.getOperand(0))
          .addReg(RISCV::X0)
          .addImm(ImmValue);
    } else {
      // First case
      unsigned BinImmOpcode;
      switch (Root.getOpcode()) {
      default:
        return false;
      case RISCV::AND:
        BinImmOpcode = RISCV::ANDI;
        break;
      case RISCV::OR:
        BinImmOpcode = RISCV::ORI;
        break;
      case RISCV::XOR:
        BinImmOpcode = RISCV::XORI;
        break;
      case RISCV::ADD:
        BinImmOpcode = RISCV::ADDI;
        break;
      case RISCV::ADDW:
        BinImmOpcode = RISCV::ADDIW;
        break;
      }

      BuildMI(MBB, Root, Root.getDebugLoc(), TII->get(BinImmOpcode))
          .add(Root.getOperand(0))
          .add(Root.getOperand(OtherPos))
          .addImm(Inst->getOperand(2).getImm());
    }
  } else {
    // Third case
    const MachineOperand &ToBePropagated = Inst->getOperand(
        Inst->getOperand(2).isImm() || Inst->getOperand(2).getReg() == RISCV::X0
            ? 1
            : 2);

    MachineInstr *NewInstr;
    if (Pos == 1)
      NewInstr =
          BuildMI(MBB, Root, Root.getDebugLoc(), TII->get(Root.getOpcode()))
              .add(Root.getOperand(0))
              .add(ToBePropagated)
              .add(Root.getOperand(OtherPos))
              .getInstr();
    else
      NewInstr =
          BuildMI(MBB, Root, Root.getDebugLoc(), TII->get(Root.getOpcode()))
              .add(Root.getOperand(0))
              .add(Root.getOperand(OtherPos))
              .add(ToBePropagated)
              .getInstr();

    // Postpone the kill flag of the register in Inst to the Root
    // e.g. %0 = ADDI %x2, 0
    //      %1 = DIV %x1, kill %x2
    //      %2 = XOR %x3, %0
    //      can be reduced to:
    //      %1 = DIV %x1, kill %x2 <- the kill flag needs to be moved to %x2 in next inst.
    //      %2 = XOR %x3, %x2
    //      finally:
    //      %1 = DIV %x1, %x2
    //      %2 = XOR %x3, kill %x2

    for (MachineInstr &MI :
         make_range(Root.getReverseIterator(), Inst->getReverseIterator()))
      for (auto &Op : MI.operands()) {
        MachineOperand &PropagatedOp = NewInstr->getOperand(Pos);
        if (Op.isReg() && Op.getReg() == PropagatedOp.getReg() && Op.isKill() &&
            !PropagatedOp.isDef()) {
          Op.setIsKill(false);
          PropagatedOp.setIsKill(true);
        }
      }
  }

  Inst->eraseFromParent();
  Root.eraseFromParent();
  return true;
}

bool RISCVMachineConstPropagation::optimizeBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;) {
    bool Folded = tryFoldBinOp(TII, *(I++));
    if (Folded)
      I--;

    Changed |= Folded;
  }

  return Changed;
}

bool RISCVMachineConstPropagation::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeBlock(MBB);

  return Changed;
}

FunctionPass *llvm::createRISCVMachineConstPropagationPass() {
  return new RISCVMachineConstPropagation();
}
