//===----- SiFive_RISCVTypePromotion.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This is an opcode based type promotion pass for i32 types that would
/// otherwise be promoted during legalisation for i64. This works around the
/// limitations of SelectionDAG for cyclic regions. The search begins from icmp
/// instruction operands where a tree is built, checked and promoted if
/// possible.
///
/// This is based on the target independent TypePromotion pass, but customized
/// for RISC-V which is more interested in sext.
///
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "riscv-type-promotion"
#define PASS_NAME "RISC-V Type Promotion"

using namespace llvm;

static cl::opt<bool>
    DisablePromotion("riscv-disable-type-promotion", cl::Hidden,
                     cl::init(false),
                     cl::desc("Disable RISC-V type promotion pass"));

namespace {
class IRPromoter {
  IntegerType *OrigTy;
  unsigned PromotedWidth;
  const SetVector<Value *> &Visited;
  const SetVector<Value *> &Sources;
  const SetVector<Instruction *> &Sinks;
  IntegerType *ExtTy;
  SmallPtrSet<Value *, 8> NewInsts;
  SmallPtrSet<Instruction *, 4> InstsToRemove;
  DenseMap<Value *, SmallVector<Type *, 4>> TruncTysMap;
  SmallPtrSet<Value *, 8> Promoted;

public:
  IRPromoter(IntegerType *Ty, unsigned Width, const SetVector<Value *> &visited,
             const SetVector<Value *> &sources,
             const SetVector<Instruction *> &sinks)
      : OrigTy(Ty), PromotedWidth(Width), Visited(visited), Sources(sources),
        Sinks(sinks) {
    ExtTy = IntegerType::get(Ty->getContext(), PromotedWidth);
    assert(OrigTy->getPrimitiveSizeInBits().getFixedValue() <
               ExtTy->getPrimitiveSizeInBits().getFixedValue() &&
           "Original type not smaller than extended type");
  }

  void Mutate();

private:
  void ReplaceAllUsersOfWith(Value *From, Value *To,
                             bool SkipIdenticalCheck = false);
  void ExtendSources();
  void PromoteTree();
  void TruncateSinks();
  void Cleanup();
};

class RISCVTypePromotion : public FunctionPass {
  SmallPtrSet<Value *, 16> AllVisited;

public:
  static char ID;

  RISCVTypePromotion() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override { return PASS_NAME; }

  bool runOnFunction(Function &F) override;

private:
  // Should V be a leaf in the promote tree?
  bool isSource(Value *V);
  // Is I a user that should be promoted.
  bool isPromotableOperation(Instruction *I);
  // Should I be a root in the promotion tree?
  bool isSink(Instruction *I);
  bool TryToPromote(Instruction *I, unsigned PromotedWidth);
};

} // end anonymous namespace

void IRPromoter::ReplaceAllUsersOfWith(Value *From, Value *To,
                                       bool SkipIdenticalCheck) {
  SmallVector<Instruction *, 4> Users;
  Instruction *InstTo = dyn_cast<Instruction>(To);
  bool ReplacedAll = true;

  LLVM_DEBUG(dbgs() << "RISC-V Promotion: Replacing " << *From << " with "
                    << *To << "\n");

  for (Use &U : From->uses()) {
    auto *User = cast<Instruction>(U.getUser());
    if (User == To ||
        (!SkipIdenticalCheck && InstTo && User->isIdenticalTo(InstTo))) {
      ReplacedAll = false;
      continue;
    }
    Users.push_back(User);
  }

  for (Instruction *U : Users)
    U->replaceUsesOfWith(From, To);

  if (ReplacedAll)
    if (auto *I = dyn_cast<Instruction>(From))
      InstsToRemove.insert(I);
}

void IRPromoter::ExtendSources() {
  auto InsertSExt = [&](Value *V, Instruction *InsertPt) {
    assert(V->getType() != ExtTy && "sext already extends to ExtTy");
    LLVM_DEBUG(dbgs() << "RISC-V Promotion: Inserting SExt for " << *V << "\n");
    IRBuilder<> Builder(InsertPt);
    if (auto *I = dyn_cast<Instruction>(V))
      Builder.SetCurrentDebugLocation(I->getDebugLoc());
    else
      Builder.SetCurrentDebugLocation(DebugLoc());

    Value *SExt = Builder.CreateSExt(V, ExtTy);
    if (auto *I = dyn_cast<Instruction>(SExt)) {
      if (isa<Instruction>(V))
        I->moveAfter(InsertPt);
      NewInsts.insert(I);
    }

    ReplaceAllUsersOfWith(V, SExt, /*SkipIdenticalCheck*/ true);
  };

  // Now, insert extending instructions between the sources and their users.
  LLVM_DEBUG(dbgs() << "RISC-V Promotion: Promoting sources:\n");
  for (Value *V : Sources) {
    LLVM_DEBUG(dbgs() << " - " << *V << "\n");
    if (auto *I = dyn_cast<Instruction>(V)) {
      InsertSExt(I, I);
    } else if (auto *Arg = dyn_cast<Argument>(V)) {
      BasicBlock &BB = Arg->getParent()->getEntryBlock();
      // Skip over allocas.
      BasicBlock::iterator I = BB.begin();
      while (isa<AllocaInst>(I))
        ++I;
      InsertSExt(Arg, &*I);
    } else {
      llvm_unreachable("unhandled source that needs extending");
    }
    Promoted.insert(V);
  }
}

void IRPromoter::PromoteTree() {
  LLVM_DEBUG(dbgs() << "RISC-V Promotion: Mutating the tree..\n");

  // Mutate the instructions that aren't sources or sinks.
  for (Value *V : Visited) {
    if (Sources.count(V))
      continue;

    auto *I = cast<Instruction>(V);
    if (Sinks.count(I))
      continue;

    const DataLayout &DL = I->getModule()->getDataLayout();
    // Update constant and undef operands.
    for (unsigned i = 0, e = I->getNumOperands(); i < e; ++i) {
      Value *Op = I->getOperand(i);
      if ((Op->getType() == ExtTy) || !isa<IntegerType>(Op->getType()))
        continue;

      if (auto *Const = dyn_cast<ConstantInt>(Op)) {
        Constant *NewConst = ConstantFoldCastOperand(Instruction::SExt, Const, ExtTy, DL);;
        I->setOperand(i, NewConst);
      } else if (isa<UndefValue>(Op))
        I->setOperand(i, UndefValue::get(ExtTy));
    }

    // Mutate the result type, unless this is an icmp or switch.
    if (!isa<ICmpInst>(I) && !isa<SwitchInst>(I)) {
      assert(!I->getType()->isVoidTy() && "Unexpected type!");
      I->mutateType(ExtTy);
      Promoted.insert(I);
    }

    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      Function *F = Intrinsic::getOrInsertDeclaration(II->getModule(),
                                                      II->getIntrinsicID(), ExtTy);
      II->setCalledFunction(F);
      // FIXME: Promotion should apply to Range.
      II->removeRetAttr(llvm::Attribute::Range);
    }
  }
}

void IRPromoter::TruncateSinks() {
  LLVM_DEBUG(dbgs() << "RISC-V Promotion: Fixing up the sinks:\n");

  auto InsertTrunc = [&](Value *V, Type *TruncTy) -> Instruction * {
    assert(TruncTy && "Null Type");

    if (!isa<Instruction>(V) || !isa<IntegerType>(V->getType()))
      return nullptr;

    if ((!Promoted.count(V) && !NewInsts.count(V)) || Sources.count(V))
      return nullptr;

    LLVM_DEBUG(dbgs() << "RISC-V Promotion: Creating " << *TruncTy
                      << " Trunc for " << *V << "\n");
    IRBuilder<> Builder(cast<Instruction>(V));
    Builder.SetCurrentDebugLocation(DebugLoc());
    auto *Trunc = dyn_cast<Instruction>(Builder.CreateTrunc(V, TruncTy));
    if (Trunc)
      NewInsts.insert(Trunc);
    return Trunc;
  };

  // Fix up any stores or returns that use the results of the promoted
  // chain.
  for (Instruction *I : Sinks) {
    LLVM_DEBUG(dbgs() << "RISC-V Promotion: For Sink: " << *I << "\n");

    // Handle calls separately as we need to iterate over arg operands.
    if (auto *Call = dyn_cast<CallInst>(I)) {
      for (unsigned i = 0; i < Call->arg_size(); ++i) {
        Value *Arg = Call->getArgOperand(i);
        Type *Ty = TruncTysMap[Call][i];
        if (Instruction *Trunc = InsertTrunc(Arg, Ty)) {
          Trunc->moveBefore(Call);
          Call->setArgOperand(i, Trunc);
        }
      }
      continue;
    }

    // Now handle the others.
    for (unsigned i = 0; i < I->getNumOperands(); ++i) {
      Type *Ty = TruncTysMap[I][i];
      if (Instruction *Trunc = InsertTrunc(I->getOperand(i), Ty)) {
        Trunc->moveBefore(I);
        I->setOperand(i, Trunc);
      }
    }
  }
}

void IRPromoter::Cleanup() {
  LLVM_DEBUG(dbgs() << "RISC-V Promotion: Cleanup..\n");
  // Some sexts will now have become redundant, along with their trunc
  // operands, so remove them
  for (Value *V : Visited) {
    auto *SExt = dyn_cast<SExtInst>(V);
    if (!SExt || SExt->getDestTy() != ExtTy)
      continue;

    Value *Src = SExt->getOperand(0);
    if (SExt->getSrcTy() == SExt->getDestTy()) {
      LLVM_DEBUG(dbgs() << "IR Promotion: Removing unnecessary cast: " << *SExt
                        << "\n");
      ReplaceAllUsersOfWith(SExt, Src);
      continue;
    }

    // Unless they produce a value that is narrower than ExtTy, we can
    // replace the result of the sext with the input of a newly inserted
    // trunc.
    if (NewInsts.count(Src) && isa<TruncInst>(Src) &&
        Src->getType() == OrigTy) {
      auto *Trunc = cast<TruncInst>(Src);
      assert(Trunc->getOperand(0)->getType() == ExtTy &&
             "expected inserted trunc to be operating on i32");
      ReplaceAllUsersOfWith(SExt, Trunc->getOperand(0));
    }
  }

  for (Instruction *I : InstsToRemove) {
    LLVM_DEBUG(dbgs() << "RISC-V Promotion: Removing " << *I << "\n");
    I->dropAllReferences();
    I->eraseFromParent();
  }
}

void IRPromoter::Mutate() {
  LLVM_DEBUG(dbgs() << "RISC-V Promotion: Promoting use-def chains from "
                    << OrigTy->getBitWidth() << " to " << PromotedWidth
                    << "-bits\n");

  // Cache original types of the values that will likely need truncating
  for (Instruction *I : Sinks) {
    if (auto *Call = dyn_cast<CallInst>(I)) {
      for (unsigned i = 0; i < Call->arg_size(); ++i)
        TruncTysMap[Call].push_back(Call->getArgOperand(i)->getType());
    } else {
      for (unsigned i = 0; i < I->getNumOperands(); ++i)
        TruncTysMap[I].push_back(I->getOperand(i)->getType());
    }
  }

  // Insert sext instructions between sources and their users.
  ExtendSources();

  // Promote visited instructions, mutating their types in place.
  PromoteTree();

  // Insert trunc instructions for use by calls, stores etc...
  TruncateSinks();

  // Finall, y remove unnecessary sexts and trucsn, delete old instructions and
  // clear the data structures.
  Cleanup();

  LLVM_DEBUG(dbgs() << "RISC-V Promotion: Mutation complete\n");
}

static bool isMinMaxIntrinsic(Instruction *I) {
  auto *II = dyn_cast<IntrinsicInst>(I);
  if (!II)
    return false;

  switch (II->getIntrinsicID()) {
  case Intrinsic::smax:
  case Intrinsic::smin:
  case Intrinsic::umax:
  case Intrinsic::umin:
    return true;
  }

  return false;
}

static bool isSExtAnd(Instruction *I) {
  // If one operand is a constant that wil sign extend with 0s, then the result
  // is always sign extended.
  if (auto *C = dyn_cast<ConstantInt>(I->getOperand(0)))
    return C->getValue().isNonNegative();
  if (auto *C = dyn_cast<ConstantInt>(I->getOperand(1)))
    return C->getValue().isNonNegative();
  return false;
}

static bool isSExtOr(Instruction *I) {
  // If one operand is a constant that wil sign extend with 1s, then the result
  // is always sign extended.
  if (auto *C = dyn_cast<ConstantInt>(I->getOperand(0)))
    return C->getValue().isNegative();
  if (auto *C = dyn_cast<ConstantInt>(I->getOperand(1)))
    return C->getValue().isNegative();
  return false;
}

/// Return true if the given value is a source in the use-def chain.
/// These values will be sext to start the promotion of the tree to i32.
bool RISCVTypePromotion::isSource(Value *V) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    switch (I->getOpcode()) {
    default:
      // Every binary operator except and, or, xor is a source.
      return isa<BinaryOperator>(I) &&
             !cast<BinaryOperator>(I)->isBitwiseLogicOp();
    case Instruction::And:
      return isSExtAnd(I);
    case Instruction::Or:
      return isSExtOr(I);
    case Instruction::BitCast:
    case Instruction::Load:
    case Instruction::Trunc:
    case Instruction::SExt:
    case Instruction::ZExt:
      // TODO: fptosi/fptoui
      return true;
    case Instruction::Call:
      return !isMinMaxIntrinsic(cast<CallInst>(V));
    }
  }

  return isa<Argument>(V);
}

bool RISCVTypePromotion::isPromotableOperation(Instruction *I) {
  switch (I->getOpcode()) {
  default:
    // Bitwise logic ops should be promoted.
    return isa<BinaryOperator>(I) &&
           cast<BinaryOperator>(I)->isBitwiseLogicOp();
  case Instruction::And:
    return !isSExtAnd(I);
  case Instruction::Or:
    return !isSExtOr(I);
  case Instruction::Select:
  case Instruction::PHI:
  case Instruction::ICmp:
  case Instruction::Switch:
    return true;
  case Instruction::Call:
    return isMinMaxIntrinsic(I);
  }

  return false;
}

/// Return true if V will require any promoted values to be truncated for the
/// the IR to remain valid. We can't mutate the value type of these
/// instructions.
bool RISCVTypePromotion::isSink(Instruction *I) {
  switch (I->getOpcode()) {
  default:
    // Every binary operator except and, or, xor is a sink.
    return isa<BinaryOperator>(I) &&
           !cast<BinaryOperator>(I)->isBitwiseLogicOp();
  case Instruction::And:
    return isSExtAnd(I);
  case Instruction::Or:
    return isSExtOr(I);
  case Instruction::Store:
  case Instruction::Ret:
  case Instruction::Trunc:
  case Instruction::SExt:
  case Instruction::ZExt:
    // TODO: Add sitofp/uitofp.
    return true;
  case Instruction::Call:
    return !isMinMaxIntrinsic(I);
  }

  return false;
}

bool RISCVTypePromotion::TryToPromote(Instruction *I, unsigned PromotedWidth) {
  IntegerType *OrigTy = cast<IntegerType>(I->getType());
  unsigned TypeSize = OrigTy->getBitWidth();

  LLVM_DEBUG(dbgs() << "RISC-V Promotion: TryToPromote: " << *I << ", from "
                    << TypeSize << " bits to " << PromotedWidth << "\n");

  if (!isSource(I) && !isPromotableOperation(I)) {
    LLVM_DEBUG(dbgs() << "Not supported" << *I << "\n");
    return false;
  }

  SmallVector<Value *> WorkList;
  SetVector<Value *> CurrentVisited;
  SetVector<Value *> Sources;
  SetVector<Instruction *> Sinks;
  WorkList.push_back(I);
  CurrentVisited.insert(I);

  auto AddToWorklist = [&](Value *V) {
    if (!CurrentVisited.insert(V))
      return;
    WorkList.push_back(V);
  };

  // Iterate through, and add to, a tree of operands and users in the use-def.
  while (!WorkList.empty()) {
    Value *V = WorkList.pop_back_val();

    assert((isa<Instruction>(V) || isa<Argument>(V)) &&
           "Expected Instruction or Argument.");

    LLVM_DEBUG(dbgs() << "RISC-V Promotion: Visiting: " << *V << "\n");

    // If we've already visited this value from somewhere, bail now because
    // the tree has already been explored.
    if (!AllVisited.insert(V).second)
      return false;

    if (isSource(V)) {
      Sources.insert(V);
    } else {
      // If this isn't a source, see if it is something we can promote.
      auto *I = cast<Instruction>(V);
      // Visit operands of any instruction visited.
      for (unsigned i = 0, e = I->getNumOperands(); i < e; ++i) {
        // Skip condition of select.
        if (isa<SelectInst>(I) && i == 0)
          continue;
        // Skip called operand of Calls.
        if (isa<CallInst>(I) && i >= cast<CallInst>(I)->arg_size())
          continue;
        Value *Op = I->getOperand(i);
        // Skip BasicBlock operands of PHINode and SwitchInst.
        if (isa<BasicBlock>(Op))
          continue;
        // If this operand doesn't have the same type, end the search.
        auto *IntTy = dyn_cast<IntegerType>(Op->getType());
        if (!IntTy || IntTy->getBitWidth() != TypeSize) {
          LLVM_DEBUG(dbgs()
                     << "RISC-V Promotion: Can't handle def: " << *Op << "\n");
          return false;
        }
        // Skip ConstantInts and undef.
        if (isa<ConstantInt>(Op) || isa<UndefValue>(Op))
          continue;
        // We can handle sources or promotable operations.
        if (!isSource(Op) && !(isa<Instruction>(Op) &&
                               isPromotableOperation(cast<Instruction>(Op)))) {
          LLVM_DEBUG(dbgs()
                     << "RISC-V Promotion: Can't handle def: " << *Op << "\n");
          return false;
        }
        AddToWorklist(Op);
      }
    }

    // Don't visit the users of icmps since it returns an i1.
    if (!isa<ICmpInst>(V)) {
      for (Use &U : V->uses()) {
        Instruction *VUser = cast<Instruction>(U.getUser());
        if (isPromotableOperation(VUser)) {
          AddToWorklist(VUser);
        } else if (isSink(VUser)) {
          Sinks.insert(VUser);
        } else {
          LLVM_DEBUG(dbgs() << "RISC-V Promotion: Can't handle user: " << *VUser
                            << "\n");
          return false;
        }
      }
    }
  }

  // Put the Sinks the Visited set.
  for (Instruction *I : Sinks)
    CurrentVisited.insert(I);

  LLVM_DEBUG(dbgs() << "RISC-V Promotion: Visited nodes:\n";
             for (auto *I
                  : CurrentVisited) I->dump(););

  unsigned ToPromote = 0;
  unsigned NonFreeArgs = 0;
  SmallSet<BasicBlock *, 4> Blocks;
  for (auto *V : CurrentVisited) {
    if (auto *I = dyn_cast<Instruction>(V))
      Blocks.insert(I->getParent());

    if (Sources.count(V)) {
      if (auto *Arg = dyn_cast<Argument>(V))
        if (!Arg->hasSExtAttr())
          ++NonFreeArgs;
      continue;
    }

    if (Sinks.count(cast<Instruction>(V)))
      continue;
    ++ToPromote;
  }

  // FIXME: This is a modified version of the TypePromotion heuristic. May
  // need tuning.
  if (ToPromote < 2 || (Blocks.size() == 1 && NonFreeArgs > 0))
    return false;

  IRPromoter Promoter(OrigTy, PromotedWidth, CurrentVisited, Sources, Sinks);
  Promoter.Mutate();

  return true;
}

// Look for (icmp eq (and (shl 1, X), Y), 0).
static bool isBitTest(ICmpInst *ICmp) {
  using namespace llvm::PatternMatch;

  if (!ICmp->isEquality())
    return false;

  // Must be a compare with 0.
  if (!match(ICmp->getOperand(1), m_ZeroInt()))
    return false;

  Instruction *I = dyn_cast<Instruction>(ICmp->getOperand(0));
  if (!I || I->getOpcode() != Instruction::And || !I->hasOneUse())
    return false;

  Value *LHS = I->getOperand(0);
  Value *RHS = I->getOperand(1);

  // If either operand is a shift of 1, this is a bit test.
  return match(LHS, m_OneUse(m_Shl(m_SpecificInt(1), m_Value()))) ||
         match(RHS, m_OneUse(m_Shl(m_SpecificInt(1), m_Value())));
}

bool RISCVTypePromotion::runOnFunction(Function &F) {
  if (skipFunction(F) || DisablePromotion)
    return false;

  LLVM_DEBUG(dbgs() << "RISC-V Promotion: Running on " << F.getName() << "\n");

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  auto &TM = TPC->getTM<RISCVTargetMachine>();
  const RISCVSubtarget *ST = TM.getSubtargetImpl(F);
  // FIXME: Only handle RV64 for now.
  if (!ST->is64Bit())
    return false;

  bool MadeChange = false;

  // Search up from icmps to try to promote their operands.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (AllVisited.count(&I))
        continue;

      auto *ICmp = dyn_cast<ICmpInst>(&I);
      if (!ICmp)
        continue;

      // Look for i32 compares.
      if (!ICmp->getOperand(0)->getType()->isIntegerTy(32))
        continue;

      LLVM_DEBUG(dbgs() << "RISC-V Promotion: Searching from: " << *ICmp
                        << "\n");

      if (isBitTest(ICmp)) {
        LLVM_DEBUG(dbgs() << "Skipping bittest\n");
        continue;
      }

      for (auto &Op : ICmp->operands()) {
        auto *I = dyn_cast<Instruction>(Op);
        if (!I)
          continue;

        MadeChange |= TryToPromote(I, ST->getXLen());
        break;
      }
    }
    LLVM_DEBUG(if (verifyFunction(F, &dbgs())) {
      dbgs() << F;
      report_fatal_error("Broken function after type promotion");
    });
  }

  AllVisited.clear();

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(RISCVTypePromotion, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_END(RISCVTypePromotion, DEBUG_TYPE, PASS_NAME, false, false)

char RISCVTypePromotion::ID = 0;

FunctionPass *llvm::createRISCVTypePromotionPass() {
  return new RISCVTypePromotion();
}
