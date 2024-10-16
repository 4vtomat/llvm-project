//===----- SiFive_RISCVLateCodeGenPrepare.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a RISC-V specific version of CodeGenPrepare.
// It munges the code in the input function to better prepare it for
// SelectionDAG-based code generation. This works around limitations in it's
// basic-block-at-a-time approach.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "riscv-late-codegenprepare"
#define PASS_NAME "RISC-V Late CodeGenPrepare"

STATISTIC(NumUnknownSizeMemmove,
          "Number of unknown size memmove call expanded");
STATISTIC(NumUnknownSizeAlignedMemmove,
          "Number of unknown size aligned memmove call expanded");
STATISTIC(NumKnownSizeMemmove, "Number of known size memmove call expanded");
STATISTIC(NumUnknownSizeMemcpy,
          "Number of unknown size memcpy call expanded");
STATISTIC(NumUnknownSizeAlignedMemcpy,
          "Number of unknown size aligned memcpy call expanded");
STATISTIC(NumKnownSizeMemcpy, "Number of known size memcpy call expanded");
STATISTIC(NumUnknownSizeMemset,
          "Number of unknown size memset call expanded");
STATISTIC(NumUnknownSizeAlignedMemset,
          "Number of unknown size aligned memset call expanded");
STATISTIC(NumKnownSizeMemset, "Number of known size memset call expanded");

#define CREATE_BASIC_BLOCKS_WO_BACKWARD(NAME)                                  \
  BasicBlock *PreLoopBB = M->getParent();                                      \
  BasicBlock *PostLoopBB = PreLoopBB->splitBasicBlock(M, NAME "-post-loop");   \
  BasicBlock *ForwardLoopBB =                                                  \
      BasicBlock::Create(PreLoopBB->getContext(), NAME "-forward-loop",        \
                         PreLoopBB->getParent(), PostLoopBB);

#define CREATE_BASIC_BLOCKS(NAME)                                              \
  CREATE_BASIC_BLOCKS_WO_BACKWARD(NAME)                                        \
  BasicBlock *BWPreLoopBB =                                                    \
      BasicBlock::Create(PreLoopBB->getContext(), NAME "-backward-pre-loop",   \
                         PreLoopBB->getParent(), PostLoopBB);                  \
  BasicBlock *BackwardLoopBB =                                                 \
      BasicBlock::Create(PreLoopBB->getContext(), NAME "-backward-loop",       \
                         PreLoopBB->getParent(), PostLoopBB);

using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<bool>
    MemToRVVOpt("riscv-mem-to-rvv", cl::Hidden,
                cl::desc("Expand mem intrinsic to vector instructions."),
                cl::init(true));
static cl::opt<bool>
    MemAlignOpt("riscv-mem-to-rvv-dlen-align", cl::Hidden,
                cl::desc("Let expansion mem intrinsic can align on DLEN."),
                cl::init(true));
static cl::opt<unsigned>
    MemLMUL("riscv-mem-to-rvv-lmul", cl::Hidden,
            cl::desc("Configure LMUL for memcpy/memmove/memset expansion "
                     "(default value: CPU specific heuristic)."),
            cl::init(0));

static cl::opt<unsigned>
    PreferMemcpyLMUL("riscv-memcpy-to-rvv-lmul", cl::Hidden,
                     cl::desc("Configure LMUL for memcpy expansion "
                              "(default value: riscv-mem-to-rvv)."),
                     cl::init(0));

static cl::opt<unsigned>
    PreferMemsetLMUL("riscv-memset-to-rvv-lmul", cl::Hidden,
                     cl::desc("Configure LMUL for memset expansion "
                              "(default value: riscv-mem-to-rvv)."),
                     cl::init(0));

static cl::opt<unsigned>
    PreferMemmoveLMUL("riscv-memmove-to-rvv-lmul", cl::Hidden,
                      cl::desc("Configure LMUL for memmove expansion "
                               "(default value: riscv-mem-to-rvv)."),
                      cl::init(0));

static cl::opt<unsigned>
    PreferUnrollThreshold("riscv-mem-to-rvv-unroll-threshold", cl::Hidden,
                          cl::desc("Configure size for unroll expansion "
                                   "(default value: 8 * vlen)."),
                          cl::init(0));

namespace {

class RISCVLateCodeGenPrepare
    : public FunctionPass,
      public InstVisitor<RISCVLateCodeGenPrepare, bool> {
  const DataLayout *DL;
  const RISCVSubtarget *ST;
  unsigned AlignBytes;

  SmallVector<MemIntrinsic *, 4> MemCalls;

public:
  static char ID;

  RISCVLateCodeGenPrepare() : FunctionPass(ID) {}

  StringRef getPassName() const override { return PASS_NAME; }

  bool runOnFunction(Function &F) override;

  static constexpr unsigned MinCopySize = 64;

  unsigned UnrollThreshold;
  unsigned MemcpyLMUL;
  unsigned MemsetLMUL;
  unsigned MemmoveLMUL;

  bool visitInstruction(Instruction &I) { return false; }
  bool visitZExtInst(ZExtInst &I);
  bool optimizeZExtWUses(ZExtInst &I);
  bool visitAnd(BinaryOperator &BO);
  bool visitXor(BinaryOperator &BO);
  bool optimizeAndUses(BinaryOperator &BO);
  bool visitICmp(ICmpInst &ICmp);
  bool visitIntrinsicInst(IntrinsicInst &I);
  bool visitMemIntrinsic(MemIntrinsic &MI);
  bool expandMemIntrinsic(MemIntrinsic *MI);
  void expandMemCpyUnknownSize(MemCpyInst *MCI);
  void expandMemCpyUnknownSizewithAlign(MemCpyInst *MCI);
  void expandMemCpyKnownSize(MemCpyInst *MCI);
  void expandMemSetUnknownSize(MemSetInst *MSI);
  void expandMemSetUnknownSizeAligned(MemSetInst *MSI);
  void expandMemSetKnownSize(MemSetInst *MSI);
  void createMemsetLoopBody(BasicBlock *LoopBody, BasicBlock *PreLoopBB,
                            BasicBlock *PostLoopBB, Value *Val, Value *DstAddr,
                            Value *CopyLen, uint64_t UnrollCount = 1);
  void expandMemmoveUnknownSize(MemMoveInst *MCI);
  void expandMemmoveUnknownSizeAligned(MemMoveInst *MCI);
  void expandMemmoveKnownSize(MemMoveInst *MCI);
  void createMemcpyLoopBody(BasicBlock *LoopBody, BasicBlock *PreLoopBB,
                            BasicBlock *PostLoopBB, Value *SrcAddr,
                            Value *DstAddr, Value *CopyLen, unsigned LMUL,
                            bool IsBackward = false, uint64_t UnrollCount = 1);
  void getMemToRVVConfig();
};

} // end anonymous namespace

// If the result of a zext.w is used by a GEP in another basic block, duplicate
// the zext to enable add.uw or shXadd.uw.
bool RISCVLateCodeGenPrepare::optimizeZExtWUses(ZExtInst &I) {
  if (!ST->hasStdExtZba())
    return false;

  BasicBlock *DefBB = I.getParent();

  Value *Src = I.getOperand(0);

  // Needs to be a zext.w.
  if (!Src->getType()->isIntegerTy(32) || !I.getType()->isIntegerTy(64))
    return false;

  // Make sure all users are GEPs or left shifts by constant.
  // NOTE: This isn't strictly necessary, but it ensures we don't extend the
  // live range of Src without removing all non-local users of I.
  bool HasNonLocalUser = false;
  for (auto *U : I.users()) {
    auto *UserI = cast<Instruction>(U);

    if (!isa<GetElementPtrInst>(UserI) &&
        !(UserI->getOpcode() == Instruction::Shl &&
          isa<ConstantInt>(UserI->getOperand(1))))
      return false;

    if (UserI->getParent() != DefBB)
      HasNonLocalUser = true;
  }

  // If all users are local we don't need to do anything.
  if (!HasNonLocalUser)
    return false;

  DenseMap<BasicBlock *, ZExtInst *> InsertedZExts;

  bool MadeChange = false;
  for (auto UI = I.user_begin(), E = I.user_end(); UI != E;) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    BasicBlock *UserBB = User->getParent();

    // If this user is in the same block as the zext, don't create a new zext.
    if (UserBB == DefBB)
      continue;

    // If we have already inserted a zext into this block, use it.
    ZExtInst *&InsertedZExt = InsertedZExts[UserBB];

    if (!InsertedZExt) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      assert(InsertPt != UserBB->end());
      InsertedZExt =
          new ZExtInst(Src, I.getType(), "", InsertPt->getIterator());
      // Propagate the debug info.
      InsertedZExt->setDebugLoc(I.getDebugLoc());
    }

    // Replace a use of the zext with a use of the new zext.
    TheUse = InsertedZExt;
    MadeChange = true;
  }

  // If the original zext has become dead, remove it.
  if (I.use_empty()) {
    I.eraseFromParent();
    MadeChange = true;
  }

  return MadeChange;
}

bool RISCVLateCodeGenPrepare::visitZExtInst(ZExtInst &ZExt) {
  if (!ST->is64Bit())
    return false;

  return optimizeZExtWUses(ZExt);
}

// If the result of a and with 0xffffffff is used by a GEP in another basic
// block, duplicate the and to enable add.uw or shXadd.uw.
bool RISCVLateCodeGenPrepare::optimizeAndUses(BinaryOperator &BO) {
  assert(BO.getOpcode() == Instruction::And);

  if (!ST->hasStdExtZba())
    return false;

  // We're looking for AND with 0xffffffffff.
  auto *CI = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!CI || CI->getZExtValue() != UINT64_C(0xffffffff))
    return false;

  BasicBlock *DefBB = BO.getParent();

  // Make sure all users are GEPs or left shifts by constant.
  // NOTE: This isn't strictly necessary, but it ensures we don't extend the
  // live range of Src without removing all non-local users of I.
  bool HasNonLocalUser = false;
  for (auto *U : BO.users()) {
    auto *UserI = cast<Instruction>(U);

    if (!isa<GetElementPtrInst>(UserI) &&
        !(UserI->getOpcode() == Instruction::Shl &&
          isa<ConstantInt>(UserI->getOperand(1))))
      return false;

    if (UserI->getParent() != DefBB)
      HasNonLocalUser = true;
  }

  // If all users are local we don't need to do anything.
  if (!HasNonLocalUser)
    return false;

  DenseMap<BasicBlock *, BinaryOperator *> InsertedAnds;

  bool MadeChange = false;
  for (auto UI = BO.user_begin(), E = BO.user_end(); UI != E;) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    BasicBlock *UserBB = User->getParent();

    // If this user is in the same block as the and, don't create a new and.
    if (UserBB == DefBB)
      continue;

    // If we have already inserted an and into this block, use it.
    BinaryOperator *&InsertedAnd = InsertedAnds[UserBB];

    if (!InsertedAnd) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      assert(InsertPt != UserBB->end());
      InsertedAnd = BinaryOperator::CreateAnd(
          BO.getOperand(0), BO.getOperand(1), "", InsertPt->getIterator());
      // Propagate the debug info.
      InsertedAnd->setDebugLoc(BO.getDebugLoc());
    }

    // Replace a use of the and with a use of the new and.
    TheUse = InsertedAnd;
    MadeChange = true;
  }

  // If the original and has become dead, remove it.
  if (BO.use_empty()) {
    BO.eraseFromParent();
    MadeChange = true;
  }

  return MadeChange;
}

bool RISCVLateCodeGenPrepare::visitAnd(BinaryOperator &BO) {
  if (!ST->is64Bit())
    return false;

  if (!BO.getType()->isIntegerTy(64))
    return false;

  return optimizeAndUses(BO);
}

// Try to sink Xor with 255 if it helps SelectionDAG recognize a blend pattern
// like (add (mul (xor (zext A), 255), (zext X)), (mul (zext A), (zext Y))).
// This is needed so SelectionDAG can recognize that the pattern produces a 16
// bit result. Naive analysis in computeKnownBits think it produces a 17 bit
// result. SelectionDAG needs the xor sunk so everything is in one basic block.
// MachineCSE/MachineLICM should hoist it again in the motivating case.
bool RISCVLateCodeGenPrepare::visitXor(BinaryOperator &BO) {
  if (!BO.getType()->isIntegerTy(16))
    return false;

  auto *CI = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!CI || CI->getZExtValue() != 255)
    return false;

  KnownBits Known = computeKnownBits(BO.getOperand(0), *DL);
  if (Known.countMaxActiveBits() > 8)
    return false;

  DenseMap<BasicBlock *, BinaryOperator *> InsertedXors;

  BasicBlock *DefBB = BO.getParent();

  bool MadeChange = false;
  for (auto UI = BO.user_begin(), E = BO.user_end(); UI != E;) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    BasicBlock *UserBB = User->getParent();

    // If this user is in the same block as the xor, don't create a new xor.
    if (UserBB == DefBB)
      continue;

    if (User->getOpcode() != Instruction::Mul || !User->hasOneUse())
      continue;

    Instruction *User2 = cast<Instruction>(*User->user_begin());
    Value *A, *X, *Y;
    if (!match(User2,
               m_c_Add(m_c_Mul(m_c_Xor(m_ZExt(m_Value(A)), m_SpecificInt(255)),
                               m_ZExt(m_Value(X))),
                       m_c_Mul(m_ZExt(m_Deferred(A)), m_ZExt(m_Value(Y))))))
      continue;

    // If we have already inserted an xor into this block, use it.
    BinaryOperator *&InsertedXor = InsertedXors[UserBB];

    if (!InsertedXor) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      assert(InsertPt != UserBB->end());
      InsertedXor = BinaryOperator::CreateXor(
          BO.getOperand(0), BO.getOperand(1), "", InsertPt->getIterator());
      // Propagate the debug info.
      InsertedXor->setDebugLoc(BO.getDebugLoc());
    }

    // Replace a use of the xor with a use of the new xor.
    TheUse = InsertedXor;
    MadeChange = true;
  }

  // If the original xor has become dead, remove it.
  if (BO.use_empty()) {
    BO.eraseFromParent();
    MadeChange = true;
  }

  return MadeChange;
}

bool RISCVLateCodeGenPrepare::visitICmp(ICmpInst &ICmp) {
  if (ST->hasStdExtZbb())
    return false;

  auto *BO = dyn_cast<BinaryOperator>(ICmp.getOperand(0));
  if (!BO)
    return false;

  // Fold (icmp sgt (A + 1), Op1) -> (icmp sge A, Op1) if the add won't wrap.
  // InstCombine normally does this, but it is disabled if the add is part of a
  // min pattern. Without Zbb, the min will be turned into control flow so it
  // is better to separate the add from the cmp so we can sink it.
  if (ICmp.getPredicate() == ICmpInst::ICMP_SGT &&
      BO->getOpcode() == Instruction::Add && BO->hasNoSignedWrap() &&
      match(BO->getOperand(1), m_One())) {
    IRBuilder<> Builder(&ICmp);
    Value *NewICmp =
        Builder.CreateICmpSGE(BO->getOperand(0), ICmp.getOperand(1));
    NewICmp->takeName(&ICmp);
    ICmp.replaceAllUsesWith(NewICmp);
    ICmp.eraseFromParent();
    return true;
  }

  return false;
}

void RISCVLateCodeGenPrepare::expandMemCpyUnknownSize(MemCpyInst *M) {
  ++NumUnknownSizeMemcpy;
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memcpy")

  Value *SrcAddr = M->getRawSource();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();


  IRBuilder<> Builder(PreLoopBB->getTerminator());

  // Main Loop, expand Memcpy() to RVV instructions.
  // The RVV instructions like below:
  // preloop:
  //   jump loop
  // loop:
  //   vsetvli VL, CopyLen, e8, m8, tu, mu
  //   vle8.v vData, (Src)
  //   vse8.v vData, (Dst)
  //   add Src, Src, VL
  //   add Dst, Dst, VL
  //   sub CopyLen, CopyLen, VL
  //   bgtu CopyLen, zero, loop

  Builder.CreateBr(ForwardLoopBB);
  createMemcpyLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, SrcAddr, DstAddr,
                       CopyLen, MemcpyLMUL, false);
  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemSetUnknownSize(MemSetInst *M) {
  ++NumUnknownSizeMemset;
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memset")

  Value *Val = M->getValue();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();

  createMemsetLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, Val, DstAddr,
                       CopyLen);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemmoveUnknownSize(MemMoveInst *M) {
  ++NumUnknownSizeMemmove;
  CREATE_BASIC_BLOCKS("memmove")

  Value *SrcAddr = M->getRawSource();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();
  Type *Int8Type = Type::getInt8Ty(M->getParent()->getContext());

  IRBuilder<> Builder(PreLoopBB->getTerminator());

  // Expand memmove to RVV instructions.
  // preloop:
  //   sub Diff, Dst, Src
  //   blt Diff, CopyLen, backward-pre-loop
  //
  // forward-loop:
  //   perform forward mem copy
  //
  // backward-pre-loop
  //   perform last copy element calculation
  //
  // backward-loop:
  //   perform backward mem copy
  //
  // postloop:
  //   return
  Value *DstAddrInt = Builder.CreatePtrToInt(DstAddr, Builder.getIntPtrTy(*DL));
  Value *SrcAddrInt = Builder.CreatePtrToInt(SrcAddr, Builder.getIntPtrTy(*DL));
  Value *Diff = Builder.CreateSub(DstAddrInt, SrcAddrInt);
  Value *ULT = Builder.CreateICmpULT(Diff, CopyLen);
  Builder.CreateCondBr(ULT, BWPreLoopBB, ForwardLoopBB);

  Builder.SetInsertPoint(BWPreLoopBB);
  Value *SrcEndAddr = Builder.CreateGEP(Int8Type, SrcAddr, CopyLen);
  Value *DstEndAddr = Builder.CreateGEP(Int8Type, DstAddr, CopyLen);
  Builder.CreateBr(BackwardLoopBB);

  createMemcpyLoopBody(BackwardLoopBB, BWPreLoopBB, PostLoopBB, SrcEndAddr,
                       DstEndAddr, CopyLen, MemmoveLMUL, true);
  createMemcpyLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, SrcAddr, DstAddr,
                       CopyLen, MemmoveLMUL, false);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

// If the address is not aligned, we may need one more vle and one more vse per
// iteration. Thus the precalculation of the elements that are not aligned to
// DLEN is beneficial in some cases.
void RISCVLateCodeGenPrepare::expandMemmoveUnknownSizeAligned(MemMoveInst *M) {
  ++NumUnknownSizeAlignedMemmove;
  CREATE_BASIC_BLOCKS("memmove")

  Value *SrcAddr = M->getRawSource();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();
  Type *Int8Type = Type::getInt8Ty(M->getParent()->getContext());

  IRBuilder<> Builder(PreLoopBB->getTerminator());

  // Expand memmove to RVV instructions.
  // preloop:
  //   sub Diff, Dst, Src
  //   blt Diff, CopyLen, backward-pre-loop
  //
  // forward-preloop:
  //   andi Dlenelement, SrcAddr, Dlen - 1
  //   sub DlenElement, AlignBytes, DlenElement
  //   minu AlignLen, CopyLen, Dlenelement
  //   vsetvli VL, AlignLen, e8, m8, tu, mu
  //   vle8.v vData, (Src)
  //   vse8.v vData, (Dst)
  //   add Src, Src, VL
  //   add Dst, Dst, VL
  //   sub NewCopyLen, CopyLen, VL
  //   beqz NewCopyLen, postloop
  //
  // forwardloop:
  //   perform forward mem copy
  //
  // backward-preloop:
  //   add SrcEndAddr, SrcAddr, CopyLen
  //   andi Dlenelement, SrcEndAddr, Dlen - 1
  //   minu AlignLen, CopyLen, Dlenelement
  //   vsetvli VL, AlignLen, e8, m8, tu, mu
  //   sub NewCopyLen, CopyLen, VL
  //   add SrcLastElemAddr, SrcAddr, NewCopyLen
  //   add DstLastElemAddr, DstAddr, NewCopyLen
  //   vle8.v vData, (SrcLastElemAddr)
  //   vse8.v vData, (DstLastElemAddr)
  //   sub SrcLastElemAddr, SrcLastElemAddr, VL
  //   sub DstLastElemAddr, DstLastElemAddr, VL
  //   beqz NewCopyLen, postloop
  //
  // backwardloop:
  //   perform backward mem copy
  //
  // postloop:
  //   return
  BasicBlock *FWPreLoopBB =
      BasicBlock::Create(PreLoopBB->getContext(), "memmove-forward-pre-loop",
                         PreLoopBB->getParent(), PostLoopBB);
  Value *DstAddrInt = Builder.CreatePtrToInt(DstAddr, Builder.getIntPtrTy(*DL));
  Value *SrcAddrInt = Builder.CreatePtrToInt(SrcAddr, Builder.getIntPtrTy(*DL));
  Value *Diff = Builder.CreateSub(DstAddrInt, SrcAddrInt);
  Value *ULT = Builder.CreateICmpULT(Diff, CopyLen);
  Builder.CreateCondBr(ULT, BWPreLoopBB, FWPreLoopBB);

  ScalableVectorType *VTy = ScalableVectorType::get(
      Int8Type, RISCV::RVVBitsPerBlock / 8 * MemmoveLMUL);
  Type *CopyLenType = CopyLen->getType();
  IntegerType *ILengthType = cast<IntegerType>(CopyLenType);

  Value *Sew8 = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  Value *Lmul =
      ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(MemmoveLMUL, false));
  ConstantInt *Zero = ConstantInt::get(ILengthType, 0U);

  Builder.SetInsertPoint(BWPreLoopBB);
  Value *NewCopyLen, *SrcLastElemAddr, *DstLastElemAddr;
  // Backward pre-loop
  {
    Value *SrcEndAddr = Builder.CreateGEP(Int8Type, SrcAddr, CopyLen);
    Value *DLenElement =
        Builder.CreateAnd(Builder.CreatePtrToInt(SrcEndAddr, ILengthType),
                          ConstantInt::get(ILengthType, AlignBytes - 1));

    Value *AlignLen = Builder.CreateBinaryIntrinsic(
        Intrinsic::umin, DLenElement, CopyLen, nullptr, "length.select");
    Value *VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                        {AlignLen, Sew8, Lmul});
    NewCopyLen = Builder.CreateSub(CopyLen, VL);

    SrcLastElemAddr = Builder.CreateGEP(Int8Type, SrcAddr, NewCopyLen);
    Value *Load =
        Builder.CreateIntrinsic(Intrinsic::riscv_vle, {VTy, CopyLenType},
                                {UndefValue::get(VTy), SrcLastElemAddr, VL});

    DstLastElemAddr = Builder.CreateGEP(Int8Type, DstAddr, NewCopyLen);
    Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                            {Load, DstLastElemAddr, VL});

    Builder.CreateCondBr(Builder.CreateICmpNE(NewCopyLen, Zero), BackwardLoopBB,
                         PostLoopBB);
  }

  // Backward loop
  createMemcpyLoopBody(BackwardLoopBB, BWPreLoopBB, PostLoopBB, SrcLastElemAddr,
                       DstLastElemAddr, NewCopyLen, MemmoveLMUL, true);

  Builder.SetInsertPoint(FWPreLoopBB);
  Value *SrcFirstElemAddr, *DstFirstElemAddr;
  // Forward pre-loop
  {
    Value *DLenElement = Builder.CreateSub(
        ConstantInt::get(ILengthType, AlignBytes),
        Builder.CreateAnd(Builder.CreatePtrToInt(SrcAddr, ILengthType),
                          ConstantInt::get(ILengthType, AlignBytes - 1)));

    Value *AlignLen = Builder.CreateBinaryIntrinsic(
        Intrinsic::umin, DLenElement, CopyLen, nullptr, "length.select");
    Value *VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                        {AlignLen, Sew8, Lmul});
    NewCopyLen = Builder.CreateSub(CopyLen, VL);

    Value *Load =
        Builder.CreateIntrinsic(Intrinsic::riscv_vle, {VTy, CopyLenType},
                                {UndefValue::get(VTy), SrcAddr, VL});

    Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                            {Load, DstAddr, VL});

    SrcFirstElemAddr = Builder.CreateGEP(Int8Type, SrcAddr, VL);
    DstFirstElemAddr = Builder.CreateGEP(Int8Type, DstAddr, VL);

    Builder.CreateCondBr(Builder.CreateICmpNE(NewCopyLen, Zero), ForwardLoopBB,
                         PostLoopBB);
  }

  // Forward loop
  createMemcpyLoopBody(ForwardLoopBB, FWPreLoopBB, PostLoopBB, SrcFirstElemAddr,
                       DstFirstElemAddr, NewCopyLen, MemmoveLMUL);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemmoveKnownSize(MemMoveInst *M) {
  ++NumKnownSizeMemmove;
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memmove")

  Value *SrcAddr = M->getRawSource();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();
  Type *Int8Type = Type::getInt8Ty(M->getParent()->getContext());

  IRBuilder<> Builder(PreLoopBB->getTerminator());

  // Expand memmove to RVV instructions.
  // preloop:
  //   sub Diff, Dst, Src
  //   blt Diff, CopyLen, backward-pre-loop
  //
  // forward-loop:
  //   perform forward mem copy
  //
  // backward-pre-loop
  //   perform last copy element calculation
  //
  // backward-loop:
  //   perform backward mem copy
  //
  // postloop:
  //   return

  auto *CI = cast<ConstantInt>(CopyLen);
  unsigned UnrollCount =
      divideCeil(CI->getZExtValue(), (ST->getRealMinVLen() / 8) * MemmoveLMUL);

  if (UnrollCount > 1) {
    BasicBlock *BWPreLoopBB =
        BasicBlock::Create(PreLoopBB->getContext(), "backward-pre-loop",
                           PreLoopBB->getParent(), PostLoopBB);
    BasicBlock *BackwardLoopBB =
        BasicBlock::Create(PreLoopBB->getContext(), "backward-loop",
                           PreLoopBB->getParent(), PostLoopBB);
    Value *DstAddrInt =
        Builder.CreatePtrToInt(DstAddr, Builder.getIntPtrTy(*DL));
    Value *SrcAddrInt =
        Builder.CreatePtrToInt(SrcAddr, Builder.getIntPtrTy(*DL));
    Value *Diff = Builder.CreateSub(DstAddrInt, SrcAddrInt);
    Value *ULT = Builder.CreateICmpULT(Diff, CopyLen);
    Builder.CreateCondBr(ULT, BWPreLoopBB, ForwardLoopBB);

    Builder.SetInsertPoint(BWPreLoopBB);
    Value *SrcEndAddr = Builder.CreateGEP(Int8Type, SrcAddr, CopyLen);
    Value *DstEndAddr = Builder.CreateGEP(Int8Type, DstAddr, CopyLen);
    Builder.CreateBr(BackwardLoopBB);

    createMemcpyLoopBody(BackwardLoopBB, BWPreLoopBB, PostLoopBB, SrcEndAddr,
                         DstEndAddr, CopyLen, MemmoveLMUL, true, UnrollCount);
  } else
    Builder.CreateBr(ForwardLoopBB);

  createMemcpyLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, SrcAddr, DstAddr,
                       CopyLen, MemmoveLMUL, false, UnrollCount);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::createMemcpyLoopBody(
    BasicBlock *LoopBody, BasicBlock *PreLoopBB, BasicBlock *PostLoopBB,
    Value *SrcAddr, Value *DstAddr, Value *CopyLen, unsigned LMUL,
    bool IsBackward, uint64_t UnrollCount) {
  BasicBlock *EpilogBB;

  // We only deal with 8-bits width of memory at a time.
  Type *Int8Type = Type::getInt8Ty(LoopBody->getContext());
  // Initial vector type for <vscale x (LMUL * RVVBitsPerBlock / 8) x i8>,
  // LMUL=8, SEW=8.
  ScalableVectorType *VTy =
      ScalableVectorType::get(Int8Type, RISCV::RVVBitsPerBlock / 8 * LMUL);
  Type *CopyLenType = CopyLen->getType();

  // Set SEW to 8 bits.
  Value *Sew8 = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  Value *Lmul =
      ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(LMUL, false));

  bool FullyUnrolled = false;
  Value *EpilogLen = nullptr;
  // Max copy size we can deal with each round: DataVLen * LMUL
  int64_t MaxCopySize = (ST->getRealMinVLen() / 8) * LMUL;
  int64_t KnownCurrentLen = -MaxCopySize;
  if (auto *CI = dyn_cast<ConstantInt>(CopyLen)) {
    KnownCurrentLen = CI->getZExtValue();
    uint64_t FullCopies = CI->getZExtValue() / MaxCopySize;
    uint64_t Remainings = CI->getZExtValue() % MaxCopySize;
    uint64_t TotalCopiesNeeded = FullCopies + (Remainings ? 1 : 0);

    // UnrollCount must be smaller or equal to CopyLen / MaxCopySize,
    // otherwise there would be redundant instuctions generated.
    if (UnrollCount == TotalCopiesNeeded)
      FullyUnrolled = true;
    else if (UnrollCount > TotalCopiesNeeded) {
      createMemcpyLoopBody(LoopBody, PreLoopBB, PostLoopBB, SrcAddr, DstAddr,
                           CopyLen, LMUL, IsBackward, UnrollCount - 1);
      return;
    } else if (FullCopies % UnrollCount) {
      uint64_t EpilogCnt = CI->getZExtValue() % (UnrollCount * MaxCopySize);
      if (EpilogCnt) {
        CopyLen = ConstantInt::get(CopyLenType, CI->getZExtValue() - EpilogCnt);
        EpilogLen = ConstantInt::get(CopyLenType, EpilogCnt);
        EpilogBB =
            BasicBlock::Create(PreLoopBB->getContext(), "epilog-basic-block",
                               PreLoopBB->getParent(), PostLoopBB);
      }
    }
  } else
    // We only deal with unroll with constant CopyLen so we are able to
    // calculate epilog
    assert(UnrollCount == 1);

  IRBuilder<> Builder(LoopBody);

  Builder.SetInsertPoint(LoopBody);
  Value *LoopCount = CopyLen;
  Value *SrcIndex = SrcAddr;
  Value *DstIndex = DstAddr;
  if (!FullyUnrolled) {
    LoopCount = Builder.CreatePHI(CopyLenType, 2, "loop-cnt");
    cast<PHINode>(LoopCount)->addIncoming(CopyLen, PreLoopBB);
    SrcIndex = Builder.CreatePHI(SrcAddr->getType(), 2, "src-addr");
    cast<PHINode>(SrcIndex)->addIncoming(SrcAddr, PreLoopBB);
    DstIndex = Builder.CreatePHI(DstAddr->getType(), 2, "dst-addr");
    cast<PHINode>(DstIndex)->addIncoming(DstAddr, PreLoopBB);
  }

  // To support a larger VLEN than the minimum VLEN.
  // Use minimum VLEN with LMUL8 to setup VLEN.
  Value *MinLenLmulM8 =
      ConstantInt::get(CopyLenType, std::min(KnownCurrentLen, MaxCopySize));
  Value *NewLoopCount = !FullyUnrolled ? LoopCount : MinLenLmulM8;
  Value *SrcIndexTmp = SrcIndex;
  Value *DstIndexTmp = DstIndex;
  Value *VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                      {NewLoopCount, Sew8, Lmul});
  while (UnrollCount--) {
    if (KnownCurrentLen != -MaxCopySize) {
      KnownCurrentLen -= MaxCopySize;
      if (KnownCurrentLen < 0)
        VL = Builder.CreateIntrinsic(
            Intrinsic::riscv_vsetvli, {CopyLenType},
            {ConstantInt::get(CopyLenType, KnownCurrentLen + MaxCopySize), Sew8,
             Lmul});
    }

    if (IsBackward) {
      Value *NegVL = Builder.CreateNeg(VL);
      SrcIndexTmp = Builder.CreateGEP(Int8Type, SrcIndexTmp, NegVL);
      DstIndexTmp = Builder.CreateGEP(Int8Type, DstIndexTmp, NegVL);
    }

    Value *Load =
        Builder.CreateIntrinsic(Intrinsic::riscv_vle, {VTy, CopyLenType},
                                {UndefValue::get(VTy), SrcIndexTmp, VL});

    Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                            {Load, DstIndexTmp, VL});

    if (FullyUnrolled && !UnrollCount)
      break;

    NewLoopCount = Builder.CreateSub(NewLoopCount, VL);

    if (!IsBackward) {
      SrcIndexTmp = Builder.CreateGEP(Int8Type, SrcIndexTmp, VL);
      DstIndexTmp = Builder.CreateGEP(Int8Type, DstIndexTmp, VL);
    }
  }

  if (!FullyUnrolled) {
    cast<PHINode>(SrcIndex)->addIncoming(SrcIndexTmp, LoopBody);
    cast<PHINode>(DstIndex)->addIncoming(DstIndexTmp, LoopBody);
    cast<PHINode>(LoopCount)->addIncoming(NewLoopCount, LoopBody);
  }

  IntegerType *ILengthType = cast<IntegerType>(CopyLenType);
  ConstantInt *Zero = ConstantInt::get(ILengthType, 0U);

  if (FullyUnrolled)
    Builder.CreateBr(PostLoopBB);
  else
    Builder.CreateCondBr(Builder.CreateICmpNE(NewLoopCount, Zero), LoopBody,
                         EpilogLen ? EpilogBB : PostLoopBB);

  // Create epilog
  if (EpilogLen) {
    Builder.SetInsertPoint(EpilogBB);

    LoopCount = Builder.CreatePHI(CopyLenType, 2, "loop-cnt");
    cast<PHINode>(LoopCount)->addIncoming(EpilogLen, LoopBody);
    SrcIndex = Builder.CreatePHI(SrcAddr->getType(), 2, "src-addr");
    cast<PHINode>(SrcIndex)->addIncoming(SrcIndexTmp, LoopBody);
    DstIndex = Builder.CreatePHI(DstAddr->getType(), 2, "dst-addr");
    cast<PHINode>(DstIndex)->addIncoming(DstIndexTmp, LoopBody);

    Value *VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                        {LoopCount, Sew8, Lmul});

    Value *Load =
        Builder.CreateIntrinsic(Intrinsic::riscv_vle, {VTy, CopyLenType},
                                {UndefValue::get(VTy), SrcIndex, VL});

    Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                            {Load, DstIndex, VL});

    NewLoopCount = Builder.CreateSub(LoopCount, VL);

    if (IsBackward)
      VL = Builder.CreateNeg(VL);

    SrcIndexTmp = Builder.CreateGEP(Int8Type, SrcIndex, VL);
    DstIndexTmp = Builder.CreateGEP(Int8Type, DstIndex, VL);

    cast<PHINode>(LoopCount)->addIncoming(NewLoopCount, EpilogBB);
    cast<PHINode>(SrcIndex)->addIncoming(SrcIndexTmp, EpilogBB);
    cast<PHINode>(DstIndex)->addIncoming(DstIndexTmp, EpilogBB);

    Builder.CreateCondBr(Builder.CreateICmpNE(NewLoopCount, Zero), EpilogBB,
                         PostLoopBB);
  }
}

void RISCVLateCodeGenPrepare::expandMemCpyKnownSize(MemCpyInst *M) {
  ++NumKnownSizeMemcpy;
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memcpy")

  Value *SrcAddr = M->getRawSource();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();

  IRBuilder<> Builder(PreLoopBB->getTerminator());

  // Expand Memcpy() to RVV instructions.
  // If copy length can unroll 2 times
  // then the RVV instructions like below:
  //
  //   vsetvli VL, CopyLen, e8, m8, tu, mu
  //   vle8.v vData, (Src)
  //   vse8.v vData, (Dst)
  //   add Src, Src, VL
  //   add Dst, Dst, VL
  //   sub CopyLen, CopyLen, VL
  //   vsetvli VL, CopyLen, e8, m8, tu, mu
  //   vle8.v vData, (Src)
  //   vse8.v vData, (Dst)

  auto *CI = cast<ConstantInt>(CopyLen);
  unsigned UnrollCount =
      divideCeil(CI->getZExtValue(), (ST->getRealMinVLen() / 8) * MemcpyLMUL);

  Builder.CreateBr(ForwardLoopBB);
  createMemcpyLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, SrcAddr, DstAddr,
                       CopyLen, MemcpyLMUL, false, UnrollCount);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemSetKnownSize(MemSetInst *M) {
  ++NumKnownSizeMemset;
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memset")

  Value *Val = M->getValue();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();

  auto *CI = cast<ConstantInt>(CopyLen);
  unsigned UnrollCount =
      divideCeil(CI->getZExtValue(), (ST->getRealMinVLen() / 8) * MemsetLMUL);

  createMemsetLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, Val, DstAddr,
                       CopyLen, UnrollCount);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemCpyUnknownSizewithAlign(MemCpyInst *M) {
  ++NumUnknownSizeAlignedMemcpy;
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memcpy")

  Value *SrcAddr = M->getRawSource();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();
  Type *Int8Type = Type::getInt8Ty(M->getParent()->getContext());

  IRBuilder<> Builder(PreLoopBB->getTerminator());

  // Expand Memcpy() to RVV instructions.
  // The RVV instructions like below:
  // preloop:
  //   andi AndRem, SrcAddr, Dlen - 1
  //   sub  DLenelement, Dlen, AndRem
  //   minu AlignLen, CopyLen, Dlenelement
  //   vsetvli VL, AlignLen, e8, m8, tu, mu
  //   vle8.v vData, (Src)
  //   vse8.v vData, (Dst)
  //   add Src, Src, VL
  //   add Dst, Dst, VL
  //   sub NewCopyLen, CopyLen, VL
  //   beqz NewCopyLen, postloop
  // loop:
  //   vsetvli VL, NewCopyLen, e8, m8, tu, mu
  //   vle8.v vData, (Src)
  //   vse8.v vData, (Dst)
  //   add Src, Src, VL
  //   add Dst, Dst, VL
  //   sub NewCopyLen, NewCopyLen, VL
  //   bgtu NewCopyLen, zero, loop
  // postloop:

  ScalableVectorType *VTy = ScalableVectorType::get(
      Int8Type, RISCV::RVVBitsPerBlock / 8 * MemcpyLMUL);
  Type *CopyLenType = CopyLen->getType();
  IntegerType *ILengthType = cast<IntegerType>(CopyLenType);

  Value *SEW = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  Value *LMUL =
      ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(MemcpyLMUL, false));

  Value *Addr = Builder.CreatePtrToInt(SrcAddr, ILengthType);
  Value *And =
      Builder.CreateAnd(Addr, ConstantInt::get(ILengthType, AlignBytes - 1));

  Value *DLenElement =
      Builder.CreateSub(ConstantInt::get(ILengthType, AlignBytes), And);

  Value *AlignLen = Builder.CreateBinaryIntrinsic(
      Intrinsic::umin, DLenElement, CopyLen, nullptr, "length.select");
  Value *AlignVL = Builder.CreateIntrinsic(
      Intrinsic::riscv_vsetvli, {CopyLenType}, {AlignLen, SEW, LMUL});
  Value *Load =
      Builder.CreateIntrinsic(Intrinsic::riscv_vle, {VTy, CopyLenType},
                              {UndefValue::get(VTy), SrcAddr, AlignVL});
  Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                          {Load, DstAddr, AlignVL});

  Value *AlignSrcGEP = Builder.CreateGEP(Int8Type, SrcAddr, AlignVL);
  Value *AlignDstGEP = Builder.CreateGEP(Int8Type, DstAddr, AlignVL);
  Value *NewCopyLen = Builder.CreateSub(CopyLen, AlignVL);

  ConstantInt *Zero = ConstantInt::get(ILengthType, 0U);
  Builder.CreateCondBr(Builder.CreateICmpNE(NewCopyLen, Zero), ForwardLoopBB,
                       PostLoopBB);

  createMemcpyLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, AlignSrcGEP,
                       AlignDstGEP, NewCopyLen, MemcpyLMUL);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemSetUnknownSizeAligned(MemSetInst *M) {
  ++NumUnknownSizeAlignedMemset;
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memset")

  Value *Val = M->getValue();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();

  Type *Int8Type = Type::getInt8Ty(PreLoopBB->getContext());
  ScalableVectorType *VTy = ScalableVectorType::get(
      Int8Type, RISCV::RVVBitsPerBlock / 8 * MemsetLMUL);
  Type *CopyLenType = CopyLen->getType();
  IntegerType *ILengthType = cast<IntegerType>(CopyLenType);

  Value *SEW = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  Value *LMUL =
      ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(MemsetLMUL, false));

  IRBuilder<> Builder(PreLoopBB->getTerminator());

  Value *VLMax = nullptr;
  if (ST->getRealMinVLen() == ST->getRealMaxVLen())
    VLMax =
        ConstantInt::get(CopyLenType, ST->getRealMinVLen() / 8 * MemsetLMUL);
  else
    VLMax = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvlimax, {CopyLenType},
                                    {SEW, LMUL});
  Value *TmpVL =
      Builder.CreateBinaryIntrinsic(Intrinsic::umin, VLMax, CopyLen);

  Value *TmpVal =
      Builder.CreateIntrinsic(Intrinsic::riscv_vmv_v_x, {VTy, CopyLenType},
                              {UndefValue::get(VTy), Val, TmpVL});

  Value *DLenElement = Builder.CreateSub(
      ConstantInt::get(ILengthType, AlignBytes),
      Builder.CreateAnd(Builder.CreatePtrToInt(DstAddr, ILengthType),
                        ConstantInt::get(ILengthType, AlignBytes - 1)));

  Value *AlignLen = Builder.CreateBinaryIntrinsic(
      Intrinsic::umin, DLenElement, CopyLen, nullptr, "length.select");

  Value *AlignVL = Builder.CreateIntrinsic(
      Intrinsic::riscv_vsetvli, {CopyLenType}, {AlignLen, SEW, LMUL});
  CopyLen = Builder.CreateSub(CopyLen, AlignVL);
  Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                          {TmpVal, DstAddr, AlignVL});

  DstAddr = Builder.CreateGEP(Int8Type, DstAddr, AlignVL);

  createMemsetLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, TmpVal, DstAddr,
                       CopyLen, 1);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::createMemsetLoopBody(
    BasicBlock *LoopBody, BasicBlock *PreLoopBB, BasicBlock *PostLoopBB,
    Value *Val, Value *DstAddr, Value *CopyLen, uint64_t UnrollCount) {
  BasicBlock *EpilogBB;

  // We only deal with 8-bits width of memory at a time.
  Type *Int8Type = Type::getInt8Ty(LoopBody->getContext());
  // Initial vector type for <vscale x (LMUL * RVVBitsPerBlock / 8) x i8>, SEW=8.
  ScalableVectorType *VTy = ScalableVectorType::get(
      Int8Type, RISCV::RVVBitsPerBlock / 8 * MemsetLMUL);
  Type *CopyLenType = CopyLen->getType();

  // Set SEW to 8 bits.
  Value *SEW = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  Value *LMUL =
      ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(MemsetLMUL, false));

  bool FullyUnrolled = false;
  Value *EpilogLen = nullptr;
  // Max copy size we can deal with each round: DataVLen * LMUL
  int64_t MaxCopySize = (ST->getRealMinVLen() / 8) * MemsetLMUL;
  int64_t KnownCurrentLen = -MaxCopySize;
  if (auto *CI = dyn_cast<ConstantInt>(CopyLen)) {
    KnownCurrentLen = CI->getZExtValue();
    uint64_t FullCopies = CI->getZExtValue() / MaxCopySize;
    uint64_t Remainings = CI->getZExtValue() % MaxCopySize;
    uint64_t TotalCopiesNeeded = FullCopies + (Remainings ? 1 : 0);

    // UnrollCount must be smaller or equal to CopyLen / MaxCopySize,
    // otherwise there would be redundant instuctions generated.
    if (UnrollCount == TotalCopiesNeeded)
      FullyUnrolled = true;
    else if (UnrollCount > TotalCopiesNeeded) {
      createMemsetLoopBody(LoopBody, PreLoopBB, PostLoopBB, Val, DstAddr,
                           CopyLen, UnrollCount - 1);
      return;
    } else if (FullCopies % UnrollCount) {
      uint64_t EpilogCnt = KnownCurrentLen % (UnrollCount * MaxCopySize);
      if (EpilogCnt) {
        CopyLen = ConstantInt::get(CopyLenType, KnownCurrentLen - EpilogCnt);
        EpilogLen = ConstantInt::get(CopyLenType, EpilogCnt);
        EpilogBB =
            BasicBlock::Create(PreLoopBB->getContext(), "epilog-basic-block",
                               PreLoopBB->getParent(), PostLoopBB);
      }
    }
  } else
    // We only deal with unroll with constant CopyLen so we are able to
    // calculate epilog
    assert(UnrollCount == 1);

  IRBuilder<> Builder(PreLoopBB->getTerminator());
  // To support a larger VLEN than the minimum VLEN.
  // Use minimum VLEN with LMUL8 to setup VLEN.
  Value *MinLenLmulM8 =
      ConstantInt::get(CopyLenType, std::min(KnownCurrentLen, MaxCopySize));
  Value *LoopCount = !FullyUnrolled ? CopyLen : MinLenLmulM8;
  Value *VL = nullptr;
  // If it already copied(broadcasted) the scalar value into a vector in
  // previous blocks, then we can use it directly, otherwise we have to do it.
  if (!isa<ScalableVectorType>(Val->getType())) {
    VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                 {LoopCount, SEW, LMUL});
    Val = Builder.CreateIntrinsic(Intrinsic::riscv_vmv_v_x, {VTy, CopyLenType},
                                  {UndefValue::get(VTy), Val, VL});
  }

  Builder.CreateBr(LoopBody);

  Builder.SetInsertPoint(LoopBody);

  Value *DstIndex = DstAddr;
  if (!FullyUnrolled) {
    LoopCount = Builder.CreatePHI(CopyLenType, 2, "loop-cnt");
    cast<PHINode>(LoopCount)->addIncoming(CopyLen, PreLoopBB);
    DstIndex = Builder.CreatePHI(DstAddr->getType(), 2, "dst-addr");
    cast<PHINode>(DstIndex)->addIncoming(DstAddr, PreLoopBB);
    VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                 {LoopCount, SEW, LMUL});
  }

  Value *NewLoopCount = LoopCount;
  Value *DstIndexTmp = DstIndex;
  SmallVector<Value *> DstIndices;
  uint64_t TmpUC = UnrollCount;
  if (!VL)
    VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                 {LoopCount, SEW, LMUL});
  while (TmpUC--) {
    DstIndices.push_back(DstIndexTmp);
    if (KnownCurrentLen != -MaxCopySize)
      KnownCurrentLen -= MaxCopySize;
    if (TmpUC != 0 && UnrollCount != 1)
      DstIndexTmp = Builder.CreateGEP(Int8Type, DstIndexTmp, VL);
  }
  while (++TmpUC < UnrollCount) {
    if (TmpUC == UnrollCount - 1 && UnrollCount != 1 &&
        (KnownCurrentLen != -MaxCopySize && KnownCurrentLen < 0))
      VL = Builder.CreateIntrinsic(
          Intrinsic::riscv_vsetvli, {CopyLenType},
          {ConstantInt::get(CopyLenType, KnownCurrentLen + MaxCopySize), SEW,
           LMUL});
    Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                            {Val, DstIndices[TmpUC], VL});
  }

  DstIndexTmp = Builder.CreateGEP(Int8Type, DstIndexTmp, VL);

  if (!FullyUnrolled) {
    NewLoopCount = Builder.CreateSub(NewLoopCount, VL);
    cast<PHINode>(DstIndex)->addIncoming(DstIndexTmp, LoopBody);
    cast<PHINode>(LoopCount)->addIncoming(NewLoopCount, LoopBody);
  }

  IntegerType *ILengthType = cast<IntegerType>(CopyLenType);
  ConstantInt *Zero = ConstantInt::get(ILengthType, 0U);

  if (FullyUnrolled)
    Builder.CreateBr(PostLoopBB);
  else
    Builder.CreateCondBr(Builder.CreateICmpNE(NewLoopCount, Zero), LoopBody,
                         EpilogLen ? EpilogBB : PostLoopBB);

  // Create epilog
  if (EpilogLen) {
    Builder.SetInsertPoint(EpilogBB);

    LoopCount = Builder.CreatePHI(CopyLenType, 2, "loop-cnt");
    cast<PHINode>(LoopCount)->addIncoming(EpilogLen, LoopBody);
    DstIndex = Builder.CreatePHI(DstAddr->getType(), 2, "dst-addr");
    cast<PHINode>(DstIndex)->addIncoming(DstIndexTmp, LoopBody);

    Value *VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                        {LoopCount, SEW, LMUL});

    Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                            {Val, DstIndex, VL});

    NewLoopCount = Builder.CreateSub(LoopCount, VL);

    DstIndexTmp = Builder.CreateGEP(Int8Type, DstIndex, VL);

    cast<PHINode>(LoopCount)->addIncoming(NewLoopCount, EpilogBB);
    cast<PHINode>(DstIndex)->addIncoming(DstIndexTmp, EpilogBB);

    Builder.CreateCondBr(Builder.CreateICmpNE(NewLoopCount, Zero), EpilogBB,
                         PostLoopBB);
  }
}

bool RISCVLateCodeGenPrepare::expandMemIntrinsic(MemIntrinsic *MI) {

  switch (MI->getIntrinsicID()) {
  case Intrinsic::memcpy: {
    if (auto *CI = dyn_cast<ConstantInt>(MI->getLength())) {
      // If Copy length within MinCopySize, then use scalar load and store.
      if (CI->getZExtValue() < MinCopySize)
        return false;
      if (CI->getZExtValue() <= UnrollThreshold) {
        expandMemCpyKnownSize(cast<MemCpyInst>(MI));
        return true;
      }
    }

    if (MemAlignOpt && ST->hasKnownDLen())
      expandMemCpyUnknownSizewithAlign(cast<MemCpyInst>(MI));
    else
      expandMemCpyUnknownSize(cast<MemCpyInst>(MI));

    break;
  }
  case Intrinsic::memset: {
    if (auto *CI = dyn_cast<ConstantInt>(MI->getLength())) {
      // If Copy length within MinCopySize, then use scalar load and store.
      if (CI->getZExtValue() < MinCopySize)
        return false;
      if (CI->getZExtValue() <= UnrollThreshold) {
        expandMemSetKnownSize(cast<MemSetInst>(MI));
        return true;
      }
    }

    if (MemAlignOpt && ST->hasKnownDLen())
      expandMemSetUnknownSizeAligned(cast<MemSetInst>(MI));
    else
      expandMemSetUnknownSize(cast<MemSetInst>(MI));

    break;
  }
  case Intrinsic::memmove: {
    if (auto *CI = dyn_cast<ConstantInt>(MI->getLength())) {
      // If Copy length within MinCopySize, then use scalar load and store.
      if (CI->getZExtValue() < MinCopySize)
        return false;
      if (CI->getZExtValue() <= UnrollThreshold) {
        expandMemmoveKnownSize(cast<MemMoveInst>(MI));
        return true;
      }
    }

    if (MemAlignOpt && ST->hasKnownDLen())
      expandMemmoveUnknownSizeAligned(cast<MemMoveInst>(MI));
    else
      expandMemmoveUnknownSize(cast<MemMoveInst>(MI));

    break;
  }
  default:
    return false;
  }

  return true;
}

// Look for (vp_mul (vp_zext X), (splat Y)) where vp_zext is doubling the
// element size and Y is known to be zero extended. Replace with
// (vp_mul (vp_zext X), (vp_zext (splat (trunc Y)))) to encourage the use of
// widening multiply. Only do this if the splat isn't already in the same
// basic block as the vp_mul. We put the vp_zext with the vp_mul and the new
// splat and trunc in basic block with the original splat.
bool RISCVLateCodeGenPrepare::visitIntrinsicInst(IntrinsicInst &I) {
  if (!ST->hasVInstructions())
    return false;

  if (I.getIntrinsicID() != Intrinsic::vp_mul)
    return false;

  Value *LHS = I.getArgOperand(0);
  Value *RHS = I.getArgOperand(1);

  Value *Mask = cast<VPIntrinsic>(I).getMaskParam();
  Value *VL = cast<VPIntrinsic>(I).getVectorLengthParam();

  // Canonicalize an Intrinsic operand to the LHS.
  if (isa<IntrinsicInst>(RHS))
    std::swap(LHS, RHS);

  // LHS should be a vp_zext.
  Value *ZExtSrc;
  if (!match(LHS, m_Intrinsic<Intrinsic::vp_zext>(
                      m_Value(ZExtSrc), m_Specific(Mask), m_Specific(VL))))
    return false;

  // vp_zext should be in the same basic block as the vp_mul.
  if (cast<Instruction>(LHS)->getParent() != I.getParent())
    return false;

  // The extend should be doubling.
  unsigned Size = I.getType()->getScalarSizeInBits();
  unsigned SrcSize = ZExtSrc->getType()->getScalarSizeInBits();
  if (Size != SrcSize * 2)
    return false;

  // Types should be legal.
  if (Size != 64 && Size != 32 && Size != 16)
    return false;
  if (Size == 64 && !ST->hasVInstructionsI64())
    return false;

  // RHS should be a splat shuffle.
  Value *SplatVal;
  if (!match(RHS, m_OneUse(m_Shuffle(
                      m_InsertElt(m_Undef(), m_Value(SplatVal), m_ZeroInt()),
                      m_Undef(), m_ZeroMask()))))
    return false;

  auto *RHSI = cast<Instruction>(RHS);

  // Splat should be in another basic block.
  if (RHSI->getParent() == I.getParent())
    return false;

  // Make sure we can freely truncate the value.
  KnownBits Known = computeKnownBits(SplatVal, *DL);
  if (Known.countMaxActiveBits() > SrcSize)
    return false;

  VectorType *VecTy = cast<VectorType>(ZExtSrc->getType());
  Type *ScalarTy = VecTy->getElementType();
  IRBuilder<> Builder(RHSI);
  Value *Splat = Builder.CreateVectorSplat(
      VecTy->getElementCount(), Builder.CreateTrunc(SplatVal, ScalarTy));

  Builder.SetInsertPoint(&I);
  Value *NewZExt = Builder.CreateIntrinsic(
      Intrinsic::vp_zext, {I.getType(), Splat->getType()}, {Splat, Mask, VL});

  RHSI->replaceAllUsesWith(NewZExt);
  RHSI->eraseFromParent();

  return true;
}

bool RISCVLateCodeGenPrepare::visitMemIntrinsic(MemIntrinsic &MI) {
  Function &F = *MI.getFunction();
  if (!F.hasFnAttribute(Attribute::NoImplicitFloat) && !F.hasOptSize() &&
      ST->hasVInstructions() && MemToRVVOpt)
    MemCalls.push_back(&MI);

  return false;
}

void RISCVLateCodeGenPrepare::getMemToRVVConfig() {
  unsigned MemLMULLocal = ST->getMemToRVVLMUL();
  if (MemLMUL.getNumOccurrences())
    MemLMULLocal = MemLMUL;

  if (MemLMULLocal != 8 && MemLMULLocal != 4 && MemLMULLocal != 2 &&
      MemLMULLocal != 1) {
    errs() << "Invalid LMUL for memcpy/memmove/memset expansion,"
           << "set to default value: 8.\n";
    MemLMULLocal = 8;
  }

  MemcpyLMUL = MemLMULLocal;
  if (PreferMemcpyLMUL.getNumOccurrences()) {
    if (isPowerOf2_64(PreferMemcpyLMUL) && PreferMemcpyLMUL <= 8)
      MemcpyLMUL = PreferMemcpyLMUL;
    else
      errs()
          << "Invalid LMUL for memcpy expansion, set to default lmul value.\n";
  }

  MemsetLMUL = MemLMULLocal;
  if (PreferMemsetLMUL.getNumOccurrences()) {
    if (isPowerOf2_64(PreferMemsetLMUL) && PreferMemsetLMUL <= 8)
      MemsetLMUL = PreferMemsetLMUL;
    else
      errs()
          << "Invalid LMUL for memset expansion, set to default lmul value.\n";
  }

  MemmoveLMUL = MemLMULLocal;
  if (PreferMemmoveLMUL.getNumOccurrences()) {
    if (isPowerOf2_64(PreferMemmoveLMUL) && PreferMemmoveLMUL <= 8)
      MemmoveLMUL = PreferMemmoveLMUL;
    else
      errs()
          << "Invalid LMUL for memmove expansion, set to default lmul value.\n";
  }

  if (ST->isSiFiveMallardCPU()) {
    unsigned CacheLineSize = ST->getCacheLineSize();
    AlignBytes = CacheLineSize ? CacheLineSize : 64;
  } else if (ST->hasKnownDLen())
    AlignBytes = ST->getDLen() / 8;

  // This is old threshold 8 * MemLMULLocal * MinVLenInBytes - 1
  UnrollThreshold = 8 * MemLMULLocal * (ST->getRealMinVLen() / 8) - 1;
  // FIXME: Tune this threshold for each sifive cpu.
  if ((ST->getProcFamily() == RISCVSubtarget::SiFiveP400 ||
       ST->getProcFamily() == RISCVSubtarget::SiFiveLeopard) && ST->hasKnownDLen())
    UnrollThreshold = 4 * (ST->getDLen() / 8);

  // TODO: Maybe need specific options for memset/memcpy/memmove?
  if (PreferUnrollThreshold.getNumOccurrences())
    UnrollThreshold = PreferUnrollThreshold;
}

bool RISCVLateCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  auto &TM = TPC->getTM<RISCVTargetMachine>();

  ST = TM.getSubtargetImpl(F);

  DL = &F.getParent()->getDataLayout();

  MemCalls.clear();

  if (ST->hasVInstructions())
    getMemToRVVConfig();

  bool MadeChange = false;
  for (auto &BB : F)
    for (Instruction &I : llvm::make_early_inc_range(BB))
      MadeChange |= visit(I);

  for (MemIntrinsic *MemCall : MemCalls)
    MadeChange |= expandMemIntrinsic(MemCall);

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(RISCVLateCodeGenPrepare, DEBUG_TYPE, PASS_NAME, false,
                      false)
INITIALIZE_PASS_END(RISCVLateCodeGenPrepare, DEBUG_TYPE, PASS_NAME, false,
                    false)

char RISCVLateCodeGenPrepare::ID = 0;

FunctionPass *llvm::createRISCVLateCodeGenPreparePass() {
  return new RISCVLateCodeGenPrepare();
}
