//===----- SiFive_RISCVLateCodeGenPrepare.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a RISCV specific version of CodeGenPrepare.
// It munges the code in the input function to better prepare it for
// SelectionDAG-based code generation. This works around limitations in it's
// basic-block-at-a-time approach.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "riscv-late-codegenprepare"
#define PASS_NAME "RISCV Late CodeGenPrepare"

#define CREATE_BASIC_BLOCKS_WO_BACKWARD(NAME)                                  \
  BasicBlock *PreLoopBB = M->getParent();                                      \
  BasicBlock *PostLoopBB = PreLoopBB->splitBasicBlock(M, NAME "-post-loop");   \
  BasicBlock *ForwardLoopBB =                                                  \
      BasicBlock::Create(PreLoopBB->getContext(), NAME "-forward-loop",        \
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

namespace {

class RISCVLateCodeGenPrepare
    : public FunctionPass,
      public InstVisitor<RISCVLateCodeGenPrepare, bool> {
  const DataLayout *DL;
  const RISCVSubtarget *ST;

  SmallVector<MemIntrinsic *, 4> MemCalls;

public:
  static char ID;

  RISCVLateCodeGenPrepare() : FunctionPass(ID) {}

  StringRef getPassName() const override { return PASS_NAME; }

  bool runOnFunction(Function &F) override;

  static constexpr unsigned MinCopySize = 64;
  static constexpr unsigned MaxUnrollTimes = 8;

  bool visitInstruction(Instruction &I) { return false; }
  bool visitZExtInst(ZExtInst &I);
  bool optimizeZExtWUses(ZExtInst &I);
  bool visitAnd(BinaryOperator &BO);
  bool optimizeAndUses(BinaryOperator &BO);
  bool visitICmp(ICmpInst &ICmp);
  bool visitMemIntrinsic(MemIntrinsic &MI);
  bool expandMemIntrinsic(MemIntrinsic *MI);
  void expandMemCpyUnknownSize(MemCpyInst *MCI);
  void expandMemCpyUnknownSizewithAlign(MemCpyInst *MCI);
  void expandMemCpyKnownSize(MemCpyInst *MCI, unsigned AVL);
  void expandMemSetUnknownSize(MemSetInst *MSI);
  void expandMemSetUnknownSizeAligned(MemSetInst *MSI);
  void expandMemSetKnownSize(MemSetInst *MSI);
  void createMemsetLoopBody(BasicBlock *LoopBody, BasicBlock *PreLoopBB,
                            BasicBlock *PostLoopBB, Value *Val, Value *DstAddr,
                            Value *CopyLen, uint64_t UnrollCount = 1);
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
      InsertedZExt = new ZExtInst(Src, I.getType(), "", &*InsertPt);
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
      InsertedAnd = BinaryOperator::CreateAnd(BO.getOperand(0),
                                              BO.getOperand(1), "", &*InsertPt);
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
  Value *SrcAddr = M->getRawSource();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();
  unsigned SrcAS = SrcAddr->getType()->getPointerAddressSpace();
  unsigned DstAS = DstAddr->getType()->getPointerAddressSpace();

  BasicBlock *PreLoopBB = M->getParent();
  BasicBlock *PostLoopBB =
      PreLoopBB->splitBasicBlock(M, "post-loop-memcpy-expansion");
  BasicBlock *LoopBB =
      BasicBlock::Create(PreLoopBB->getContext(), "loop-memcpy-expansion",
                         PreLoopBB->getParent(), PostLoopBB);

  // We only deal with 8-bits width of memory at a time.
  Type *Int8Type = Type::getInt8Ty(PreLoopBB->getContext());
  // Initial vector type for <vscale x 64 x i8>, LMUL=8, SEW=8.
  ScalableVectorType *VTy =
      ScalableVectorType::get(Int8Type, RISCV::RVVBitsPerBlock);
  Type *CopyLenType = CopyLen->getType();

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

  IRBuilder<> Builder(PreLoopBB->getTerminator());
  Builder.CreateBr(LoopBB);
  PreLoopBB->getTerminator()->eraseFromParent();

  Builder.SetInsertPoint(LoopBB);
  PHINode *LoopIndex = Builder.CreatePHI(CopyLenType, 2, "loop-index");
  LoopIndex->addIncoming(CopyLen, PreLoopBB);
  PHINode *SrcIndex = Builder.CreatePHI(SrcAddr->getType(), 2, "src-addr");
  SrcIndex->addIncoming(SrcAddr, PreLoopBB);
  PHINode *DstIndex = Builder.CreatePHI(DstAddr->getType(), 2, "dst-addr");
  DstIndex->addIncoming(DstAddr, PreLoopBB);

  // Set SEW to 8 bits.
  Value *SEW = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  // Set LMUL to 8 registers.
  Value *LMUL = ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(8, false));

  Value *VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                      {LoopIndex, SEW, LMUL});
  Value *SrcCast =
      Builder.CreatePointerCast(SrcIndex, PointerType::get(VTy, SrcAS));
  Value *Load =
      Builder.CreateIntrinsic(Intrinsic::riscv_vle, {VTy, CopyLenType},
                              {UndefValue::get(VTy), SrcCast, VL});
  Value *DstCast =
      Builder.CreatePointerCast(DstIndex, PointerType::get(VTy, DstAS));
  Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                          {Load, DstCast, VL});

  Value *SrcGEP = Builder.CreateGEP(Int8Type, SrcIndex, VL);
  SrcIndex->addIncoming(SrcGEP, LoopBB);

  Value *DstGEP = Builder.CreateGEP(Int8Type, DstIndex, VL);
  DstIndex->addIncoming(DstGEP, LoopBB);

  Value *NewIndex = Builder.CreateSub(LoopIndex, VL);
  LoopIndex->addIncoming(NewIndex, LoopBB);

  IntegerType *ILengthType = cast<IntegerType>(CopyLenType);
  ConstantInt *Zero = ConstantInt::get(ILengthType, 0U);
  Builder.CreateCondBr(Builder.CreateICmpUGT(NewIndex, Zero), LoopBB,
                       PostLoopBB);
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemSetUnknownSize(MemSetInst *M) {
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memset")

  Value *Val = M->getValue();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();

  createMemsetLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, Val, DstAddr,
                       CopyLen);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemCpyKnownSize(MemCpyInst *M,
                                                    unsigned AVL) {
  Value *SrcAddr = M->getRawSource();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();
  unsigned SrcAS = SrcAddr->getType()->getPointerAddressSpace();
  unsigned DstAS = DstAddr->getType()->getPointerAddressSpace();

  Type *Int8Type = Type::getInt8Ty(M->getParent()->getContext());
  Type *CopyLenType = CopyLen->getType();
  ScalableVectorType *VTy =
      ScalableVectorType::get(Int8Type, RISCV::RVVBitsPerBlock);

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

  unsigned MinVLenInBytes = ST->getRealMinVLen() / 8;
  // The unroll count is equal to AVL / (MinVLen * LMUL).
  unsigned UnrollCount = divideCeil(AVL, MinVLenInBytes * 8);

  IRBuilder<> Builder(M);
  Value *SEW = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  Value *LMUL = ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(8, false));

  while (UnrollCount != 0) {
    Value *VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                        {CopyLen, SEW, LMUL});
    Value *SrcCast =
        Builder.CreatePointerCast(SrcAddr, PointerType::get(VTy, SrcAS));
    Value *Load =
        Builder.CreateIntrinsic(Intrinsic::riscv_vle, {VTy, CopyLenType},
                                {UndefValue::get(VTy), SrcCast, VL});

    Value *DstCast =
        Builder.CreatePointerCast(DstAddr, PointerType::get(VTy, DstAS));
    Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                            {Load, DstCast, VL});

    // Last time, we don't need to update address and copy length.
    if (UnrollCount != 1) {
      SrcAddr = Builder.CreateGEP(Int8Type, SrcAddr, VL);
      DstAddr = Builder.CreateGEP(Int8Type, DstAddr, VL);
      CopyLen = Builder.CreateSub(CopyLen, VL);
    }
    UnrollCount--;
  }

  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemSetKnownSize(MemSetInst *M) {
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memset")

  Value *Val = M->getValue();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();

  auto *CI = dyn_cast<ConstantInt>(CopyLen);
  unsigned UnrollCount = divideCeil(CI->getZExtValue(), ST->getRealMinVLen());

  createMemsetLoopBody(ForwardLoopBB, PreLoopBB, PostLoopBB, Val, DstAddr,
                       CopyLen, UnrollCount);

  PreLoopBB->getTerminator()->eraseFromParent();
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemCpyUnknownSizewithAlign(MemCpyInst *M) {
  Value *SrcAddr = M->getRawSource();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();
  unsigned SrcAS = SrcAddr->getType()->getPointerAddressSpace();
  unsigned DstAS = DstAddr->getType()->getPointerAddressSpace();

  BasicBlock *PreLoopBB = M->getParent();
  BasicBlock *PostLoopBB =
      PreLoopBB->splitBasicBlock(M, "post-loop-memcpy-expansion");
  BasicBlock *LoopBB =
      BasicBlock::Create(PreLoopBB->getContext(), "loop-memcpy-expansion",
                         PreLoopBB->getParent(), PostLoopBB);

  Type *Int8Type = Type::getInt8Ty(PreLoopBB->getContext());
  ScalableVectorType *VTy =
      ScalableVectorType::get(Int8Type, RISCV::RVVBitsPerBlock);
  Type *CopyLenType = CopyLen->getType();
  IntegerType *ILengthType = cast<IntegerType>(CopyLenType);

  Value *SEW = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  Value *LMUL = ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(8, false));

  unsigned AlignBytes = ST->getDLen() / 8;

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

  IRBuilder<> Builder(PreLoopBB->getTerminator());

  Value *Addr = Builder.CreatePtrToInt(SrcAddr, ILengthType);
  Value *And =
      Builder.CreateAnd(Addr, ConstantInt::get(ILengthType, AlignBytes - 1));

  Value *DLenElement =
      Builder.CreateSub(ConstantInt::get(ILengthType, AlignBytes), And);

  Value *Cmp = Builder.CreateICmpUGT(DLenElement, CopyLen);
  Value *AlignLen =
      Builder.CreateSelect(Cmp, CopyLen, DLenElement, "length.select");
  Value *AlignVL = Builder.CreateIntrinsic(
      Intrinsic::riscv_vsetvli, {CopyLenType}, {AlignLen, SEW, LMUL});
  Value *SrcCast =
      Builder.CreatePointerCast(SrcAddr, PointerType::get(VTy, SrcAS));
  Value *Load =
      Builder.CreateIntrinsic(Intrinsic::riscv_vle, {VTy, CopyLenType},
                              {UndefValue::get(VTy), SrcCast, AlignVL});
  Value *DstCast =
      Builder.CreatePointerCast(DstAddr, PointerType::get(VTy, DstAS));
  Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                          {Load, DstCast, AlignVL});

  Value *AlignSrcGEP = Builder.CreateGEP(Int8Type, SrcAddr, AlignVL);
  Value *AlignDstGEP = Builder.CreateGEP(Int8Type, DstAddr, AlignVL);
  Value *NewCopyLen = Builder.CreateSub(CopyLen, AlignVL);

  ConstantInt *Zero = ConstantInt::get(ILengthType, 0U);
  Builder.CreateCondBr(Builder.CreateICmpNE(NewCopyLen, Zero),
                       LoopBB, PostLoopBB);
  PreLoopBB->getTerminator()->eraseFromParent();

  Builder.SetInsertPoint(LoopBB);
  PHINode *LoopIndex = Builder.CreatePHI(CopyLenType, 2, "loop-index");
  LoopIndex->addIncoming(NewCopyLen, PreLoopBB);
  PHINode *SrcIndex = Builder.CreatePHI(SrcAddr->getType(), 2, "src-addr");
  SrcIndex->addIncoming(AlignSrcGEP, PreLoopBB);
  PHINode *DstIndex = Builder.CreatePHI(DstAddr->getType(), 2, "dst-addr");
  DstIndex->addIncoming(AlignDstGEP, PreLoopBB);

  Value *LoopVL = Builder.CreateIntrinsic(
      Intrinsic::riscv_vsetvli, {CopyLenType}, {LoopIndex, SEW, LMUL});
  SrcCast = Builder.CreatePointerCast(SrcIndex, PointerType::get(VTy, SrcAS));

  Load = Builder.CreateIntrinsic(Intrinsic::riscv_vle, {VTy, CopyLenType},
                                 {UndefValue::get(VTy), SrcCast, LoopVL});
  DstCast = Builder.CreatePointerCast(DstIndex, PointerType::get(VTy, DstAS));
  Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                          {Load, DstCast, LoopVL});

  Value *SrcGEP = Builder.CreateGEP(Int8Type, SrcIndex, LoopVL);
  SrcIndex->addIncoming(SrcGEP, LoopBB);

  Value *DstGEP = Builder.CreateGEP(Int8Type, DstIndex, LoopVL);
  DstIndex->addIncoming(DstGEP, LoopBB);

  Value *NewIndex = Builder.CreateSub(LoopIndex, LoopVL);
  LoopIndex->addIncoming(NewIndex, LoopBB);

  Builder.CreateCondBr(Builder.CreateICmpUGT(NewIndex, Zero),
                       LoopBB, PostLoopBB);
  M->eraseFromParent();
}

void RISCVLateCodeGenPrepare::expandMemSetUnknownSizeAligned(MemSetInst *M) {
  CREATE_BASIC_BLOCKS_WO_BACKWARD("memset")

  Value *Val = M->getValue();
  Value *DstAddr = M->getRawDest();
  Value *CopyLen = M->getLength();

  unsigned DstAS = DstAddr->getType()->getPointerAddressSpace();
  unsigned AlignBytes = ST->getDLen() / 8;

  Type *Int8Type = Type::getInt8Ty(PreLoopBB->getContext());
  ScalableVectorType *VTy =
      ScalableVectorType::get(Int8Type, RISCV::RVVBitsPerBlock);
  Type *CopyLenType = CopyLen->getType();
  IntegerType *ILengthType = cast<IntegerType>(CopyLenType);

  Value *SEW = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  Value *LMUL = ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(8, false));

  IRBuilder<> Builder(PreLoopBB->getTerminator());

  Value *DLenElement = Builder.CreateSub(
      ConstantInt::get(ILengthType, AlignBytes),
      Builder.CreateAnd(Builder.CreatePtrToInt(DstAddr, ILengthType),
                        ConstantInt::get(ILengthType, AlignBytes - 1)));

  Value *Cmp = Builder.CreateICmpUGT(DLenElement, CopyLen);
  Value *AlignLen =
      Builder.CreateSelect(Cmp, CopyLen, DLenElement, "length.select");

  Value *TmpVal =
      Builder.CreateIntrinsic(Intrinsic::riscv_vmv_v_x, {VTy, CopyLenType},
                              {UndefValue::get(VTy), Val, CopyLen});

  Value *VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                      {AlignLen, SEW, LMUL});
  Value *DstCast =
      Builder.CreatePointerCast(DstAddr, PointerType::get(VTy, DstAS));
  Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                          {TmpVal, DstCast, VL});

  CopyLen = Builder.CreateSub(CopyLen, VL);

  DstAddr = Builder.CreateGEP(Int8Type, DstAddr, VL);

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
  // Initial vector type for <vscale x 64 x i8>, LMUL=8, SEW=8.
  ScalableVectorType *VTy =
      ScalableVectorType::get(Int8Type, RISCV::RVVBitsPerBlock);
  Type *CopyLenType = CopyLen->getType();

  // Set SEW to 8 bits.
  Value *SEW = ConstantInt::get(CopyLenType, RISCVVType::encodeSEW(8));
  // Set LMUL to 8 registers.
  Value *LMUL = ConstantInt::get(CopyLenType, RISCVVType::encodeLMUL(8, false));

  bool FullyUnrolled = false;
  Value *EpilogLen = nullptr;
  // Max copy size we can deal with each round: DataVLen * LMUL
  int64_t MaxCopySize = (ST->getRealMinVLen() / 8) * 8;
  int64_t KnownCurrentLen = -MaxCopySize;
  if (auto *CI = dyn_cast<ConstantInt>(CopyLen)) {
    KnownCurrentLen = CI->getZExtValue();
    uint64_t TotalCopiesNeeded = divideCeil(KnownCurrentLen, MaxCopySize);

    // UnrollCount must be smaller or equal to CopyLen / MaxCopySize,
    // otherwise there would be redundant instuctions generated.
    if (UnrollCount == TotalCopiesNeeded)
      FullyUnrolled = true;
    else if (UnrollCount > TotalCopiesNeeded) {
      createMemsetLoopBody(LoopBody, PreLoopBB, PostLoopBB, Val, DstAddr,
                           CopyLen, UnrollCount - 1);
      return;
    } else if (TotalCopiesNeeded % UnrollCount) {
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

  Value *LoopCount = CopyLen;
  Value *VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                                      {LoopCount, SEW, LMUL});
  // If it already copied(broadcasted) the scalar value into a vector in
  // previous blocks, then we can use it directly, otherwise we have to do it.
  if (!dyn_cast<ScalableVectorType>(Val->getType()))
    Val = Builder.CreateIntrinsic(Intrinsic::riscv_vmv_v_x, {VTy, CopyLenType},
                                  {UndefValue::get(VTy), Val, VL});

  Builder.CreateBr(LoopBody);

  Builder.SetInsertPoint(LoopBody);

  Value *DstIndex = DstAddr;
  if (!FullyUnrolled) {
    LoopCount = Builder.CreatePHI(CopyLenType, 2, "loop-cnt");
    cast<PHINode>(LoopCount)->addIncoming(CopyLen, PreLoopBB);
    DstIndex = Builder.CreatePHI(DstAddr->getType(), 2, "dst-addr");
    cast<PHINode>(DstIndex)->addIncoming(DstAddr, PreLoopBB);
  }

  unsigned DstAS = DstAddr->getType()->getPointerAddressSpace();

  Value *NewLoopCount = LoopCount;
  Value *DstIndexTmp = DstIndex;
  SmallVector<Value *> DstIndices;
  VL = Builder.CreateIntrinsic(Intrinsic::riscv_vsetvli, {CopyLenType},
                               {NewLoopCount, SEW, LMUL});
  uint64_t TmpUC = UnrollCount;
  while (TmpUC--) {
    DstIndices.push_back(DstIndexTmp);
    if (KnownCurrentLen != -MaxCopySize)
      KnownCurrentLen -= MaxCopySize;
    if (TmpUC != 0 && UnrollCount != 1)
      DstIndexTmp = Builder.CreateGEP(Int8Type, DstIndexTmp, VL);
  }
  while (++TmpUC < UnrollCount) {
    Value *DstCast = Builder.CreatePointerCast(DstIndices[TmpUC],
                                               PointerType::get(VTy, DstAS));
    if (TmpUC == UnrollCount - 1 &&
        (KnownCurrentLen != -MaxCopySize && KnownCurrentLen < 0))
      VL = Builder.CreateIntrinsic(
          Intrinsic::riscv_vsetvli, {CopyLenType},
          {ConstantInt::get(CopyLenType, KnownCurrentLen + MaxCopySize), SEW,
           LMUL});
    Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                            {Val, DstCast, VL});
  }

  DstIndexTmp = Builder.CreateGEP(Int8Type, DstIndexTmp, VL);

  if (!FullyUnrolled) {
    NewLoopCount = Builder.CreateSub(
        NewLoopCount, ConstantInt::get(CopyLenType, UnrollCount * MaxCopySize));
    cast<PHINode>(DstIndex)->addIncoming(DstIndexTmp, LoopBody);
    cast<PHINode>(LoopCount)->addIncoming(NewLoopCount, LoopBody);
  }

  IntegerType *ILengthType = cast<IntegerType>(CopyLenType);
  ConstantInt *Zero = ConstantInt::get(ILengthType, 0U);

  if (FullyUnrolled)
    Builder.CreateBr(PostLoopBB);
  else
    Builder.CreateCondBr(Builder.CreateICmpSGT(NewLoopCount, Zero), LoopBody,
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

    Value *DstCast =
        Builder.CreatePointerCast(DstIndexTmp, PointerType::get(VTy, DstAS));
    Builder.CreateIntrinsic(Intrinsic::riscv_vse, {VTy, CopyLenType},
                            {Val, DstCast, VL});

    NewLoopCount = Builder.CreateSub(LoopCount, VL);

    DstIndexTmp = Builder.CreateGEP(Int8Type, DstIndexTmp, VL);

    cast<PHINode>(LoopCount)->addIncoming(NewLoopCount, EpilogBB);
    cast<PHINode>(DstIndex)->addIncoming(DstIndexTmp, EpilogBB);

    Builder.CreateCondBr(Builder.CreateICmpUGT(NewLoopCount, Zero), EpilogBB,
                         PostLoopBB);
  }
}

bool RISCVLateCodeGenPrepare::expandMemIntrinsic(MemIntrinsic *MI) {

  switch (MI->getIntrinsicID()) {
  case Intrinsic::memcpy: {
    if (auto *CI = dyn_cast<ConstantInt>(MI->getLength())) {
      unsigned MinVLenInBytes = ST->getRealMinVLen() / 8;

      // If Copy length within MinCopySize, then use scalar load and store.
      if (CI->getZExtValue() < MinCopySize)
        return false;
      // We only deal with the size of VLen * LMUL * MaxUnrollTimes.
      if (CI->getZExtValue() < (MinVLenInBytes * 8 * MaxUnrollTimes)) {
        expandMemCpyKnownSize(cast<MemCpyInst>(MI), CI->getZExtValue());
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
      unsigned MinVLenInBytes = ST->getRealMinVLen() / 8;

      // If Copy length within MinCopySize, then use scalar load and store.
      if (CI->getZExtValue() < MinCopySize)
        return false;
      // We only deal with the size of VLen * LMUL * MaxUnrollTimes.
      if (CI->getZExtValue() < (MinVLenInBytes * 8 * MaxUnrollTimes)) {
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
  case Intrinsic::memmove:
  default:
    return false;
  }

  return true;
}

bool RISCVLateCodeGenPrepare::visitMemIntrinsic(MemIntrinsic &MI) {
  Function &F = *MI.getFunction();
  if (!F.hasFnAttribute(Attribute::NoImplicitFloat) && !F.hasOptSize() &&
      ST->hasVInstructions() && MemToRVVOpt)
    MemCalls.push_back(&MI);

  return false;
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
