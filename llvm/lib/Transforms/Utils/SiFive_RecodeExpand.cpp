//===---------------------- SiFive_RecodeExpand.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands Neon intrinsics to LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SiFive_RecodeExpand.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

static cl::opt<bool> DisableSiFiveRecode(
    "disable-sifive-recode",
    cl::desc("Disable expand aarch64 NEON into LLVM IR or RISC-V Intrinsic"),
    cl::Hidden, cl::init(false));

static CallInst *toScalableVector(const TargetTransformInfo &TTI,
                                  IRBuilder<> &Builder, Value *Vec) {
  Type *ScalableVecTy = TTI.getScalableVectorFromFixed(Vec->getType());
  return Builder.CreateInsertVector(
      ScalableVecTy, PoisonValue::get(ScalableVecTy), Vec, Builder.getInt64(0));
}

static Value *widen(IRBuilder<> &Builder, Value *V, unsigned DesNumElements) {
  VectorType *SrcTy = cast<VectorType>(V->getType());
  PoisonValue *Des =
      PoisonValue::get(VectorType::get(SrcTy->getElementType(), DesNumElements,
                                       SrcTy->getElementCount().isScalable()));
  return Builder.CreateInsertVector(Des->getType(), Des, V,
                                    Builder.getInt64(0));
}

static Value *glue(IRBuilder<> &Builder, SmallVector<Value *, 4> Src) {
  assert(Src.size() != 0);
  SmallVector<Value *, 4> Temp;
  while (Src.size() != 1) {
    while (1 < Src.size()) {
      unsigned Src0NumElements =
          cast<FixedVectorType>(Src[0]->getType())->getNumElements();
      unsigned Src1NumElements =
          cast<FixedVectorType>(Src[1]->getType())->getNumElements();
      if (Src0NumElements > Src1NumElements) {
        Src[1] = widen(Builder, Src[1], Src0NumElements);
      } else if (Src0NumElements < Src1NumElements) {
        Src[0] = widen(Builder, Src[0], Src1NumElements);
      }
      std::vector<int> Mask(Src0NumElements + Src1NumElements);
      std::iota(Mask.begin(), Mask.end(), 0);
      Temp.push_back(Builder.CreateShuffleVector(Src[0], Src[1], Mask));
      Src.erase(Src.begin(), Src.begin() + 2);
    }
    if (Src.size() != 0)
      Temp.push_back(Src[0]);
    Src = std::move(Temp);
    Temp.clear();
  }
  return Src[0];
}

static Value *narrow(IRBuilder<> &Builder, Value *V, unsigned DesNumElements) {
  VectorType *SrcTy = cast<VectorType>(V->getType());
  return Builder.CreateExtractVector(
      VectorType::get(SrcTy->getElementType(), DesNumElements,
                      SrcTy->getElementCount().isScalable()),
      V, Builder.getInt64(0));
}

static SmallVector<int, 8> increasingSequenceByN(int Init, int N, size_t Size) {
  SmallVector<int, 8> Mask(Size);
  for (size_t i = 0; i != Size; ++i)
    Mask[i] = Init + i * N;
  return Mask;
}

static Value *smaxZero(IRBuilder<> &Builder, Value *Src) {
  assert(isa<FixedVectorType>(Src->getType()));
  return Builder.CreateIntrinsic(Intrinsic::smax, {Src->getType()},
                                 {Src, ConstantInt::get(Src->getType(), 0)});
}

static Value *createFclass(const TargetTransformInfo &TTI, IRBuilder<> &Builder,
                           Value *V, ConstantInt *VL) {
  Value *ScalableV = toScalableVector(TTI, Builder, V);
  Value *Fclass = Builder.CreateIntrinsic(
      Intrinsic::riscv_vfclass, {ScalableV->getType(), VL->getType()},
      {PoisonValue::get(ScalableVectorType::getInteger(
           cast<ScalableVectorType>(ScalableV->getType()))),
       ScalableV, VL});
  return Builder.CreateExtractVector(
      FixedVectorType::getInteger(cast<FixedVectorType>(V->getType())), Fclass,
      Builder.getInt64(0));
}

static Value *expandShl(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                        bool IsSigned) {
  auto toVectorIfScalar = [&](Value *V) {
    if (V->getType()->isVectorTy())
      return V;
    return Builder.CreateVectorSplat(
        cast<FixedVectorType>(Op0->getType())->getElementCount(), V);
  };
  Type *VecTy = Op0->getType();
  Type *Ty = Op1->getType();
  unsigned Size = Ty->getScalarSizeInBits();
  Value *RShiftAmount;
  if (Size == 8) {
    RShiftAmount = Builder.CreateSub(ConstantInt::get(Ty, 0), Op1);
  } else {
    // Only lower 8 bit is effective for Op1.
    Op1 = Builder.CreateAnd(Op1, 255);
    RShiftAmount = Builder.CreateSub(ConstantInt::get(Ty, 256), Op1);
  }
  // If Op1 is in [0, 127], we do Op0 << Op1.
  // If Op1 is in [128, 255], we do Op0 >> Op1.
  Value *IsRight = Builder.CreateICmpUGT(Op1, ConstantInt::get(Ty, 127));
  // Because LLVM IR only takes log2(Size) bit to do shift. If a shift amount is
  // greater than or equal to log2(Size), we need to handle it by ourselves.
  Value *MaxShift = ConstantInt::get(Ty, Size - 1);
  // For a left shift, when a shift amount is greater than or equal to
  // log2(Size), Op0 << Op1 is 0.
  Value *LShift = Builder.CreateSelect(
      Builder.CreateICmpUGT(Op1, MaxShift), ConstantInt::get(VecTy, 0),
      Builder.CreateShl(Op0, toVectorIfScalar(Op1)));
  Value *RShift;
  // For a right shift, the result is different because of signedness.
  if (IsSigned) {
    RShift = Builder.CreateAShr(
        Op0, toVectorIfScalar(Builder.CreateIntrinsic(
                 Intrinsic::umin, {Ty}, {RShiftAmount, MaxShift})));
  } else {
    // We split a lshr into two, and set the maximum shift amount for the second
    // lshr to (Size - 1). The result is same. But no conditional branch inside
    // if Ty is scalar type. It also uses 1 less vector instructions if Ty is
    // vector type.
    Op0 = Builder.CreateLShr(Op0, ConstantInt::get(VecTy, 1));
    RShift = Builder.CreateLShr(
        Op0, toVectorIfScalar(Builder.CreateIntrinsic(
                 Intrinsic::umin, {Ty},
                 {Builder.CreateSub(RShiftAmount, ConstantInt::get(Ty, 1)),
                  MaxShift})));
  }
  return Builder.CreateSelect(IsRight, RShift, LShift);
}

bool SiFiveRecodePass::requireExpand(IntrinsicInst *II) {
  switch (II->getIntrinsicID()) {
  case Intrinsic::aarch64_neon_abs:
  case Intrinsic::aarch64_neon_addp:
  case Intrinsic::aarch64_neon_fabd:
  case Intrinsic::aarch64_neon_facgt:
  case Intrinsic::aarch64_neon_faddp:
  case Intrinsic::aarch64_neon_faddv:
  case Intrinsic::aarch64_neon_fcvtas:
  case Intrinsic::aarch64_neon_fcvtau:
  case Intrinsic::aarch64_neon_fcvtms:
  case Intrinsic::aarch64_neon_fcvtmu:
  case Intrinsic::aarch64_neon_fcvtns:
  case Intrinsic::aarch64_neon_fcvtnu:
  case Intrinsic::aarch64_neon_fcvtps:
  case Intrinsic::aarch64_neon_fcvtpu:
  case Intrinsic::aarch64_neon_fcvtzs:
  case Intrinsic::aarch64_neon_fcvtzu:
  case Intrinsic::aarch64_neon_fmaxp:
  case Intrinsic::aarch64_neon_fmaxv:
  case Intrinsic::aarch64_neon_fminp:
  case Intrinsic::aarch64_neon_fminv:
  case Intrinsic::aarch64_neon_frecpe:
  case Intrinsic::aarch64_neon_frecps:
  case Intrinsic::aarch64_neon_frsqrte:
  case Intrinsic::aarch64_neon_frsqrts:
  case Intrinsic::aarch64_neon_ld1x2:
  case Intrinsic::aarch64_neon_ld1x3:
  case Intrinsic::aarch64_neon_ld1x4:
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_ld2lane:
  case Intrinsic::aarch64_neon_ld2r:
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_ld3lane:
  case Intrinsic::aarch64_neon_ld3r:
  case Intrinsic::aarch64_neon_ld4:
  case Intrinsic::aarch64_neon_ld4lane:
  case Intrinsic::aarch64_neon_ld4r:
  case Intrinsic::aarch64_neon_sabd:
  case Intrinsic::aarch64_neon_saddlp:
  case Intrinsic::aarch64_neon_saddlv:
  case Intrinsic::aarch64_neon_saddv:
  case Intrinsic::aarch64_neon_sdot:
  case Intrinsic::aarch64_neon_smax:
  case Intrinsic::aarch64_neon_smaxp:
  case Intrinsic::aarch64_neon_smaxv:
  case Intrinsic::aarch64_neon_smin:
  case Intrinsic::aarch64_neon_sminp:
  case Intrinsic::aarch64_neon_sminv:
  case Intrinsic::aarch64_neon_smull:
  case Intrinsic::aarch64_neon_sqadd:
  case Intrinsic::aarch64_neon_sqdmulh:
  case Intrinsic::aarch64_neon_sqrdmulh:
  case Intrinsic::aarch64_neon_sqrshrun:
  case Intrinsic::aarch64_neon_sqshlu:
  case Intrinsic::aarch64_neon_sqshrun:
  case Intrinsic::aarch64_neon_sqsub:
  case Intrinsic::aarch64_neon_sqxtun:
  case Intrinsic::aarch64_neon_sshl:
  case Intrinsic::aarch64_neon_st1x2:
  case Intrinsic::aarch64_neon_st1x3:
  case Intrinsic::aarch64_neon_st1x4:
  case Intrinsic::aarch64_neon_st2:
  case Intrinsic::aarch64_neon_st2lane:
  case Intrinsic::aarch64_neon_st3:
  case Intrinsic::aarch64_neon_st3lane:
  case Intrinsic::aarch64_neon_st4:
  case Intrinsic::aarch64_neon_st4lane:
  case Intrinsic::aarch64_neon_tbl1:
  case Intrinsic::aarch64_neon_tbl2:
  case Intrinsic::aarch64_neon_tbl3:
  case Intrinsic::aarch64_neon_tbl4:
  case Intrinsic::aarch64_neon_tbx1:
  case Intrinsic::aarch64_neon_tbx2:
  case Intrinsic::aarch64_neon_tbx3:
  case Intrinsic::aarch64_neon_tbx4:
  case Intrinsic::aarch64_neon_uabd:
  case Intrinsic::aarch64_neon_uaddlp:
  case Intrinsic::aarch64_neon_uaddlv:
  case Intrinsic::aarch64_neon_uaddv:
  case Intrinsic::aarch64_neon_udot:
  case Intrinsic::aarch64_neon_umax:
  case Intrinsic::aarch64_neon_umaxp:
  case Intrinsic::aarch64_neon_umaxv:
  case Intrinsic::aarch64_neon_umin:
  case Intrinsic::aarch64_neon_uminp:
  case Intrinsic::aarch64_neon_uminv:
  case Intrinsic::aarch64_neon_umull:
  case Intrinsic::aarch64_neon_uqadd:
  case Intrinsic::aarch64_neon_uqsub:
  case Intrinsic::aarch64_neon_ushl:
  case Intrinsic::aarch64_neon_vcvtfp2hf:
  case Intrinsic::aarch64_neon_vcvthf2fp:
  case Intrinsic::aarch64_neon_vsli:
  case Intrinsic::aarch64_neon_vsri:
    return true;
  }
  return false;
}

PreservedAnalyses SiFiveRecodePass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  if (DisableSiFiveRecode)
    return PreservedAnalyses::all();

  const TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  bool MadeChange = false;
  unsigned XLEN =
      TTI.isTypeLegal(IntegerType::get(F.getContext(), 64)) ? 64 : 32;
  const DataLayout &DL = F.getParent()->getDataLayout();
  for (Instruction &Inst : llvm::make_early_inc_range(instructions(F))) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(&Inst);
    if (II && requireExpand(II)) {
      MadeChange = true;
      IRBuilder<> Builder(II);

      switch (II->getIntrinsicID()) {
      case Intrinsic::aarch64_neon_abs: {
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            Intrinsic::abs, {II->getArgOperand(0)->getType()},
            {II->getArgOperand(0), Builder.getInt1(false)}));
        break;
      }
      case Intrinsic::aarch64_neon_addp:
      case Intrinsic::aarch64_neon_faddp: {
        Value *Concatenate =
            glue(Builder, {II->getArgOperand(0), II->getArgOperand(1)});
        unsigned DesVecNumElements =
            cast<FixedVectorType>(II->getType())->getNumElements();
        Value *Input[2];
        for (int i = 0; i != 2; ++i)
          Input[i] = Builder.CreateShuffleVector(
              Concatenate, increasingSequenceByN(i, 2, DesVecNumElements));
        if (II->getIntrinsicID() == Intrinsic::aarch64_neon_addp)
          II->replaceAllUsesWith(Builder.CreateAdd(Input[0], Input[1]));
        else
          II->replaceAllUsesWith(Builder.CreateFAdd(Input[0], Input[1]));
        break;
      }
      case Intrinsic::aarch64_neon_fabd: {
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            Intrinsic::fabs, {II->getArgOperand(0)->getType()},
            {Builder.CreateFSub(II->getArgOperand(0), II->getArgOperand(1))}));
        break;
      }
      case Intrinsic::aarch64_neon_facgt: {
        // If either operand is a NaN, return false and set invalid flag.
        CallInst *Abs0 = Builder.CreateIntrinsic(
            Intrinsic::fabs, {II->getArgOperand(0)->getType()},
            {II->getArgOperand(0)});
        CallInst *Abs1 = Builder.CreateIntrinsic(
            Intrinsic::fabs, {II->getArgOperand(1)->getType()},
            {II->getArgOperand(1)});
        II->replaceAllUsesWith(Builder.CreateSExt(
            Builder.CreateFCmpOGT(Abs0, Abs1), II->getType()));
        break;
      }
      case Intrinsic::aarch64_neon_faddv: {
        Value *Src = II->getArgOperand(0);
        unsigned VecNumElements =
            cast<FixedVectorType>(Src->getType())->getNumElements();
        if (VecNumElements == 4) {
          Value *P0 = Builder.CreateShuffleVector(Src, {0, 2});
          Value *P1 = Builder.CreateShuffleVector(Src, {1, 3});
          Src = Builder.CreateFAdd(P0, P1);
          VecNumElements /= 2;
        }
        assert(VecNumElements == 2);
        Value *P0 = Builder.CreateShuffleVector(Src, {0});
        Value *P1 = Builder.CreateShuffleVector(Src, {1});
        II->replaceAllUsesWith(Builder.CreateExtractElement(
            Builder.CreateFAdd(P0, P1), static_cast<uint64_t>(0)));
        break;
      }
      case Intrinsic::aarch64_neon_fcvtas:
      case Intrinsic::aarch64_neon_fcvtau:
      case Intrinsic::aarch64_neon_fcvtms:
      case Intrinsic::aarch64_neon_fcvtmu:
      case Intrinsic::aarch64_neon_fcvtns:
      case Intrinsic::aarch64_neon_fcvtnu:
      case Intrinsic::aarch64_neon_fcvtps:
      case Intrinsic::aarch64_neon_fcvtpu: {
        // a: rounding to nearest with ties to Away
        // m: rounding toward minus infinity
        // n: rounding to nearest with ties to even
        // p: rounding toward plus infinity
        // out of range -> clamp the value and set invalid flag
        // inf -> clamp the value and set invalid flag
        // nan -> 0, set invalid flag
        // Inexact flag may be set.
        Intrinsic::ID RoundID;
        Intrinsic::ID FPToI;
        switch (II->getIntrinsicID()) {
        default:
          llvm_unreachable("Unexpected intrinsic");
        case Intrinsic::aarch64_neon_fcvtas:
          RoundID = Intrinsic::round;
          FPToI = Intrinsic::fptosi_sat;
          break;
        case Intrinsic::aarch64_neon_fcvtau:
          RoundID = Intrinsic::round;
          FPToI = Intrinsic::fptoui_sat;
          break;
        case Intrinsic::aarch64_neon_fcvtms:
          RoundID = Intrinsic::floor;
          FPToI = Intrinsic::fptosi_sat;
          break;
        case Intrinsic::aarch64_neon_fcvtmu:
          RoundID = Intrinsic::floor;
          FPToI = Intrinsic::fptoui_sat;
          break;
        case Intrinsic::aarch64_neon_fcvtns:
          RoundID = Intrinsic::roundeven;
          FPToI = Intrinsic::fptosi_sat;
          break;
        case Intrinsic::aarch64_neon_fcvtnu:
          RoundID = Intrinsic::roundeven;
          FPToI = Intrinsic::fptoui_sat;
          break;
        case Intrinsic::aarch64_neon_fcvtps:
          RoundID = Intrinsic::ceil;
          FPToI = Intrinsic::fptosi_sat;
          break;
        case Intrinsic::aarch64_neon_fcvtpu:
          RoundID = Intrinsic::ceil;
          FPToI = Intrinsic::fptoui_sat;
          break;
        }
        CallInst *Rint = Builder.CreateIntrinsic(
            RoundID, {II->getArgOperand(0)->getType()}, {II->getArgOperand(0)});
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            FPToI, {II->getType(), Rint->getType()}, {Rint}));
        break;
      }
      case Intrinsic::aarch64_neon_fcvtzs:
      case Intrinsic::aarch64_neon_fcvtzu: {
        // rounding towards zero
        // out of range -> clamp the value and set invalid flag
        // inf -> clamp the value and set invalid flag
        // nan -> 0, set invalid flag
        // Inexact flag may be set.
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            II->getIntrinsicID() == Intrinsic::aarch64_neon_fcvtzs
                ? Intrinsic::fptosi_sat
                : Intrinsic::fptoui_sat,
            {II->getType(), II->getArgOperand(0)->getType()},
            {II->getArgOperand(0)}));
        break;
      }
      case Intrinsic::aarch64_neon_fmaxp:
      case Intrinsic::aarch64_neon_fminp:
      case Intrinsic::aarch64_neon_smaxp:
      case Intrinsic::aarch64_neon_sminp:
      case Intrinsic::aarch64_neon_umaxp:
      case Intrinsic::aarch64_neon_uminp: {
        Value *Concatenate =
            glue(Builder, {II->getArgOperand(0), II->getArgOperand(1)});
        unsigned DesVecNumElements =
            cast<FixedVectorType>(II->getType())->getNumElements();
        Value *Input[2];
        for (int i = 0; i != 2; ++i)
          Input[i] = Builder.CreateShuffleVector(
              Concatenate, increasingSequenceByN(i, 2, DesVecNumElements));
        Intrinsic::ID Op;
        switch (II->getIntrinsicID()) {
        default:
          llvm_unreachable("Unexpected intrinsic");
        case Intrinsic::aarch64_neon_fmaxp:
          Op = Intrinsic::aarch64_neon_fmax;
          break;
        case Intrinsic::aarch64_neon_fminp:
          Op = Intrinsic::aarch64_neon_fmin;
          break;
        case Intrinsic::aarch64_neon_smaxp:
          Op = Intrinsic::smax;
          break;
        case Intrinsic::aarch64_neon_sminp:
          Op = Intrinsic::smin;
          break;
        case Intrinsic::aarch64_neon_umaxp:
          Op = Intrinsic::umax;
          break;
        case Intrinsic::aarch64_neon_uminp:
          Op = Intrinsic::umin;
          break;
        }
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            Op, {II->getArgOperand(0)->getType()}, {Input[0], Input[1]}));
        break;
      }
      case Intrinsic::aarch64_neon_fmaxv:
      case Intrinsic::aarch64_neon_fminv: {
        Value *Src = II->getArgOperand(0);
        FixedVectorType *VecTy = cast<FixedVectorType>(Src->getType());
        unsigned VectorNumElements = VecTy->getNumElements();
        assert((VectorNumElements) % 2 == 0);
        Intrinsic::ID Op = II->getIntrinsicID() == Intrinsic::aarch64_neon_fmaxv
                               ? Intrinsic::aarch64_neon_fmax
                               : Intrinsic::aarch64_neon_fmin;
        Type *DesTy = FixedVectorType::get(VecTy->getElementType(), 1);
        Value *Des =
            Builder.CreateExtractVector(DesTy, Src, Builder.getInt64(0));
        for (unsigned i = 1; i != VectorNumElements; ++i)
          Des = Builder.CreateIntrinsic(
              Op, {Des->getType()},
              {Des,
               Builder.CreateExtractVector(DesTy, Src, Builder.getInt64(i))});
        II->replaceAllUsesWith(
            Builder.CreateExtractElement(Des, static_cast<uint64_t>(0)));
        break;
      }
      case Intrinsic::aarch64_neon_frecpe:
      case Intrinsic::aarch64_neon_frsqrte: {
        FixedVectorType *VecTy =
            cast<FixedVectorType>(II->getArgOperand(0)->getType());
        unsigned VecNumElements = VecTy->getNumElements();
        CallInst *Src = toScalableVector(TTI, Builder, II->getArgOperand(0));
        ConstantInt *VL = Builder.getIntN(XLEN, VecNumElements);
        II->replaceAllUsesWith(Builder.CreateExtractVector(
            VecTy,
            Builder.CreateIntrinsic(
                II->getIntrinsicID() == Intrinsic::aarch64_neon_frecpe
                    ? Intrinsic::riscv_vfrec7
                    : Intrinsic::riscv_vfrsqrt7,
                {Src->getType(), VL->getType()},
                {PoisonValue::get(Src->getType()), Src, VL}),
            Builder.getInt64(0)));
        break;
      }
      case Intrinsic::aarch64_neon_frecps:
        // If either left or right is NaN, return NaN.
        // If either operand is sNaN, set invalid flag.
        //               |         right
        //               |----------------------
        //               |  inf  |   0   | other
        // --------------+-------+-------+------
        //       |  inf  | ?inf  |  +2   | ?inf
        //  left |   0   |  +2   |  op   |  op
        //       | other | ?inf  |  op   |  op
        // ? is signedness, it depends on the signedness of left and right.
        // op = 2 - left * right
        // op is a fully fused multiply-add.
      case Intrinsic::aarch64_neon_frsqrts: {
        // If either left or right is NaN, return NaN.
        // If either operand is sNaN, set invalid flag.
        //               |         right
        //               |----------------------
        //               |  inf  |   0   | other
        // --------------+-------+-------+------
        //       |  inf  | ?inf  | +1.5  | ?inf
        //  left |   0   | +1.5  |  op   |  op
        //       | other | ?inf  |  op   |  op
        // ? is signedness, it depends on the signedness of left and right.
        // op = (3 - left * right) / 2
        // op is a fully fused multiply-add.
        bool Isfrecps = II->getIntrinsicID() == Intrinsic::aarch64_neon_frecps;
        FixedVectorType *VecTy =
            cast<FixedVectorType>(II->getArgOperand(0)->getType());
        unsigned VecNumElements = VecTy->getNumElements();
        Value *Op0 = II->getArgOperand(0);
        Value *Op1 = II->getArgOperand(1);
        Value *FabsOp0 =
            Builder.CreateIntrinsic(Intrinsic::fabs, {VecTy}, {Op0});
        Value *FabsOp1 =
            Builder.CreateIntrinsic(Intrinsic::fabs, {VecTy}, {Op1});
        ConstantInt *VL = Builder.getIntN(XLEN, VecNumElements);
        ConstantInt *Agnostic = Builder.getIntN(XLEN, 1);
        Value *FclassOp0 = createFclass(TTI, Builder, FabsOp0, VL);
        Value *FclassOp1 = createFclass(TTI, Builder, FabsOp1, VL);
        Value *FclassOr = Builder.CreateOr(FclassOp0, FclassOp1);
        // 144 = 128 (+inf) + 16 (+0)
        Value *IsNotInfAnd0 = Builder.CreateICmpNE(
            FclassOr, ConstantInt::get(FclassOr->getType(), 144));
        Value *ScalableOp0 = toScalableVector(TTI, Builder, Op0);
        Value *ScalableOp1 = toScalableVector(TTI, Builder, Op1);
        Value *ScalableIsNotInfAnd0 =
            toScalableVector(TTI, Builder, IsNotInfAnd0);
        Value *Vfmacc = Builder.CreateExtractVector(
            VecTy,
            Builder.CreateIntrinsic(
                Intrinsic::riscv_vfnmsac_mask,
                {ScalableOp0->getType(), ScalableOp0->getType(), VL->getType()},
                {ConstantFP::get(ScalableOp0->getType(), Isfrecps ? 2 : 3),
                 ScalableOp0, ScalableOp1, ScalableIsNotInfAnd0, VL, Agnostic}),
            Builder.getInt64(0));
        if (Isfrecps)
          II->replaceAllUsesWith(Vfmacc);
        else
          II->replaceAllUsesWith(Builder.CreateFMul(
              Vfmacc, ConstantFP::get(Vfmacc->getType(), 0.5)));
        break;
      }
      case Intrinsic::aarch64_neon_ld1x2:
      case Intrinsic::aarch64_neon_ld1x3:
      case Intrinsic::aarch64_neon_ld1x4: {
        StructType *DesTy = cast<StructType>(II->getType());
        unsigned StructNumElements = DesTy->getNumElements();
        FixedVectorType *StructElementType =
            cast<FixedVectorType>(DesTy->getElementType(0));
        unsigned VectorNumElements = StructElementType->getNumElements();
        FixedVectorType *ConcatenateTy =
            FixedVectorType::get(StructElementType->getElementType(),
                                 StructNumElements * VectorNumElements);
        LoadInst *Load = Builder.CreateAlignedLoad(
            ConcatenateTy,
            Builder.CreateBitCast(II->getArgOperand(0),
                                  ConcatenateTy->getPointerTo()),
            DL.getABITypeAlign(DesTy->getElementType(0)->getScalarType()));
        Value *Des = PoisonValue::get(DesTy);
        for (unsigned i = 0; i != StructNumElements; ++i) {
          Des = Builder.CreateInsertValue(
              Des,
              Builder.CreateExtractVector(
                  StructElementType, Load,
                  Builder.getInt64(i * VectorNumElements)),
              i);
        }
        II->replaceAllUsesWith(Des);
        break;
      }
      case Intrinsic::aarch64_neon_ld2:
      case Intrinsic::aarch64_neon_ld2r:
      case Intrinsic::aarch64_neon_ld3:
      case Intrinsic::aarch64_neon_ld3r:
      case Intrinsic::aarch64_neon_ld4:
      case Intrinsic::aarch64_neon_ld4r: {
        bool IsDup;
        switch (II->getIntrinsicID()) {
        default:
          llvm_unreachable("Unexpected intrinsic");
        case Intrinsic::aarch64_neon_ld2:
        case Intrinsic::aarch64_neon_ld3:
        case Intrinsic::aarch64_neon_ld4:
          IsDup = false;
          break;
        case Intrinsic::aarch64_neon_ld2r:
        case Intrinsic::aarch64_neon_ld3r:
        case Intrinsic::aarch64_neon_ld4r:
          IsDup = true;
          break;
        }
        StructType *DesTy = cast<StructType>(II->getType());
        unsigned StructNumElements = DesTy->getNumElements();
        FixedVectorType *StructElementType =
            cast<FixedVectorType>(DesTy->getElementType(0));
        unsigned VectorNumElements = StructElementType->getNumElements();
        static const Intrinsic::ID Vlseg[3] = {Intrinsic::riscv_vlseg2,
                                               Intrinsic::riscv_vlseg3,
                                               Intrinsic::riscv_vlseg4};
        static const Intrinsic::ID Vlsseg[3] = {Intrinsic::riscv_vlsseg2,
                                                Intrinsic::riscv_vlsseg3,
                                                Intrinsic::riscv_vlsseg4};
        Type *ScalableStructElementType =
            TTI.getScalableVectorFromFixed(StructElementType);
        SmallVector<Value *, 6> Ops;
        for (unsigned i = 0; i != StructNumElements; ++i)
          Ops.push_back(PoisonValue::get(ScalableStructElementType));
        Ops.push_back(II->getArgOperand(0));
        if (IsDup)
          Ops.push_back(Builder.getIntN(XLEN, 0));
        ConstantInt *VL = Builder.getIntN(XLEN, VectorNumElements);
        Ops.push_back(VL);
        CallInst *NewLoad = Builder.CreateIntrinsic(
            IsDup ? Vlsseg[StructNumElements - 2]
                  : Vlseg[StructNumElements - 2],
            {ScalableStructElementType, VL->getType()}, Ops);
        Value *NewDes = PoisonValue::get(DesTy);
        for (unsigned i = 0; i != StructNumElements; ++i)
          NewDes = Builder.CreateInsertValue(
              NewDes,
              Builder.CreateExtractVector(
                  StructElementType, Builder.CreateExtractValue(NewLoad, i),
                  Builder.getInt64(0)),
              i);
        II->replaceAllUsesWith(NewDes);
        break;
      }
      case Intrinsic::aarch64_neon_ld2lane:
      case Intrinsic::aarch64_neon_ld3lane:
      case Intrinsic::aarch64_neon_ld4lane: {
        unsigned StructNumElements = II->arg_size() - 2;
        FixedVectorType *ConcatenateTy = FixedVectorType::get(
            II->getArgOperand(0)->getType()->getScalarType(),
            StructNumElements);
        LoadInst *Load = Builder.CreateAlignedLoad(
            ConcatenateTy,
            Builder.CreateBitCast(II->getArgOperand(II->arg_size() - 1),
                                  ConcatenateTy->getPointerTo()),
            Align(1));
        ConstantInt *Lane =
            cast<ConstantInt>(II->getArgOperand(StructNumElements));
        Value *Des = PoisonValue::get(II->getType());
        for (unsigned i = 0; i != StructNumElements; ++i)
          Des = Builder.CreateInsertValue(
              Des,
              Builder.CreateInsertElement(II->getArgOperand(i),
                                          Builder.CreateExtractElement(Load, i),
                                          Lane),
              i);
        II->replaceAllUsesWith(Des);
        break;
      }
      case Intrinsic::aarch64_neon_sabd:
      case Intrinsic::aarch64_neon_uabd: {
        Intrinsic::ID MaxID;
        Intrinsic::ID MinID;
        if (II->getIntrinsicID() == Intrinsic::aarch64_neon_sabd) {
          MaxID = Intrinsic::smax;
          MinID = Intrinsic::smin;
        } else {
          MaxID = Intrinsic::umax;
          MinID = Intrinsic::umin;
        }
        CallInst *Max = Builder.CreateIntrinsic(
            MaxID, {II->getArgOperand(0)->getType()},
            {II->getArgOperand(0), II->getArgOperand(1)});
        CallInst *Min = Builder.CreateIntrinsic(
            MinID, {II->getArgOperand(0)->getType()},
            {II->getArgOperand(0), II->getArgOperand(1)});
        II->replaceAllUsesWith(Builder.CreateSub(Max, Min));
        break;
      }
      case Intrinsic::aarch64_neon_saddlp:
      case Intrinsic::aarch64_neon_uaddlp: {
        unsigned DesVecNumElements =
            cast<FixedVectorType>(II->getType())->getNumElements();
        Value *Input[2];
        for (int i = 0; i != 2; ++i)
          Input[i] = Builder.CreateShuffleVector(
              II->getArgOperand(0),
              increasingSequenceByN(i, 2, DesVecNumElements));
        if (II->getIntrinsicID() == Intrinsic::aarch64_neon_saddlp) {
          Input[0] = Builder.CreateSExt(Input[0], II->getType());
          Input[1] = Builder.CreateSExt(Input[1], II->getType());
        } else {
          Input[0] = Builder.CreateZExt(Input[0], II->getType());
          Input[1] = Builder.CreateZExt(Input[1], II->getType());
        }
        II->replaceAllUsesWith(Builder.CreateAdd(Input[0], Input[1]));
        break;
      }
      case Intrinsic::aarch64_neon_saddlv: {
        Value *SExt = Builder.CreateSExt(
            II->getArgOperand(0),
            FixedVectorType::get(
                II->getType(),
                cast<FixedVectorType>(II->getArgOperand(0)->getType())));
        II->replaceAllUsesWith(Builder.CreateAddReduce(SExt));
        break;
      }
      case Intrinsic::aarch64_neon_saddv: {
        CallInst *Reduce = Builder.CreateAddReduce(II->getArgOperand(0));
        Value *SExt = Builder.CreateSExt(Reduce, II->getType());
        II->replaceAllUsesWith(SExt);
        break;
      }
      case Intrinsic::aarch64_neon_sdot:
      case Intrinsic::aarch64_neon_udot: {
        // mul[0] = Op1[0] * Op2[0]
        // mul[1] = Op1[1] * Op2[1]
        // ...
        // out[0] = Op0[0] + mul[0] + mul[1] + mul[2] + mul[3]
        // out[1] = Op0[1] + mul[4] + mul[5] + mul[6] + mul[7]
        Instruction::CastOps ExtID =
            II->getIntrinsicID() == Intrinsic::aarch64_neon_sdot
                ? Instruction::SExt
                : Instruction::ZExt;
        unsigned VectorNumElements =
            cast<FixedVectorType>(II->getType())->getNumElements();
        Value *Ext0 = Builder.CreateCast(
            ExtID, II->getArgOperand(1),
            FixedVectorType::getExtendedElementVectorType(
                cast<FixedVectorType>(II->getArgOperand(1)->getType())));
        Value *Ext1 = Builder.CreateCast(
            ExtID, II->getArgOperand(2),
            FixedVectorType::getExtendedElementVectorType(
                cast<FixedVectorType>(II->getArgOperand(2)->getType())));
        Value *Mul = Builder.CreateMul(Ext0, Ext1);
        Value *Even = Builder.CreateShuffleVector(
            Mul, increasingSequenceByN(0, 2, VectorNumElements * 2));
        Even = Builder.CreateCast(ExtID, Even,
                                  FixedVectorType::getExtendedElementVectorType(
                                      cast<FixedVectorType>(Even->getType())));
        Value *Odd = Builder.CreateShuffleVector(
            Mul, increasingSequenceByN(1, 2, VectorNumElements * 2));
        Odd = Builder.CreateCast(ExtID, Odd,
                                 FixedVectorType::getExtendedElementVectorType(
                                     cast<FixedVectorType>(Odd->getType())));
        Value *HalfAdd = Builder.CreateAdd(Even, Odd);
        HalfAdd = Builder.CreateAdd(
            Builder.CreateShuffleVector(
                HalfAdd, increasingSequenceByN(0, 2, VectorNumElements)),
            Builder.CreateShuffleVector(
                HalfAdd, increasingSequenceByN(1, 2, VectorNumElements)));
        II->replaceAllUsesWith(
            Builder.CreateAdd(II->getArgOperand(0), HalfAdd));
        break;
      }
      case Intrinsic::aarch64_neon_smax:
      case Intrinsic::aarch64_neon_smin:
      case Intrinsic::aarch64_neon_umax:
      case Intrinsic::aarch64_neon_umin: {
        Intrinsic::ID Op;
        switch (II->getIntrinsicID()) {
        default:
          llvm_unreachable("Unexpected intrinsic");
        case Intrinsic::aarch64_neon_smax:
          Op = Intrinsic::smax;
          break;
        case Intrinsic::aarch64_neon_smin:
          Op = Intrinsic::smin;
          break;
        case Intrinsic::aarch64_neon_umax:
          Op = Intrinsic::umax;
          break;
        case Intrinsic::aarch64_neon_umin:
          Op = Intrinsic::umin;
          break;
        }
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            Op, {II->getArgOperand(0)->getType()},
            {II->getArgOperand(0), II->getArgOperand(1)}));
        break;
      }
      case Intrinsic::aarch64_neon_smaxv: {
        II->replaceAllUsesWith(Builder.CreateSExt(
            Builder.CreateIntMaxReduce(II->getArgOperand(0), true),
            II->getType()));
        break;
      }
      case Intrinsic::aarch64_neon_sminv: {
        II->replaceAllUsesWith(Builder.CreateSExt(
            Builder.CreateIntMinReduce(II->getArgOperand(0), true),
            II->getType()));
        break;
      }
      case Intrinsic::aarch64_neon_smull: {
        II->replaceAllUsesWith(Builder.CreateMul(
            Builder.CreateSExt(II->getArgOperand(0), II->getType()),
            Builder.CreateSExt(II->getArgOperand(1), II->getType())));
        break;
      }
      case Intrinsic::aarch64_neon_sqadd:
      case Intrinsic::aarch64_neon_sqsub:
      case Intrinsic::aarch64_neon_uqadd:
      case Intrinsic::aarch64_neon_uqsub: {
        // Intrinsics associated with the Q-bit and their feature macro
        // __ARM_FEATURE_QBIT are deprecated in ACLE 2.0 for A-profile. They are
        // fully supported for M-profile and R-profile. This macro is defined
        // for AArch32 only. Recode is targeted on AArch64. __ARM_FEATURE_QBIT
        // will not be supported.
        Intrinsic::ID Op;
        switch (II->getIntrinsicID()) {
        default:
          llvm_unreachable("Unexpected intrinsic");
        case Intrinsic::aarch64_neon_sqadd:
          Op = Intrinsic::sadd_sat;
          break;
        case Intrinsic::aarch64_neon_sqsub:
          Op = Intrinsic::ssub_sat;
          break;
        case Intrinsic::aarch64_neon_uqadd:
          Op = Intrinsic::uadd_sat;
          break;
        case Intrinsic::aarch64_neon_uqsub:
          Op = Intrinsic::usub_sat;
          break;
        }
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            Op, {II->getArgOperand(0)->getType()},
            {II->getArgOperand(0), II->getArgOperand(1)}));
        break;
      }
      case Intrinsic::aarch64_neon_sqdmulh:
      case Intrinsic::aarch64_neon_sqrdmulh: {
        Value *Smull = Builder.CreateMul(
            Builder.CreateSExt(
                II->getArgOperand(0),
                II->getArgOperand(0)->getType()->getExtendedType()),
            Builder.CreateSExt(
                II->getArgOperand(1),
                II->getArgOperand(1)->getType()->getExtendedType()));
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            II->getIntrinsicID() == Intrinsic::aarch64_neon_sqdmulh
                ? Intrinsic::aarch64_neon_sqshrn
                : Intrinsic::aarch64_neon_sqrshrn,
            {II->getType()},
            {Smull,
             Builder.getInt32(II->getType()->getScalarSizeInBits() - 1)}));
        break;
      }
      case Intrinsic::aarch64_neon_sqrshrun:
      case Intrinsic::aarch64_neon_sqshrun: {
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            II->getIntrinsicID() == Intrinsic::aarch64_neon_sqrshrun
                ? Intrinsic::aarch64_neon_uqrshrn
                : Intrinsic::aarch64_neon_uqshrn,
            {II->getType()},
            {smaxZero(Builder, II->getArgOperand(0)), II->getArgOperand(1)}));
        break;
      }
      case Intrinsic::aarch64_neon_sqshlu: {
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            Intrinsic::ushl_sat, {II->getType()},
            {smaxZero(Builder, II->getArgOperand(0)), II->getArgOperand(1)}));
        break;
      }
      case Intrinsic::aarch64_neon_sqxtun: {
        II->replaceAllUsesWith(Builder.CreateIntrinsic(
            Intrinsic::aarch64_neon_uqxtn, {II->getType()},
            {smaxZero(Builder, II->getArgOperand(0))}));
        break;
      }
      case Intrinsic::aarch64_neon_sshl:
      case Intrinsic::aarch64_neon_ushl: {
        Value *Op1 = II->getArgOperand(1);
        if (auto *Op1SplatValue = getSplatValue(Op1))
          Op1 = Op1SplatValue;
        II->replaceAllUsesWith(
            expandShl(Builder, II->getArgOperand(0), Op1,
                      II->getIntrinsicID() == Intrinsic::aarch64_neon_sshl));
        break;
      }
      case Intrinsic::aarch64_neon_st1x2:
      case Intrinsic::aarch64_neon_st1x3:
      case Intrinsic::aarch64_neon_st1x4: {
        unsigned StructNumElements = II->arg_size() - 1;
        FixedVectorType *VecTy =
            cast<FixedVectorType>(II->getArgOperand(0)->getType());
        Type *VecElementTy = VecTy->getElementType();
        unsigned VecNumElements = VecTy->getNumElements();
        FixedVectorType *ConcatenateTy = FixedVectorType::get(
            VecElementTy, StructNumElements * VecNumElements);
        SmallVector<Value *, 4> Arg;
        for (unsigned i = 0; i != StructNumElements; ++i)
          Arg.push_back(II->getArgOperand(i));
        II->replaceAllUsesWith(Builder.CreateAlignedStore(
            glue(Builder, Arg),
            Builder.CreateBitCast(II->getArgOperand(StructNumElements),
                                  ConcatenateTy->getPointerTo()),
            DL.getABITypeAlign(VecElementTy)));
        break;
      }
      case Intrinsic::aarch64_neon_st2:
      case Intrinsic::aarch64_neon_st3:
      case Intrinsic::aarch64_neon_st4: {
        unsigned StructNumElements = II->arg_size() - 1;
        unsigned VectorNumElements =
            cast<FixedVectorType>(II->getArgOperand(0)->getType())
                ->getNumElements();
        static const Intrinsic::ID Vsseg[3] = {Intrinsic::riscv_vsseg2,
                                               Intrinsic::riscv_vsseg3,
                                               Intrinsic::riscv_vsseg4};
        SmallVector<Value *, 6> Ops;
        for (unsigned i = 0; i != StructNumElements; ++i)
          Ops.push_back(toScalableVector(TTI, Builder, II->getArgOperand(i)));
        Ops.push_back(II->getArgOperand(StructNumElements));
        ConstantInt *VL = Builder.getIntN(XLEN, VectorNumElements);
        Ops.push_back(VL);
        II->replaceAllUsesWith(
            Builder.CreateIntrinsic(Vsseg[StructNumElements - 2],
                                    {Ops[0]->getType(), VL->getType()}, Ops));
        break;
      }
      case Intrinsic::aarch64_neon_st2lane:
      case Intrinsic::aarch64_neon_st3lane:
      case Intrinsic::aarch64_neon_st4lane: {
        unsigned StructNumElements = II->arg_size() - 2;
        FixedVectorType *DesTy = FixedVectorType::get(
            II->getArgOperand(0)->getType()->getScalarType(),
            StructNumElements);
        ConstantInt *Lane =
            cast<ConstantInt>(II->getArgOperand(StructNumElements));
        Value *Des = PoisonValue::get(DesTy);
        for (unsigned i = 0; i != StructNumElements; ++i)
          Des = Builder.CreateInsertElement(
              Des, Builder.CreateExtractElement(II->getArgOperand(i), Lane), i);
        II->replaceAllUsesWith(Builder.CreateAlignedStore(
            Des,
            Builder.CreateBitCast(II->getArgOperand(StructNumElements + 1),
                                  DesTy->getPointerTo()),
            Align(1)));
        break;
      }
      case Intrinsic::aarch64_neon_tbl1:
      case Intrinsic::aarch64_neon_tbl2:
      case Intrinsic::aarch64_neon_tbl3:
      case Intrinsic::aarch64_neon_tbl4:
      case Intrinsic::aarch64_neon_tbx1:
      case Intrinsic::aarch64_neon_tbx2:
      case Intrinsic::aarch64_neon_tbx3:
      case Intrinsic::aarch64_neon_tbx4: {
        bool IsTbl;
        unsigned TableOperandBegin;
        switch (II->getIntrinsicID()) {
        default:
          llvm_unreachable("Unexpected intrinsic");
        case Intrinsic::aarch64_neon_tbl1:
        case Intrinsic::aarch64_neon_tbl2:
        case Intrinsic::aarch64_neon_tbl3:
        case Intrinsic::aarch64_neon_tbl4:
          IsTbl = true;
          TableOperandBegin = 0;
          break;
        case Intrinsic::aarch64_neon_tbx1:
        case Intrinsic::aarch64_neon_tbx2:
        case Intrinsic::aarch64_neon_tbx3:
        case Intrinsic::aarch64_neon_tbx4:
          IsTbl = false;
          TableOperandBegin = 1;
          break;
        }
        const unsigned TableSize = 16;
        size_t TableNum = II->arg_size() - (IsTbl ? 1 : 2);
        Value *Index = II->getArgOperand(II->arg_size() - 1);
        unsigned IndexNumElements =
            cast<FixedVectorType>(Index->getType())->getNumElements();
        SmallVector<Value *, 4> Arg;
        for (size_t i = 0; i != TableNum; ++i)
          Arg.push_back(II->getArgOperand(i + TableOperandBegin));
        Value *Concatenate = glue(Builder, Arg);
        // TableNum may be 3. Widen Concatenate to a proper size.
        unsigned VrgatherNumElements = TableSize * PowerOf2Ceil(TableNum);
        Concatenate = widen(Builder, Concatenate, VrgatherNumElements);
        // IndexNumElements may be smaller than TableSize. We need to widen
        // Index.
        Value *WidenIndex = widen(Builder, Index, VrgatherNumElements);
        Type *VrgatherTy =
            TTI.getScalableVectorFromFixed(Concatenate->getType());
        ConstantInt *VL = Builder.getIntN(XLEN, IndexNumElements);
        Value *Vrgather = Builder.CreateIntrinsic(
            Intrinsic::riscv_vrgather_vv, {VrgatherTy, VL->getType()},
            {PoisonValue::get(VrgatherTy),
             toScalableVector(TTI, Builder, Concatenate),
             toScalableVector(TTI, Builder, WidenIndex), VL});
        Vrgather = Builder.CreateExtractVector(Concatenate->getType(), Vrgather,
                                               Builder.getInt64(0));
        // Only <IndexNumElements x i8> is meaningful.
        Vrgather = narrow(Builder, Vrgather, IndexNumElements);
        // If Index is greater than or equal to TableSize * TableNum, return 0.
        Value *CC = Builder.CreateICmpUGT(
            Index,
            Builder.CreateVectorSplat(
                IndexNumElements, Builder.getInt8(TableSize * TableNum - 1)));
        Value *TrueVal = IsTbl ? Builder.CreateVectorSplat(IndexNumElements,
                                                           Builder.getInt8(0))
                               : II->getArgOperand(0);
        II->replaceAllUsesWith(Builder.CreateSelect(CC, TrueVal, Vrgather));
        break;
      }
      case Intrinsic::aarch64_neon_uaddlv: {
        Value *ZExt = Builder.CreateZExt(
            II->getArgOperand(0),
            FixedVectorType::get(
                II->getType(),
                cast<FixedVectorType>(II->getArgOperand(0)->getType())));
        II->replaceAllUsesWith(Builder.CreateAddReduce(ZExt));
        break;
      }
      case Intrinsic::aarch64_neon_uaddv: {
        CallInst *Reduce = Builder.CreateAddReduce(II->getArgOperand(0));
        Value *ZExt = Builder.CreateZExt(Reduce, II->getType());
        II->replaceAllUsesWith(ZExt);
        break;
      }
      case Intrinsic::aarch64_neon_umaxv: {
        II->replaceAllUsesWith(Builder.CreateSExt(
            Builder.CreateIntMaxReduce(II->getArgOperand(0)), II->getType()));
        break;
      }
      case Intrinsic::aarch64_neon_uminv: {
        II->replaceAllUsesWith(Builder.CreateSExt(
            Builder.CreateIntMinReduce(II->getArgOperand(0)), II->getType()));
        break;
      }
      case Intrinsic::aarch64_neon_umull: {
        II->replaceAllUsesWith(Builder.CreateMul(
            Builder.CreateZExt(II->getArgOperand(0), II->getType()),
            Builder.CreateZExt(II->getArgOperand(1), II->getType())));
        break;
      }
      case Intrinsic::aarch64_neon_vcvtfp2hf: {
        II->replaceAllUsesWith(Builder.CreateBitCast(
            Builder.CreateFPTrunc(
                II->getArgOperand(0),
                FixedVectorType::get(
                    Type::getHalfTy(II->getContext()),
                    cast<FixedVectorType>(II->getArgOperand(0)->getType()))),
            II->getType()));
        break;
      }
      case Intrinsic::aarch64_neon_vcvthf2fp: {
        II->replaceAllUsesWith(Builder.CreateFPExt(
            Builder.CreateBitCast(
                II->getArgOperand(0),
                FixedVectorType::get(
                    Type::getHalfTy(II->getContext()),
                    cast<FixedVectorType>(II->getArgOperand(0)->getType()))),
            FixedVectorType::get(
                Type::getFloatTy(II->getContext()),
                cast<FixedVectorType>(II->getArgOperand(0)->getType()))));
        break;
      }
      case Intrinsic::aarch64_neon_vsli: {
        uint64_t Shift =
            cast<ConstantInt>(II->getArgOperand(2))->getZExtValue();
        // (II->getArgOperand(1) << Shift) |
        //     (II->getArgOperand(0) & ((1 << Shift) - 1))
        Value *NBit = Builder.CreateAnd(
            II->getArgOperand(0),
            ConstantInt::get(II->getArgOperand(0)->getType(),
                             (static_cast<uint64_t>(1) << Shift) - 1));
        II->replaceAllUsesWith(Builder.CreateOr(
            NBit,
            Builder.CreateShl(
                II->getArgOperand(1),
                ConstantInt::get(II->getArgOperand(1)->getType(), Shift))));
        break;
      }
      case Intrinsic::aarch64_neon_vsri: {
        FixedVectorType *VecTy =
            cast<FixedVectorType>(II->getArgOperand(0)->getType());
        uint64_t ShiftAmount =
            cast<ConstantInt>(II->getArgOperand(2))->getZExtValue();
        unsigned ScalarSizeInBits = VecTy->getScalarSizeInBits();
        // (II->getArgOperand(0) & (-1 << (ScalarSizeInBits - ShiftAmount))) |
        //     (II->getArgOperand(1) >> ShiftAmount)
        if (ShiftAmount == ScalarSizeInBits) {
          II->replaceAllUsesWith(II->getArgOperand(0));
          break;
        }
        Value *RShift = Builder.CreateLShr(
            II->getArgOperand(1), ConstantInt::get(VecTy, ShiftAmount));
        Value *Left = Builder.CreateAnd(
            II->getArgOperand(0),
            ConstantInt::get(VecTy, static_cast<uint64_t>(-1)
                                        << (ScalarSizeInBits - ShiftAmount)));
        II->replaceAllUsesWith(Builder.CreateOr(Left, RShift));
        break;
      }
      default:
        break;
      }
      II->eraseFromParent();
    }
  }

  if (MadeChange) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }

  return PreservedAnalyses::all();
}
