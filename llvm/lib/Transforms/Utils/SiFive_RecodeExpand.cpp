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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

static CallInst *toScalableVector(const TargetTransformInfo &TTI,
                                  IRBuilder<> &Builder, Value *Vec) {
  Type *ScalableVecTy = TTI.getScalableVectorFromFixed(Vec->getType());
  return Builder.CreateInsertVector(
      ScalableVecTy, UndefValue::get(ScalableVecTy), Vec, Builder.getInt64(0));
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

bool SiFiveRecodePass::requireExpand(IntrinsicInst *II) {
  switch (II->getIntrinsicID()) {
  case Intrinsic::aarch64_neon_ld1x2:
  case Intrinsic::aarch64_neon_ld1x3:
  case Intrinsic::aarch64_neon_ld1x4:
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_ld4:
  case Intrinsic::aarch64_neon_st1x2:
  case Intrinsic::aarch64_neon_st1x3:
  case Intrinsic::aarch64_neon_st1x4:
  case Intrinsic::aarch64_neon_st2:
  case Intrinsic::aarch64_neon_st3:
  case Intrinsic::aarch64_neon_st4:
    return true;
  }
  return false;
}

PreservedAnalyses SiFiveRecodePass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
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
      case Intrinsic::aarch64_neon_ld3:
      case Intrinsic::aarch64_neon_ld4: {
        StructType *DesTy = cast<StructType>(II->getType());
        unsigned StructNumElements = DesTy->getNumElements();
        FixedVectorType *StructElementType =
            cast<FixedVectorType>(DesTy->getElementType(0));
        unsigned VectorNumElements = StructElementType->getNumElements();
        static const Intrinsic::ID Vlseg[3] = {Intrinsic::riscv_vlseg2,
                                               Intrinsic::riscv_vlseg3,
                                               Intrinsic::riscv_vlseg4};
        Type *ScalableStructElementType =
            TTI.getScalableVectorFromFixed(StructElementType);
        SmallVector<Value *, 6> Ops;
        for (unsigned i = 0; i != StructNumElements; ++i)
          Ops.push_back(UndefValue::get(ScalableStructElementType));
        Ops.push_back(II->getArgOperand(0));
        ConstantInt *VL = Builder.getIntN(XLEN, VectorNumElements);
        Ops.push_back(VL);
        CallInst *NewLoad = Builder.CreateIntrinsic(
            Vlseg[StructNumElements - 2],
            {ScalableStructElementType, VL->getType()}, Ops);
        Value *NewDes = UndefValue::get(DesTy);
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
