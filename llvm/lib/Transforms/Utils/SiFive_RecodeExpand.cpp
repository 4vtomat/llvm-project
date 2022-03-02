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

bool SiFiveRecodePass::requireExpand(IntrinsicInst *II) {
  switch (II->getIntrinsicID()) {
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_ld4:
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
  for (Instruction &Inst : llvm::make_early_inc_range(instructions(F))) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(&Inst);
    if (II && requireExpand(II)) {
      MadeChange = true;
      IRBuilder<> Builder(II);

      switch (II->getIntrinsicID()) {
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
