#ifndef LLVM_TRANSFORMS_UTILS_SIFIVERECODEEXPAND_H
#define LLVM_TRANSFORMS_UTILS_SIFIVERECODEEXPAND_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class IntrinsicInst;

class SiFiveRecodePass : public PassInfoMixin<SiFiveRecodePass> {
public:
  // O0 requires this.
  static bool isRequired() { return true; }
  // Return true if a Neon intrinsic will be expanded by this pass.
  // Not all Neon intrinsics is required expand. (e.g., aarch64_neon_fmax is
  // lowered to RISCVISD::FMAXNUM_VL)
  static bool requireExpand(IntrinsicInst *II);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif
