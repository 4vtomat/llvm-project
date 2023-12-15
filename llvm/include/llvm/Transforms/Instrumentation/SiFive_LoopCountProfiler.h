// This file provides the interface for the Loop Profiler pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LOOP_COUNT_PROFILER_H
#define LLVM_LOOP_COUNT_PROFILER_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class LoopCountProfilerPass : public PassInfoMixin<LoopCountProfilerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // namespace llvm

#endif // LLVM_LOOP_COUNT_PROFILER_H
