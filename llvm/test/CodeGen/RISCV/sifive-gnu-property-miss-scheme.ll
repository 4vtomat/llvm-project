; RUN: not --crash llc -mtriple riscv32 -mattr=+experimental-zicfilp < %s 2>&1 | FileCheck %s --check-prefixes=CHECK
; RUN: not --crash llc -mtriple riscv64 -mattr=+experimental-zicfilp < %s 2>&1 | FileCheck %s --check-prefixes=CHECK

; CHECK:  LLVM ERROR: cf-branch-label-scheme is not specified but cf-protection-branch is enabled

define dso_local void @foo() {
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"cf-protection-branch", i32 1}
