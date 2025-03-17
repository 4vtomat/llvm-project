; RUN: not --crash llc -mtriple riscv32 -mattr=+experimental-zicfilp < %s 2>&1 | FileCheck %s --check-prefixes=CHECK
; RUN: not --crash llc -mtriple riscv64 -mattr=+experimental-zicfilp < %s 2>&1 | FileCheck %s --check-prefixes=CHECK

; CHECK:  LLVM ERROR: Unknown value for cf-branch-label-scheme

define dso_local void @foo() {
  ret void
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"wrong-scheme"}
