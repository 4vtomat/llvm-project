; RUN: llc -mtriple riscv32 -mattr=+experimental-zicfilp < %s | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple riscv64 -mattr=+experimental-zicfilp < %s | FileCheck %s --check-prefixes=CHECK,RV64

define dso_local void @foo() {
  ret void
}

!llvm.module.flags = !{!0, !1}

; CHECK:        .size   foo, .Lfunc_end0-foo
; CHECK:        .cfi_endproc
; CHECK:                                        # -- End function
; CHECK:        .section        ".note.GNU-stack","",@progbits
; CHECK:        .section        .note.gnu.property,"a",@note
; RV32:         .p2align        2, 0x0
; RV64:         .p2align        3, 0x0
; CHECK:        .word   4
; RV32:         .word   12
; RV64:         .word   16
; CHECK:        .word   5
; CHECK:        .asciz  "GNU"
; CHECK:        .word   3221225472
; CHECK:        .word   4
; CHECK:        .word   4
; RV64:         .word   0
; CHECK:.Lsec_end0:
; CHECK:        .section        ".note.GNU-stack","",@progbits


!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"fixed-one"}
