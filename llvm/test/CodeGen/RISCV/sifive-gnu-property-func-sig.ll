; RUN: llc -mtriple riscv32 -mattr=+experimental-zicfiss < %s | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple riscv64 -mattr=+experimental-zicfiss < %s | FileCheck %s --check-prefixes=CHECK,RV64

define dso_local void @foo() {
  ret void
}

!llvm.module.flags = !{!0}

; CHECK:              .size   foo, .Lfunc_end0-foo
; CHECK:              .cfi_endproc
; CHECK:                                              # -- End function
; CHECK:              .section        ".note.GNU-stack","",@progbits
; CHECK-NEXT:         .section        .note.gnu.property,"a",@note
; RV32-NEXT:          .p2align        2, 0x0
; RV64-NEXT:          .p2align        3, 0x0
; CHECK-NEXT:         .word   4
; RV32-NEXT:          .word   12
; RV64-NEXT:          .word   16
; CHECK-NEXT:         .word   5
; CHECK-NEXT:         .asciz  "GNU"
; CHECK-NEXT:         .word   3221225472
; CHECK-NEXT:         .word   4
; CHECK-NEXT:         .word   2
; RV32-NEXT:          .p2align        2, 0x0
; RV64-NEXT:          .p2align        3, 0x0
; CHECK-NEXT:         .section        ".note.GNU-stack","",@progbits


!0 = !{i32 8, !"cf-protection-return", i32 1}
