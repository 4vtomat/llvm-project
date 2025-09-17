; RUN: llc -mtriple=riscv64 -mattr=+lui-addi-fusion -riscv-collect-macro-fusion-stats -pass-remarks-output=%t.yaml -pass-remarks-filter=riscv-macro-fusion-stats %s -o %t.s
; RUN: FileCheck --input-file=%t.yaml %s

; CHECK: --- !Analysis
; CHECK-NEXT: Pass:            riscv-macro-fusion-stats
; CHECK-NEXT: Name:            TuneLUIADDIFusion
; CHECK-NEXT: Function:        imm64_shifted
; CHECK-NEXT: Args:
; CHECK-NEXT:  - NumOccurrences:  '1'

define i64 @imm64_shifted() nounwind {
  ret i64 1311768464867721216 ; 0x1234_5678_0000_0000
}
