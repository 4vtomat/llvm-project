; RUN:   llc -mtriple=riscv64 -mattr=+d -riscv-enable-small-rodata-section=true %s -o - | \
; RUN:   FileCheck -check-prefix=CHECK-ENABLE %s
; RUN:   llc -mtriple=riscv64 -mattr=+d -riscv-enable-small-rodata-section=false %s -o -| \
; RUN:   FileCheck -check-prefix=CHECK-DISABLE %s

define dso_local float @foof() {
entry:
  ret float 0x400A08ACA0000000
}

define dso_local double @foo() {
entry:
  ret double 0x400A08AC91C3E242
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"SmallDataLimit", i32 8}

; CHECK-ENABLE:    .section        .srodata.cst4
; CHECK-ENABLE:    .section        .srodata.cst8
; CHECK-DISABLE:    .section        .sdata
