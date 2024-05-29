; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define void @a() {
  unreachable
}

define void @b() !riscv_cfi_type !0 {
  unreachable
}

; CHECK: function must have a single !riscv_cfi_type attachment
define void @f0() !riscv_cfi_type !0 !riscv_cfi_type !0 {
  unreachable
}
!0 = !{i32 10}

; CHECK: !riscv_cfi_type must have exactly one operand
define void @f1() !riscv_cfi_type !1 {
  unreachable
}
!1 = !{!"string", i32 0}

; CHECK: expected a constant operand for !riscv_cfi_type
define void @f2() !riscv_cfi_type !2 {
  unreachable
}
!2 = !{!"string"}

; CHECK: expected a constant integer operand for !riscv_cfi_type
define void @f3() !riscv_cfi_type !3 {
  unreachable
}
!3 = !{ptr @f3}

; CHECK: expected a 32-bit integer constant operand for !riscv_cfi_type
define void @f4() !riscv_cfi_type !4 {
  unreachable
}
!4 = !{i64 10}

define void @f5() !riscv_cfi_type !5 {
  unreachable
}
!5 = !{i32 -1}

; CHECK: expected a 20-bit integer constant or minus one operand for !riscv_cfi_type
define void @f6() !riscv_cfi_type !6 {
  unreachable
}
!6 = !{i32 1048576}

; CHECK: expected a 20-bit integer constant or minus one operand for !riscv_cfi_type
define void @f7() !riscv_cfi_type !7 {
  unreachable
}
!7 = !{i32 -5}
