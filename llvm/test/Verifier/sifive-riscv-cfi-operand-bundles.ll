; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

define void @test_riscv_cfi_bundle(i64 %arg0, i32 %arg1, ptr %arg2) {
; CHECK: Multiple riscv_cfi operand bundles
; CHECK-NEXT: call void %arg2() [ "riscv_cfi"(i32 42), "riscv_cfi"(i32 42) ]
  call void %arg2() [ "riscv_cfi"(i32 42), "riscv_cfi"(i32 42) ]

; CHECK: riscv_cfi bundle operand must be an i32 constant
; CHECK-NEXT: call void %arg2() [ "riscv_cfi"(i64 42) ]
  call void %arg2() [ "riscv_cfi"(i64 42) ]

; CHECK: riscv_cfi bundle operand must be fit to 20-bits or minus one
; CHECK-NEXT: call void %arg2() [ "riscv_cfi"(i32 1048576) ]
  call void %arg2() [ "riscv_cfi"(i32 1048576) ]

; CHECK: riscv_cfi bundle operand must be fit to 20-bits or minus one
; CHECK-NEXT: call void %arg2() [ "riscv_cfi"(i32 -5) ]
  call void %arg2() [ "riscv_cfi"(i32 -5) ]

; CHECK-NOT: call void
  call void %arg2() [ "riscv_cfi"(i32 42) ] ; OK
  call void %arg2() [ "riscv_cfi"(i32 -1) ] ; OK
  ret void
}
