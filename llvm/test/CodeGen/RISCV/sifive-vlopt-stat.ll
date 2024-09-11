; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 -mattr=+v -stats 2>&1 1>/dev/null | FileCheck %s
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 -mattr=+v -stats 2>&1 1>/dev/null | FileCheck %s

; REQUIRES: asserts

declare <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, iXLen)

define <vscale x 4 x i32> @reduce_vl(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
  %v = call <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32(<vscale x 4 x i32> poison, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, iXLen 5)
  %w = call <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32(<vscale x 4 x i32> poison, <vscale x 4 x i32> %v, <vscale x 4 x i32> %a, iXLen 4)
  ret <vscale x 4 x i32> %w
}

; CHECK: 1 riscv-vl-optimizer    - Number of VLs that were reduced

