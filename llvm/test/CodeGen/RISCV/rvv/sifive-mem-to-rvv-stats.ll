; REQUIRES: asserts
; RUN: opt -riscv-late-codegenprepare -stats -mattr=+v -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
; RUN: opt -riscv-late-codegenprepare -stats -mattr=+v,+dlen128b -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-ALIGN

target triple = "riscv64"

define void @KnownSize(i8* nocapture readonly %src, i8* nocapture %dst, i8 %val) {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %dst, i8* align 1 %src, i64 384, i1 false)
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* align 1 %dst, i8* align 1 %src, i64 384, i1 false)
  tail call void @llvm.memset.p0i8.p0i8.i64(i8* align 1 %dst, i8 %val, i64 384, i1 false)
  ret void
}

define void @UnknownSize(i8* nocapture readonly %src, i8* nocapture %dst, i64 %sz, i8 %val) {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %dst, i8* align 1 %src, i64 %sz, i1 false)
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* align 1 %dst, i8* align 1 %src, i64 %sz, i1 false)
  tail call void @llvm.memset.p0i8.p0i8.i64(i8* align 1 %dst, i8 %val, i64 %sz, i1 false)
  ret void
}

; CHECK: 1 riscv-late-codegenprepare - Number of known size memcpy call expanded
; CHECK-NEXT: 1 riscv-late-codegenprepare - Number of known size memmove call expanded
; CHECK-NEXT: 1 riscv-late-codegenprepare - Number of known size memset call expanded

; CHECK: 1 riscv-late-codegenprepare - Number of unknown size memcpy call expanded
; CHECK-NEXT: 1 riscv-late-codegenprepare - Number of unknown size memmove call expanded
; CHECK-NEXT: 1 riscv-late-codegenprepare - Number of unknown size memset call expanded

; CHECK-ALIGN: 1 riscv-late-codegenprepare - Number of known size memcpy call expanded
; CHECK-ALIGN-NEXT: 1 riscv-late-codegenprepare - Number of known size memmove call expanded
; CHECK-ALIGN-NEXT: 1 riscv-late-codegenprepare - Number of known size memset call expanded

; CHECK-ALIGN: 1 riscv-late-codegenprepare - Number of unknown size aligned memcpy call expanded
; CHECK-ALIGN-NEXT: 1 riscv-late-codegenprepare - Number of unknown size aligned memmove call expanded
; CHECK-ALIGN-NEXT: 1 riscv-late-codegenprepare - Number of unknown size aligned memset call expanded
