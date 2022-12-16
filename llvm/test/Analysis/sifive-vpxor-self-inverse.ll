; RUN: opt -passes=instsimplify -S -o - < %s | FileCheck %s

declare <256 x i32> @llvm.vp.xor.v256i32(<256 x i32>, <256 x i32>, <256 x i1>, i32)

define <256 x i32> @foo1(<256 x i32> %i0, <256 x i32> %i1, <256 x i1> %m, i32 %n) {
; CHECK-LABEL: @foo1(
; CHECK-NEXT: ret <256 x i32> %i0
;
  %r0 = call <256 x i32> @llvm.vp.xor.v256i32(<256 x i32> %i0, <256 x i32> %i1, <256 x i1> %m, i32 %n)
  %r1 = call <256 x i32> @llvm.vp.xor.v256i32(<256 x i32> %r0, <256 x i32> %i1, <256 x i1> %m, i32 %n)
  ret <256 x i32> %r1
}

define <256 x i32> @foo2(<256 x i32> %i0, <256 x i32> %i1, <256 x i1> %m, i32 %n) {
; CHECK-LABEL: @foo2(
; CHECK-NEXT: ret <256 x i32> %i0
;
  %r0 = call <256 x i32> @llvm.vp.xor.v256i32(<256 x i32> %i0, <256 x i32> %i1, <256 x i1> %m, i32 %n)
  %r1 = call <256 x i32> @llvm.vp.xor.v256i32(<256 x i32> %i1, <256 x i32> %r0, <256 x i1> %m, i32 %n)
  ret <256 x i32> %r1
}

define <256 x i32> @foo3(<256 x i32> %i0, <256 x i32> %i1, <256 x i1> %m, i32 %n) {
; CHECK-LABEL: @foo3(
; CHECK-NEXT: ret <256 x i32> %i0
;
  %r0 = call <256 x i32> @llvm.vp.xor.v256i32(<256 x i32> %i1, <256 x i32> %i0, <256 x i1> %m, i32 %n)
  %r1 = call <256 x i32> @llvm.vp.xor.v256i32(<256 x i32> %r0, <256 x i32> %i1, <256 x i1> %m, i32 %n)
  ret <256 x i32> %r1
}

define <256 x i32> @foo4(<256 x i32> %i0, <256 x i32> %i1, <256 x i1> %m, i32 %n) {
; CHECK-LABEL: @foo4(
; CHECK-NEXT: ret <256 x i32> %i0
;
  %r0 = call <256 x i32> @llvm.vp.xor.v256i32(<256 x i32> %i0, <256 x i32> %i1, <256 x i1> %m, i32 %n)
  %r1 = call <256 x i32> @llvm.vp.xor.v256i32(<256 x i32> %i1, <256 x i32> %r0, <256 x i1> %m, i32 %n)
  ret <256 x i32> %r1
}

declare <vscale x 1 x i32> @llvm.vp.xor.nxv1i32(<vscale x 1 x i32>, <vscale x 1 x i32>, <vscale x 1 x i1>, i32)

define <vscale x 1 x i32> @foo5(<vscale x 1 x i32> %i0, <vscale x 1 x i32> %i1, <vscale x 1 x i1> %m, i32 %n) {
; CHECK-LABEL: @foo5(
; CHECK-NEXT: ret <vscale x 1 x i32> %i0
;
  %r0 = call <vscale x 1 x i32> @llvm.vp.xor.nxv1i32(<vscale x 1 x i32> %i0, <vscale x 1 x i32> %i1, <vscale x 1 x i1> %m, i32 %n)
  %r1 = call <vscale x 1 x i32> @llvm.vp.xor.nxv1i32(<vscale x 1 x i32> %r0, <vscale x 1 x i32> %i1, <vscale x 1 x i1> %m, i32 %n)
  ret <vscale x 1 x i32> %r1
}

define <vscale x 1 x i32> @foo6(<vscale x 1 x i32> %i0, <vscale x 1 x i32> %i1, <vscale x 1 x i1> %m, i32 %n) {
; CHECK-LABEL: @foo6(
; CHECK-NEXT: ret <vscale x 1 x i32> %i0
;
  %r0 = call <vscale x 1 x i32> @llvm.vp.xor.nxv1i32(<vscale x 1 x i32> %i0, <vscale x 1 x i32> %i1, <vscale x 1 x i1> %m, i32 %n)
  %r1 = call <vscale x 1 x i32> @llvm.vp.xor.nxv1i32(<vscale x 1 x i32> %i1, <vscale x 1 x i32> %r0, <vscale x 1 x i1> %m, i32 %n)
  ret <vscale x 1 x i32> %r1
}

define <vscale x 1 x i32> @foo7(<vscale x 1 x i32> %i0, <vscale x 1 x i32> %i1, <vscale x 1 x i1> %m, i32 %n) {
; CHECK-LABEL: @foo7(
; CHECK-NEXT: ret <vscale x 1 x i32> %i0
;
  %r0 = call <vscale x 1 x i32> @llvm.vp.xor.nxv1i32(<vscale x 1 x i32> %i1, <vscale x 1 x i32> %i0, <vscale x 1 x i1> %m, i32 %n)
  %r1 = call <vscale x 1 x i32> @llvm.vp.xor.nxv1i32(<vscale x 1 x i32> %r0, <vscale x 1 x i32> %i1, <vscale x 1 x i1> %m, i32 %n)
  ret <vscale x 1 x i32> %r1
}

define <vscale x 1 x i32> @foo8(<vscale x 1 x i32> %i0, <vscale x 1 x i32> %i1, <vscale x 1 x i1> %m, i32 %n) {
; CHECK-LABEL: @foo8(
; CHECK-NEXT: ret <vscale x 1 x i32> %i0
;
  %r0 = call <vscale x 1 x i32> @llvm.vp.xor.nxv1i32(<vscale x 1 x i32> %i0, <vscale x 1 x i32> %i1, <vscale x 1 x i1> %m, i32 %n)
  %r1 = call <vscale x 1 x i32> @llvm.vp.xor.nxv1i32(<vscale x 1 x i32> %i1, <vscale x 1 x i32> %r0, <vscale x 1 x i1> %m, i32 %n)
  ret <vscale x 1 x i32> %r1
}
