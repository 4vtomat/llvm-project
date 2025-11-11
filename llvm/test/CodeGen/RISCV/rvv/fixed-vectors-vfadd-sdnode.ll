define <64 x bfloat> @vfadd_vv_v64bf16(<64 x bfloat> %va, <64 x bfloat> %vb) {
; CHECK-LABEL: vfadd_vv_v64bf16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a0, 64
; CHECK-NEXT:    vsetvli zero, a0, e16alt, m8, ta, ma
; CHECK-NEXT:    vfadd.vv v8, v8, v16
; CHECK-NEXT:    ret
  %vc = fadd <64 x bfloat> %va, %vb
  ret <64 x bfloat> %vc
}
