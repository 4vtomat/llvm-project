; RUN: llc %s -mtriple=riscv64 -mcpu=sifive-p470 -O3 -riscv-limit-load-cluster-succ-size=4 -debug-only=machine-scheduler -o /dev/null 2>&1 | \
; RUN:  FileCheck %s
; REQUIRES: asserts

; Check that -riscv-limit-load-cluster-succ-size=4 effectively disables all load clusterings
; such that the only clusterings happen here are store clustering.

; CHECK: Cluster ld/st SU([[S1:[0-9]+]]) - SU([[S2:[0-9]+]])
; CHECK: Cluster ld/st SU([[S2]]) - SU([[S3:[0-9]+]])
; CHECK: Cluster ld/st SU([[S3]]) - SU([[S4:[0-9]+]])

define void @eo_fermion_force.for.body.i() {
entry:
  %0 = load ptr, ptr null, align 8
  %imag11.i = getelementptr i8, ptr %0, i64 8
  %1 = load double, ptr null, align 8
  %2 = load double, ptr inttoptr (i64 8 to ptr), align 8
  %3 = load double, ptr %0, align 8
  %4 = load double, ptr %imag11.i, align 8
  %mul13.i = fmul double %4, %1
  %mul14.i = fmul contract double %3, %2
  %add.i = fadd contract double %mul13.i, %mul14.i
; CHECK: SU([[S1]]):   SD
  store double 0.000000e+00, ptr null, align 8
; CHECK: SU([[S2]]):   FSD
  store double %add.i, ptr inttoptr (i64 8 to ptr), align 8
; CHECK: SU([[S3]]):   SD
  store double 0.000000e+00, ptr inttoptr (i64 16 to ptr), align 8
; CHECK: SU([[S4]]):   SD
  store double 0.000000e+00, ptr inttoptr (i64 24 to ptr), align 8
  ret void
}
