; RUN: opt < %s -S -passes=loop-data-layout -loop-data-layout-enable -mtriple=riscv64-unknown-linux-gnu | FileCheck %s
; This is just a simple compile test for avoiding a crash on FreezeInst.

define i32 @main() {
; CHECK-LABEL: @main(
;
entry:
  br label %cleanup
if.then3:
  %call122 = call fastcc i32 @scanmanager()
  br label %cleanup
cleanup:
  ret i32 0
}
define internal fastcc i32 @scanmanager() {
; CHECK-LABEL: @scanmanager(
;
entry:
  br label %cleanup370
if.then267:
  %call268 = call fastcc i32 @scanstdin()
  br label %cleanup370
cleanup370:
  ret i32 0
}
define internal fastcc i32 @scanstdin() {
; CHECK-LABEL: @scanstdin(
;
entry:
  br label %cleanup
while.cond:
  %call.i.i = call fastcc i32 @cli_magic_scandesc()
  br label %cleanup
cleanup:
  ret i32 0
}
define internal fastcc i32 @cli_scanpe() {
; CHECK-LABEL: @cli_scanpe(
;
entry:
  br label %cleanup4880
if.end4039:
  %call4043 = call fastcc i32 @unspin()
  br label %cleanup4880
cleanup4880:
  ret i32 0
}
define internal fastcc i32 @cli_magic_scandesc() {
; CHECK-LABEL: @cli_magic_scandesc(
;
entry:
  br label %cleanup
if.then413:
  %call414 = call fastcc i32 @cli_scanpe()
  br label %cleanup
cleanup:
  ret i32 0
}
define internal fastcc i32 @unspin() {
; CHECK-LABEL: @unspin(
;
entry:
  br label %cleanup791
if.then.i35:
  %add.ptr179.fr = freeze ptr null
  br label %cleanup791
land.lhs.true336:
  %call356 = call fastcc i8 @exec86(ptr %add.ptr179.fr)
  br label %cleanup791
cleanup791:
  ret i32 0
}
define internal fastcc i8 @exec86(ptr %curremu) {
; CHECK-LABEL: @exec86(
;
entry:
  ret i8 0
}
