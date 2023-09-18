; RUN: llc -mtriple=riscv64 -mattr=+cmov-branch-opt < %s

; Test it can run correctly.
define i64 @foo(i64 %x, i64 %y) {
entry:
  %tobool.not = icmp eq i64 %x, 0
  %cond = select i1 %tobool.not, i64 %y, i64 400
  ret i64 %cond
}
