; Test that -inline-param-size works.
; RUN: opt < %s -O2 -inline-param-size=true -S  | FileCheck %s
; RUN: opt < %s -O3 -inline-param-size=true -S  | FileCheck %s
; RUN: opt < %s -Os -inline-param-size=true -S  | FileCheck %s
; RUN: opt < %s -Oz -inline-param-size=true -S  | FileCheck %s

@a = global i32 4

define i32 @simpleInline(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) #0 {
entry:
  %a1 = load volatile i32, ptr @a
  %x1 = add i32 %a1,  %a
  ret i32 %x1
}

define i32 @simpleNoInline(i32 %a) #0 {
entry:
  %a1 = load volatile i32, ptr @a
  %x1 = add i32 %a1,  %a
  ret i32 %x1
}

; Function Attrs: nounwind readnone uwtable
define i32 @bar(i32 %a) #0 {
; CHECK-LABEL: @bar
; CHECK: load volatile
; CHECK-NEXT: add i32
; CHECK: ret
entry:
  %i = tail call i32 @simpleInline(i32 6, i32 7, i32 8, i32 9, i32 10, i32 11) "function-inline-cost"="749"
  ret i32 %i
}

; Function Attrs: nounwind readnone uwtable
define i32 @foo(i32 %a) #0 {
; CHECK-LABEL: @foo
; CHECK: call i32 @simpleNoInline
; CHECK: ret
entry:
  %i = tail call i32 @simpleNoInline(i32 6) "function-inline-cost"="749"
  ret i32 %i
}


attributes #0 = { nounwind readnone uwtable }
attributes #0 = { nounwind readnone uwtable }
