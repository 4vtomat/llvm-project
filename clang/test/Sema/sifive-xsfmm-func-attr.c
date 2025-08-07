// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +xsfmmbase -fsyntax-only -verify %s

int xsfmm_in_callee(void) __xsfmm_in;
int xsfmm_out_callee(void) __xsfmm_out;
int xsfmm_inout_callee(void) __xsfmm_inout;
int xsfmm_preserves_callee(void) __xsfmm_preserves;
int xsfmm_new_callee(void) __xsfmm_new;

void valid_new(void) {
  xsfmm_new_callee();
}

void valid_preserves_in_in(void) __xsfmm_in {
  xsfmm_preserves_callee();
}

void valid_in_in_in(void) __xsfmm_in {
  xsfmm_in_callee();
}

void valid_in_in_out(void) __xsfmm_out {
  xsfmm_in_callee();
}

void valid_out_in_out(void) __xsfmm_out {
  xsfmm_out_callee();
}

void valid_inout_in_out(void) __xsfmm_out {
  xsfmm_inout_callee();
}

void valid_preserves_in_out(void) __xsfmm_out {
  xsfmm_preserves_callee();
}

void valid_preserves_in_preserves(void) __xsfmm_preserves {
  xsfmm_preserves_callee();
}


void invalid_mutual_exclusive1(void) __xsfmm_in __xsfmm_out { // expected-error {{mutual exclusive attributes for state '__xsfmm_out' and '__xsfmm_in'}}
}

void invalid_mutual_exclusive2(void) __xsfmm_in __xsfmm_inout { // expected-error {{mutual exclusive attributes for state '__xsfmm_inout' and '__xsfmm_in'}}
}

void invalid_mutual_exclusive3(void) __xsfmm_in __xsfmm_preserves { // expected-error {{mutual exclusive attributes for state '__xsfmm_preserves' and '__xsfmm_in'}}
}

void invalid_mutual_exclusive4(void) __xsfmm_in __xsfmm_new { // expected-error {{mutual exclusive attributes for state '__xsfmm_new' and '__xsfmm_in'}}
}

void invalid_mutual_exclusive5(void) __xsfmm_out __xsfmm_inout { // expected-error {{mutual exclusive attributes for state '__xsfmm_inout' and '__xsfmm_out'}}
}

void invalid_mutual_exclusive6(void) __xsfmm_out __xsfmm_preserves { // expected-error {{mutual exclusive attributes for state '__xsfmm_preserves' and '__xsfmm_out'}}
}

void invalid_mutual_exclusive7(void) __xsfmm_out __xsfmm_new { // expected-error {{mutual exclusive attributes for state '__xsfmm_new' and '__xsfmm_out'}}
}

void invalid_mutual_exclusive8(void) __xsfmm_inout __xsfmm_preserves { // expected-error {{mutual exclusive attributes for state '__xsfmm_preserves' and '__xsfmm_inout'}}
}

void invalid_mutual_exclusive9(void) __xsfmm_inout __xsfmm_new { // expected-error {{mutual exclusive attributes for state '__xsfmm_new' and '__xsfmm_inout'}}
}

void invalid_mutual_exclusive10(void) __xsfmm_preserves __xsfmm_new { // expected-error {{mutual exclusive attributes for state '__xsfmm_new' and '__xsfmm_preserves'}}
}

void invalid_in_in_preserves(void) __xsfmm_preserves {
  xsfmm_in_callee(); // expected-error {{calling function with attribute __xsfmm_in within function with attribute '__xsfmm_preserves'}}
}

void invalid_out_in_preserves(void) __xsfmm_preserves {
  xsfmm_out_callee(); // expected-error {{calling function with attribute __xsfmm_out within function with attribute '__xsfmm_preserves'}}
}

void invalid_inout_in_preserves(void) __xsfmm_preserves {
  xsfmm_inout_callee(); // expected-error {{calling function with attribute __xsfmm_inout within function with attribute '__xsfmm_preserves'}}
}

void invalid_new_in_preserves(void) __xsfmm_preserves {
  xsfmm_new_callee(); // expected-error {{calling function with attribute __xsfmm_new within function with attribute '__xsfmm_preserves'}}
}

void invalid_out_in_in(void) __xsfmm_in {
  xsfmm_out_callee(); // expected-error {{calling function with attribute __xsfmm_out within function with attribute '__xsfmm_in'}}
}

void invalid_inout_in_in(void) __xsfmm_in {
  xsfmm_inout_callee(); // expected-error {{calling function with attribute __xsfmm_inout within function with attribute '__xsfmm_in'}}
}

void invalid_new_in_in(void) __xsfmm_in {
  xsfmm_new_callee(); // expected-error {{calling function with attribute __xsfmm_new within function with attribute '__xsfmm_in'}}
}

void invalid_new_in_out(void) __xsfmm_in {
  xsfmm_new_callee(); // expected-error {{calling function with attribute __xsfmm_new within function with attribute '__xsfmm_in'}}
}

void invalid_new_in_new(void) __xsfmm_new {
  xsfmm_new_callee(); // expected-error {{calling function with attribute __xsfmm_new within function with attribute '__xsfmm_new'}}
}

