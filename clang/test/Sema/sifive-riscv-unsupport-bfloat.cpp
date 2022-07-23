// RUN: %clang_cc1 %s -fsyntax-only -triple riscv64-elf -verify
// RUN: %clang_cc1 %s -fsyntax-only -triple riscv64-elf \
// RUN:            -target-feature +xsfvfwmaccqqq -target-feature +zve32f \
// RUN:            -target-feature +f -verify
// RUN: %clang_cc1 %s -fsyntax-only -triple riscv64-elf \
// RUN:            -target-feature +xsfvfhbfmin -target-feature +zve32f \
// RUN:            -target-feature +f -verify

// REQUIRES: riscv-registered-target

__bf16 a; // expected-error {{__bf16 is not supported on this target}}
