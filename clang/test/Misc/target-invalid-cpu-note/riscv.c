// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple riscv32 -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix RISCV32
// RISCV32: error: unknown target CPU 'not-a-cpu'
// RISCV32-NEXT: note: valid target CPU values are:
// RISCV32-SAME: {{^}} generic-rv32
// RISCV32-SAME: {{^}}, rocket-rv32
// RISCV32-SAME: {{^}}, rp2350-hazard3
// RISCV32-SAME: {{^}}, sifive-e20
// RISCV32-SAME: {{^}}, sifive-e20r
// RISCV32-SAME: {{^}}, sifive-e21
// RISCV32-SAME: {{^}}, sifive-e21r
// RISCV32-SAME: {{^}}, sifive-e24
// RISCV32-SAME: {{^}}, sifive-e24r
// RISCV32-SAME: {{^}}, sifive-e31
// RISCV32-SAME: {{^}}, sifive-e34
// RISCV32-SAME: {{^}}, sifive-e61
// RISCV32-SAME: {{^}}, sifive-e61r
// RISCV32-SAME: {{^}}, sifive-e76
// RISCV32-SAME: {{^}}, sifive-e76m
// RISCV32-SAME: {{^}}, sifive-e76n
// RISCV32-SAME: {{^}}, sifive-e76r
// RISCV32-SAME: {{^}}, syntacore-scr1-base
// RISCV32-SAME: {{^}}, syntacore-scr1-max
// RISCV32-SAME: {{^}}, syntacore-scr3-rv32
// RISCV32-SAME: {{^}}, syntacore-scr4-rv32
// RISCV32-SAME: {{^}}, syntacore-scr5-rv32
// RISCV32-SAME: {{$}}

// RUN: not %clang_cc1 -triple riscv64 -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix RISCV64
// RISCV64: error: unknown target CPU 'not-a-cpu'
// RISCV64-NEXT: note: valid target CPU values are:
// RISCV64-SAME: {{^}} generic-rv64
// RISCV64-SAME: {{^}}, rocket-rv64
// RISCV64-SAME: {{^}}, sifive-cheetah
// RISCV64-SAME: {{^}}, sifive-leopard
// RISCV64-SAME: {{^}}, sifive-lion
// RISCV64-SAME: {{^}}, sifive-p270
// RISCV64-SAME: {{^}}, sifive-p270n
// RISCV64-SAME: {{^}}, sifive-p450
// RISCV64-SAME: {{^}}, sifive-p470
// RISCV64-SAME: {{^}}, sifive-p550
// RISCV64-SAME: {{^}}, sifive-p650
// RISCV64-SAME: {{^}}, sifive-p670
// RISCV64-SAME: {{^}}, sifive-s21
// RISCV64-SAME: {{^}}, sifive-s21r
// RISCV64-SAME: {{^}}, sifive-s51
// RISCV64-SAME: {{^}}, sifive-s54
// RISCV64-SAME: {{^}}, sifive-s61
// RISCV64-SAME: {{^}}, sifive-s61r
// RISCV64-SAME: {{^}}, sifive-s76
// RISCV64-SAME: {{^}}, sifive-s76m
// RISCV64-SAME: {{^}}, sifive-s76n
// RISCV64-SAME: {{^}}, sifive-s76r
// RISCV64-SAME: {{^}}, sifive-tiger
// RISCV64-SAME: {{^}}, sifive-u54
// RISCV64-SAME: {{^}}, sifive-u64
// RISCV64-SAME: {{^}}, sifive-u64r
// RISCV64-SAME: {{^}}, sifive-u74
// RISCV64-SAME: {{^}}, sifive-u74m
// RISCV64-SAME: {{^}}, sifive-u74n
// RISCV64-SAME: {{^}}, sifive-u74r
// RISCV64-SAME: {{^}}, sifive-x280
// RISCV64-SAME: {{^}}, sifive-x280n
// RISCV64-SAME: {{^}}, sifive-x280o
// RISCV64-SAME: {{^}}, sifive-x390-fast-fp64
// RISCV64-SAME: {{^}}, sifive-x390
// RISCV64-SAME: {{^}}, spacemit-x60
// RISCV64-SAME: {{^}}, syntacore-scr3-rv64
// RISCV64-SAME: {{^}}, syntacore-scr4-rv64
// RISCV64-SAME: {{^}}, syntacore-scr5-rv64
// RISCV64-SAME: {{^}}, syntacore-scr7
// RISCV64-SAME: {{^}}, veyron-v1
// RISCV64-SAME: {{^}}, xiangshan-nanhu
// RISCV64-SAME: {{$}}

// RUN: not %clang_cc1 -triple riscv32 -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE-RISCV32
// TUNE-RISCV32: error: unknown target CPU 'not-a-cpu'
// TUNE-RISCV32-NEXT: note: valid target CPU values are:
// TUNE-RISCV32-SAME: {{^}} generic-rv32
// TUNE-RISCV32-SAME: {{^}}, rocket-rv32
// TUNE-RISCV32-SAME: {{^}}, rp2350-hazard3
// TUNE-RISCV32-SAME: {{^}}, sifive-e20
// TUNE-RISCV32-SAME: {{^}}, sifive-e20r
// TUNE-RISCV32-SAME: {{^}}, sifive-e21
// TUNE-RISCV32-SAME: {{^}}, sifive-e21r
// TUNE-RISCV32-SAME: {{^}}, sifive-e24
// TUNE-RISCV32-SAME: {{^}}, sifive-e24r
// TUNE-RISCV32-SAME: {{^}}, sifive-e31
// TUNE-RISCV32-SAME: {{^}}, sifive-e34
// TUNE-RISCV32-SAME: {{^}}, sifive-e61
// TUNE-RISCV32-SAME: {{^}}, sifive-e61r
// TUNE-RISCV32-SAME: {{^}}, sifive-e76
// TUNE-RISCV32-SAME: {{^}}, sifive-e76m
// TUNE-RISCV32-SAME: {{^}}, sifive-e76n
// TUNE-RISCV32-SAME: {{^}}, sifive-e76r
// TUNE-RISCV32-SAME: {{^}}, syntacore-scr1-base
// TUNE-RISCV32-SAME: {{^}}, syntacore-scr1-max
// TUNE-RISCV32-SAME: {{^}}, syntacore-scr3-rv32
// TUNE-RISCV32-SAME: {{^}}, syntacore-scr4-rv32
// TUNE-RISCV32-SAME: {{^}}, syntacore-scr5-rv32
// TUNE-RISCV32-SAME: {{^}}, generic
// TUNE-RISCV32-SAME: {{^}}, rocket
// TUNE-RISCV32-SAME: {{^}}, sifive-6-series
// TUNE-RISCV32-SAME: {{^}}, sifive-7-series
// TUNE-RISCV32-SAME: {{^}}, sifive-7m-series
// TUNE-RISCV32-SAME: {{^}}, sifive-7n-series
// TUNE-RISCV32-SAME: {{^}}, sifive-7o-series
// TUNE-RISCV32-SAME: {{^}}, sifive-p400-series
// TUNE-RISCV32-SAME: {{^}}, sifive-p500-series
// TUNE-RISCV32-SAME: {{^}}, sifive-p600-series
// TUNE-RISCV32-SAME: {{$}}

// RUN: not %clang_cc1 -triple riscv64 -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE-RISCV64
// TUNE-RISCV64: error: unknown target CPU 'not-a-cpu'
// TUNE-RISCV64-NEXT: note: valid target CPU values are:
// TUNE-RISCV64-SAME: {{^}} generic-rv64
// TUNE-RISCV64-SAME: {{^}}, rocket-rv64
// TUNE-RISCV64-SAME: {{^}}, sifive-cheetah
// TUNE-RISCV64-SAME: {{^}}, sifive-leopard
// TUNE-RISCV64-SAME: {{^}}, sifive-lion
// TUNE-RISCV64-SAME: {{^}}, sifive-p270
// TUNE-RISCV64-SAME: {{^}}, sifive-p270n
// TUNE-RISCV64-SAME: {{^}}, sifive-p450
// TUNE-RISCV64-SAME: {{^}}, sifive-p470
// TUNE-RISCV64-SAME: {{^}}, sifive-p550
// TUNE-RISCV64-SAME: {{^}}, sifive-p650
// TUNE-RISCV64-SAME: {{^}}, sifive-p670
// TUNE-RISCV64-SAME: {{^}}, sifive-s21
// TUNE-RISCV64-SAME: {{^}}, sifive-s21r
// TUNE-RISCV64-SAME: {{^}}, sifive-s51
// TUNE-RISCV64-SAME: {{^}}, sifive-s54
// TUNE-RISCV64-SAME: {{^}}, sifive-s61
// TUNE-RISCV64-SAME: {{^}}, sifive-s61r
// TUNE-RISCV64-SAME: {{^}}, sifive-s76
// TUNE-RISCV64-SAME: {{^}}, sifive-s76m
// TUNE-RISCV64-SAME: {{^}}, sifive-s76n
// TUNE-RISCV64-SAME: {{^}}, sifive-s76r
// TUNE-RISCV64-SAME: {{^}}, sifive-tiger
// TUNE-RISCV64-SAME: {{^}}, sifive-u54
// TUNE-RISCV64-SAME: {{^}}, sifive-u64
// TUNE-RISCV64-SAME: {{^}}, sifive-u64r
// TUNE-RISCV64-SAME: {{^}}, sifive-u74
// TUNE-RISCV64-SAME: {{^}}, sifive-u74m
// TUNE-RISCV64-SAME: {{^}}, sifive-u74n
// TUNE-RISCV64-SAME: {{^}}, sifive-u74r
// TUNE-RISCV64-SAME: {{^}}, sifive-x280
// TUNE-RISCV64-SAME: {{^}}, sifive-x280n
// TUNE-RISCV64-SAME: {{^}}, sifive-x280o
// TUNE-RISCV64-SAME: {{^}}, sifive-x390-fast-fp64
// TUNE-RISCV64-SAME: {{^}}, sifive-x390
// TUNE-RISCV64-SAME: {{^}}, spacemit-x60
// TUNE-RISCV64-SAME: {{^}}, syntacore-scr3-rv64
// TUNE-RISCV64-SAME: {{^}}, syntacore-scr4-rv64
// TUNE-RISCV64-SAME: {{^}}, syntacore-scr5-rv64
// TUNE-RISCV64-SAME: {{^}}, syntacore-scr7
// TUNE-RISCV64-SAME: {{^}}, veyron-v1
// TUNE-RISCV64-SAME: {{^}}, xiangshan-nanhu
// TUNE-RISCV64-SAME: {{^}}, generic
// TUNE-RISCV64-SAME: {{^}}, rocket
// TUNE-RISCV64-SAME: {{^}}, sifive-6-series
// TUNE-RISCV64-SAME: {{^}}, sifive-7-series
// TUNE-RISCV64-SAME: {{^}}, sifive-7m-series
// TUNE-RISCV64-SAME: {{^}}, sifive-7n-series
// TUNE-RISCV64-SAME: {{^}}, sifive-7o-series
// TUNE-RISCV64-SAME: {{^}}, sifive-p400-series
// TUNE-RISCV64-SAME: {{^}}, sifive-p500-series
// TUNE-RISCV64-SAME: {{^}}, sifive-p600-series
// TUNE-RISCV64-SAME: {{$}}
