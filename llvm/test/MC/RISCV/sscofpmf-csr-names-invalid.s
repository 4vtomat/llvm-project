# RUN: not llvm-mc -triple riscv64 -mattr=+zicsr < %s 2>&1 \
# RUN:   | FileCheck -check-prefixes=CHECK-NEED-RV32 %s

# These machine mode CSR register names are RV32 only.

csrrs t1, mhpmevent3h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent4h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent5h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent6h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent7h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent8h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent9h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent10h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent11h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent12h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent13h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent14h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent15h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent16h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent17h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent18h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent19h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent20h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent21h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent22h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent23h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent24h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent25h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent26h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent27h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent28h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent29h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent30h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
csrrs t1, mhpmevent31h, zero # CHECK-NEED-RV32: :[[@LINE]]:11: error: system register use requires an option to be enabled
