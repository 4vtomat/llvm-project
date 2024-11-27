# RUN: not llvm-mc -triple riscv32 -mattr=+xsfsci %s 2>&1 | FileCheck %s

sf.sci 1, 0, a0, a1, a2 # CHECK: :[[@LINE]]:8: error: immediate must be a multiple of 2 bytes in the range [0, 6]
