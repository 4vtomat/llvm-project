# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELAX-RELOC %s

    .section .text.__global_pointer__,"aGM",@progbits,8,__global_pointer__,comdat
    .p2align 3
    .globl  __global_pointer__
    .hidden __global_pointer__
    .type   __global_pointer__,@object
__global_pointer__:
.Ltmp0:
    .quad   __global_pointer$ - .Ltmp0
# RELAX-RELOC: R_RISCV_ALIGN - 0x4
# RELAX-RELOC: R_RISCV_ADD64 __global_pointer$ 0x0
# RELAX-RELOC: R_RISCV_SUB64 .Ltmp0 0x0
