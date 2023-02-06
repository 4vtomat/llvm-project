# RUN: llvm-mc -triple riscv64-unknown-linux-gnu -filetype obj -o - %s \
# RUN:   | llvm-readobj -r - | FileCheck %s

# CHECK: Relocations [
# CHECK:         0x0 R_RISCV_ADD64 __global_pointer$ 0x0
# CHECK-NEXT:    0x0 R_RISCV_SUB64 .Ltmp0 0x0
# CHECK:  }
# CHECK-NEXT:]
        .section        .text.__global_pointer__,"aGM",@progbits,8,__global_pointer__,comdat
        .p2align        3, 0x0
        .globl  __global_pointer__
        .hidden __global_pointer__
        .type   __global_pointer__,@object
__global_pointer__:
.Ltmp0:
        .quad   __global_pointer$-.Ltmp0
