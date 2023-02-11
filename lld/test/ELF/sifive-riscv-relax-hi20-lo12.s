# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax %s -o %t.rv64.o
# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+c,+relax %s -o %t.rv32c.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+c,+relax %s -o %t.rv64c.o

# RUN: echo 'SECTIONS { .text : { *(.text) } .sdata 0x200000 : { foo = .; } }' > %t.lds
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ %t.rv32.o %t.lds -o %t.rv32
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ %t.rv64.o %t.lds -o %t.rv64
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t.rv32 | FileCheck --check-prefix=GP %s
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t.rv64 | FileCheck --check-prefix=GP %s
# GP-NOT:  lui
# GP:      addi    a0, gp, -2048
# GP-NEXT: lw      a0, -2048(gp)
# GP-NEXT: sw      a0, -2048(gp)

# RUN: echo 'SECTIONS { .text : { *(.text) } .sdata 0x200000 : { foo = . + 4096; } }' > %t-out-of-range.lds
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ %t.rv32.o %t-out-of-range.lds -o %t.rv32-out-of-range
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ %t.rv64.o %t-out-of-range.lds -o %t.rv64-out-of-range
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t.rv32-out-of-range | FileCheck --check-prefix=NORELAX %s
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t.rv64-out-of-range | FileCheck --check-prefix=NORELAX %s
# NORELAX:      lui     a0, 513
# NORELAX-NEXT: addi    a0, a0, 0
# NORELAX-NEXT: lw      a0, 0(a0)
# NORELAX-NEXT: sw      a0, 0(a0)

.global _start
_start:
  lui a0, %hi(foo)
  addi a0, a0, %lo(foo)
  lw a0, %lo(foo)(a0)
  sw a0, %lo(foo)(a0)
