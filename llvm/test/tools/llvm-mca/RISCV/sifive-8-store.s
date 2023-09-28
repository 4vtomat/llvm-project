# RUN: llvm-mca -mtriple=riscv64 -mcpu=sifive-p550 -timeline -iterations=1 < %s | FileCheck %s

# LLVM-MCA-BEGIN
sb      a0, 0(a2)
sh      a0, 0(a2)
sw      a0, 0(a2)
sd      a0, 0(a2)
fsw    fa0, 0(a2)
fsd    fa0, 0(a2)
# LLVM-MCA-END

# CHECK:      Timeline view:
# CHECK-NEXT: Index     012345678

# CHECK:      [0,0]     DeER .  .   sb	a0, 0(a2)
# CHECK-NEXT: [0,1]     D=eER.  .   sh	a0, 0(a2)
# CHECK-NEXT: [0,2]     D==eER  .   sw	a0, 0(a2)
# CHECK-NEXT: [0,3]     .D==eER .   sd	a0, 0(a2)
# CHECK-NEXT: [0,4]     .D===eER.   fsw	fa0, 0(a2)
# CHECK-NEXT: [0,5]     .D====eER   fsd	fa0, 0(a2)
