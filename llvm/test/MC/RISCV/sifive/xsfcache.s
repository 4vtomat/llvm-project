# RUN: llvm-mc %s -triple=riscv32 --mattr=+experimental-xsifivecflushdlone --mattr=+experimental-xsifivecdiscarddlone \
# RUN:    -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 --mattr=+experimental-xsifivecflushdlone --mattr=+experimental-xsifivecdiscarddlone \
# RUN:    -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 --mattr=+experimental-xsifivecflushdlone --mattr=+experimental-xsifivecdiscarddlone < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xsifivecflushdlone --mattr=+experimental-xsifivecdiscarddlone -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 --mattr=+experimental-xsifivecflushdlone --mattr=+experimental-xsifivecdiscarddlone < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xsifivecflushdlone --mattr=+experimental-xsifivecdiscarddlone -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: cflush.d.l1
# CHECK: encoding: [0x73,0x00,0x05,0xfc]
cflush.d.l1 x10

# CHECK-INST: cdiscard.d.l1
# CHECK: encoding: [0x73,0x00,0x25,0xfc]
cdiscard.d.l1 x10

# CHECK-INST: cflush.i.l1
# CHECK: encoding: [0x73,0x00,0x10,0xfc]
cflush.i.l1
