# RUN: llvm-mc %s -triple=riscv32 --mattr=+zicsr -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 --mattr=+zicsr < %s \
# RUN:     | llvm-objdump -d --mattr=+zicsr - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s

##################################
# Sscofpmf CSRs
##################################

# mhpmevent3h
# name
# CHECK-INST: csrrs t1, mhpmevent3h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent3h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent3h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent3h
# name
csrrs t1, mhpmevent3h, zero
# uimm12
csrrs t2, 0x723, zero

# mhpmevent4h
# name
# CHECK-INST: csrrs t1, mhpmevent4h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent4h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent4h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent4h
# name
csrrs t1, mhpmevent4h, zero
# uimm12
csrrs t2, 0x724, zero

# mhpmevent5h
# name
# CHECK-INST: csrrs t1, mhpmevent5h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent5h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent5h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent5h
# name
csrrs t1, mhpmevent5h, zero
# uimm12
csrrs t2, 0x725, zero

# mhpmevent6h
# name
# CHECK-INST: csrrs t1, mhpmevent6h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent6h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent6h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent6h
# name
csrrs t1, mhpmevent6h, zero
# uimm12
csrrs t2, 0x726, zero

# mhpmevent7h
# name
# CHECK-INST: csrrs t1, mhpmevent7h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent7h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent7h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent7h
# name
csrrs t1, mhpmevent7h, zero
# uimm12
csrrs t2, 0x727, zero

# mhpmevent8h
# name
# CHECK-INST: csrrs t1, mhpmevent8h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent8h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent8h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent8h
# name
csrrs t1, mhpmevent8h, zero
# uimm12
csrrs t2, 0x728, zero

# mhpmevent9h
# name
# CHECK-INST: csrrs t1, mhpmevent9h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent9h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent9h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent9h
# name
csrrs t1, mhpmevent9h, zero
# uimm12
csrrs t2, 0x729, zero

# mhpmevent10h
# name
# CHECK-INST: csrrs t1, mhpmevent10h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent10h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent10h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent10h
# name
csrrs t1, mhpmevent10h, zero
# uimm12
csrrs t2, 0x72a, zero

# mhpmevent11h
# name
# CHECK-INST: csrrs t1, mhpmevent11h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent11h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent11h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent11h
# name
csrrs t1, mhpmevent11h, zero
# uimm12
csrrs t2, 0x72b, zero

# mhpmevent12h
# name
# CHECK-INST: csrrs t1, mhpmevent12h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent12h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent12h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent12h
# name
csrrs t1, mhpmevent12h, zero
# uimm12
csrrs t2, 0x72c, zero

# mhpmevent13h
# name
# CHECK-INST: csrrs t1, mhpmevent13h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent13h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent13h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent13h
# name
csrrs t1, mhpmevent13h, zero
# uimm12
csrrs t2, 0x72d, zero

# mhpmevent14h
# name
# CHECK-INST: csrrs t1, mhpmevent14h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent14h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent14h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent14h
# name
csrrs t1, mhpmevent14h, zero
# uimm12
csrrs t2, 0x72e, zero

# mhpmevent15h
# name
# CHECK-INST: csrrs t1, mhpmevent15h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0x72]
# CHECK-INST-ALIAS: csrr t1, mhpmevent15h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent15h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0x72]
# CHECK-INST-ALIAS: csrr t2, mhpmevent15h
# name
csrrs t1, mhpmevent15h, zero
# uimm12
csrrs t2, 0x72f, zero

# mhpmevent16h
# name
# CHECK-INST: csrrs t1, mhpmevent16h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x00,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent16h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent16h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x00,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent16h
# name
csrrs t1, mhpmevent16h, zero
# uimm12
csrrs t2, 0x730, zero

# mhpmevent17h
# name
# CHECK-INST: csrrs t1, mhpmevent17h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent17h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent17h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent17h
# name
csrrs t1, mhpmevent17h, zero
# uimm12
csrrs t2, 0x731, zero

# mhpmevent18h
# name
# CHECK-INST: csrrs t1, mhpmevent18h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x20,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent18h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent18h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x20,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent18h
# name
csrrs t1, mhpmevent18h, zero
# uimm12
csrrs t2, 0x732, zero

# mhpmevent19h
# name
# CHECK-INST: csrrs t1, mhpmevent19h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x30,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent19h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent19h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x30,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent19h
# name
csrrs t1, mhpmevent19h, zero
# uimm12
csrrs t2, 0x733, zero

# mhpmevent20h
# name
# CHECK-INST: csrrs t1, mhpmevent20h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x40,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent20h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent20h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x40,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent20h
# name
csrrs t1, mhpmevent20h, zero
# uimm12
csrrs t2, 0x734, zero

# mhpmevent21h
# name
# CHECK-INST: csrrs t1, mhpmevent21h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x50,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent21h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent21h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x50,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent21h
# name
csrrs t1, mhpmevent21h, zero
# uimm12
csrrs t2, 0x735, zero

# mhpmevent22h
# name
# CHECK-INST: csrrs t1, mhpmevent22h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x60,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent22h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent22h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x60,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent22h
# name
csrrs t1, mhpmevent22h, zero
# uimm12
csrrs t2, 0x736, zero

# mhpmevent23h
# name
# CHECK-INST: csrrs t1, mhpmevent23h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x70,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent23h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent23h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x70,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent23h
# name
csrrs t1, mhpmevent23h, zero
# uimm12
csrrs t2, 0x737, zero

# mhpmevent24h
# name
# CHECK-INST: csrrs t1, mhpmevent24h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x80,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent24h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent24h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x80,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent24h
# name
csrrs t1, mhpmevent24h, zero
# uimm12
csrrs t2, 0x738, zero

# mhpmevent25h
# name
# CHECK-INST: csrrs t1, mhpmevent25h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x90,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent25h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent25h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x90,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent25h
# name
csrrs t1, mhpmevent25h, zero
# uimm12
csrrs t2, 0x739, zero

# mhpmevent26h
# name
# CHECK-INST: csrrs t1, mhpmevent26h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xa0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent26h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent26h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xa0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent26h
# name
csrrs t1, mhpmevent26h, zero
# uimm12
csrrs t2, 0x73a, zero

# mhpmevent27h
# name
# CHECK-INST: csrrs t1, mhpmevent27h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xb0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent27h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent27h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xb0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent27h
# name
csrrs t1, mhpmevent27h, zero
# uimm12
csrrs t2, 0x73b, zero

# mhpmevent28h
# name
# CHECK-INST: csrrs t1, mhpmevent28h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xc0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent28h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent28h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xc0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent28h
# name
csrrs t1, mhpmevent28h, zero
# uimm12
csrrs t2, 0x73c, zero

# mhpmevent29h
# name
# CHECK-INST: csrrs t1, mhpmevent29h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xd0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent29h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent29h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xd0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent29h
# name
csrrs t1, mhpmevent29h, zero
# uimm12
csrrs t2, 0x73d, zero

# mhpmevent30h
# name
# CHECK-INST: csrrs t1, mhpmevent30h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xe0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent30h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent30h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xe0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent30h
# name
csrrs t1, mhpmevent30h, zero
# uimm12
csrrs t2, 0x73e, zero

# mhpmevent31h
# name
# CHECK-INST: csrrs t1, mhpmevent31h, zero
# CHECK-ENC:  encoding: [0x73,0x23,0xf0,0x73]
# CHECK-INST-ALIAS: csrr t1, mhpmevent31h
# uimm12
# CHECK-INST: csrrs t2, mhpmevent31h, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0xf0,0x73]
# CHECK-INST-ALIAS: csrr t2, mhpmevent31h
# name
csrrs t1, mhpmevent31h, zero
# uimm12
csrrs t2, 0x73f, zero
