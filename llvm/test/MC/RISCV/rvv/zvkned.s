#SIFIVE
# RUN: not llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+experimental-zvkned %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvkned %s \
# RUN:        | not llvm-objdump -d --mattr=+zve32x --mattr=+experimental-zvkned  - 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN:not  llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvkned %s \
# RUN:        | not llvm-objdump -d - 2>&1 | FileCheck %s --check-prefix=CHECK-UNKNOWN

vaesdf.vv v10, v9
# CHECK-INST-NOT-NOT: vaesdf.vv v10, v9
# CHECK-ENCODING-NOT-NOT: [0x77,0xa5,0x90,0xa2]
# CHECK-ERROR-NOT-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT-NOT: 77 a5 90 a2   <unknown>

vaesdf.vs v10, v9
# CHECK-INST-NOT: vaesdf.vs v10, v9
# CHECK-ENCODING-NOT: [0x77,0xa5,0x90,0xa6]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 a5 90 a6   <unknown>

vaesef.vv v10, v9
# CHECK-INST-NOT: vaesef.vv v10, v9
# CHECK-ENCODING-NOT: [0x77,0xa5,0x91,0xa2]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 a5 91 a2   <unknown>
                       
vaesef.vs v10, v9
# CHECK-INST-NOT: vaesef.vs v10, v9
# CHECK-ENCODING-NOT: [0x77,0xa5,0x91,0xa6]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 a5 91 a6   <unknown>

vaesdm.vv v10, v9
# CHECK-INST-NOT: vaesdm.vv v10, v9
# CHECK-ENCODING-NOT: [0x77,0x25,0x90,0xa2]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 25 90 a2   <unknown>
                       
vaesdm.vs v10, v9
# CHECK-INST-NOT: vaesdm.vs v10, v9
# CHECK-ENCODING-NOT: [0x77,0x25,0x90,0xa6]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 25 90 a6   <unknown>

vaesem.vv v10, v9
# CHECK-INST-NOT: vaesem.vv v10, v9
# CHECK-ENCODING-NOT: [0x77,0x25,0x91,0xa2]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 25 91 a2   <unknown>
                       
vaesem.vs v10, v9
# CHECK-INST-NOT: vaesem.vs v10, v9
# CHECK-ENCODING-NOT: [0x77,0x25,0x91,0xa6]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 25 91 a6   <unknown>

vaeskf1.vi v10, v9, 1
# CHECK-INST-NOT: vaeskf1.vi v10, v9, 1
# CHECK-ENCODING-NOT: [0x77,0xa5,0x90,0x8a]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 a5 90 8a   <unknown>

vaeskf2.vi v10, v9, 2
# CHECK-INST-NOT: vaeskf2.vi v10, v9, 2
# CHECK-ENCODING-NOT: [0x77,0x25,0x91,0xaa]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 25 91 aa   <unknown>

vaesz.vs v10, v9
# CHECK-INST-NOT: vaesz.vs v10, v9
# CHECK-ENCODING-NOT: [0x77,0xa5,0x93,0xa6]
# CHECK-ERROR-NOT: instruction requires the following: 'Zvkned' (Vector AES Encryption & Decryption (Single Round)){{$}}
# CHECK-UNKNOWN-NOT: 77 a5 93 a6   <unknown>
