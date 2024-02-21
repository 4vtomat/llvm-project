## Arch string without version.

# RUN: llvm-mc %s -triple=riscv32 -filetype=asm | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm \
# RUN:     | FileCheck --check-prefixes=CHECK %s

.attribute arch, "rv32izicclsm"
# CHECK: attribute      5, "rv32i2p1_zicclsm1p0"

.attribute arch, "rv32iziccif"
# CHECK: attribute      5, "rv32i2p1_ziccif1p0"

.attribute arch, "rv32iziccamoa"
# CHECK: attribute      5, "rv32i2p1_ziccamoa1p0"

.attribute arch, "rv32iziccrse"
# CHECK: attribute      5, "rv32i2p1_ziccrse1p0"

.attribute arch, "rv32iza64rs"
# CHECK: attribute      5, "rv32i2p1_za64rs1p0"

.attribute arch, "rv32izic64b"
# CHECK: attribute      5, "rv32i2p1_zic64b1p0"

.attribute arch, "rv32izjid0p0"
# CHECK: attribute      5, "rv32i2p1_zjid0p0"

.attribute arch, "rv32i_zicsr_zicntr"
# CHECK: attribute      5, "rv32i2p1_zicntr2p0_zicsr2p0"

.attribute arch, "rv32i_zicsr_zihpm"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zihpm2p0"

.attribute arch, "rv32isdext"
# CHECK: attribute      5, "rv32i2p1_sdext1p0"

.attribute arch, "rv32isdtrig"
# CHECK: attribute      5, "rv32i2p1_sdtrig1p0"

.attribute arch, "rv32iss"
# CHECK: attribute      5, "rv32i2p1_ss1p12"

.attribute arch, "rv32isvbare"
# CHECK: attribute      5, "rv32i2p1_svbare1p0"

.attribute arch, "rv32isvade"
# CHECK: attribute      5, "rv32i2p1_svade1p0"

.attribute arch, "rv32isvadu"
# CHECK: attribute      5, "rv32i2p1_svadu1p0"

.attribute arch, "rv32issccptr"
# CHECK: attribute      5, "rv32i2p1_ssccptr1p0"

.attribute arch, "rv32isstvecd"
# CHECK: attribute      5, "rv32i2p1_sstvecd1p0"

.attribute arch, "rv32isstvala"
# CHECK: attribute      5, "rv32i2p1_sstvala1p0"

.attribute arch, "rv32isscounterenw"
# CHECK: attribute      5, "rv32i2p1_sscounterenw1p0"

.attribute arch, "rv32issu64xl"
# CHECK: attribute      5, "rv32i2p1_ssu64xl1p0"

.attribute arch, "rv32isstc"
# CHECK: attribute      5, "rv32i2p1_sstc1p0"

.attribute arch, "rv32issstateen"
# CHECK: attribute      5, "rv32i2p1_ssstateen1p0"

.attribute arch, "rv32ismstateen"
# CHECK: attribute      5, "rv32i2p1_smstateen1p0"

.attribute arch, "rv32ishcounterenw"
# CHECK: attribute      5, "rv32i2p1_shcounterenw1p0"

.attribute arch, "rv32ishvstvala"
# CHECK: attribute      5, "rv32i2p1_shvstvala1p0"

.attribute arch, "rv32ishtvala"
# CHECK: attribute      5, "rv32i2p1_shtvala1p0"

.attribute arch, "rv32ishvstvecd"
# CHECK: attribute      5, "rv32i2p1_shvstvecd1p0"

.attribute arch, "rv32ishvsatpa"
# CHECK: attribute      5, "rv32i2p1_shvsatpa1p0"

.attribute arch, "rv32ishgatpa"
# CHECK: attribute      5, "rv32i2p1_shgatpa1p0"

.attribute arch, "rv32ismwg0p3"
# CHECK: attribute      5, "rv32i2p1_smwg0p3"

.attribute arch, "rv32i_smwg0p3_smwgd0p3"
# CHECK: attribute      5, "rv32i2p1_smwg0p3_smwgd0p3"

.attribute arch, "rv32i_sswg0p3"
# CHECK: attribute      5, "rv32i2p1_sswg0p3"

.attribute arch, "rv32i_ssnpm0p8"
# CHECK: attribute      5, "rv32i2p1_ssnpm0p8"

.attribute arch, "rv32i_sscofpmf"
# CHECK: attribute      5, "rv32i2p1_sscofpmf1p0"

.attribute arch, "rv32i_zve32x_zvkb0p1"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkb0p1_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvkg0p1"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkg0p1_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvknha0p1"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvknha0p1_zvl32b1p0"

.attribute arch, "rv32i_zve64x_zvknhb0p1"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvknhb0p1_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32i_zve32x_zvkns0p1"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkns0p1_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvksed0p1"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvksed0p1_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvksh0p1"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvksh0p1_zvl32b1p0"

.attribute arch, "rv32ifv_xsfvfnrclipxfqf0p1_xsfvfwmaccqqq0p1"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfbfmin1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfvfnrclipxfqf0p1_xsfvfwmaccqqq0p1"

.attribute arch, "rv32i_xsfpgflushdlone0p1"
# CHECK: attribute      5, "rv32i2p1_xsfpgflushdlone0p1"

.attribute arch, "rv32iv_zvl32b"
# CHECK: attribute     5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32iv_zvl64b"
# CHECK: attribute     5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32iv_zvl128b"
# CHECK: attribute     5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32iv_zvl256b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32iv_zvl512b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32iv_zvl1024b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32iv_zvl2048b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32iv_zvl4096b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32iv_zvl8192b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32iv_zvl16384b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32iv_zvl32768b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32768b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32iv_zvl65536b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32768b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl65536b1p0_zvl8192b1p0"

.attribute arch, "rv32i_zve32x"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32i_zve64x"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"
