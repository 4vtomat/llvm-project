## Arch string without version.

# RUN: llvm-mc %s -triple=riscv32 -filetype=asm | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm | FileCheck %s

.attribute arch, "rv32i"
# CHECK: attribute      5, "rv32i2p1"

.attribute arch, "rv32i2p1"
# CHECK: attribute      5, "rv32i2p1"

.attribute arch, "rv32i2p1_m2"
# CHECK: attribute      5, "rv32i2p1_m2p0"

.attribute arch, "rv32i2p1_ma"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1"

.attribute arch, "rv32g"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zifencei2p0"

.attribute arch, "rv32imafdc"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"

.attribute arch, "rv32i2p1_mafdc"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"

.attribute arch, "rv32ima2p1_fdc"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"

.attribute arch, "rv32ima2p1_fdc"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"

.attribute arch, "rv32iv"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl32b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl64b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl128b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl256b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl512b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl1024b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl2048b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl4096b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl8192b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32ivzvl16384b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32ivzvl32768b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32768b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32ivzvl65536b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32768b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl65536b1p0_zvl8192b1p0"

.attribute arch, "rv32izve32x"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32ifzve32f"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32izve64x"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ifzve64f"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ifdzve64d"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32izicbom"
# CHECK: attribute      5, "rv32i2p1_zicbom1p0"

.attribute arch, "rv32izicboz"
# CHECK: attribute      5, "rv32i2p1_zicboz1p0"

.attribute arch, "rv32izicbop"
# CHECK: attribute      5, "rv32i2p1_zicbop1p0"

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

.attribute arch, "rv32i_zicsr_zicntr"
# CHECK: attribute      5, "rv32i2p1_zicntr1p0_zicsr2p0"

.attribute arch, "rv32i_zicsr_zihpm"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zihpm1p0"

.attribute arch, "rv32iss"
# CHECK: attribute      5, "rv32i2p1_ss1p12"

.attribute arch, "rv32isvbare"
# CHECK: attribute      5, "rv32i2p1_svbare1p0"

.attribute arch, "rv32issptead"
# CHECK: attribute      5, "rv32i2p1_ssptead1p0"

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

## SIFIVE_CUSTOMIZATION
.attribute arch, "rv32i_sscofpmf"
# CHECK: attribute      5, "rv32i2p1_sscofpmf1p0"
## end SIFIVE_CUSTOMIZATION

## Experimental extensions require version string to be explicitly specified

.attribute arch, "rv32izba1p0"
# CHECK: attribute      5, "rv32i2p1_zba1p0"

.attribute arch, "rv32izbb1p0"
# CHECK: attribute      5, "rv32i2p1_zbb1p0"

.attribute arch, "rv32izbc1p0"
# CHECK: attribute      5, "rv32i2p1_zbc1p0"

.attribute arch, "rv32i_zve32x_zvkb0p1"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zvkb0p1_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvkg0p1"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zvkg0p1_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvknha0p1"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zvknha0p1_zvl32b1p0"

.attribute arch, "rv32i_zve64x_zvknhb0p1"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zve64x1p0_zvknhb0p1_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32i_zve32x_zvkns0p1"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zvkns0p1_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvksed0p1"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zvksed0p1_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvksh0p1"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zvksh0p1_zvl32b1p0"

.attribute arch, "rv32izbs1p0"
# CHECK: attribute      5, "rv32i2p1_zbs1p0"

.attribute arch, "rv32ifzfhmin1p0"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zfhmin1p0"

.attribute arch, "rv32ifzfh1p0"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zfh1p0"

.attribute arch, "rv32izfinx"
# CHECK: attribute      5, "rv32i2p1_zfinx1p0"

.attribute arch, "rv32izfinx_zdinx"
# CHECK: attribute      5, "rv32i2p1_zfinx1p0_zdinx1p0"

.attribute arch, "rv32izfinx_zhinxmin"
# CHECK: attribute      5, "rv32i2p1_zfinx1p0_zhinxmin1p0"

.attribute arch, "rv32izfinx_zhinx1p0"
# CHECK: attribute      5, "rv32i2p1_zfinx1p0_zhinx1p0"

.attribute arch, "rv32i_zbkb1p0"
# CHECK: attribute      5, "rv32i2p1_zbkb1p0"

.attribute arch, "rv32i_zbkc1p0"
# CHECK: attribute      5, "rv32i2p1_zbkc1p0"

.attribute arch, "rv32i_zbkx1p0"
# CHECK: attribute      5, "rv32i2p1_zbkx1p0"

.attribute arch, "rv32i_zknd1p0"
# CHECK: attribute      5, "rv32i2p1_zknd1p0"

.attribute arch, "rv32i_zkne1p0"
# CHECK: attribute      5, "rv32i2p1_zkne1p0"

.attribute arch, "rv32i_zknh1p0"
# CHECK: attribute      5, "rv32i2p1_zknh1p0"

.attribute arch, "rv32i_zksed1p0"
# CHECK: attribute      5, "rv32i2p1_zksed1p0"

.attribute arch, "rv32i_zksh1p0"
# CHECK: attribute      5, "rv32i2p1_zksh1p0"

.attribute arch, "rv32i_zkr1p0"
# CHECK: attribute      5, "rv32i2p1_zkr1p0"

.attribute arch, "rv32i_zkn1p0"
# CHECK: attribute      5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"

.attribute arch, "rv32i_zks1p0"
# CHECK: attribute      5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"

.attribute arch, "rv32i_zkt1p0"
# CHECK: attribute      5, "rv32i2p1_zkt1p0"

.attribute arch, "rv32i_zk1p0"
# CHECK: attribute      5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"

.attribute arch, "rv32izihintntl0p2"
# CHECK: attribute      5, "rv32i2p1_zihintntl0p2"

.attribute arch, "rv32iczihintntl0p2"
# CHECK: attribute      5, "rv32i2p1_c2p0_zihintntl0p2"

.attribute arch, "rv32if_zkt1p0_zve32f1p0_zve32x1p0_zvl32b1p0"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zkt1p0_zve32f1p0_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32izca0p70"
# CHECK: attribute      5, "rv32i2p1_zca0p70"

.attribute arch, "rv32ifv_xsfvfnrclipxfqf_xsfvfwmaccqqq"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0_xsfvfnrclipxfqf0p1_xsfvfwmaccqqq0p1"

.attribute arch, "rv64i_xsfvcp"
# CHECK: attribute      5, "rv64i2p1_xsfvcp0p1"

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
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32i_zve64x"
# CHECK: attribute      5, "rv32i2p1_zve32x1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32izawrs1p0"
# SIFIVE_CUSTOMIZATION
# CHECK: attribute      5, "rv32i2p1_zawrs1p0"
#end SIFIVE_CUSTOMIZATION

.attribute arch, "rv32iztso0p1"
# CHECK: attribute      5, "rv32i2p1_ztso0p1"
