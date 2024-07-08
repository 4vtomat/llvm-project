;; Generate ELF attributes from llc.

; RUN: llc -mtriple=riscv32 -mattr=+zicclsm %s -o - | FileCheck --check-prefixes=CHECK,RV32ZICCLSM %s
; RUN: llc -mtriple=riscv32 -mattr=+ziccif %s -o - | FileCheck --check-prefixes=CHECK,RV32ZICCIF %s
; RUN: llc -mtriple=riscv32 -mattr=+ziccamoa %s -o - | FileCheck --check-prefixes=CHECK,RV32ZICCAMOA %s
; RUN: llc -mtriple=riscv32 -mattr=+ziccrse %s -o - | FileCheck --check-prefixes=CHECK,RV32ZICCRSE %s
; RUN: llc -mtriple=riscv32 -mattr=+za64rs %s -o - | FileCheck --check-prefixes=CHECK,RV32ZA64RS %s
; RUN: llc -mtriple=riscv32 -mattr=+zama16b %s -o - | FileCheck --check-prefixes=CHECK,RV32ZAMA16B %s
; RUN: llc -mtriple=riscv32 -mattr=+zic64b %s -o - | FileCheck --check-prefixes=CHECK,RV32ZIC64B %s
; RUN: llc -mtriple=riscv32 -mattr=+sdext %s -o - | FileCheck --check-prefixes=CHECK,RV32SDEXT %s
; RUN: llc -mtriple=riscv32 -mattr=+sdtrig %s -o - | FileCheck --check-prefixes=CHECK,RV32SDTRIG %s
; RUN: llc -mtriple=riscv32 -mattr=+ss %s -o - | FileCheck --check-prefixes=CHECK,RV32SS %s
; RUN: llc -mtriple=riscv32 -mattr=+svbare %s -o - | FileCheck --check-prefixes=CHECK,RV32SVBARE %s
; RUN: llc -mtriple=riscv32 -mattr=+svade %s -o - | FileCheck --check-prefixes=CHECK,RV32SVADE %s
; RUN: llc -mtriple=riscv32 -mattr=+svadu %s -o - | FileCheck --check-prefixes=CHECK,RV32SVADU %s
; RUN: llc -mtriple=riscv32 -mattr=+ssccptr %s -o - | FileCheck --check-prefixes=CHECK,RV32SSCCPTR %s
; RUN: llc -mtriple=riscv32 -mattr=+sstvecd %s -o - | FileCheck --check-prefixes=CHECK,RV32SSTVECD %s
; RUN: llc -mtriple=riscv32 -mattr=+sstvala %s -o - | FileCheck --check-prefixes=CHECK,RV32SSTVALA %s
; RUN: llc -mtriple=riscv32 -mattr=+sscounterenw %s -o - | FileCheck --check-prefixes=CHECK,RV32SSCOUNTERENW %s
; RUN: llc -mtriple=riscv32 -mattr=+ssu64xl %s -o - | FileCheck --check-prefixes=CHECK,RV32SSU64XL %s
; RUN: llc -mtriple=riscv32 -mattr=+sstc %s -o - | FileCheck --check-prefixes=CHECK,RV32SSTC %s
; RUN: llc -mtriple=riscv32 -mattr=+ssstateen %s -o - | FileCheck --check-prefixes=CHECK,RV32SSSTATEEN %s
; RUN: llc -mtriple=riscv32 -mattr=+smstateen %s -o - | FileCheck --check-prefixes=CHECK,RV32SMSTATEEN %s
; RUN: llc -mtriple=riscv32 -mattr=+shcounterenw %s -o - | FileCheck --check-prefixes=CHECK,RV32SHCOUNTERENW %s
; RUN: llc -mtriple=riscv32 -mattr=+shvstvala %s -o - | FileCheck --check-prefixes=CHECK,RV32SHVSTVALA %s
; RUN: llc -mtriple=riscv32 -mattr=+shtvala %s -o - | FileCheck --check-prefixes=CHECK,RV32SHTVALA %s
; RUN: llc -mtriple=riscv32 -mattr=+shvstvecd %s -o - | FileCheck --check-prefixes=CHECK,RV32SHVSTVECD %s
; RUN: llc -mtriple=riscv32 -mattr=+shvsatpa %s -o - | FileCheck --check-prefixes=CHECK,RV32SHVSATPA %s
; RUN: llc -mtriple=riscv32 -mattr=+shgatpa %s -o - | FileCheck --check-prefixes=CHECK,RV32SHGATPA %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+zvkb0p1 %s -o - | FileCheck --check-prefix=RV32ZVKB0P1 %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+zvkg0p1 %s -o - | FileCheck --check-prefix=RV32ZVKG0P1 %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zjid %s -o - | FileCheck --check-prefixes=CHECK,RV32ZJID %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-smrnmi %s -o - | FileCheck --check-prefix=RV32SMRNMI %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-smwg %s -o - | FileCheck --check-prefixes=CHECK,RV32SMWG %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-smwg -mattr=+experimental-smwgd %s -o - | FileCheck --check-prefixes=CHECK,RV32SMWGD %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-sswg %s -o - | FileCheck --check-prefixes=CHECK,RV32SSWG %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-ssnpm %s -o - | FileCheck --check-prefixes=CHECK,RV32SSNPM %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfpmpmt %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFPMPMT %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfvfexpa %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFVFEXPA %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfvfexpa64e %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFVFEXPA64E %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfpgflushdlone %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFPGFLUSHDLONE %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfcease %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFCEASE %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm128t %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM128T %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm16t %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM16T %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a4i %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A4I %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a8i %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A8I %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a8f %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A8F %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a16f %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A16F %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32a32f %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32A32F %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32ea %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32EA %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm32t %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM32T %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm64a64f %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM64A64F %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmm64t %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMM64T %s
; RUN: llc -mtriple=riscv32 -mattr=+xsfmmbase %s -o - | FileCheck --check-prefixes=CHECK,RV32XSFMMBASE %s
; RUN: llc -mtriple=riscv64 -mattr=+zicclsm %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICCLSM %s
; RUN: llc -mtriple=riscv64 -mattr=+ziccif %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICCIF %s
; RUN: llc -mtriple=riscv64 -mattr=+ziccamoa %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICCAMOA %s
; RUN: llc -mtriple=riscv64 -mattr=+ziccrse %s -o - | FileCheck --check-prefixes=CHECK,RV64ZICCRSE %s
; RUN: llc -mtriple=riscv64 -mattr=+za64rs %s -o - | FileCheck --check-prefixes=CHECK,RV64ZA64RS %s
; RUN: llc -mtriple=riscv64 -mattr=+zama16b %s -o - | FileCheck --check-prefixes=CHECK,RV64ZAMA16B %s
; RUN: llc -mtriple=riscv64 -mattr=+zic64b %s -o - | FileCheck --check-prefixes=CHECK,RV64ZIC64B %s
; RUN: llc -mtriple=riscv64 -mattr=+sdext %s -o - | FileCheck --check-prefixes=CHECK,RV64SDEXT %s
; RUN: llc -mtriple=riscv64 -mattr=+sdtrig %s -o - | FileCheck --check-prefixes=CHECK,RV64SDTRIG %s
; RUN: llc -mtriple=riscv64 -mattr=+ss %s -o - | FileCheck --check-prefixes=CHECK,RV64SS %s
; RUN: llc -mtriple=riscv64 -mattr=+svbare %s -o - | FileCheck --check-prefixes=CHECK,RV64SVBARE %s
; RUN: llc -mtriple=riscv64 -mattr=+svade %s -o - | FileCheck --check-prefixes=CHECK,RV64SVADE %s
; RUN: llc -mtriple=riscv64 -mattr=+svadu %s -o - | FileCheck --check-prefixes=CHECK,RV64SVADU %s
; RUN: llc -mtriple=riscv64 -mattr=+ssccptr %s -o - | FileCheck --check-prefixes=CHECK,RV64SSCCPTR %s
; RUN: llc -mtriple=riscv64 -mattr=+sstvecd %s -o - | FileCheck --check-prefixes=CHECK,RV64SSTVECD %s
; RUN: llc -mtriple=riscv64 -mattr=+sstvala %s -o - | FileCheck --check-prefixes=CHECK,RV64SSTVALA %s
; RUN: llc -mtriple=riscv64 -mattr=+sscounterenw %s -o - | FileCheck --check-prefixes=CHECK,RV64SSCOUNTERENW %s
; RUN: llc -mtriple=riscv64 -mattr=+ssu64xl %s -o - | FileCheck --check-prefixes=CHECK,RV64SSU64XL %s
; RUN: llc -mtriple=riscv64 -mattr=+sstc %s -o - | FileCheck --check-prefixes=CHECK,RV64SSTC %s
; RUN: llc -mtriple=riscv64 -mattr=+ssstateen %s -o - | FileCheck --check-prefixes=CHECK,RV64SSSTATEEN %s
; RUN: llc -mtriple=riscv64 -mattr=+smstateen %s -o - | FileCheck --check-prefixes=CHECK,RV64SMSTATEEN %s
; RUN: llc -mtriple=riscv64 -mattr=+shcounterenw %s -o - | FileCheck --check-prefixes=CHECK,RV64SHCOUNTERENW %s
; RUN: llc -mtriple=riscv64 -mattr=+shvstvala %s -o - | FileCheck --check-prefixes=CHECK,RV64SHVSTVALA %s
; RUN: llc -mtriple=riscv64 -mattr=+shtvala %s -o - | FileCheck --check-prefixes=CHECK,RV64SHTVALA %s
; RUN: llc -mtriple=riscv64 -mattr=+shvstvecd %s -o - | FileCheck --check-prefixes=CHECK,RV64SHVSTVECD %s
; RUN: llc -mtriple=riscv64 -mattr=+shvsatpa %s -o - | FileCheck --check-prefixes=CHECK,RV64SHVSATPA %s
; RUN: llc -mtriple=riscv64 -mattr=+shgatpa %s -o - | FileCheck --check-prefixes=CHECK,RV64SHGATPA %s
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -mattr=+zvkb0p1 %s -o - | FileCheck --check-prefix=RV64ZVKB0P1 %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+zvkg0p1 %s -o - | FileCheck --check-prefix=RV64ZVKG0P1 %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zjid %s -o - | FileCheck --check-prefixes=CHECK,RV64ZJID %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-smrnmi %s -o - | FileCheck --check-prefix=RV64SMRNMI %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-smwg %s -o - | FileCheck --check-prefixes=CHECK,RV64SMWG %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-smwg -mattr=+experimental-smwgd %s -o - | FileCheck --check-prefixes=CHECK,RV64SMWGD %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-sswg %s -o - | FileCheck --check-prefixes=CHECK,RV64SSWG %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-ssnpm %s -o - | FileCheck --check-prefixes=CHECK,RV64SSNPM %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfpmpmt %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFPMPMT %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfvfexpa %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFVFEXPA %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfvfexpa64e %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFVFEXPA64E %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfpgflushdlone %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFPGFLUSHDLONE %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfcease %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFCEASE %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm128t %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM128T %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm16t %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM16T %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a4i %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A4I %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a8i %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A8I %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a8f %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A8F %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a16f %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A16F %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32a32f %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32A32F %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32ea %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32EA %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm32t %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM32T %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm64a64f %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM64A64F %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmm64t %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMM64T %s
; RUN: llc -mtriple=riscv64 -mattr=+xsfmmbase %s -o - | FileCheck --check-prefixes=CHECK,RV64XSFMMBASE %s

; RUN: llc -mtriple=riscv64 -mattr=+sfa23 %s -o - | FileCheck --check-prefix=SFA23 %s
; RUN: llc -mtriple=riscv64 -mattr=+sfx23 %s -o - | FileCheck --check-prefix=SFX23 %s
; RUN: llc -mtriple=riscv64 -mattr=+sfp23 %s -o - | FileCheck --check-prefix=SFP23 %s
; RUN: llc -mtriple=riscv32 -mattr=+sfe23 %s -o - | FileCheck --check-prefix=SFE23 %s
; RUN: llc -mtriple=riscv64 -mattr=+sfs23 %s -o - | FileCheck --check-prefix=SFS23 %s
; RUN: llc -mtriple=riscv64 -mattr=+sfb23 %s -o - | FileCheck --check-prefix=SFB23 %s

; CHECK: .attribute 4, 16

; RV32ZICCLSM: .attribute 5, "rv32i2p1_zicclsm1p0"
; RV32ZICCIF: .attribute 5, "rv32i2p1_ziccif1p0"
; RV32ZICCAMOA: .attribute 5, "rv32i2p1_ziccamoa1p0"
; RV32ZICCRSE: .attribute 5, "rv32i2p1_ziccrse1p0"
; RV32ZA64RS: .attribute 5, "rv32i2p1_za64rs1p0"
; RV32ZAMA16B: .attribute 5, "rv32i2p1_zama16b1p0"
; RV32ZIC64B: .attribute 5, "rv32i2p1_zic64b1p0"
; RV32ZJID: .attribute 5, "rv32i2p1_zjid0p0"
; RV32ZVKB0P1: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb0p1_zvl32b1p0_zvl64b1p0"
; RV32ZVKG0P1: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkg0p1_zvl32b1p0"
; RV32SDEXT: .attribute 5, "rv32i2p1_sdext1p0"
; RV32SDTRIG: .attribute 5, "rv32i2p1_sdtrig1p0"
; RV32SS: .attribute 5, "rv32i2p1_ss1p13"
; RV32SVBARE: .attribute 5, "rv32i2p1_svbare1p0"
; RV32SVADE: .attribute 5, "rv32i2p1_svade1p0"
; RV32SVADU: .attribute 5, "rv32i2p1_svadu1p0"
; RV32SSCCPTR: .attribute 5, "rv32i2p1_ssccptr1p0"
; RV32SSTVECD: .attribute 5, "rv32i2p1_sstvecd1p0"
; RV32SSTVALA: .attribute 5, "rv32i2p1_sstvala1p0"
; RV32SSCOUNTERENW: .attribute 5, "rv32i2p1_sscounterenw1p0"
; RV32SSU64XL: .attribute 5, "rv32i2p1_ssu64xl1p0"
; RV32SSTC: .attribute 5, "rv32i2p1_sstc1p0"
; RV32SSSTATEEN: .attribute 5, "rv32i2p1_ssstateen1p0"
; RV32SMSTATEEN: .attribute 5, "rv32i2p1_smstateen1p0"
; RV32SHCOUNTERENW: .attribute 5, "rv32i2p1_shcounterenw1p0"
; RV32SHVSTVALA: .attribute 5, "rv32i2p1_shvstvala1p0"
; RV32SHTVALA: .attribute 5, "rv32i2p1_shtvala1p0"
; RV32SHVSTVECD: .attribute 5, "rv32i2p1_shvstvecd1p0"
; RV32SHVSATPA: .attribute 5, "rv32i2p1_shvsatpa1p0"
; RV32SHGATPA: .attribute 5, "rv32i2p1_shgatpa1p0"
; RV32SMRNMI: .attribute 5, "rv32i2p1_smrnmi0p5"
; RV32SMWG: .attribute 5, "rv32i2p1_smwg0p3"
; RV32SMWGD: .attribute 5, "rv32i2p1_smwg0p3_smwgd0p3"
; RV32SSWG: .attribute 5, "rv32i2p1_sswg0p3"
; RV32SSNPM: .attribute 5, "rv32i2p1_ssnpm0p8"
; RV32XSFPMPMT: .attribute 5, "rv32i2p1_xsfpmpmt0p1"
; RV32XSFVFEXPA: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfvfexpa0p2"
; RV32XSFVFEXPA64E: .attribute 5, "rv32i2p1_f2p2_d2p2_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0_xsfvfexpa0p2_xsfvfexpa64e0p2"
; RV32XSFPGFLUSHDLONE: .attribute 5, "rv32i2p1_xsfpgflushdlone0p1"
; RV32XSFCEASE: .attribute 5, "rv32i2p1_xsfcease0p1"
; RV32XSFMM128T: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm128t0p6_xsfmmbase0p6"
; RV32XSFMM16T: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm16t0p6_xsfmmbase0p6"
; RV32XSFMM32A: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a0p6_xsfmm32a16f0p6_xsfmm32a32f0p6_xsfmm32a8i0p6_xsfmmbase0p6"
; RV32XSFMM32A4I: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a4i0p6_xsfmmbase0p6"
; RV32XSFMM32A8I: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a8i0p6_xsfmmbase0p6"
; RV32XSFMM32A8F: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a8f0p6_xsfmmbase0p6"
; RV32XSFMM32A16F: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a16f0p6_xsfmmbase0p6"
; RV32XSFMM32A32F: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a32f0p6_xsfmmbase0p6"
; RV32XSFMM32EA: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32ea0p6"
; RV32XSFMM32T: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32t0p6_xsfmmbase0p6"
; RV32XSFMM64A64F: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm64a64f0p6_xsfmmbase0p6"
; RV32XSFMM64T: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm64t0p6_xsfmmbase0p6"
; RV32XSFMMBASE: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmmbase0p6"
; RV64ZICCLSM: .attribute 5, "rv64i2p1_zicclsm1p0"
; RV64ZICCIF: .attribute 5, "rv64i2p1_ziccif1p0"
; RV64ZICCAMOA: .attribute 5, "rv64i2p1_ziccamoa1p0"
; RV64ZICCRSE: .attribute 5, "rv64i2p1_ziccrse1p0"
; RV64ZA64RS: .attribute 5, "rv64i2p1_za64rs1p0"
; RV64ZAMA16B: .attribute 5, "rv64i2p1_zama16b1p0"
; RV64ZIC64B: .attribute 5, "rv64i2p1_zic64b1p0"
; RV64ZJID: .attribute 5, "rv64i2p1_zjid0p0"
; RV64SDEXT: .attribute 5, "rv64i2p1_sdext1p0"
; RV64SDTRIG: .attribute 5, "rv64i2p1_sdtrig1p0"
; RV64SS: .attribute 5, "rv64i2p1_ss1p13"
; RV64SVBARE: .attribute 5, "rv64i2p1_svbare1p0"
; RV64SVADE: .attribute 5, "rv64i2p1_svade1p0"
; RV64SVADU: .attribute 5, "rv64i2p1_svadu1p0"
; RV64SSCCPTR: .attribute 5, "rv64i2p1_ssccptr1p0"
; RV64SSTVECD: .attribute 5, "rv64i2p1_sstvecd1p0"
; RV64SSTVALA: .attribute 5, "rv64i2p1_sstvala1p0"
; RV64SSCOUNTERENW: .attribute 5, "rv64i2p1_sscounterenw1p0"
; RV64SSU64XL: .attribute 5, "rv64i2p1_ssu64xl1p0"
; RV64SSTC: .attribute 5, "rv64i2p1_sstc1p0"
; RV64SSSTATEEN: .attribute 5, "rv64i2p1_ssstateen1p0"
; RV64SMSTATEEN: .attribute 5, "rv64i2p1_smstateen1p0"
; RV64SHCOUNTERENW: .attribute 5, "rv64i2p1_shcounterenw1p0"
; RV64SHVSTVALA: .attribute 5, "rv64i2p1_shvstvala1p0"
; RV64SHTVALA: .attribute 5, "rv64i2p1_shtvala1p0"
; RV64SHVSTVECD: .attribute 5, "rv64i2p1_shvstvecd1p0"
; RV64SHVSATPA: .attribute 5, "rv64i2p1_shvsatpa1p0"
; RV64SHGATPA: .attribute 5, "rv64i2p1_shgatpa1p0"
; RV64ZVKB0P1: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb0p1_zvl32b1p0_zvl64b1p0"
; RV64ZVKG0P1: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvkg0p1_zvl32b1p0"
; RV64SMRNMI: .attribute 5, "rv64i2p1_smrnmi0p5"
; RV64SMWG: .attribute 5, "rv64i2p1_smwg0p3"
; RV64SMWGD: .attribute 5, "rv64i2p1_smwg0p3_smwgd0p3"
; RV64SSWG: .attribute 5, "rv64i2p1_sswg0p3"
; RV64SSNPM: .attribute 5, "rv64i2p1_ssnpm0p8"
; RV64XSFPMPMT: .attribute 5, "rv64i2p1_xsfpmpmt0p1"
; RV64XSFVFEXPA: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfvfexpa0p2"
; RV64XSFVFEXPA64E: .attribute 5, "rv64i2p1_f2p2_d2p2_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0_xsfvfexpa0p2_xsfvfexpa64e0p2"
; RV64XSFPGFLUSHDLONE: .attribute 5, "rv64i2p1_xsfpgflushdlone0p1"
; RV64XSFCEASE: .attribute 5, "rv64i2p1_xsfcease0p1"
; RV64XSFMM128T: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm128t0p6_xsfmmbase0p6"
; RV64XSFMM16T: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm16t0p6_xsfmmbase0p6"
; RV64XSFMM32A: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a0p6_xsfmm32a16f0p6_xsfmm32a32f0p6_xsfmm32a8i0p6_xsfmmbase0p6"
; RV64XSFMM32A4I: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a4i0p6_xsfmmbase0p6"
; RV64XSFMM32A8I: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a8i0p6_xsfmmbase0p6"
; RV64XSFMM32A8F: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a8f0p6_xsfmmbase0p6"
; RV64XSFMM32A16F: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a16f0p6_xsfmmbase0p6"
; RV64XSFMM32A32F: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32a32f0p6_xsfmmbase0p6"
; RV64XSFMM32EA: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32ea0p6"
; RV64XSFMM32T: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32t0p6_xsfmmbase0p6"
; RV64XSFMM64A64F: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm64a64f0p6_xsfmmbase0p6"
; RV64XSFMM64T: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm64t0p6_xsfmmbase0p6"
; RV64XSFMMBASE: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmmbase0p6"

; SFA23: .attribute 5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_h1p0_zic64b1p0_zicbom1p0_zicbop1p0_zicboz1p0_ziccamoa1p0_ziccif1p0_zicclsm1p0_ziccrse1p0_zicfilp0p4_zicfiss0p4_zicntr2p0_zicond1p0_zicsr2p0_zifencei2p0_zihintntl1p0_zihintpause2p0_zihpm2p0_zimop1p0_za64rs1p0_zawrs1p0_zfa1p0_zfhmin1p0_zca1p0_zcb1p0_zcmop1p0_zba1p0_zbb1p0_zbs1p0_zkr1p0_zkt1p0_zvbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfhmin1p0_zvkb1p0_zvkt1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_shcounterenw1p0_shgatpa1p0_shtvala1p0_shvsatpa1p0_shvstvala1p0_shvstvecd1p0_ssaia1p0_ssccptr1p0_sscofpmf1p0_sscounterenw1p0_ssnpm0p8_ssqosid1p0_ssstateen1p0_sstc1p0_sstvala1p0_sstvecd1p0_ssu64xl1p0_svade1p0_svbare1p0_svinval1p0_svnapot1p0_svpbmt1p0"
; SFX23: .attribute 5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_h1p0_zic64b1p0_zicbom1p0_zicbop1p0_zicboz1p0_ziccamoa1p0_ziccif1p0_zicclsm1p0_ziccrse1p0_zicfilp0p4_zicfiss0p4_zicntr2p0_zicond1p0_zicsr2p0_zifencei2p0_zihintntl1p0_zihintpause2p0_zihpm2p0_zimop1p0_za64rs1p0_zawrs1p0_zfa1p0_zfbfmin1p0_zfh1p0_zfhmin1p0_zca1p0_zcb1p0_zcmop1p0_zba1p0_zbb1p0_zbs1p0_zkr1p0_zkt1p0_zvbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfbfmin1p0_zvfbfwma1p0_zvfh1p0_zvfhmin1p0_zvkb1p0_zvkt1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_shcounterenw1p0_shgatpa1p0_shtvala1p0_shvsatpa1p0_shvstvala1p0_shvstvecd1p0_ssaia1p0_ssccptr1p0_sscofpmf1p0_sscounterenw1p0_ssnpm0p8_ssqosid1p0_ssstateen1p0_sstc1p0_sstvala1p0_sstvecd1p0_ssu64xl1p0_svade1p0_svbare1p0_svinval1p0_svnapot1p0_svpbmt1p0_xsfvfexpa0p2_xsfvfnrclipxfqf1p0"
; SFP23: .attribute 5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_h1p0_zic64b1p0_zicbom1p0_zicbop1p0_zicboz1p0_ziccamoa1p0_ziccif1p0_zicclsm1p0_ziccrse1p0_zicfilp0p4_zicfiss0p4_zicntr2p0_zicond1p0_zicsr2p0_zifencei2p0_zihintntl1p0_zihintpause2p0_zihpm2p0_zimop1p0_za64rs1p0_zawrs1p0_zfa1p0_zfbfmin1p0_zfh1p0_zfhmin1p0_zca1p0_zcb1p0_zcmop1p0_zba1p0_zbb1p0_zbs1p0_zkr1p0_zkt1p0_zvbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfbfmin1p0_zvfbfwma1p0_zvfh1p0_zvfhmin1p0_zvkb1p0_zvkt1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_shcounterenw1p0_shgatpa1p0_shtvala1p0_shvsatpa1p0_shvstvala1p0_shvstvecd1p0_ssaia1p0_ssccptr1p0_sscofpmf1p0_sscounterenw1p0_ssnpm0p8_ssqosid1p0_ssstateen1p0_sstc1p0_sstvala1p0_sstvecd1p0_ssu64xl1p0_svade1p0_svbare1p0_svinval1p0_svnapot1p0_svpbmt1p0_xsfvqdotq0p1"
; SFE23: .attribute 5, "rv32i2p1_m2p0_c2p0_zicbop1p0_zicntr2p0_zicond1p0_zicsr2p0_zifencei2p0_zihintntl1p0_zimop1p0_zaamo1p0_zca1p0_zcb1p0_zcmop1p0_zba1p0_zbb1p0_zbs1p0_zkt1p0"
; SFS23: .attribute 5, "rv64i2p1_m2p0_c2p0_zicbop1p0_zicntr2p0_zicond1p0_zicsr2p0_zifencei2p0_zihintntl1p0_zimop1p0_zaamo1p0_zca1p0_zcb1p0_zcmop1p0_zba1p0_zbb1p0_zbs1p0_zkt1p0"
; SFB23: .attribute 5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zic64b1p0_zicbom1p0_zicbop1p0_zicboz1p0_ziccamoa1p0_ziccif1p0_zicclsm1p0_ziccrse1p0_zicfilp0p4_zicfiss0p4_zicntr2p0_zicond1p0_zicsr2p0_zifencei2p0_zihintntl1p0_zihintpause2p0_zihpm2p0_zimop1p0_za64rs1p0_zawrs1p0_zfa1p0_zfhmin1p0_zca1p0_zcb1p0_zcmop1p0_zba1p0_zbb1p0_zbs1p0_zkt1p0_ssccptr1p0_sscofpmf1p0_sscounterenw1p0_sstc1p0_sstvala1p0_sstvecd1p0_ssu64xl1p0_svade1p0_svbare1p0_svinval1p0_svnapot1p0_svpbmt1p0"

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}
