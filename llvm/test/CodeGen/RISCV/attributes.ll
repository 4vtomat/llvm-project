;; Generate ELF attributes from llc.

; RUN: llc -mtriple=riscv32 -mattr=+m %s -o - | FileCheck --check-prefix=RV32M %s
; RUN: llc -mtriple=riscv32 -mattr=+zmmul %s -o - | FileCheck --check-prefix=RV32ZMMUL %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+zmmul %s -o - | FileCheck --check-prefix=RV32MZMMUL %s
; RUN: llc -mtriple=riscv32 -mattr=+a %s -o - | FileCheck --check-prefix=RV32A %s
; RUN: llc -mtriple=riscv32 -mattr=+f %s -o - | FileCheck --check-prefix=RV32F %s
; RUN: llc -mtriple=riscv32 -mattr=+d %s -o - | FileCheck --check-prefix=RV32D %s
; RUN: llc -mtriple=riscv32 -mattr=+c %s -o - | FileCheck --check-prefix=RV32C %s
; RUN: llc -mtriple=riscv32 -mattr=+zihintpause %s -o - | FileCheck --check-prefix=RV32ZIHINTPAUSE %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zihintntl %s -o - | FileCheck --check-prefix=RV32ZIHINTNTL %s
; RUN: llc -mtriple=riscv32 -mattr=+zfhmin %s -o - | FileCheck --check-prefix=RV32ZFHMIN %s
; RUN: llc -mtriple=riscv32 -mattr=+zfh %s -o - | FileCheck --check-prefix=RV32ZFH %s
; RUN: llc -mtriple=riscv32 -mattr=+zba %s -o - | FileCheck --check-prefix=RV32ZBA %s
; RUN: llc -mtriple=riscv32 -mattr=+zbb %s -o - | FileCheck --check-prefix=RV32ZBB %s
; RUN: llc -mtriple=riscv32 -mattr=+zbc %s -o - | FileCheck --check-prefix=RV32ZBC %s
; RUN: llc -mtriple=riscv32 -mattr=+zbs %s -o - | FileCheck --check-prefix=RV32ZBS %s
; RUN: llc -mtriple=riscv32 -mattr=+v %s -o - | FileCheck --check-prefix=RV32V %s
; RUN: llc -mtriple=riscv32 -mattr=+zbb,+zfh,+v,+f %s -o - | FileCheck --check-prefix=RV32COMBINED %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb %s -o - | FileCheck --check-prefix=RV32ZBKB %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkc %s -o - | FileCheck --check-prefix=RV32ZBKC %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkx %s -o - | FileCheck --check-prefix=RV32ZBKX %s
; RUN: llc -mtriple=riscv32 -mattr=+zknd %s -o - | FileCheck --check-prefix=RV32ZKND %s
; RUN: llc -mtriple=riscv32 -mattr=+zkne %s -o - | FileCheck --check-prefix=RV32ZKNE %s
; RUN: llc -mtriple=riscv32 -mattr=+zknh %s -o - | FileCheck --check-prefix=RV32ZKNH %s
; RUN: llc -mtriple=riscv32 -mattr=+zksed %s -o - | FileCheck --check-prefix=RV32ZKSED %s
; RUN: llc -mtriple=riscv32 -mattr=+zksh %s -o - | FileCheck --check-prefix=RV32ZKSH %s
; RUN: llc -mtriple=riscv32 -mattr=+zkr %s -o - | FileCheck --check-prefix=RV32ZKR %s
; RUN: llc -mtriple=riscv32 -mattr=+zkn %s -o - | FileCheck --check-prefix=RV32ZKN %s
; RUN: llc -mtriple=riscv32 -mattr=+zks %s -o - | FileCheck --check-prefix=RV32ZKS %s
; RUN: llc -mtriple=riscv32 -mattr=+zkt %s -o - | FileCheck --check-prefix=RV32ZKT %s
; RUN: llc -mtriple=riscv32 -mattr=+zk %s -o - | FileCheck --check-prefix=RV32ZK %s
; RUN: llc -mtriple=riscv32 -mattr=+zkn,+zkr,+zkt %s -o - | FileCheck --check-prefix=RV32COMBINEINTOZK %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb,+zbkc,+zbkx,+zkne,+zknd,+zknh %s -o - | FileCheck --check-prefix=RV32COMBINEINTOZKN %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb,+zbkc,+zbkx,+zksed,+zksh %s -o - | FileCheck --check-prefix=RV32COMBINEINTOZKS %s
; RUN: llc -mtriple=riscv32 -mattr=+zicbom %s -o - | FileCheck --check-prefix=RV32ZICBOM %s
; RUN: llc -mtriple=riscv32 -mattr=+zicboz %s -o - | FileCheck --check-prefix=RV32ZICBOZ %s
; RUN: llc -mtriple=riscv32 -mattr=+zicbop %s -o - | FileCheck --check-prefix=RV32ZICBOP %s
; RUN: llc -mtriple=riscv32 -mattr=+zicclsm %s -o - | FileCheck --check-prefix=RV32ZICCLSM %s
; RUN: llc -mtriple=riscv32 -mattr=+ziccif %s -o - | FileCheck --check-prefix=RV32ZICCIF %s
; RUN: llc -mtriple=riscv32 -mattr=+ziccamoa %s -o - | FileCheck --check-prefix=RV32ZICCAMOA %s
; RUN: llc -mtriple=riscv32 -mattr=+ziccrse %s -o - | FileCheck --check-prefix=RV32ZICCRSE %s
; RUN: llc -mtriple=riscv32 -mattr=+za64rs %s -o - | FileCheck --check-prefix=RV32ZA64RS %s
; RUN: llc -mtriple=riscv32 -mattr=+zic64b %s -o - | FileCheck --check-prefix=RV32ZIC64B %s
; RUN: llc -mtriple=riscv32 -mattr=+zicsr,+zicntr %s -o - | FileCheck --check-prefix=RV32ZICNTR %s
; RUN: llc -mtriple=riscv32 -mattr=+zicsr,+zihpm %s -o - | FileCheck --check-prefix=RV32ZIHPM %s
; RUN: llc -mtriple=riscv32 -mattr=+svnapot %s -o - | FileCheck --check-prefix=RV32SVNAPOT %s
; RUN: llc -mtriple=riscv32 -mattr=+svpbmt %s -o - | FileCheck --check-prefix=RV32SVPBMT %s
; RUN: llc -mtriple=riscv32 -mattr=+svinval %s -o - | FileCheck --check-prefix=RV32SVINVAL %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zca %s -o - | FileCheck --check-prefix=RV32ZCA %s

; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+experimental-zvkb %s -o - | FileCheck --check-prefix=RV32ZVKB %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+experimental-zvkg %s -o - | FileCheck --check-prefix=RV32ZVKG %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+experimental-zvknha %s -o - | FileCheck --check-prefix=RV32ZVKNHA %s
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -mattr=+experimental-zvknhb %s -o - | FileCheck --check-prefix=RV32ZVKNHB %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+experimental-zvkns %s -o - | FileCheck --check-prefix=RV32ZVKNS %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+experimental-zvksed %s -o - | FileCheck --check-prefix=RV32ZVKSED %s
; RUN: llc -mtriple=riscv32 -mattr=+zve32x -mattr=+experimental-zvksh %s -o - | FileCheck --check-prefix=RV32ZVKSH %s
; RUN: llc -mtriple=riscv64 -mattr=+m %s -o - | FileCheck --check-prefix=RV64M %s
; RUN: llc -mtriple=riscv64 -mattr=+zmmul %s -o - | FileCheck --check-prefix=RV64ZMMUL %s
; RUN: llc -mtriple=riscv64 -mattr=+m,+zmmul %s -o - | FileCheck --check-prefix=RV64MZMMUL %s
; RUN: llc -mtriple=riscv64 -mattr=+a %s -o - | FileCheck --check-prefix=RV64A %s
; RUN: llc -mtriple=riscv64 -mattr=+f %s -o - | FileCheck --check-prefix=RV64F %s
; RUN: llc -mtriple=riscv64 -mattr=+d %s -o - | FileCheck --check-prefix=RV64D %s
; RUN: llc -mtriple=riscv64 -mattr=+c %s -o - | FileCheck --check-prefix=RV64C %s
; RUN: llc -mtriple=riscv64 -mattr=+zihintpause %s -o - | FileCheck --check-prefix=RV64ZIHINTPAUSE %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zihintntl %s -o - | FileCheck --check-prefix=RV64ZIHINTNTL %s
; RUN: llc -mtriple=riscv64 -mattr=+zfhmin %s -o - | FileCheck --check-prefix=RV64ZFHMIN %s
; RUN: llc -mtriple=riscv64 -mattr=+zfh %s -o - | FileCheck --check-prefix=RV64ZFH %s
; RUN: llc -mtriple=riscv64 -mattr=+zba %s -o - | FileCheck --check-prefix=RV64ZBA %s
; RUN: llc -mtriple=riscv64 -mattr=+zbb %s -o - | FileCheck --check-prefix=RV64ZBB %s
; RUN: llc -mtriple=riscv64 -mattr=+zbc %s -o - | FileCheck --check-prefix=RV64ZBC %s
; RUN: llc -mtriple=riscv64 -mattr=+zbs %s -o - | FileCheck --check-prefix=RV64ZBS %s
; RUN: llc -mtriple=riscv64 -mattr=+v %s -o - | FileCheck --check-prefix=RV64V %s
; RUN: llc -mtriple=riscv64 -mattr=+zbb,+zfh,+v,+f %s -o - | FileCheck --check-prefix=RV64COMBINED %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb %s -o - | FileCheck --check-prefix=RV64ZBKB %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkc %s -o - | FileCheck --check-prefix=RV64ZBKC %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkx %s -o - | FileCheck --check-prefix=RV64ZBKX %s
; RUN: llc -mtriple=riscv64 -mattr=+zknd %s -o - | FileCheck --check-prefix=RV64ZKND %s
; RUN: llc -mtriple=riscv64 -mattr=+zkn,+zkr,+zkt %s -o - | FileCheck --check-prefix=RV64COMBINEINTOZK %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb,+zbkc,+zbkx,+zkne,+zknd,+zknh %s -o - | FileCheck --check-prefix=RV64COMBINEINTOZKN %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb,+zbkc,+zbkx,+zksed,+zksh %s -o - | FileCheck --check-prefix=RV64COMBINEINTOZKS %s
; RUN: llc -mtriple=riscv64 -mattr=+zkne %s -o - | FileCheck --check-prefix=RV64ZKNE %s
; RUN: llc -mtriple=riscv64 -mattr=+zknh %s -o - | FileCheck --check-prefix=RV64ZKNH %s
; RUN: llc -mtriple=riscv64 -mattr=+zksed %s -o - | FileCheck --check-prefix=RV64ZKSED %s
; RUN: llc -mtriple=riscv64 -mattr=+zksh %s -o - | FileCheck --check-prefix=RV64ZKSH %s
; RUN: llc -mtriple=riscv64 -mattr=+zkr %s -o - | FileCheck --check-prefix=RV64ZKR %s
; RUN: llc -mtriple=riscv64 -mattr=+zkn %s -o - | FileCheck --check-prefix=RV64ZKN %s
; RUN: llc -mtriple=riscv64 -mattr=+zks %s -o - | FileCheck --check-prefix=RV64ZKS %s
; RUN: llc -mtriple=riscv64 -mattr=+zkt %s -o - | FileCheck --check-prefix=RV64ZKT %s
; RUN: llc -mtriple=riscv64 -mattr=+zk %s -o - | FileCheck --check-prefix=RV64ZK %s
; RUN: llc -mtriple=riscv64 -mattr=+zkn,+zkr,+zkt %s -o - | FileCheck --check-prefix=RV64COMBINEINTOZK %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb,+zbkc,+zbkx,+zkne,+zknd,+zknh %s -o - | FileCheck --check-prefix=RV64COMBINEINTOZKN %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb,+zbkc,+zbkx,+zksed,+zksh %s -o - | FileCheck --check-prefix=RV64COMBINEINTOZKS %s
; RUN: llc -mtriple=riscv64 -mattr=+zicbom %s -o - | FileCheck --check-prefix=RV64ZICBOM %s
; RUN: llc -mtriple=riscv64 -mattr=+zicboz %s -o - | FileCheck --check-prefix=RV64ZICBOZ %s
; RUN: llc -mtriple=riscv64 -mattr=+zicclsm %s -o - | FileCheck --check-prefix=RV64ZICCLSM %s
; RUN: llc -mtriple=riscv64 -mattr=+ziccif %s -o - | FileCheck --check-prefix=RV64ZICCIF %s
; RUN: llc -mtriple=riscv64 -mattr=+ziccamoa %s -o - | FileCheck --check-prefix=RV64ZICCAMOA %s
; RUN: llc -mtriple=riscv64 -mattr=+ziccrse %s -o - | FileCheck --check-prefix=RV64ZICCRSE %s
; RUN: llc -mtriple=riscv64 -mattr=+za64rs %s -o - | FileCheck --check-prefix=RV64ZA64RS %s
; RUN: llc -mtriple=riscv64 -mattr=+zic64b %s -o - | FileCheck --check-prefix=RV64ZIC64B %s
; RUN: llc -mtriple=riscv64 -mattr=+zicsr,+zicntr %s -o - | FileCheck --check-prefix=RV64ZICNTR %s
; RUN: llc -mtriple=riscv64 -mattr=+zicsr,+zihpm %s -o - | FileCheck --check-prefix=RV64ZIHPM %s
; RUN: llc -mtriple=riscv64 -mattr=+zicbop %s -o - | FileCheck --check-prefix=RV64ZICBOP %s
; RUN: llc -mtriple=riscv64 -mattr=+svnapot %s -o - | FileCheck --check-prefix=RV64SVNAPOT %s
; RUN: llc -mtriple=riscv64 -mattr=+svpbmt %s -o - | FileCheck --check-prefix=RV64SVPBMT %s
; RUN: llc -mtriple=riscv64 -mattr=+svinval %s -o - | FileCheck --check-prefix=RV64SVINVAL %s
; RUN: llc -mtriple=riscv64 -mattr=+xventanacondops %s -o - | FileCheck --check-prefix=RV64XVENTANACONDOPS %s
; RUN: llc -mtriple=riscv64 -mattr=+xtheadvdot %s -o - | FileCheck --check-prefix=RV64XTHEADVDOT %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zawrs %s -o - | FileCheck --check-prefix=RV64ZAWRS %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-ztso %s -o - | FileCheck --check-prefix=RV64ZTSO %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zca %s -o - | FileCheck --check-prefix=RV64ZCA %s
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -mattr=+experimental-zvkb %s -o - | FileCheck --check-prefix=RV64ZVKB %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+experimental-zvkg %s -o - | FileCheck --check-prefix=RV64ZVKG %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+experimental-zvknha %s -o - | FileCheck --check-prefix=RV64ZVKNHA %s
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -mattr=+experimental-zvknhb %s -o - | FileCheck --check-prefix=RV64ZVKNHB %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+experimental-zvkns %s -o - | FileCheck --check-prefix=RV64ZVKNS %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+experimental-zvksed %s -o - | FileCheck --check-prefix=RV64ZVKSED %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x -mattr=+experimental-zvksh %s -o - | FileCheck --check-prefix=RV64ZVKSH %s

; SIFIVE_CUSTOMIZATION
; RV32M: .attribute 5, "rv32i2p1_m2p0"
; RV32ZMMUL: .attribute 5, "rv32i2p1_zmmul1p0"
; RV32MZMMUL: .attribute 5, "rv32i2p1_m2p0_zmmul1p0"
; RV32A: .attribute 5, "rv32i2p1_a2p1"
; RV32F: .attribute 5, "rv32i2p1_f2p2_zicsr2p0"
; RV32D: .attribute 5, "rv32i2p1_f2p2_d2p2_zicsr2p0"
; RV32C: .attribute 5, "rv32i2p1_c2p0"
; RV32ZIHINTPAUSE: .attribute 5, "rv32i2p1_zihintpause2p0"
; RV32ZIHINTNTL: .attribute 5, "rv32i2p1_zihintntl0p2"
; RV32ZFHMIN: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zfhmin1p0"
; RV32ZFH: .attribute 5, "rv32i2p1_f2p2_zicsr2p0_zfh1p0"
; RV32ZBA: .attribute 5, "rv32i2p1_zba1p0"
; RV32ZBB: .attribute 5, "rv32i2p1_zbb1p0"
; RV32ZBC: .attribute 5, "rv32i2p1_zbc1p0"
; RV32ZBS: .attribute 5, "rv32i2p1_zbs1p0"
; RV32V: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV32COMBINED: .attribute 5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zfh1p0_zbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV32ZBKB: .attribute 5, "rv32i2p1_zbkb1p0"
; RV32ZBKC: .attribute 5, "rv32i2p1_zbkc1p0"
; RV32ZBKX: .attribute 5, "rv32i2p1_zbkx1p0"
; RV32ZKND: .attribute 5, "rv32i2p1_zknd1p0"
; RV32ZKNE: .attribute 5, "rv32i2p1_zkne1p0"
; RV32ZKNH: .attribute 5, "rv32i2p1_zknh1p0"
; RV32ZKSED: .attribute 5, "rv32i2p1_zksed1p0"
; RV32ZKSH: .attribute 5, "rv32i2p1_zksh1p0"
; RV32ZKR: .attribute 5, "rv32i2p1_zkr1p0"
; RV32ZKN: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV32ZKS: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV32ZKT: .attribute 5, "rv32i2p1_zkt1p0"
; RV32ZK: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV32COMBINEINTOZK: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV32COMBINEINTOZKN: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV32COMBINEINTOZKS: .attribute 5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV32ZICBOM: .attribute 5, "rv32i2p1_zicbom1p0"
; RV32ZICBOZ: .attribute 5, "rv32i2p1_zicboz1p0"
; RV32ZICBOP: .attribute 5, "rv32i2p1_zicbop1p0"
; RV32SVNAPOT: .attribute 5, "rv32i2p1_svnapot1p0"
; RV32SVPBMT: .attribute 5, "rv32i2p1_svpbmt1p0"
; RV32SVINVAL: .attribute 5, "rv32i2p1_svinval1p0"
; RV32ZCA: .attribute 5, "rv32i2p1_zca0p70"
; RV32ZICCLSM: .attribute 5, "rv32i2p1_zicclsm1p0"
; RV32ZICCIF: .attribute 5, "rv32i2p1_ziccif1p0"
; RV32ZICCAMOA: .attribute 5, "rv32i2p1_ziccamoa1p0"
; RV32ZICCRSE: .attribute 5, "rv32i2p1_ziccrse1p0"
; RV32ZA64RS: .attribute 5, "rv32i2p1_za64rs1p0"
; RV32ZIC64B: .attribute 5, "rv32i2p1_zic64b1p0"
; RV32ZVKB: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb0p1_zvl32b1p0_zvl64b1p0"
; RV32ZVKG: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkg0p1_zvl32b1p0"
; RV32ZVKNHA: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvknha0p1_zvl32b1p0"
; RV32ZVKNHB: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvknha0p1_zvknhb0p1_zvl32b1p0_zvl64b1p0"
; RV32ZVKNS: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkns0p1_zvl32b1p0"
; RV32ZVKSED: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvksed0p1_zvl32b1p0"
; RV32ZVKSH: .attribute 5, "rv32i2p1_zicsr2p0_zve32x1p0_zvksh0p1_zvl32b1p0"
; RV32ZICNTR: .attribute 5, "rv32i2p1_zicntr1p0_zicsr2p0"
; RV32ZIHPM: .attribute 5, "rv32i2p1_zicsr2p0_zihpm1p0"

; RV64M: .attribute 5, "rv64i2p1_m2p0"
; RV64ZMMUL: .attribute 5, "rv64i2p1_zmmul1p0"
; RV64MZMMUL: .attribute 5, "rv64i2p1_m2p0_zmmul1p0"
; RV64A: .attribute 5, "rv64i2p1_a2p1"
; RV64F: .attribute 5, "rv64i2p1_f2p2_zicsr2p0"
; RV64D: .attribute 5, "rv64i2p1_f2p2_d2p2_zicsr2p0"
; RV64C: .attribute 5, "rv64i2p1_c2p0"
; RV64ZIHINTPAUSE: .attribute 5, "rv64i2p1_zihintpause2p0"
; RV64ZIHINTNTL: .attribute 5, "rv64i2p1_zihintntl0p2"
; RV64ZFHMIN: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zfhmin1p0"
; RV64ZFH: .attribute 5, "rv64i2p1_f2p2_zicsr2p0_zfh1p0"
; RV64ZBA: .attribute 5, "rv64i2p1_zba1p0"
; RV64ZBB: .attribute 5, "rv64i2p1_zbb1p0"
; RV64ZBC: .attribute 5, "rv64i2p1_zbc1p0"
; RV64ZBS: .attribute 5, "rv64i2p1_zbs1p0"
; RV64V: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV64COMBINED: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zfh1p0_zbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV64ZBKB: .attribute 5, "rv64i2p1_zbkb1p0"
; RV64ZBKC: .attribute 5, "rv64i2p1_zbkc1p0"
; RV64ZBKX: .attribute 5, "rv64i2p1_zbkx1p0"
; RV64ZKND: .attribute 5, "rv64i2p1_zknd1p0"
; RV64ZKNE: .attribute 5, "rv64i2p1_zkne1p0"
; RV64ZKNH: .attribute 5, "rv64i2p1_zknh1p0"
; RV64ZKSED: .attribute 5, "rv64i2p1_zksed1p0"
; RV64ZKSH: .attribute 5, "rv64i2p1_zksh1p0"
; RV64ZKR: .attribute 5, "rv64i2p1_zkr1p0"
; RV64ZKN: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV64ZKS: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV64ZKT: .attribute 5, "rv64i2p1_zkt1p0"
; RV64ZK: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV64COMBINEINTOZK: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV64COMBINEINTOZKN: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV64COMBINEINTOZKS: .attribute 5, "rv64i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV64XSFQMACC: .attribute 5, "rv64i2p1_f2p0_d2p0_v0p10_zve32f0p10_zve32x0p10_zve64d0p10_zve64f0p10_zve64x0p10_xsfvqmaccdod0p1_xsfvqmaccqoq0p1"
; RV64ZICBOM: .attribute 5, "rv64i2p1_zicbom1p0"
; RV64ZICBOZ: .attribute 5, "rv64i2p1_zicboz1p0"
; RV64ZICCLSM: .attribute 5, "rv64i2p1_zicclsm1p0"
; RV64ZICCIF: .attribute 5, "rv64i2p1_ziccif1p0"
; RV64ZICCAMOA: .attribute 5, "rv64i2p1_ziccamoa1p0"
; RV64ZICCRSE: .attribute 5, "rv64i2p1_ziccrse1p0"
; RV64ZA64RS: .attribute 5, "rv64i2p1_za64rs1p0"
; RV64ZIC64B: .attribute 5, "rv64i2p1_zic64b1p0"
; RV64ZICNTR: .attribute 5, "rv64i2p1_zicntr1p0_zicsr2p0"
; RV64ZIHPM: .attribute 5, "rv64i2p1_zicsr2p0_zihpm1p0"
; RV64ZAWRS: .attribute 5, "rv64i2p1_zawrs1p0"
; RV64ZICBOP: .attribute 5, "rv64i2p1_zicbop1p0"
; RV64SVNAPOT: .attribute 5, "rv64i2p1_svnapot1p0"
; RV64SVPBMT: .attribute 5, "rv64i2p1_svpbmt1p0"
; RV64SVINVAL: .attribute 5, "rv64i2p1_svinval1p0"
; RV64XVENTANACONDOPS: .attribute 5, "rv64i2p1_xventanacondops1p0"
; RV64XTHEADVDOT: .attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xtheadvdot1p0"
; RV64ZTSO: .attribute 5, "rv64i2p1_ztso0p1"
; RV64ZCA: .attribute 5, "rv64i2p1_zca0p70"
; RV64ZVKB: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb0p1_zvl32b1p0_zvl64b1p0"
; RV64ZVKG: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvkg0p1_zvl32b1p0"
; RV64ZVKNHA: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvknha0p1_zvl32b1p0"
; RV64ZVKNHB: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvknha0p1_zvknhb0p1_zvl32b1p0_zvl64b1p0"
; RV64ZVKNS: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvkns0p1_zvl32b1p0"
; RV64ZVKSED: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvksed0p1_zvl32b1p0"
; RV64ZVKSH: .attribute 5, "rv64i2p1_zicsr2p0_zve32x1p0_zvksh0p1_zvl32b1p0"
; end SIFIVE_CUSTOMIZATION

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}
