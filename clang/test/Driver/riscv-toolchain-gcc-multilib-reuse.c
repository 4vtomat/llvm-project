// Test case for scanning input of GCC output as multilib config
// Skip this test on Windows, we can't create a dummy GCC to output
// multilib config, ExecuteAndWait only execute *.exe file.
// UNSUPPORTED: system-windows

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk2 \
// RUN:   --print-multi-lib \
// RUN:   | FileCheck -check-prefix=C-GCC-MULTI-LIB %s
// C-GCC-MULTI-LIB:  rv32ic/ilp32;@march=rv32ic@mabi=ilp32
// C-GCC-MULTI-LIB-NEXT: rv32im/ilp32;@march=rv32im@mabi=ilp32
// C-GCC-MULTI-LIB-NEXT: rv32iac/ilp32;@march=rv32iac@mabi=ilp32
// C-GCC-MULTI-LIB-NEXT: rv32if/ilp32f;@march=rv32if@mabi=ilp32f
// C-GCC-MULTI-LIB-NEXT: rv32ifc/ilp32;@march=rv32ifc@mabi=ilp32
// C-GCC-MULTI-LIB-NEXT: rv32imafc/ilp32f;@march=rv32imafc@mabi=ilp32f
// C-GCC-MULTI-LIB-NEXT: rv64imafdc/lp64d;@march=rv64imafdc@mabi=lp64d
// C-GCC-MULTI-LIB-NOT:  {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imc -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMC-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32IMC-ILP32: rv32im/ilp32
// GCC-MULTI-LIB-REUSE-RV32IMC-ILP32-NOT:  {{^.+$}}

// Check rv32imac won't reuse rv32im or rv32ic
// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imac -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMAC-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32IMAC-ILP32: rv32imac/ilp32
// GCC-MULTI-LIB-REUSE-RV32IMAC-ILP32--NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32iac -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IAC-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32IAC-ILP32: rv32iac/ilp32
// GCC-MULTI-LIB-REUSE-RV32IAC-ILP32-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk2 \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32ifc -mabi=ilp32f \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IFC-ILP32F %s
// GCC-MULTI-LIB-REUSE-RV32IFC-ILP32F: rv32if/ilp32f
// GCC-MULTI-LIB-REUSE-RV32IFC-ILP32F-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imafdc -mabi=ilp32f \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32F %s
// GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32F: rv32imafc/ilp32f
// GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32F-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imafdc -mabi=ilp32d \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32D %s
// GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32D: .
// GCC-MULTI-LIB-REUSE-RV32IMAFDC-ILP32D-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imafc -mabi=lp64 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMAFC-LP64 %s
// GCC-MULTI-LIB-REUSE-RV64IMAFC-LP64: rv64imac/lp64
// GCC-MULTI-LIB-REUSE-RV64IMAFC-LP64-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32imafc_zfh -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32IMAFC_ZFH-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32IMAFC_ZFH-ILP32: rv32imac/ilp32
// GCC-MULTI-LIB-REUSE-RV32IMAFC_ZFH-ILP32-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv32iv_zvkb -mabi=ilp32 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV32I_ZVKB-ILP32 %s
// GCC-MULTI-LIB-REUSE-RV32I_ZVKB-ILP32: rv32i/ilp32
// GCC-MULTI-LIB-REUSE-RV32I_ZVKB-ILP32-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imfc -mabi=lp64 \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMFC-LP64 %s
// GCC-MULTI-LIB-REUSE-RV64IMFC-LP64: .
// GCC-MULTI-LIB-REUSE-RV64IMFC-LP64-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk3 \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imafc -mabi=lp64 -mcmodel=compact \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMAFC-LP64-COMPACT %s
// GCC-MULTI-LIB-REUSE-RV64IMAFC-LP64-COMPACT: rv64imac/lp64/compact
// GCC-MULTI-LIB-REUSE-RV64IMAFC-LP64-COMPACT-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imafdc_zicfilp_zicfiss -mabi=lp64d -fcf-protection \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMAFDC_ZICFILP_ZICFISS-LP64D-CFI %s

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imafdc_zicfilp_zicfiss -mabi=lp64d -fcf-protection=full \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMAFDC_ZICFILP_ZICFISS-LP64D-CFI %s
// GCC-MULTI-LIB-REUSE-RV64IMAFDC_ZICFILP_ZICFISS-LP64D-CFI: rv64imafdc_zicfiss_zicfilp/lp64d/cfi
// GCC-MULTI-LIB-REUSE-RV64IMAFDC_ZICFILP_ZICFISS-LP64D-CFI-NOT: {{^.+$}}

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imafdc_zicfilp_zicfiss -mabi=lp64d -fcf-protection=branch \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMAFDC_ZICFILP_ZICFISS-LP64D-NONE-CFI %s

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imafdc_zicfilp_zicfiss -mabi=lp64d -fcf-protection=return \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMAFDC_ZICFILP_ZICFISS-LP64D-NONE-CFI %s

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk \
// RUN:   --print-multi-directory \
// RUN:   -march=rv64imafdc_zicfilp_zicfiss -mabi=lp64d -fcf-protection=none \
// RUN:   | FileCheck -check-prefix=GCC-MULTI-LIB-REUSE-RV64IMAFDC_ZICFILP_ZICFISS-LP64D-NONE-CFI %s
// GCC-MULTI-LIB-REUSE-RV64IMAFDC_ZICFILP_ZICFISS-LP64D-NONE-CFI: rv64imafdc/lp64d
// GCC-MULTI-LIB-REUSE-RV64IMAFDC_ZICFILP_ZICFISS-LP64D-NONE-CFI-NOT: {{^.+$}}
