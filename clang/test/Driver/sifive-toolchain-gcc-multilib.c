// Test case for scanning input of GCC output as multilib config
// Skip this test on Windows, we can't create a dummy GCC to output
// multilib config, ExecuteAndWait only execute *.exe file.
// UNSUPPORTED: system-windows

// RUN: %clang %s \
// RUN:   -target riscv32-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv32_elf_sdk \
// RUN:   --print-multi-lib \
// RUN:   | FileCheck -check-prefix=C-RV32-GCC-MULTI-LIB %s
// C-RV32-GCC-MULTI-LIB: rv32iac/ilp32;@march=rv32iac@mabi=ilp32
// C-RV32-GCC-MULTI-LIB-NEXT: rv32imafc/ilp32f;@march=rv32iac@mabi=ilp32f
// C-RV32-GCC-MULTI-LIB-NEXT: rv64imafdc/lp64d;@march=rv64imafdc@mabi=lp64d
// C-RV32-GCC-MULTI-LIB-NOT:  {{^.+$}}

// RUN: %clang %s \
// RUN:   -### -v \
// RUN:   -target riscv32-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv32_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32-GCC-MULTI-LIB-V %s
// C-RV32-GCC-MULTI-LIB-V: Candidate multilib: rv32iac/ilp32;@march=rv32iac@mabi=ilp32
// C-RV32-GCC-MULTI-LIB-V-NEXT: Candidate multilib: rv32imafc/ilp32f;@march=rv32iac@mabi=ilp32f
// C-RV32-GCC-MULTI-LIB-V-NEXT: Candidate multilib: rv64imafdc/lp64d;@march=rv64imafdc@mabi=lp64d

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk \
// RUN:   --print-multi-lib \
// RUN:   | FileCheck -check-prefix=C-RV64-GCC-MULTI-LIB %s
// C-RV64-GCC-MULTI-LIB: rv32iac/ilp32;@march=rv32iac@mabi=ilp32
// C-RV64-GCC-MULTI-LIB-NEXT: rv32imafc/ilp32f;@march=rv32iac@mabi=ilp32f
// C-RV64-GCC-MULTI-LIB-NEXT: rv64imafdc/lp64d;@march=rv64imafdc@mabi=lp64d
// C-RV64-GCC-MULTI-LIB-NEXT: rv64imafdc_zicfiss_zicfilp/lp64d/cfi;@march=rv64imafdc_zicfiss_zicfilp@mabi=lp64d@fcf-protection=full
// C-RV64-GCC-MULTI-LIB-NOT:  {{^.+$}}

// RUN: %clang %s \
// RUN:   -### -v \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-GCC-MULTI-LIB-V %s
// C-RV64-GCC-MULTI-LIB-V: Candidate multilib: rv32iac/ilp32;@march=rv32iac@mabi=ilp32
// C-RV64-GCC-MULTI-LIB-V-NEXT: Candidate multilib: rv32imafc/ilp32f;@march=rv32iac@mabi=ilp32f
// C-RV64-GCC-MULTI-LIB-V-NEXT: Candidate multilib: rv64imafdc/lp64d;@march=rv64imafdc@mabi=lp64d
// C-RV64-GCC-MULTI-LIB-V-NEXT: Candidate multilib: rv64imafdc_zicfiss_zicfilp/lp64d/cfi;@march=rv64imafdc_zicfiss_zicfilp@mabi=lp64d@fcf-protection=full

// RUN: %clang %s \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk_bad \
// RUN:   --print-multi-lib 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-GCC-MULTI-LIB-BAD %s
// C-RV64-GCC-MULTI-LIB-BAD: warning: xxx option unrecognized in multi-lib configuration when parsing config from GCC, falling back to built-in multi-lib configuration [-Wmultilib-fallback]

// RUN: %clang %s \
// RUN:   -### -v \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk_bad2 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-GCC-MULTI-LIB-BAD-V %s
// C-RV64-GCC-MULTI-LIB-BAD-V: Attempt to obtain the multilib configuration from '{{.*}}gcc{{(.exe)?}}'
// C-RV64-GCC-MULTI-LIB-BAD-V-NEXT: Failed to execute '{{.*}}gcc{{(.exe)?}}' in an attempt to obtain the multilib configuration from GCC

// RUN: %clang %s \
// RUN:   -### -v \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv64_elf_sdk_bad3 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-GCC-NOT-FOUND %s
// C-RV64-GCC-NOT-FOUND: Failed to find GCC in an attempt to obtain the multilib configuration
