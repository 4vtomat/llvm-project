// Check target CPUs are correctly passed.

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=rocket-rv32 | FileCheck -check-prefix=MCPU-ROCKET32 %s
// MCPU-ROCKET32: "-nostdsysteminc" "-target-cpu" "rocket-rv32"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=rocket-rv64 | FileCheck -check-prefix=MCPU-ROCKET64 %s
// MCPU-ROCKET64: "-nostdsysteminc" "-target-cpu" "rocket-rv64"
// MCPU-ROCKET64: "-target-feature" "+64bit"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=syntacore-scr1-base | FileCheck -check-prefix=MCPU-SYNTACORE-SCR1-BASE %s
// MCPU-SYNTACORE-SCR1-BASE: "-target-cpu" "syntacore-scr1-base"
// MCPU-SYNTACORE-SCR1-BASE: "-target-feature" "+c" "-target-feature" "-64bit"
// MCPU-SYNTACORE-SCR1-BASE: "-target-abi" "ilp32"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=syntacore-scr1-max | FileCheck -check-prefix=MCPU-SYNTACORE-SCR1-MAX %s
// MCPU-SYNTACORE-SCR1-MAX: "-target-cpu" "syntacore-scr1-max"
// MCPU-SYNTACORE-SCR1-MAX: "-target-feature" "+m" "-target-feature" "+c" "-target-feature" "-64bit"
// MCPU-SYNTACORE-SCR1-MAX: "-target-abi" "ilp32"

// We cannot check much for -mcpu=native, but it should be replaced by a valid CPU string.
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=native | FileCheck -check-prefix=MCPU-NATIVE %s
// MCPU-NATIVE-NOT: "-target-cpu" "native"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=rocket-rv32 | FileCheck -check-prefix=MTUNE-ROCKET32 %s
// MTUNE-ROCKET32: "-tune-cpu" "rocket-rv32"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=rocket-rv64 | FileCheck -check-prefix=MTUNE-ROCKET64 %s
// MTUNE-ROCKET64: "-tune-cpu" "rocket-rv64"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=syntacore-scr1-base | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR1-BASE %s
// MTUNE-SYNTACORE-SCR1-BASE: "-tune-cpu" "syntacore-scr1-base"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=syntacore-scr1-max | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR1-MAX %s
// MTUNE-SYNTACORE-SCR1-MAX: "-tune-cpu" "syntacore-scr1-max"

// Check mtune alias CPU has resolved to the right CPU according XLEN.
// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=generic | FileCheck -check-prefix=MTUNE-GENERIC-32 %s
// MTUNE-GENERIC-32: "-tune-cpu" "generic"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=generic | FileCheck -check-prefix=MTUNE-GENERIC-64 %s
// MTUNE-GENERIC-64: "-tune-cpu" "generic"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=rocket | FileCheck -check-prefix=MTUNE-ROCKET-32 %s
// MTUNE-ROCKET-32: "-tune-cpu" "rocket"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=rocket | FileCheck -check-prefix=MTUNE-ROCKET-64 %s
// MTUNE-ROCKET-64: "-tune-cpu" "rocket"

// We cannot check much for -mtune=native, but it should be replaced by a valid CPU string.
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=native | FileCheck -check-prefix=MTUNE-NATIVE %s
// MTUNE-NATIVE-NOT: "-tune-cpu" "native"

// mcpu with default march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e20 | FileCheck -check-prefix=MCPU-SIFIVE-E20 %s
// MCPU-SIFIVE-E20: "-nostdsysteminc" "-target-cpu" "sifive-e20"
// MCPU-SIFIVE-E20: "-target-feature" "+m" "-target-feature" "+c"
// MCPU-SIFIVE-E20: "-target-abi" "ilp32"

// mcpu with default march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e21 | FileCheck -check-prefix=MCPU-SIFIVE-E21 %s
// MCPU-SIFIVE-E21: "-nostdsysteminc" "-target-cpu" "sifive-e21"
// MCPU-SIFIVE-E21: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+c"
// MCPU-SIFIVE-E21: "-target-abi" "ilp32"

// mcpu with default march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e24 | FileCheck -check-prefix=MCPU-SIFIVE-E24 %s
// MCPU-SIFIVE-E24: "-nostdsysteminc" "-target-cpu" "sifive-e24"
// MCPU-SIFIVE-E24: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f"
// MCPU-SIFIVE-E24: "-target-feature" "+c"
// MCPU-SIFIVE-E24: "-target-abi" "ilp32"

// mcpu with default march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e34 | FileCheck -check-prefix=MCPU-SIFIVE-E34 %s
// MCPU-SIFIVE-E34: "-nostdsysteminc" "-target-cpu" "sifive-e34"
// MCPU-SIFIVE-E34: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f"
// MCPU-SIFIVE-E34: "-target-feature" "+c"
// MCPU-SIFIVE-E34: "-target-abi" "ilp32"

// mcpu with mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-s21 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-S21 %s
// MCPU-ABI-SIFIVE-S21: "-nostdsysteminc" "-target-cpu" "sifive-s21"
// MCPU-ABI-SIFIVE-S21: "-target-feature" "+m" "-target-feature" "+a"
// MCPU-ABI-SIFIVE-S21: "-target-feature" "+c" "-target-feature" "+64bit"
// MCPU-ABI-SIFIVE-S21: "-target-abi" "lp64"

// mcpu with mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-s51 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-S51 %s
// MCPU-ABI-SIFIVE-S51: "-nostdsysteminc" "-target-cpu" "sifive-s51"
// MCPU-ABI-SIFIVE-S51: "-target-feature" "+m" "-target-feature" "+a"
// MCPU-ABI-SIFIVE-S51: "-target-feature" "+c" "-target-feature" "+64bit"
// MCPU-ABI-SIFIVE-S51: "-target-abi" "lp64"

// mcpu with default march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-s54 | FileCheck -check-prefix=MCPU-SIFIVE-S54 %s
// MCPU-SIFIVE-S54: "-nostdsysteminc" "-target-cpu" "sifive-s54"
// MCPU-SIFIVE-S54: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-S54: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+64bit"
// MCPU-SIFIVE-S54: "-target-abi" "lp64d"

// mcpu with mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-s76 | FileCheck -check-prefix=MCPU-SIFIVE-S76 %s
// MCPU-SIFIVE-S76: "-nostdsysteminc" "-target-cpu" "sifive-s76"
// MCPU-SIFIVE-S76: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-S76: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+64bit"
// MCPU-SIFIVE-S76: "-target-abi" "lp64d"

// mcpu with default march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-u54 | FileCheck -check-prefix=MCPU-SIFIVE-U54 %s
// MCPU-SIFIVE-U54: "-nostdsysteminc" "-target-cpu" "sifive-u54"
// MCPU-SIFIVE-U54: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-U54: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+64bit"
// MCPU-SIFIVE-U54: "-target-abi" "lp64d"

// mcpu with mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-u54 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-U54 %s
// MCPU-ABI-SIFIVE-U54: "-nostdsysteminc" "-target-cpu" "sifive-u54"
// MCPU-ABI-SIFIVE-U54: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-ABI-SIFIVE-U54: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+64bit"
// MCPU-ABI-SIFIVE-U54: "-target-abi" "lp64"

// mcpu with default march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e76 | FileCheck -check-prefix=MCPU-SIFIVE-E76 %s
// MCPU-SIFIVE-E76: "-nostdsysteminc" "-target-cpu" "sifive-e76"
// MCPU-SIFIVE-E76: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f"
// MCPU-SIFIVE-E76: "-target-feature" "+c"
// MCPU-SIFIVE-E76: "-target-abi" "ilp32"

// mcpu with default march
// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-e76m | FileCheck -check-prefix=MCPU-SIFIVE-E76M %s
// MCPU-SIFIVE-E76M: "-target-cpu" "sifive-e76m"
// MCPU-SIFIVE-E76M: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f"
// MCPU-SIFIVE-E76M: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zba"
// MCPU-SIFIVE-E76M: "-target-feature" "+zbb"
// MCPU-SIFIVE-E76M: "-target-abi" "ilp32"

// mcpu with mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-u74 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-U74 %s
// MCPU-ABI-SIFIVE-U74: "-nostdsysteminc" "-target-cpu" "sifive-u74"
// MCPU-ABI-SIFIVE-U74: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-ABI-SIFIVE-U74: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+64bit"
// MCPU-ABI-SIFIVE-U74: "-target-abi" "lp64"

// mcpu with mabi option
// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-u74m -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-U74M %s
// MCPU-ABI-SIFIVE-U74M: "-target-cpu" "sifive-u74m"
// MCPU-ABI-SIFIVE-U74M: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-ABI-SIFIVE-U74M: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+zba"
// MCPU-ABI-SIFIVE-U74M: "-target-feature" "+zbb" "-target-feature" "+64bit"
// MCPU-ABI-SIFIVE-U74M: "-target-abi" "lp64"

// march overwrite mcpu's default march
// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=sifive-e31 -march=rv32imc | FileCheck -check-prefix=MCPU-MARCH %s
// MCPU-MARCH: "-nostdsysteminc" "-target-cpu" "sifive-e31" "-target-feature" "+m" "-target-feature" "+c"
// MCPU-MARCH: "-target-abi" "ilp32"

// Check interaction between mcpu and mtune, mtune won't affect arch related
// target feature, but mcpu will.
//
// In this case, sifive-e31 is rv32imac, sifive-e76 is rv32imafc, so M-extension
// should not enabled.
//
// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=sifive-e31 -mtune=sifive-e76 | FileCheck -check-prefix=MTUNE-E31-MCPU-E76 %s
// MTUNE-E31-MCPU-E76: "-target-cpu" "sifive-e31"
// MTUNE-E31-MCPU-E76-NOT: "-target-feature" "+f"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+m"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+a"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+c"
// MTUNE-E31-MCPU-E76-SAME: "-tune-cpu" "sifive-e76"

// Check failed cases

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=generic-rv321 | FileCheck -check-prefix=FAIL-MCPU-NAME %s
// FAIL-MCPU-NAME: error: unsupported argument 'generic-rv321' to option '-mcpu='

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=generic-rv32 -march=rv64i | FileCheck -check-prefix=MISMATCH-ARCH %s
// MISMATCH-ARCH: error: unsupported argument 'generic-rv32' to option '-mcpu='

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=generic-rv64 | FileCheck -check-prefix=MISMATCH-MCPU %s
// MISMATCH-MCPU: error: unsupported argument 'generic-rv64' to option '-mcpu='

// mcpu with default march include experimental extensions
// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-x280 | FileCheck -check-prefix=MCPU-SIFIVE-X280 %s
// MCPU-SIFIVE-X280: "-nostdsysteminc" "-target-cpu" "sifive-x280"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+c" "-target-feature" "+v"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zfh"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zba" "-target-feature" "+zbb"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+experimental-zvfh"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zvl128b"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zvl256b"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zvl32b"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zvl512b"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zvl64b"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+xsfvfhbfmin"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+xsfvqmaccqoq"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+64bit"
// MCPU-SIFIVE-X280-SAME: "-target-abi" "lp64d"

// mcpu with default march include experimental extensions
// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-x280n | FileCheck -check-prefix=MCPU-SIFIVE-X280N %s
// MCPU-SIFIVE-X280N: "-nostdsysteminc" "-target-cpu" "sifive-x280n"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+c" "-target-feature" "+v"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+zfh"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+zba" "-target-feature" "+zbb"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+experimental-zvfh"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+zvl128b"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+zvl256b" "-target-feature" "+zvl32b"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+zvl512b" "-target-feature" "+zvl64b"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+xsfvfhbfmin"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+xsfvfnrclipxfqf" "-target-feature" "+xsfvfwmaccqqq"
// MCPU-SIFIVE-X280N-SAME: "-target-feature" "+xsfvqmaccqoq" "-target-feature" "+64bit"
// MCPU-SIFIVE-X280N-SAME: "-target-abi" "lp64d"

// mcpu with default march include experimental extensions
// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-x280o | FileCheck -check-prefix=MCPU-SIFIVE-X280O %s
// MCPU-SIFIVE-X280O: "-nostdsysteminc" "-target-cpu" "sifive-x280o"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+c" "-target-feature" "+v"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+zfh"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+zba" "-target-feature" "+zbb"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+experimental-zvfh"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+zvl128b"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+zvl256b" "-target-feature" "+zvl32b"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+zvl512b" "-target-feature" "+zvl64b"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+xsfvfhbfmin"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+xsfvfnrclipxfqf" "-target-feature" "+xsfvfwmaccqqq"
// MCPU-SIFIVE-X280O-SAME: "-target-feature" "+xsfvqmaccqoq" "-target-feature" "+64bit"
// MCPU-SIFIVE-X280O-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p270 | FileCheck -check-prefix=MCPU-SIFIVE-P270 %s
// MCPU-SIFIVE-P270: "-target-cpu" "sifive-p270"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+c" "-target-feature" "+v"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+zfh"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+zba" "-target-feature" "+zbb"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+experimental-zvfh"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+zvl128b"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+zvl256b"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+zvl32b"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+zvl64b"
// MCPU-SIFIVE-P270-SAME: "-target-feature" "+64bit"
// MCPU-SIFIVE-P270-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p550 | FileCheck -check-prefix=MCPU-SIFIVE-P550 %s
// MCPU-SIFIVE-P550: "-target-cpu" "sifive-p550"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+zba"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+zbb"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+64bit"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p650 | FileCheck -check-prefix=MCPU-SIFIVE-P650 %s
// MCPU-SIFIVE-P650: "-target-cpu" "sifive-p650"
// MCPU-SIFIVE-P650-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-P650-SAME: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+zfhmin"
// MCPU-SIFIVE-P650-SAME: "-target-feature" "+zba" "-target-feature" "+zbb" "-target-feature" "+zbs"
// MCPU-SIFIVE-P650-SAME: "-target-feature" "+64bit"
// MCPU-SIFIVE-P650-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p450 | FileCheck -check-prefix=MCPU-SIFIVE-P450 %s
// MCPU-SIFIVE-P450: "-target-cpu" "sifive-p450"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+zfhmin"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zba" "-target-feature" "+zbb" "-target-feature" "+zbs"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+64bit"
// MCPU-SIFIVE-P450-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p470 | FileCheck -check-prefix=MCPU-SIFIVE-P470 %s
// MCPU-SIFIVE-P470: "-target-cpu" "sifive-p470"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+c" "-target-feature" "+v" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+zfhmin"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zba" "-target-feature" "+zbb" "-target-feature" "+zbs"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+64bit"
// MCPU-SIFIVE-P470-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p670 | FileCheck -check-prefix=MCPU-SIFIVE-P670 %s
// MCPU-SIFIVE-P670: "-target-cpu" "sifive-p670"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+c" "-target-feature" "+v" "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+zfhmin"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zba" "-target-feature" "+zbb" "-target-feature" "+zbs"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+64bit"
// MCPU-SIFIVE-P670-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-e66 | FileCheck -check-prefix=MCPU-SIFIVE-E66 %s
// MCPU-SIFIVE-E66: "-target-cpu" "sifive-e66"
// MCPU-SIFIVE-E66-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+c"
// MCPU-SIFIVE-E66-SAME: "-target-feature" "+zba" "-target-feature" "+zbb"
// MCPU-SIFIVE-E66-SAME: "-target-abi" "ilp32"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-s66 | FileCheck -check-prefix=MCPU-SIFIVE-S66 %s
// MCPU-SIFIVE-S66: "-target-cpu" "sifive-s66"
// MCPU-SIFIVE-S66-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-S66-SAME: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zba" "-target-feature" "+zbb"
// MCPU-SIFIVE-S66-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-u64 | FileCheck -check-prefix=MCPU-SIFIVE-U64 %s
// MCPU-SIFIVE-U64: "-target-cpu" "sifive-u64"
// MCPU-SIFIVE-U64-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-U64-SAME: "-target-feature" "+c" "-target-feature" "+zicsr" "-target-feature" "+zba" "-target-feature" "+zbb"
// MCPU-SIFIVE-U64-SAME: "-target-abi" "lp64d"
