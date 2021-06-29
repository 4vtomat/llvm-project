// Check the --specs=nano.specs has handled correctly.

// RUN: %clang -### -target riscv32-elf \
// RUN:  --gcc-toolchain= --specs=nano.specs %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LIBC-NANO %s
// CHECK-LIBC-NANO: {{.*}} "-lc_nano"
// CHECK-LIBC-NANO-NOT: {{.*}} "-lc"

// RUN: %clang -### -target riscv32-elf \
// RUN:  --gcc-toolchain= --specs=nano.specs -lm %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LIBM-NANO %s
// CHECK-LIBM-NANO: {{.*}} "-lm_nano"
// CHECK-LIBM-NANO-NOT: {{.*}} "-lm"

// RUN: %clang -### -target riscv32-elf \
// RUN:  --gcc-toolchain= --specs=nano.specs --specs=bar.specs %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MULTI-SPEC-NANO %s
// CHECK-MULTI-SPEC-NANO: {{.*}} "-lc_nano"
// CHECK-MULTI-SPEC-NANO-NOT: {{.*}} "-lc"
