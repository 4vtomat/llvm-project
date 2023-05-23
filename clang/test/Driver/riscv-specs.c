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


// Check the --specs=gloss-segger.spes has handled correctly.

// RUN: %clang -### -target riscv32-elf \
// RUN:  --gcc-toolchain= --specs=gloss-segger.specs %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LIBC-SEGGER %s
// CHECK-LIBC-SEGGER: {{.*}} "-D__SEGGER_LIBC__"
// CHECK-LIBC-SEGGER: {{.*}} "-lc_segger"
// CHECK-LIBC-SEGGER-NOT: {{.*}} "-lc"

// RUN: %clang -### -target riscv32-elf \
// RUN:  --gcc-toolchain= --specs=nano.specs --specs=gloss-segger.specs %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-BOTH-NANO-SEGGER-SPEC %s
// CHECK-BOTH-NANO-SEGGER-SPEC: {{.*}} "-D__SEGGER_LIBC__"
// CHECK-BOTH-NANO-SEGGER-SPEC: {{.*}} "-lc_segger"
// CHECK-BOTH-NANO-SEGGER-SPEC-NOT: {{.*}} "-lc"
// CHECK-BOTH-NANO-SEGGER-SPEC-NOT: {{.*}} "-lc_nano"

// RUN: %clang -### -target riscv32-elf \
// RUN:  --gcc-toolchain= --specs=gloss-segger.specs --specs=nano.specs %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-BOTH-SEGGER-NANO-SPEC %s
// CHECK-BOTH-SEGGER-NANO-SPEC: {{.*}} "-lc_nano"
// CHECK-BOTH-SEGGER-NANO-SPEC-NOT: {{.*}} "-D__SEGGER_LIBC__"
// CHECK-BOTH-SEGGER-NANO-SPEC-NOT: {{.*}} "-lc"
// CHECK-BOTH-SEGGER-NANO-SPEC-NOT: {{.*}} "-lc_segger"
