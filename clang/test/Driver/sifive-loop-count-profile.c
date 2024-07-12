// if SIFIVE_CUSTOMIZATION
// RUN: %clang --target=riscv32-unknown-linux-gnu %s -fuse-ld=gold -flto \
// RUN:   -fsifive-loop-count-profile-generate \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=LTO-TEST
//
// LTO-TEST: "-plugin-opt=-sifive-enable-loop-count-profiler"

// endif SIFIVE_CUSTOMIZATION
