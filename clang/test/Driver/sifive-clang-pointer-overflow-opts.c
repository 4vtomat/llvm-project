// RUN: %clang -### -S -fno-strict-pointer-overflow %s 2>&1 | FileCheck %s
// COM: Make sure the driver flag is passed to the frontend.
// CHECK: -fno-strict-pointer-overflow

// RUN: %clang -### -S -fwrapv -fno-strict-pointer-overflow %s 2>&1 | FileCheck %s --check-prefix=FWRAP
// COM: Make sure we don't add redundant flags.
// FWRAP-NOT: -fno-strict-pointer-overflow
