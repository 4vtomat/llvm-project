// RUN: %clang -### -S -fwrapv-pointer-arithmetic %s 2>&1 | FileCheck %s
// COM: Make sure the driver flag is passed to the frontend.
// RUN: %clang -### -S -fno-strict-pointer-overflow %s 2>&1 | FileCheck %s
// COM: Make sure the alias works.

// CHECK: -fwrapv-pointer-arithmetic

// RUN: %clang -### -S -fwrapv-pointer -fwrapv-pointer-arithmetic %s 2>&1 | FileCheck %s --check-prefix=FWRAP
// COM: Make sure we don't add redundant flags.
// FWRAP-NOT: -fwrapv-pointer-arithmetic
