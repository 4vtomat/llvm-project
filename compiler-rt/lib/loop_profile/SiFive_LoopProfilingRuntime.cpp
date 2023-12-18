//===- LoopProfilingRuntime.cpp - Loop Profile runtime initialization -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

extern "C" {

#include "SiFive_LoopProfiling.h"
#include "stdio.h"

static int RegisterRuntime() {
  atexit(dump_loop_profile);
  return 0;
}

/* int __llvm_loop_profile_runtime  */
COMPILER_RT_VISIBILITY int LOOP_PROFILE_RUNTIME_VAR = RegisterRuntime();
}
