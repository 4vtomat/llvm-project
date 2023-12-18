#ifndef __RT_LOOPPROFILE_H
#define __RT_LOOPPROFILE_H

#include <stdlib.h>

/* LOOP_PROFILE_RUNTIME_VAR is the runtime hook which will be invoke by
 * -u<hook_var>, then the linker will invoke the loopProfile runtime.
 */
#define LOOP_PROFILE_RUNTIME_VAR __llvm_loop_profile_runtime
#define COMPILER_RT_VISIBILITY __attribute__((visibility("hidden")))

void dump_loop_profile(void);
COMPILER_RT_VISIBILITY extern int LOOP_PROFILE_RUNTIME_VAR;
#endif
