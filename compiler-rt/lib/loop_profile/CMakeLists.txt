add_compiler_rt_component(loop_profile)

set(LOOP_PROFILE_SOURCES
  SiFive_LoopProfiling.c
  SiFive_LoopProfilingRuntime.cpp
  )

set(LOOP_PROFILE_HEADERS
  SiFive_LoopProfiling.h
  )

include_directories(..)
include_directories(../../include)

if(FUCHSIA OR UNIX)
 set(EXTRA_FLAGS
     -fPIC
     -Wno-pedantic)
endif()

# We don't use the C++ Standard Library here, so avoid including it by mistake.
append_list_if(COMPILER_RT_HAS_NOSTDINCXX_FLAG -nostdinc++ EXTRA_FLAGS)
# XRay uses C++ standard library headers.
string(REGEX REPLACE "-stdlib=[a-zA-Z+]*" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# This appears to be a C-only warning banning the use of locals in aggregate
# initializers. All other compilers accept this, though.
# nonstandard extension used : 'identifier' : cannot be initialized using address of automatic variable
append_list_if(COMPILER_RT_HAS_WD4221_FLAG /wd4221 EXTRA_FLAGS)

# Disable 'nonstandard extension used: translation unit is empty'.
append_list_if(COMPILER_RT_HAS_WD4206_FLAG /wd4206 EXTRA_FLAGS)

if(APPLE)
  add_compiler_rt_runtime(clang_rt.loop_profile
    STATIC
    OS ${PROFILE_SUPPORTED_OS}
    ARCHS ${PROFILE_SUPPORTED_ARCH}
    CFLAGS ${EXTRA_FLAGS}
    SOURCES ${LOOP_PROFILE_SOURCES}
    ADDITIONAL_HEADERS ${LOOP_PROFILE_HEADERS}
    PARENT_TARGET loop_profile)
else()
  add_compiler_rt_runtime(clang_rt.loop_profile
    STATIC
    ARCHS ${PROFILE_SUPPORTED_ARCH}
    CFLAGS ${EXTRA_FLAGS}
    SOURCES ${LOOP_PROFILE_SOURCES}
    ADDITIONAL_HEADERS ${LOOP_PROFILE_HEADERS}
    PARENT_TARGET loop_profile)
endif()
