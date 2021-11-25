/*
Copyright (c) 2015 - 2021 SiFive, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to use this
Software with RISC-V based SiFive products only and not with any other
processors and platforms, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE LICENSED SOFTWARE IS PROVIDED TO LICENSEE “AS IS” WITHOUT ANY SUPPORT
SERVICES OR WARRANTIES OF ANY KIND, INCLUDING, BUT WITHOUT LIMITATION, SIFIVE
DOES NOT WARRANT TO LICENSEE THAT THE LICENSED SOFTWARE WILL OPERATE ERROR FREE
OR UNINTERRUPTED, NOR THAT IT WILL MEET YOUR REQUIREMENTS. SIFIVE SHALL NOT HAVE
ANY DUTY OR OBLIGATION TO DEFEND OR INDEMNIFY LICENSEE OR TO HOLD IT HARMLESS
FOR ANY REASON RELATED TO THE LICENSED SOFTWARE, OR OTHERWISE BE LIABLE TO
LICENSEE OR ANY THIRD PARTY FOR ANY INCIDENTAL, INDIRECT, SPECIAL, EXEMPLARY,
PUNITIVE, OR CONSEQUENTIAL DAMAGES (INCLUDING LOST PROFITS) ARISING OUT OF OR
RELATED TO THIS AGREEMENT.
*/
#ifndef SIFIVE_RECODE_NEON_H
#define SIFIVE_RECODE_NEON_H
#if defined(__riscv_v_min_vlen) && (512 <= __riscv_v_min_vlen)
#include "arm_neon_512.h"
#elif defined(__riscv_v_min_vlen) && (256 <= __riscv_v_min_vlen)
#include "arm_neon_256.h"
#elif defined(__riscv_v_min_vlen) && (128 <= __riscv_v_min_vlen)
#include "arm_neon_128.h"
#elif defined(__riscv_v_min_vlen) && (__riscv_v_min_vlen < 64)
#error "The minimal VLEN requirement for SIFIVE RECODE NEON is 64 or greater"
#else
#include "arm_neon_64.h"
#endif
#endif
