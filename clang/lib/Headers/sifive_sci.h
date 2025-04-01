/*===---- sifive_sci.h - SiFive SCI intrinsics -----------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __SIFIVE_SCI_H
#define __SIFIVE_SCI_H

#if defined(__riscv_xsfsci)

#if __riscv_xlen == 32
// Legacy interface
#define __riscv_sf_sci_x_xx_32(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_x_xx_32((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_x_xx_se_32(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_x_xx_se_32((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_xx_se_32(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_xx_se_32((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_3_r_x_xx_32(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_x_xx_32((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_3_r_x_xx_se_32(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_x_xx_se_32((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_3_r_xx_se_32(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_xx_se_32((funct3), (funct7), (x), (y));
#endif // __riscv_xlen == 32

#if __riscv_xlen == 64
// Legacy interface
#define __riscv_sf_sci_x_xx_64(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_x_xx_64((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_x_xx_se_64(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_x_xx_se_64((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_xx_se_64(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_xx_se_64((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_3_r_x_xx_64(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_x_xx_64((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_3_r_x_xx_se_64(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_x_xx_se_64((funct3), (funct7), (x), (y));

#define __riscv_sf_sci_3_r_xx_se_64(funct3, funct7, x, y) \
  __builtin_riscv_sf_sci_3_r_xx_se_64((funct3), (funct7), (x), (y));
#endif // __riscv_xlen == 64

#endif // defined(__riscv_xsfsci)

#endif
