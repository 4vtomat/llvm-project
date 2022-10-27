enum {
  __RISCV_NTLH_INNERMOST_PRIVATE = 2,
  __RISCV_NTLH_ALL_PRIVATE,
  __RISCV_NTLH_INNERMOST_SHARED,
  __RISCV_NTLH_ALL
};

#define __rv_ntl_load(PTR, DOMAIN) __builtin_riscv_ntl_load(PTR, DOMAIN)
#define __rv_ntl_store(PTR, VAL, DOMAIN)                                       \
  __builtin_riscv_ntl_store(PTR, VAL, DOMAIN)
