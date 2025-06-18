; RUN: llc -mtriple=riscv32 -mattr=help 2>&1 | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=help 2>&1 | FileCheck %s

; CHECK: Available features for this target:
<<<<<<< HEAD
; CHECK:   32bit                            - Implements RV32.
; CHECK:   64bit                            - Implements RV64.
; CHECK:   a                                - 'A' (Atomic Instructions).
; CHECK:   auipc-addi-fusion                - Enable AUIPC+ADDI macrofusion.
; CHECK:   b                                - 'B' (the collection of the Zba, Zbb, Zbs extensions).
; CHECK:   c                                - 'C' (Compressed Instructions).
; CHECK:   conditional-cmv-fusion           - Enable branch+c.mv fusion.
; CHECK:   d                                - 'D' (Double-Precision Floating-Point).
; CHECK:   disable-latency-sched-heuristic  - Disable latency scheduling heuristic.
; CHECK:   dlen-factor-2                    - Vector unit DLEN(data path width) is half of VLEN.
; CHECK:   e                                - 'E' (Embedded Instruction Set with 16 GPRs).
; CHECK:   experimental                     - Experimental intrinsics.
; CHECK:   experimental-rvm23u32            - RISC-V experimental-rvm23u32 profile.
; CHECK:   experimental-smctr               - 'Smctr' (Control Transfer Records Machine Level).
; CHECK:   experimental-ssctr               - 'Ssctr' (Control Transfer Records Supervisor Level).
; CHECK:   experimental-svukte              - 'Svukte' (Address-Independent Latency of User-Mode Faults to Supervisor Addresses).
; CHECK:   experimental-xqcia               - 'Xqcia' (Qualcomm uC Arithmetic Extension).
; CHECK:   experimental-xqcics              - 'Xqcics' (Qualcomm uC Conditional Select Extension).
; CHECK:   experimental-xqcicsr             - 'Xqcicsr' (Qualcomm uC CSR Extension).
; CHECK:   experimental-xqcilsm             - 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension).
; CHECK:   experimental-xqcisls             - 'Xqcisls' (Qualcomm uC Scaled Load Store Extension).
; CHECK:   experimental-zalasr              - 'Zalasr' (Load-Acquire and Store-Release Instructions).
; CHECK:   experimental-zicfilp             - 'Zicfilp' (Landing pad).
; CHECK:   experimental-zicfiss             - 'Zicfiss' (Shadow stack).
; CHECK:   experimental-zvbc32e             - 'Zvbc32e' (Vector Carryless Multiplication with 32-bits elements).
; CHECK:   experimental-zvkgs               - 'Zvkgs' (Vector-Scalar GCM instructions for Cryptography).
; CHECK:   f                                - 'F' (Single-Precision Floating-Point).
; CHECK:   forced-atomics                   - Assume that lock-free native-width atomics are available.
; CHECK:   h                                - 'H' (Hypervisor).
; CHECK:   i                                - 'I' (Base Integer Instruction Set).
; CHECK:   ld-add-fusion                    - Enable LD+ADD macrofusion.
; CHECK:   log-vrgather                     - Has vrgather.vv with LMUL*log2(LMUL) latency
; CHECK:   lui-addi-fusion                  - Enable LUI+ADDI macro fusion.
; CHECK:   m                                - 'M' (Integer Multiplication and Division).
; CHECK:   mips-p8700                       - MIPS p8700 processor.
; CHECK:   no-default-unroll                - Disable default unroll preference..
; CHECK:   no-rvc-hints                     - Disable RVC Hint Instructions..
; CHECK:   no-sink-splat-operands           - Disable sink splat operands to enable .vx, .vf,.wx, and .wf instructions.
; CHECK:   no-trailing-seq-cst-fence        - Disable trailing fence for seq-cst store..
; CHECK:   optimized-nf2-segment-load-store - vlseg2eN.v and vsseg2eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf3-segment-load-store - vlseg3eN.v and vsseg3eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf4-segment-load-store - vlseg4eN.v and vsseg4eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf5-segment-load-store - vlseg5eN.v and vsseg5eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf6-segment-load-store - vlseg6eN.v and vsseg6eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf7-segment-load-store - vlseg7eN.v and vsseg7eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf8-segment-load-store - vlseg8eN.v and vsseg8eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-zero-stride-load       - Optimized (perform fewer memory operations)zero-stride vector load.
; CHECK:   predictable-select-expensive     - Prefer likely predicted branches over selects.
; CHECK:   prefer-w-inst                    - Prefer instructions with W suffix.
; CHECK:   relax                            - Enable Linker relaxation..
; CHECK:   reserve-x1                       - Reserve X1.
; CHECK:   reserve-x10                      - Reserve X10.
; CHECK:   reserve-x11                      - Reserve X11.
; CHECK:   reserve-x12                      - Reserve X12.
; CHECK:   reserve-x13                      - Reserve X13.
; CHECK:   reserve-x14                      - Reserve X14.
; CHECK:   reserve-x15                      - Reserve X15.
; CHECK:   reserve-x16                      - Reserve X16.
; CHECK:   reserve-x17                      - Reserve X17.
; CHECK:   reserve-x18                      - Reserve X18.
; CHECK:   reserve-x19                      - Reserve X19.
; CHECK:   reserve-x2                       - Reserve X2.
; CHECK:   reserve-x20                      - Reserve X20.
; CHECK:   reserve-x21                      - Reserve X21.
; CHECK:   reserve-x22                      - Reserve X22.
; CHECK:   reserve-x23                      - Reserve X23.
; CHECK:   reserve-x24                      - Reserve X24.
; CHECK:   reserve-x25                      - Reserve X25.
; CHECK:   reserve-x26                      - Reserve X26.
; CHECK:   reserve-x27                      - Reserve X27.
; CHECK:   reserve-x28                      - Reserve X28.
; CHECK:   reserve-x29                      - Reserve X29.
; CHECK:   reserve-x3                       - Reserve X3.
; CHECK:   reserve-x30                      - Reserve X30.
; CHECK:   reserve-x31                      - Reserve X31.
; CHECK:   reserve-x4                       - Reserve X4.
; CHECK:   reserve-x5                       - Reserve X5.
; CHECK:   reserve-x6                       - Reserve X6.
; CHECK:   reserve-x7                       - Reserve X7.
; CHECK:   reserve-x8                       - Reserve X8.
; CHECK:   reserve-x9                       - Reserve X9.
; CHECK:   rva20s64                         - RISC-V rva20s64 profile.
; CHECK:   rva20u64                         - RISC-V rva20u64 profile.
; CHECK:   rva22s64                         - RISC-V rva22s64 profile.
; CHECK:   rva22u64                         - RISC-V rva22u64 profile.
; CHECK:   rva23s64                         - RISC-V rva23s64 profile.
; CHECK:   rva23u64                         - RISC-V rva23u64 profile.
; CHECK:   rvb23s64                         - RISC-V rvb23s64 profile.
; CHECK:   rvb23u64                         - RISC-V rvb23u64 profile.
; CHECK:   rvi20u32                         - RISC-V rvi20u32 profile.
; CHECK:   rvi20u64                         - RISC-V rvi20u64 profile.
; CHECK:   save-restore                     - Enable save/restore..
#if SIFIVE_CUSTOMIZATION
; CHECK:   sdext                            - 'Sdext' (External Debugging Extension).
; CHECK:   sdtrig                           - 'Sdtrig' (Debugging Triggers).
#endif // SIFIVE_CUSTOMIZATION
; CHECK:   sha                              - 'Sha' (Augmented Hypervisor).
; CHECK:   shcounterenw                     - 'Shcounterenw' (Support writeable hcounteren enable bit for any hpmcounter that is not read-only zero).
; CHECK:   shgatpa                          - 'Shgatpa' (SvNNx4 mode supported for all modes supported by satp, as well as Bare).
; CHECK:   shifted-zextw-fusion             - Enable SLLI+SRLI to be fused when computing (shifted) word zero extension.
; CHECK:   short-forward-branch-opt         - Enable short forward branch optimization.
; CHECK:   shtvala                          - 'Shtvala' (htval provides all needed values).
; CHECK:   shvsatpa                         - 'Shvsatpa' (vsatp supports all modes supported by satp).
; CHECK:   shvstvala                        - 'Shvstvala' (vstval provides all needed values).
; CHECK:   shvstvecd                        - 'Shvstvecd' (vstvec supports Direct mode).
; CHECK:   sifive7                          - SiFive 7-Series processors.
; CHECK:   smaia                            - 'Smaia' (Advanced Interrupt Architecture Machine Level).
; CHECK:   smcdeleg                         - 'Smcdeleg' (Counter Delegation Machine Level).
; CHECK:   smcsrind                         - 'Smcsrind' (Indirect CSR Access Machine Level).
; CHECK:   smdbltrp                         - 'Smdbltrp' (Double Trap Machine Level).
; CHECK:   smepmp                           - 'Smepmp' (Enhanced Physical Memory Protection).
; CHECK:   smmpm                            - 'Smmpm' (Machine-level Pointer Masking for M-mode).
; CHECK:   smnpm                            - 'Smnpm' (Machine-level Pointer Masking for next lower privilege mode).
; CHECK:   smrnmi                           - 'Smrnmi' (Resumable Non-Maskable Interrupts).
; CHECK:   smstateen                        - 'Smstateen' (Machine-mode view of the state-enable extension).
; CHECK:   ssaia                            - 'Ssaia' (Advanced Interrupt Architecture Supervisor Level).
; CHECK:   ssccfg                           - 'Ssccfg' (Counter Configuration Supervisor Level).
; CHECK:   ssccptr                          - 'Ssccptr' (Main memory supports page table reads).
; CHECK:   sscofpmf                         - 'Sscofpmf' (Count Overflow and Mode-Based Filtering).
; CHECK:   sscounterenw                     - 'Sscounterenw' (Support writeable scounteren enable bit for any hpmcounter that is not read-only zero).
; CHECK:   sscsrind                         - 'Sscsrind' (Indirect CSR Access Supervisor Level).
; CHECK:   ssdbltrp                         - 'Ssdbltrp' (Double Trap Supervisor Level).
; CHECK:   ssnpm                            - 'Ssnpm' (Supervisor-level Pointer Masking for next lower privilege mode).
; CHECK:   sspm                             - 'Sspm' (Indicates Supervisor-mode Pointer Masking).
; CHECK:   ssqosid                          - 'Ssqosid' (Quality-of-Service (QoS) Identifiers).
; CHECK:   ssstateen                        - 'Ssstateen' (Supervisor-mode view of the state-enable extension).
; CHECK:   ssstrict                         - 'Ssstrict' (No non-conforming extensions are present).
; CHECK:   sstc                             - 'Sstc' (Supervisor-mode timer interrupts).
; CHECK:   sstvala                          - 'Sstvala' (stval provides all needed values).
; CHECK:   sstvecd                          - 'Sstvecd' (stvec supports Direct mode).
; CHECK:   ssu64xl                          - 'Ssu64xl' (UXLEN=64 supported).
; CHECK:   supm                             - 'Supm' (Indicates User-mode Pointer Masking).
; CHECK:   svade                            - 'Svade' (Raise exceptions on improper A/D bits).
; CHECK:   svadu                            - 'Svadu' (Hardware A/D updates).
; CHECK:   svbare                           - 'Svbare' (satp mode Bare supported).
; CHECK:   svinval                          - 'Svinval' (Fine-Grained Address-Translation Cache Invalidation).
; CHECK:   svnapot                          - 'Svnapot' (NAPOT Translation Contiguity).
; CHECK:   svpbmt                           - 'Svpbmt' (Page-Based Memory Types).
; CHECK:   svvptc                           - 'Svvptc' (Obviating Memory-Management Instructions after Marking PTEs Valid).
; CHECK:   tagged-globals                   - Use an instruction sequence for taking the address of a global that allows a memory tag in the upper address bits.
; CHECK:   unaligned-scalar-mem             - Has reasonably performant unaligned scalar loads and stores.
; CHECK:   unaligned-vector-mem             - Has reasonably performant unaligned vector loads and stores.
; CHECK:   use-postra-scheduler             - Schedule again after register allocation.
; CHECK:   v                                - 'V' (Vector Extension for Application Processors).
; CHECK:   ventana-veyron                   - Ventana Veyron-Series processors.
; CHECK:   vxrm-pipeline-flush              - VXRM writes causes pipeline flush.
; CHECK:   xcvalu                           - 'XCValu' (CORE-V ALU Operations).
; CHECK:   xcvbi                            - 'XCVbi' (CORE-V Immediate Branching).
; CHECK:   xcvbitmanip                      - 'XCVbitmanip' (CORE-V Bit Manipulation).
; CHECK:   xcvelw                           - 'XCVelw' (CORE-V Event Load Word).
; CHECK:   xcvmac                           - 'XCVmac' (CORE-V Multiply-Accumulate).
; CHECK:   xcvmem                           - 'XCVmem' (CORE-V Post-incrementing Load & Store).
; CHECK:   xcvsimd                          - 'XCVsimd' (CORE-V SIMD ALU).
; CHECK:   xsfcease                         - 'XSfcease' (SiFive sf.cease Instruction).
; CHECK:   xsfvcp                           - 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions).
; CHECK:   xsfvfnrclipxfqf                  - 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions).
; CHECK:   xsfvfwmaccqqq                    - 'XSfvfwmaccqqq' (SiFive Matrix Multiply Accumulate Instruction and 4-by-4)).
; CHECK:   xsfvqmaccdod                     - 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2)).
; CHECK:   xsfvqmaccqoq                     - 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4)).
; CHECK:   xsifivecdiscarddlone             - 'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction).
; CHECK:   xsifivecflushdlone               - 'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction).
; CHECK:   xtheadba                         - 'XTHeadBa' (T-Head address calculation instructions).
; CHECK:   xtheadbb                         - 'XTHeadBb' (T-Head basic bit-manipulation instructions).
; CHECK:   xtheadbs                         - 'XTHeadBs' (T-Head single-bit instructions).
; CHECK:   xtheadcmo                        - 'XTHeadCmo' (T-Head cache management instructions).
; CHECK:   xtheadcondmov                    - 'XTHeadCondMov' (T-Head conditional move instructions).
; CHECK:   xtheadfmemidx                    - 'XTHeadFMemIdx' (T-Head FP Indexed Memory Operations).
; CHECK:   xtheadmac                        - 'XTHeadMac' (T-Head Multiply-Accumulate Instructions).
; CHECK:   xtheadmemidx                     - 'XTHeadMemIdx' (T-Head Indexed Memory Operations).
; CHECK:   xtheadmempair                    - 'XTHeadMemPair' (T-Head two-GPR Memory Operations).
; CHECK:   xtheadsync                       - 'XTHeadSync' (T-Head multicore synchronization instructions).
; CHECK:   xtheadvdot                       - 'XTHeadVdot' (T-Head Vector Extensions for Dot).
; CHECK:   xventanacondops                  - 'XVentanaCondOps' (Ventana Conditional Ops).
; CHECK:   xwchc                            - 'Xwchc' (WCH/QingKe additional compressed opcodes).
; CHECK:   za128rs                          - 'Za128rs' (Reservation Set Size of at Most 128 Bytes).
; CHECK:   za64rs                           - 'Za64rs' (Reservation Set Size of at Most 64 Bytes).
; CHECK:   zaamo                            - 'Zaamo' (Atomic Memory Operations).
; CHECK:   zabha                            - 'Zabha' (Byte and Halfword Atomic Memory Operations).
; CHECK:   zacas                            - 'Zacas' (Atomic Compare-And-Swap Instructions).
; CHECK:   zalrsc                           - 'Zalrsc' (Load-Reserved/Store-Conditional).
; CHECK:   zama16b                          - 'Zama16b' (Atomic 16-byte misaligned loads, stores and AMOs).
; CHECK:   zawrs                            - 'Zawrs' (Wait on Reservation Set).
; CHECK:   zba                              - 'Zba' (Address Generation Instructions).
; CHECK:   zbb                              - 'Zbb' (Basic Bit-Manipulation).
; CHECK:   zbc                              - 'Zbc' (Carry-Less Multiplication).
; CHECK:   zbkb                             - 'Zbkb' (Bitmanip instructions for Cryptography).
; CHECK:   zbkc                             - 'Zbkc' (Carry-less multiply instructions for Cryptography).
; CHECK:   zbkx                             - 'Zbkx' (Crossbar permutation instructions).
; CHECK:   zbs                              - 'Zbs' (Single-Bit Instructions).
; CHECK:   zca                              - 'Zca' (part of the C extension, excluding compressed floating point loads/stores).
; CHECK:   zcb                              - 'Zcb' (Compressed basic bit manipulation instructions).
; CHECK:   zcd                              - 'Zcd' (Compressed Double-Precision Floating-Point Instructions).
; CHECK:   zce                              - 'Zce' (Compressed extensions for microcontrollers).
; CHECK:   zcf                              - 'Zcf' (Compressed Single-Precision Floating-Point Instructions).
; CHECK:   zcmop                            - 'Zcmop' (Compressed May-Be-Operations).
; CHECK:   zcmp                             - 'Zcmp' (sequenced instructions for code-size reduction).
; CHECK:   zcmt                             - 'Zcmt' (table jump instructions for code-size reduction).
; CHECK:   zdinx                            - 'Zdinx' (Double in Integer).
; CHECK:   zexth-fusion                     - Enable SLLI+SRLI to be fused to zero extension of halfword.
; CHECK:   zextw-fusion                     - Enable SLLI+SRLI to be fused to zero extension of word.
; CHECK:   zfa                              - 'Zfa' (Additional Floating-Point).
; CHECK:   zfbfmin                          - 'Zfbfmin' (Scalar BF16 Converts).
; CHECK:   zfh                              - 'Zfh' (Half-Precision Floating-Point).
; CHECK:   zfhmin                           - 'Zfhmin' (Half-Precision Floating-Point Minimal).
; CHECK:   zfinx                            - 'Zfinx' (Float in Integer).
; CHECK:   zhinx                            - 'Zhinx' (Half Float in Integer).
; CHECK:   zhinxmin                         - 'Zhinxmin' (Half Float in Integer Minimal).
; CHECK:   zic64b                           - 'Zic64b' (Cache Block Size Is 64 Bytes).
; CHECK:   zicbom                           - 'Zicbom' (Cache-Block Management Instructions).
; CHECK:   zicbop                           - 'Zicbop' (Cache-Block Prefetch Instructions).
; CHECK:   zicboz                           - 'Zicboz' (Cache-Block Zero Instructions).
; CHECK:   ziccamoa                         - 'Ziccamoa' (Main Memory Supports All Atomics in A).
; CHECK:   ziccif                           - 'Ziccif' (Main Memory Supports Instruction Fetch with Atomicity Requirement).
; CHECK:   zicclsm                          - 'Zicclsm' (Main Memory Supports Misaligned Loads/Stores).
; CHECK:   ziccrse                          - 'Ziccrse' (Main Memory Supports Forward Progress on LR/SC Sequences).
; CHECK:   zicntr                           - 'Zicntr' (Base Counters and Timers).
; CHECK:   zicond                           - 'Zicond' (Integer Conditional Operations).
; CHECK:   zicsr                            - 'Zicsr' (CSRs).
; CHECK:   zifencei                         - 'Zifencei' (fence.i).
; CHECK:   zihintntl                        - 'Zihintntl' (Non-Temporal Locality Hints).
; CHECK:   zihintpause                      - 'Zihintpause' (Pause Hint).
; CHECK:   zihpm                            - 'Zihpm' (Hardware Performance Counters).
; CHECK:   zimop                            - 'Zimop' (May-Be-Operations).
; CHECK:   zk                               - 'Zk' (Standard scalar cryptography extension).
; CHECK:   zkn                              - 'Zkn' (NIST Algorithm Suite).
; CHECK:   zknd                             - 'Zknd' (NIST Suite: AES Decryption).
; CHECK:   zkne                             - 'Zkne' (NIST Suite: AES Encryption).
; CHECK:   zknh                             - 'Zknh' (NIST Suite: Hash Function Instructions).
; CHECK:   zkr                              - 'Zkr' (Entropy Source Extension).
; CHECK:   zks                              - 'Zks' (ShangMi Algorithm Suite).
; CHECK:   zksed                            - 'Zksed' (ShangMi Suite: SM4 Block Cipher Instructions).
; CHECK:   zksh                             - 'Zksh' (ShangMi Suite: SM3 Hash Function Instructions).
; CHECK:   zkt                              - 'Zkt' (Data Independent Execution Latency).
; CHECK:   zmmul                            - 'Zmmul' (Integer Multiplication).
; CHECK:   ztso                             - 'Ztso' (Memory Model - Total Store Order).
; CHECK:   zvbb                             - 'Zvbb' (Vector basic bit-manipulation instructions).
; CHECK:   zvbc                             - 'Zvbc' (Vector Carryless Multiplication).
; CHECK:   zve32f                           - 'Zve32f' (Vector Extensions for Embedded Processors with maximal 32 EEW and F extension).
; CHECK:   zve32x                           - 'Zve32x' (Vector Extensions for Embedded Processors with maximal 32 EEW).
; CHECK:   zve64d                           - 'Zve64d' (Vector Extensions for Embedded Processors with maximal 64 EEW, F and D extension).
; CHECK:   zve64f                           - 'Zve64f' (Vector Extensions for Embedded Processors with maximal 64 EEW and F extension).
; CHECK:   zve64x                           - 'Zve64x' (Vector Extensions for Embedded Processors with maximal 64 EEW).
; CHECK:   zvfbfmin                         - 'Zvfbfmin' (Vector BF16 Converts).
; CHECK:   zvfbfwma                         - 'Zvfbfwma' (Vector BF16 widening mul-add).
; CHECK:   zvfh                             - 'Zvfh' (Vector Half-Precision Floating-Point).
; CHECK:   zvfhmin                          - 'Zvfhmin' (Vector Half-Precision Floating-Point Minimal).
; CHECK:   zvkb                             - 'Zvkb' (Vector Bit-manipulation used in Cryptography).
; CHECK:   zvkg                             - 'Zvkg' (Vector GCM instructions for Cryptography).
; CHECK:   zvkn                             - 'Zvkn' (shorthand for 'Zvkned', 'Zvknhb', 'Zvkb', and 'Zvkt').
; CHECK:   zvknc                            - 'Zvknc' (shorthand for 'Zvknc' and 'Zvbc').
; CHECK:   zvkned                           - 'Zvkned' (Vector AES Encryption & Decryption (Single Round)).
; CHECK:   zvkng                            - 'Zvkng' (shorthand for 'Zvkn' and 'Zvkg').
; CHECK:   zvknha                           - 'Zvknha' (Vector SHA-2 (SHA-256 only)).
; CHECK:   zvknhb                           - 'Zvknhb' (Vector SHA-2 (SHA-256 and SHA-512)).
; CHECK:   zvks                             - 'Zvks' (shorthand for 'Zvksed', 'Zvksh', 'Zvkb', and 'Zvkt').
; CHECK:   zvksc                            - 'Zvksc' (shorthand for 'Zvks' and 'Zvbc').
; CHECK:   zvksed                           - 'Zvksed' (SM4 Block Cipher Instructions).
; CHECK:   zvksg                            - 'Zvksg' (shorthand for 'Zvks' and 'Zvkg').
; CHECK:   zvksh                            - 'Zvksh' (SM3 Hash Function Instructions).
; CHECK:   zvkt                             - 'Zvkt' (Vector Data-Independent Execution Latency).
; CHECK:   zvl1024b                         - 'Zvl1024b' (Minimum Vector Length 1024).
; CHECK:   zvl128b                          - 'Zvl128b' (Minimum Vector Length 128).
; CHECK:   zvl16384b                        - 'Zvl16384b' (Minimum Vector Length 16384).
; CHECK:   zvl2048b                         - 'Zvl2048b' (Minimum Vector Length 2048).
; CHECK:   zvl256b                          - 'Zvl256b' (Minimum Vector Length 256).
; CHECK:   zvl32768b                        - 'Zvl32768b' (Minimum Vector Length 32768).
; CHECK:   zvl32b                           - 'Zvl32b' (Minimum Vector Length 32).
; CHECK:   zvl4096b                         - 'Zvl4096b' (Minimum Vector Length 4096).
; CHECK:   zvl512b                          - 'Zvl512b' (Minimum Vector Length 512).
; CHECK:   zvl64b                           - 'Zvl64b' (Minimum Vector Length 64).
; CHECK:   zvl65536b                        - 'Zvl65536b' (Minimum Vector Length 65536).
; CHECK:   zvl8192b                         - 'Zvl8192b' (Minimum Vector Length 8192).
=======
; CHECK-EMPTY:
; CHECK-NEXT:   32bit                            - Implements RV32.
; CHECK-NEXT:   64bit                            - Implements RV64.
; CHECK-NEXT:   a                                - 'A' (Atomic Instructions).
; CHECK-NEXT:   auipc-addi-fusion                - Enable AUIPC+ADDI macrofusion.
; CHECK-NEXT:   b                                - 'B' (the collection of the Zba, Zbb, Zbs extensions).
; CHECK-NEXT:   c                                - 'C' (Compressed Instructions).
; CHECK-NEXT:   conditional-cmv-fusion           - Enable branch+c.mv fusion.
; CHECK-NEXT:   d                                - 'D' (Double-Precision Floating-Point).
; CHECK-NEXT:   disable-latency-sched-heuristic  - Disable latency scheduling heuristic.
; CHECK-NEXT:   dlen-factor-2                    - Vector unit DLEN(data path width) is half of VLEN.
; CHECK-NEXT:   e                                - 'E' (Embedded Instruction Set with 16 GPRs).
; CHECK-NEXT:   experimental                     - Experimental intrinsics.
; CHECK-NEXT:   experimental-p                   - 'P' ('Base P' (Packed SIMD)).
; CHECK-NEXT:   experimental-rvm23u32            - RISC-V experimental-rvm23u32 profile.
; CHECK-NEXT:   experimental-sdext               - 'Sdext' (External debugger).
; CHECK-NEXT:   experimental-sdtrig              - 'Sdtrig' (Debugger triggers).
; CHECK-NEXT:   experimental-smctr               - 'Smctr' (Control Transfer Records Machine Level).
; CHECK-NEXT:   experimental-ssctr               - 'Ssctr' (Control Transfer Records Supervisor Level).
; CHECK-NEXT:   experimental-svukte              - 'Svukte' (Address-Independent Latency of User-Mode Faults to Supervisor Addresses).
; CHECK-NEXT:   experimental-xqccmp              - 'Xqccmp' (Qualcomm 16-bit Push/Pop and Double Moves).
; CHECK-NEXT:   experimental-xqcia               - 'Xqcia' (Qualcomm uC Arithmetic Extension).
; CHECK-NEXT:   experimental-xqciac              - 'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension).
; CHECK-NEXT:   experimental-xqcibi              - 'Xqcibi' (Qualcomm uC Branch Immediate Extension).
; CHECK-NEXT:   experimental-xqcibm              - 'Xqcibm' (Qualcomm uC Bit Manipulation Extension).
; CHECK-NEXT:   experimental-xqcicli             - 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension).
; CHECK-NEXT:   experimental-xqcicm              - 'Xqcicm' (Qualcomm uC Conditional Move Extension).
; CHECK-NEXT:   experimental-xqcics              - 'Xqcics' (Qualcomm uC Conditional Select Extension).
; CHECK-NEXT:   experimental-xqcicsr             - 'Xqcicsr' (Qualcomm uC CSR Extension).
; CHECK-NEXT:   experimental-xqciint             - 'Xqciint' (Qualcomm uC Interrupts Extension).
; CHECK-NEXT:   experimental-xqcilb              - 'Xqcilb' (Qualcomm uC Long Branch Extension).
; CHECK-NEXT:   experimental-xqcili              - 'Xqcili' (Qualcomm uC Load Large Immediate Extension).
; CHECK-NEXT:   experimental-xqcilia             - 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension).
; CHECK-NEXT:   experimental-xqcilo              - 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension).
; CHECK-NEXT:   experimental-xqcilsm             - 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension).
; CHECK-NEXT:   experimental-xqcisim             - 'Xqcisim' (Qualcomm uC Simulation Hint Extension).
; CHECK-NEXT:   experimental-xqcisls             - 'Xqcisls' (Qualcomm uC Scaled Load Store Extension).
; CHECK-NEXT:   experimental-xqcisync            - 'Xqcisync' (Qualcomm uC Sync Delay Extension).
; CHECK-NEXT:   experimental-xrivosvisni         - 'XRivosVisni' (Rivos Vector Integer Small New).
; CHECK-NEXT:   experimental-xrivosvizip         - 'XRivosVizip' (Rivos Vector Register Zips).
; CHECK-NEXT:   experimental-zalasr              - 'Zalasr' (Load-Acquire and Store-Release Instructions).
; CHECK-NEXT:   experimental-zicfilp             - 'Zicfilp' (Landing pad).
; CHECK-NEXT:   experimental-zicfiss             - 'Zicfiss' (Shadow stack).
; CHECK-NEXT:   experimental-zvbc32e             - 'Zvbc32e' (Vector Carryless Multiplication with 32-bits elements).
; CHECK-NEXT:   experimental-zvkgs               - 'Zvkgs' (Vector-Scalar GCM instructions for Cryptography).
; CHECK-NEXT:   experimental-zvqdotq             - 'Zvqdotq' (Vector quad widening 4D Dot Product).
; CHECK-NEXT:   f                                - 'F' (Single-Precision Floating-Point).
; CHECK-NEXT:   forced-atomics                   - Assume that lock-free native-width atomics are available.
; CHECK-NEXT:   h                                - 'H' (Hypervisor).
; CHECK-NEXT:   i                                - 'I' (Base Integer Instruction Set).
; CHECK-NEXT:   ld-add-fusion                    - Enable LD+ADD macrofusion.
; CHECK-NEXT:   log-vrgather                     - Has vrgather.vv with LMUL*log2(LMUL) latency
; CHECK-NEXT:   lui-addi-fusion                  - Enable LUI+ADDI macro fusion.
; CHECK-NEXT:   m                                - 'M' (Integer Multiplication and Division).
; CHECK-NEXT:   mips-p8700                       - MIPS p8700 processor.
; CHECK-NEXT:   no-default-unroll                - Disable default unroll preference..
; CHECK-NEXT:   no-rvc-hints                     - Disable RVC Hint Instructions..
; CHECK-NEXT:   no-sink-splat-operands           - Disable sink splat operands to enable .vx, .vf,.wx, and .wf instructions.
; CHECK-NEXT:   no-trailing-seq-cst-fence        - Disable trailing fence for seq-cst store..
; CHECK-NEXT:   optimized-nf2-segment-load-store - vlseg2eN.v and vsseg2eN.v are implemented as a wide memory op and shuffle.
; CHECK-NEXT:   optimized-nf3-segment-load-store - vlseg3eN.v and vsseg3eN.v are implemented as a wide memory op and shuffle.
; CHECK-NEXT:   optimized-nf4-segment-load-store - vlseg4eN.v and vsseg4eN.v are implemented as a wide memory op and shuffle.
; CHECK-NEXT:   optimized-nf5-segment-load-store - vlseg5eN.v and vsseg5eN.v are implemented as a wide memory op and shuffle.
; CHECK-NEXT:   optimized-nf6-segment-load-store - vlseg6eN.v and vsseg6eN.v are implemented as a wide memory op and shuffle.
; CHECK-NEXT:   optimized-nf7-segment-load-store - vlseg7eN.v and vsseg7eN.v are implemented as a wide memory op and shuffle.
; CHECK-NEXT:   optimized-nf8-segment-load-store - vlseg8eN.v and vsseg8eN.v are implemented as a wide memory op and shuffle.
; CHECK-NEXT:   optimized-zero-stride-load       - Optimized (perform fewer memory operations)zero-stride vector load.
; CHECK-NEXT:   predictable-select-expensive     - Prefer likely predicted branches over selects.
; CHECK-NEXT:   prefer-vsetvli-over-read-vlenb   - Prefer vsetvli over read vlenb CSR to calculate VLEN.
; CHECK-NEXT:   prefer-w-inst                    - Prefer instructions with W suffix.
; CHECK-NEXT:   relax                            - Enable Linker relaxation..
; CHECK-NEXT:   reserve-x1                       - Reserve X1.
; CHECK-NEXT:   reserve-x10                      - Reserve X10.
; CHECK-NEXT:   reserve-x11                      - Reserve X11.
; CHECK-NEXT:   reserve-x12                      - Reserve X12.
; CHECK-NEXT:   reserve-x13                      - Reserve X13.
; CHECK-NEXT:   reserve-x14                      - Reserve X14.
; CHECK-NEXT:   reserve-x15                      - Reserve X15.
; CHECK-NEXT:   reserve-x16                      - Reserve X16.
; CHECK-NEXT:   reserve-x17                      - Reserve X17.
; CHECK-NEXT:   reserve-x18                      - Reserve X18.
; CHECK-NEXT:   reserve-x19                      - Reserve X19.
; CHECK-NEXT:   reserve-x2                       - Reserve X2.
; CHECK-NEXT:   reserve-x20                      - Reserve X20.
; CHECK-NEXT:   reserve-x21                      - Reserve X21.
; CHECK-NEXT:   reserve-x22                      - Reserve X22.
; CHECK-NEXT:   reserve-x23                      - Reserve X23.
; CHECK-NEXT:   reserve-x24                      - Reserve X24.
; CHECK-NEXT:   reserve-x25                      - Reserve X25.
; CHECK-NEXT:   reserve-x26                      - Reserve X26.
; CHECK-NEXT:   reserve-x27                      - Reserve X27.
; CHECK-NEXT:   reserve-x28                      - Reserve X28.
; CHECK-NEXT:   reserve-x29                      - Reserve X29.
; CHECK-NEXT:   reserve-x3                       - Reserve X3.
; CHECK-NEXT:   reserve-x30                      - Reserve X30.
; CHECK-NEXT:   reserve-x31                      - Reserve X31.
; CHECK-NEXT:   reserve-x4                       - Reserve X4.
; CHECK-NEXT:   reserve-x5                       - Reserve X5.
; CHECK-NEXT:   reserve-x6                       - Reserve X6.
; CHECK-NEXT:   reserve-x7                       - Reserve X7.
; CHECK-NEXT:   reserve-x8                       - Reserve X8.
; CHECK-NEXT:   reserve-x9                       - Reserve X9.
; CHECK-NEXT:   rva20s64                         - RISC-V rva20s64 profile.
; CHECK-NEXT:   rva20u64                         - RISC-V rva20u64 profile.
; CHECK-NEXT:   rva22s64                         - RISC-V rva22s64 profile.
; CHECK-NEXT:   rva22u64                         - RISC-V rva22u64 profile.
; CHECK-NEXT:   rva23s64                         - RISC-V rva23s64 profile.
; CHECK-NEXT:   rva23u64                         - RISC-V rva23u64 profile.
; CHECK-NEXT:   rvb23s64                         - RISC-V rvb23s64 profile.
; CHECK-NEXT:   rvb23u64                         - RISC-V rvb23u64 profile.
; CHECK-NEXT:   rvi20u32                         - RISC-V rvi20u32 profile.
; CHECK-NEXT:   rvi20u64                         - RISC-V rvi20u64 profile.
; CHECK-NEXT:   save-restore                     - Enable save/restore..
; CHECK-NEXT:   sha                              - 'Sha' (Augmented Hypervisor).
; CHECK-NEXT:   shcounterenw                     - 'Shcounterenw' (Support writeable hcounteren enable bit for any hpmcounter that is not read-only zero).
; CHECK-NEXT:   shgatpa                          - 'Shgatpa' (SvNNx4 mode supported for all modes supported by satp, as well as Bare).
; CHECK-NEXT:   shifted-zextw-fusion             - Enable SLLI+SRLI to be fused when computing (shifted) word zero extension.
; CHECK-NEXT:   short-forward-branch-opt         - Enable short forward branch optimization.
; CHECK-NEXT:   shtvala                          - 'Shtvala' (htval provides all needed values).
; CHECK-NEXT:   shvsatpa                         - 'Shvsatpa' (vsatp supports all modes supported by satp).
; CHECK-NEXT:   shvstvala                        - 'Shvstvala' (vstval provides all needed values).
; CHECK-NEXT:   shvstvecd                        - 'Shvstvecd' (vstvec supports Direct mode).
; CHECK-NEXT:   sifive7                          - SiFive 7-Series processors.
; CHECK-NEXT:   smaia                            - 'Smaia' (Advanced Interrupt Architecture Machine Level).
; CHECK-NEXT:   smcdeleg                         - 'Smcdeleg' (Counter Delegation Machine Level).
; CHECK-NEXT:   smcsrind                         - 'Smcsrind' (Indirect CSR Access Machine Level).
; CHECK-NEXT:   smdbltrp                         - 'Smdbltrp' (Double Trap Machine Level).
; CHECK-NEXT:   smepmp                           - 'Smepmp' (Enhanced Physical Memory Protection).
; CHECK-NEXT:   smmpm                            - 'Smmpm' (Machine-level Pointer Masking for M-mode).
; CHECK-NEXT:   smnpm                            - 'Smnpm' (Machine-level Pointer Masking for next lower privilege mode).
; CHECK-NEXT:   smrnmi                           - 'Smrnmi' (Resumable Non-Maskable Interrupts).
; CHECK-NEXT:   smstateen                        - 'Smstateen' (Machine-mode view of the state-enable extension).
; CHECK-NEXT:   ssaia                            - 'Ssaia' (Advanced Interrupt Architecture Supervisor Level).
; CHECK-NEXT:   ssccfg                           - 'Ssccfg' (Counter Configuration Supervisor Level).
; CHECK-NEXT:   ssccptr                          - 'Ssccptr' (Main memory supports page table reads).
; CHECK-NEXT:   sscofpmf                         - 'Sscofpmf' (Count Overflow and Mode-Based Filtering).
; CHECK-NEXT:   sscounterenw                     - 'Sscounterenw' (Support writeable scounteren enable bit for any hpmcounter that is not read-only zero).
; CHECK-NEXT:   sscsrind                         - 'Sscsrind' (Indirect CSR Access Supervisor Level).
; CHECK-NEXT:   ssdbltrp                         - 'Ssdbltrp' (Double Trap Supervisor Level).
; CHECK-NEXT:   ssnpm                            - 'Ssnpm' (Supervisor-level Pointer Masking for next lower privilege mode).
; CHECK-NEXT:   sspm                             - 'Sspm' (Indicates Supervisor-mode Pointer Masking).
; CHECK-NEXT:   ssqosid                          - 'Ssqosid' (Quality-of-Service (QoS) Identifiers).
; CHECK-NEXT:   ssstateen                        - 'Ssstateen' (Supervisor-mode view of the state-enable extension).
; CHECK-NEXT:   ssstrict                         - 'Ssstrict' (No non-conforming extensions are present).
; CHECK-NEXT:   sstc                             - 'Sstc' (Supervisor-mode timer interrupts).
; CHECK-NEXT:   sstvala                          - 'Sstvala' (stval provides all needed values).
; CHECK-NEXT:   sstvecd                          - 'Sstvecd' (stvec supports Direct mode).
; CHECK-NEXT:   ssu64xl                          - 'Ssu64xl' (UXLEN=64 supported).
; CHECK-NEXT:   supm                             - 'Supm' (Indicates User-mode Pointer Masking).
; CHECK-NEXT:   svade                            - 'Svade' (Raise exceptions on improper A/D bits).
; CHECK-NEXT:   svadu                            - 'Svadu' (Hardware A/D updates).
; CHECK-NEXT:   svbare                           - 'Svbare' (satp mode Bare supported).
; CHECK-NEXT:   svinval                          - 'Svinval' (Fine-Grained Address-Translation Cache Invalidation).
; CHECK-NEXT:   svnapot                          - 'Svnapot' (NAPOT Translation Contiguity).
; CHECK-NEXT:   svpbmt                           - 'Svpbmt' (Page-Based Memory Types).
; CHECK-NEXT:   svvptc                           - 'Svvptc' (Obviating Memory-Management Instructions after Marking PTEs Valid).
; CHECK-NEXT:   tagged-globals                   - Use an instruction sequence for taking the address of a global that allows a memory tag in the upper address bits.
; CHECK-NEXT:   unaligned-scalar-mem             - Has reasonably performant unaligned scalar loads and stores.
; CHECK-NEXT:   unaligned-vector-mem             - Has reasonably performant unaligned vector loads and stores.
; CHECK-NEXT:   use-postra-scheduler             - Schedule again after register allocation.
; CHECK-NEXT:   v                                - 'V' (Vector Extension for Application Processors).
; CHECK-NEXT:   ventana-veyron                   - Ventana Veyron-Series processors.
; CHECK-NEXT:   vxrm-pipeline-flush              - VXRM writes causes pipeline flush.
; CHECK-NEXT:   xcvalu                           - 'XCValu' (CORE-V ALU Operations).
; CHECK-NEXT:   xcvbi                            - 'XCVbi' (CORE-V Immediate Branching).
; CHECK-NEXT:   xcvbitmanip                      - 'XCVbitmanip' (CORE-V Bit Manipulation).
; CHECK-NEXT:   xcvelw                           - 'XCVelw' (CORE-V Event Load Word).
; CHECK-NEXT:   xcvmac                           - 'XCVmac' (CORE-V Multiply-Accumulate).
; CHECK-NEXT:   xcvmem                           - 'XCVmem' (CORE-V Post-incrementing Load & Store).
; CHECK-NEXT:   xcvsimd                          - 'XCVsimd' (CORE-V SIMD ALU).
; CHECK-NEXT:   xmipscmove                       - 'XMIPSCMove' (MIPS conditional move instruction(s) (ccmov)).
; CHECK-NEXT:   xmipslsp                         - 'XMIPSLSP' (MIPS optimization for hardware load-store bonding).
; CHECK-NEXT:   xsfcease                         - 'XSfcease' (SiFive sf.cease Instruction).
; CHECK-NEXT:   xsfvcp                           - 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions).
; CHECK-NEXT:   xsfvfnrclipxfqf                  - 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions).
; CHECK-NEXT:   xsfvfwmaccqqq                    - 'XSfvfwmaccqqq' (SiFive Matrix Multiply Accumulate Instruction and 4-by-4)).
; CHECK-NEXT:   xsfvqmaccdod                     - 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2)).
; CHECK-NEXT:   xsfvqmaccqoq                     - 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4)).
; CHECK-NEXT:   xsifivecdiscarddlone             - 'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction).
; CHECK-NEXT:   xsifivecflushdlone               - 'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction).
; CHECK-NEXT:   xtheadba                         - 'XTHeadBa' (T-Head address calculation instructions).
; CHECK-NEXT:   xtheadbb                         - 'XTHeadBb' (T-Head basic bit-manipulation instructions).
; CHECK-NEXT:   xtheadbs                         - 'XTHeadBs' (T-Head single-bit instructions).
; CHECK-NEXT:   xtheadcmo                        - 'XTHeadCmo' (T-Head cache management instructions).
; CHECK-NEXT:   xtheadcondmov                    - 'XTHeadCondMov' (T-Head conditional move instructions).
; CHECK-NEXT:   xtheadfmemidx                    - 'XTHeadFMemIdx' (T-Head FP Indexed Memory Operations).
; CHECK-NEXT:   xtheadmac                        - 'XTHeadMac' (T-Head Multiply-Accumulate Instructions).
; CHECK-NEXT:   xtheadmemidx                     - 'XTHeadMemIdx' (T-Head Indexed Memory Operations).
; CHECK-NEXT:   xtheadmempair                    - 'XTHeadMemPair' (T-Head two-GPR Memory Operations).
; CHECK-NEXT:   xtheadsync                       - 'XTHeadSync' (T-Head multicore synchronization instructions).
; CHECK-NEXT:   xtheadvdot                       - 'XTHeadVdot' (T-Head Vector Extensions for Dot).
; CHECK-NEXT:   xventanacondops                  - 'XVentanaCondOps' (Ventana Conditional Ops).
; CHECK-NEXT:   xwchc                            - 'Xwchc' (WCH/QingKe additional compressed opcodes).
; CHECK-NEXT:   za128rs                          - 'Za128rs' (Reservation Set Size of at Most 128 Bytes).
; CHECK-NEXT:   za64rs                           - 'Za64rs' (Reservation Set Size of at Most 64 Bytes).
; CHECK-NEXT:   zaamo                            - 'Zaamo' (Atomic Memory Operations).
; CHECK-NEXT:   zabha                            - 'Zabha' (Byte and Halfword Atomic Memory Operations).
; CHECK-NEXT:   zacas                            - 'Zacas' (Atomic Compare-And-Swap Instructions).
; CHECK-NEXT:   zalrsc                           - 'Zalrsc' (Load-Reserved/Store-Conditional).
; CHECK-NEXT:   zama16b                          - 'Zama16b' (Atomic 16-byte misaligned loads, stores and AMOs).
; CHECK-NEXT:   zawrs                            - 'Zawrs' (Wait on Reservation Set).
; CHECK-NEXT:   zba                              - 'Zba' (Address Generation Instructions).
; CHECK-NEXT:   zbb                              - 'Zbb' (Basic Bit-Manipulation).
; CHECK-NEXT:   zbc                              - 'Zbc' (Carry-Less Multiplication).
; CHECK-NEXT:   zbkb                             - 'Zbkb' (Bitmanip instructions for Cryptography).
; CHECK-NEXT:   zbkc                             - 'Zbkc' (Carry-less multiply instructions for Cryptography).
; CHECK-NEXT:   zbkx                             - 'Zbkx' (Crossbar permutation instructions).
; CHECK-NEXT:   zbs                              - 'Zbs' (Single-Bit Instructions).
; CHECK-NEXT:   zca                              - 'Zca' (part of the C extension, excluding compressed floating point loads/stores).
; CHECK-NEXT:   zcb                              - 'Zcb' (Compressed basic bit manipulation instructions).
; CHECK-NEXT:   zcd                              - 'Zcd' (Compressed Double-Precision Floating-Point Instructions).
; CHECK-NEXT:   zce                              - 'Zce' (Compressed extensions for microcontrollers).
; CHECK-NEXT:   zcf                              - 'Zcf' (Compressed Single-Precision Floating-Point Instructions).
; CHECK-NEXT:   zclsd                            - 'Zclsd' (Compressed Load/Store Pair Instructions).
; CHECK-NEXT:   zcmop                            - 'Zcmop' (Compressed May-Be-Operations).
; CHECK-NEXT:   zcmp                             - 'Zcmp' (sequenced instructions for code-size reduction).
; CHECK-NEXT:   zcmt                             - 'Zcmt' (table jump instructions for code-size reduction).
; CHECK-NEXT:   zdinx                            - 'Zdinx' (Double in Integer).
; CHECK-NEXT:   zexth-fusion                     - Enable SLLI+SRLI to be fused to zero extension of halfword.
; CHECK-NEXT:   zextw-fusion                     - Enable SLLI+SRLI to be fused to zero extension of word.
; CHECK-NEXT:   zfa                              - 'Zfa' (Additional Floating-Point).
; CHECK-NEXT:   zfbfmin                          - 'Zfbfmin' (Scalar BF16 Converts).
; CHECK-NEXT:   zfh                              - 'Zfh' (Half-Precision Floating-Point).
; CHECK-NEXT:   zfhmin                           - 'Zfhmin' (Half-Precision Floating-Point Minimal).
; CHECK-NEXT:   zfinx                            - 'Zfinx' (Float in Integer).
; CHECK-NEXT:   zhinx                            - 'Zhinx' (Half Float in Integer).
; CHECK-NEXT:   zhinxmin                         - 'Zhinxmin' (Half Float in Integer Minimal).
; CHECK-NEXT:   zic64b                           - 'Zic64b' (Cache Block Size Is 64 Bytes).
; CHECK-NEXT:   zicbom                           - 'Zicbom' (Cache-Block Management Instructions).
; CHECK-NEXT:   zicbop                           - 'Zicbop' (Cache-Block Prefetch Instructions).
; CHECK-NEXT:   zicboz                           - 'Zicboz' (Cache-Block Zero Instructions).
; CHECK-NEXT:   ziccamoa                         - 'Ziccamoa' (Main Memory Supports All Atomics in A).
; CHECK-NEXT:   ziccif                           - 'Ziccif' (Main Memory Supports Instruction Fetch with Atomicity Requirement).
; CHECK-NEXT:   zicclsm                          - 'Zicclsm' (Main Memory Supports Misaligned Loads/Stores).
; CHECK-NEXT:   ziccrse                          - 'Ziccrse' (Main Memory Supports Forward Progress on LR/SC Sequences).
; CHECK-NEXT:   zicntr                           - 'Zicntr' (Base Counters and Timers).
; CHECK-NEXT:   zicond                           - 'Zicond' (Integer Conditional Operations).
; CHECK-NEXT:   zicsr                            - 'Zicsr' (CSRs).
; CHECK-NEXT:   zifencei                         - 'Zifencei' (fence.i).
; CHECK-NEXT:   zihintntl                        - 'Zihintntl' (Non-Temporal Locality Hints).
; CHECK-NEXT:   zihintpause                      - 'Zihintpause' (Pause Hint).
; CHECK-NEXT:   zihpm                            - 'Zihpm' (Hardware Performance Counters).
; CHECK-NEXT:   zilsd                            - 'Zilsd' (Load/Store Pair Instructions).
; CHECK-NEXT:   zimop                            - 'Zimop' (May-Be-Operations).
; CHECK-NEXT:   zk                               - 'Zk' (Standard scalar cryptography extension).
; CHECK-NEXT:   zkn                              - 'Zkn' (NIST Algorithm Suite).
; CHECK-NEXT:   zknd                             - 'Zknd' (NIST Suite: AES Decryption).
; CHECK-NEXT:   zkne                             - 'Zkne' (NIST Suite: AES Encryption).
; CHECK-NEXT:   zknh                             - 'Zknh' (NIST Suite: Hash Function Instructions).
; CHECK-NEXT:   zkr                              - 'Zkr' (Entropy Source Extension).
; CHECK-NEXT:   zks                              - 'Zks' (ShangMi Algorithm Suite).
; CHECK-NEXT:   zksed                            - 'Zksed' (ShangMi Suite: SM4 Block Cipher Instructions).
; CHECK-NEXT:   zksh                             - 'Zksh' (ShangMi Suite: SM3 Hash Function Instructions).
; CHECK-NEXT:   zkt                              - 'Zkt' (Data Independent Execution Latency).
; CHECK-NEXT:   zmmul                            - 'Zmmul' (Integer Multiplication).
; CHECK-NEXT:   ztso                             - 'Ztso' (Memory Model - Total Store Order).
; CHECK-NEXT:   zvbb                             - 'Zvbb' (Vector basic bit-manipulation instructions).
; CHECK-NEXT:   zvbc                             - 'Zvbc' (Vector Carryless Multiplication).
; CHECK-NEXT:   zve32f                           - 'Zve32f' (Vector Extensions for Embedded Processors with maximal 32 EEW and F extension).
; CHECK-NEXT:   zve32x                           - 'Zve32x' (Vector Extensions for Embedded Processors with maximal 32 EEW).
; CHECK-NEXT:   zve64d                           - 'Zve64d' (Vector Extensions for Embedded Processors with maximal 64 EEW, F and D extension).
; CHECK-NEXT:   zve64f                           - 'Zve64f' (Vector Extensions for Embedded Processors with maximal 64 EEW and F extension).
; CHECK-NEXT:   zve64x                           - 'Zve64x' (Vector Extensions for Embedded Processors with maximal 64 EEW).
; CHECK-NEXT:   zvfbfmin                         - 'Zvfbfmin' (Vector BF16 Converts).
; CHECK-NEXT:   zvfbfwma                         - 'Zvfbfwma' (Vector BF16 widening mul-add).
; CHECK-NEXT:   zvfh                             - 'Zvfh' (Vector Half-Precision Floating-Point).
; CHECK-NEXT:   zvfhmin                          - 'Zvfhmin' (Vector Half-Precision Floating-Point Minimal).
; CHECK-NEXT:   zvkb                             - 'Zvkb' (Vector Bit-manipulation used in Cryptography).
; CHECK-NEXT:   zvkg                             - 'Zvkg' (Vector GCM instructions for Cryptography).
; CHECK-NEXT:   zvkn                             - 'Zvkn' (shorthand for 'Zvkned', 'Zvknhb', 'Zvkb', and 'Zvkt').
; CHECK-NEXT:   zvknc                            - 'Zvknc' (shorthand for 'Zvknc' and 'Zvbc').
; CHECK-NEXT:   zvkned                           - 'Zvkned' (Vector AES Encryption & Decryption (Single Round)).
; CHECK-NEXT:   zvkng                            - 'Zvkng' (shorthand for 'Zvkn' and 'Zvkg').
; CHECK-NEXT:   zvknha                           - 'Zvknha' (Vector SHA-2 (SHA-256 only)).
; CHECK-NEXT:   zvknhb                           - 'Zvknhb' (Vector SHA-2 (SHA-256 and SHA-512)).
; CHECK-NEXT:   zvks                             - 'Zvks' (shorthand for 'Zvksed', 'Zvksh', 'Zvkb', and 'Zvkt').
; CHECK-NEXT:   zvksc                            - 'Zvksc' (shorthand for 'Zvks' and 'Zvbc').
; CHECK-NEXT:   zvksed                           - 'Zvksed' (SM4 Block Cipher Instructions).
; CHECK-NEXT:   zvksg                            - 'Zvksg' (shorthand for 'Zvks' and 'Zvkg').
; CHECK-NEXT:   zvksh                            - 'Zvksh' (SM3 Hash Function Instructions).
; CHECK-NEXT:   zvkt                             - 'Zvkt' (Vector Data-Independent Execution Latency).
; CHECK-NEXT:   zvl1024b                         - 'Zvl1024b' (Minimum Vector Length 1024).
; CHECK-NEXT:   zvl128b                          - 'Zvl128b' (Minimum Vector Length 128).
; CHECK-NEXT:   zvl16384b                        - 'Zvl16384b' (Minimum Vector Length 16384).
; CHECK-NEXT:   zvl2048b                         - 'Zvl2048b' (Minimum Vector Length 2048).
; CHECK-NEXT:   zvl256b                          - 'Zvl256b' (Minimum Vector Length 256).
; CHECK-NEXT:   zvl32768b                        - 'Zvl32768b' (Minimum Vector Length 32768).
; CHECK-NEXT:   zvl32b                           - 'Zvl32b' (Minimum Vector Length 32).
; CHECK-NEXT:   zvl4096b                         - 'Zvl4096b' (Minimum Vector Length 4096).
; CHECK-NEXT:   zvl512b                          - 'Zvl512b' (Minimum Vector Length 512).
; CHECK-NEXT:   zvl64b                           - 'Zvl64b' (Minimum Vector Length 64).
; CHECK-NEXT:   zvl65536b                        - 'Zvl65536b' (Minimum Vector Length 65536).
; CHECK-NEXT:   zvl8192b                         - 'Zvl8192b' (Minimum Vector Length 8192).
; CHECK-EMPTY:
>>>>>>> 94783a8199c5e589d8efd6d4530482d72bf98f4d
