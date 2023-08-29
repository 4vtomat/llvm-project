//===-- RISCVFeatures.td - RISC-V Features and Extensions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

def FeatureStdExtZicsr
    : SubtargetFeature<"zicsr", "HasStdExtZicsr", "true",
                       "'zicsr' (CSRs)">;
def HasStdExtZicsr : Predicate<"Subtarget->hasStdExtZicsr()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicsr),
                                "'Zicsr' (CSRs)">;

#ifdef SIFIVE_CUSTOMIZATION
def FeatureStdExtSs
    : SubtargetFeature<"ss", "HasStdExtSs", "true",
                       "'Ss' (Supervisor Architecture)",
                       []>;

def FeatureStdExtSvbare
    : SubtargetFeature<"svbare", "HasStdExtSvbare", "true",
                       "'Svbare'",
                       []>;

def FeatureStdExtSvade
    : SubtargetFeature<"svade", "HasStdExtSvade", "true",
                       "'Svade' (Page-fault exceptions are raised when a page "
                       "is accessed when A bit is clear, or written when D bit "
                       "is clear)", []>;

def FeatureStdExtSsccptr
    : SubtargetFeature<"ssccptr", "HasStdExtSsccptr", "true",
                       "'Ssccptr' (Main memory regions with both the "
                       "cacheability and coherence PMAs must support hardware "
                       "page-table reads.)", []>;

def FeatureStdExtSstvecd
    : SubtargetFeature<"sstvecd", "HasStdExtSstvecd", "true",
                       "'Sstvecd' (stvec.MODE must be capable of holding the "
                       "value 0 (Direct). When stvec.MODE=Direct, stvec.BASE "
                       "must be capable of holding any valid four-byte-aligned "
                       "address.)", []>;

def FeatureStdExtSstvala
    : SubtargetFeature<"sstvala", "HasStdExtSstvala", "true",
                       "'Sstvala' (stval must be written with the faulting "
                       "virtual address for load, store, and instruction "
                       "page-fault, access-fault, and misaligned exceptions, "
                       "and for breakpoint exceptions other than those caused "
                       "by execution of the EBREAK or C.EBREAK instructions. "
                       "For illegal-instruction exceptions, stval must be "
                       "written with the faulting instruction.)", []>;

def FeatureStdExtSscounterenw
    : SubtargetFeature<"sscounterenw", "HasStdExtSscounterenw", "true",
                       "'Sscounterenw' (For any hpmcounter that is not "
                       "read-only zero, the corresponding bit in scounteren "
                       "must be writable.)", []>;

def FeatureStdExtSsu64xl
    : SubtargetFeature<"ssu64xl", "HasStdExtSsu64xl", "true",
                       "'Ssu64xl' (sstatus.UXL=64)", []>;

def FeatureStdExtSstc
    : SubtargetFeature<"sstc", "HasStdExtSstc", "true",
                       "'Sstc' (supervisor-mode timer interrupts.)", []>;

def FeatureStdExtSsstateen
    : SubtargetFeature<"ssstateen", "HasStdExtSsstateen", "true",
                       "'Ssstateen' (Supervisor-mode view of the state-enable "
                       "extension.)", []>;

def FeatureStdExtSmstateen
    : SubtargetFeature<"smstateen", "HasStdExtSmstateen", "true",
                       "'Smstateen' (Machine-mode view of the state-enable "
                       "extension.)", []>;

def FeatureStdExtShcounterenw
    : SubtargetFeature<"shcounterenw", "HasStdExtShcounterenw", "true",
                       "'Shcounterenw' (For any hpmcounter that is not read-only "
                       "zero, the corresponding bit in hcounteren must be "
                       "writable.)", []>;

def FeatureStdExtShvstvala
    : SubtargetFeature<"shvstvala", "HasStdExtShvstvala", "true",
                       "'Shvstvala' (vstval must be written in all cases "
                       "described above for stval.)", []>;

def FeatureStdExtShtvala
    : SubtargetFeature<"shtvala", "HasStdExtShtvala", "true",
                       "'Shtvala' (htval must be written with the faulting "
                       "guest physical address in all circumstances permitted "
                       "by the ISA.)", []>;

def FeatureStdExtShvstvecd
    : SubtargetFeature<"shvstvecd", "HasStdExtShvstvecd", "true",
                       "'Shvstvecd' (vstvec.MODE must be capable of holding the "
                       "value 0 (Direct). When vstvec.MODE=Direct, vstvec.BASE "
                       "must be capable of holding any valid four-byte-aligned "
                       "address.)", []>;

def FeatureStdExtShvsatpa
    : SubtargetFeature<"shvsatpa", "HasStdExtShvsatpa", "true",
                       "'Shvsatpa' (All translation modes supported in satp "
                       "must be supported in vsatp.)", []>;

def FeatureStdExtShgatpa
    : SubtargetFeature<"shgatpa", "HasStdExtShgatpa", "true",
                       "'Shgatpa' (For each supported virtual memory scheme "
                       "SvNN supported in satp, the corresponding hgatp SvNNx4 "
                       "mode must be supported. The hgatp mode Bare must also "
                       "be supported.)", []>;

def FeatureStdExtSmwg
    : SubtargetFeature<"experimental-smwg", "HasStdExtSmwg", "true",
                       "'Smwg' (The Smwg extension adds the mlwid CSR, which is "
                       "an M-mode read-write CSR, whose least-significant bits "
                       "set the WID to be used by lower-privilege modes.)", []>;

def FeatureStdExtSmwgd
    : SubtargetFeature<"experimental-smwgd", "HasStdExtSmwgd", "true",
                       "'Smwgd' (The Smwgd extension adds the mwiddeleg M-mode "
                       "read-write CSR. The mwiddeleg register represents a set "
                       "of WIDs as a bit vector with WID i represented by bit i "
                       "of the register.)", []>;

def FeatureStdExtSswg
    : SubtargetFeature<"experimental-sswg", "HasStdExtSswg", "true",
                       "'Sswg' (The Sswg extension adds the [H]S-mode read-write "
                       "slwid CSR, which sets the WID used for modes lower than "
                       "[H]S-mode.)", []>;

def FeatureStdExtXsifivecflushdlone
    : SubtargetFeature<"experimental-xsifivecflushdlone", "HasStdExtXsifivecflushdlone", "true",
                       "'Xsifivecflushdlone' (Cache Flush/Power Down Instructions)", []>;
def HasStdExtXsifivecflushdlone :
    Predicate<"Subtarget->hasStdExtXsifivecflushdlone()">,
    AssemblerPredicate<(all_of FeatureStdExtXsifivecflushdlone),
                       "'Xsifivecflushdlone' (Cache Flush/Power Down Instructions)">;

def FeatureStdExtXsifivecdiscarddlone
    : SubtargetFeature<"experimental-xsifivecdiscarddlone", "HasStdExtXsifivecdiscarddlone", "true",
                       "Cache Flush/Power Down Instructions"
                       "'Xsifivecdiscarddlone'(Cache Flush/Power Down Instructions)", []>;
def HasStdExtXsifivecdiscarddlone :
    Predicate<"Subtarget->hasStdExtXsifivecdiscarddlone()">,
    AssemblerPredicate<(all_of FeatureStdExtXsifivecdiscarddlone),
                        "'Xsifivecdiscarddlone' (Cache Flush/Power Down Instructions)">;
#endif // SIFIVE_CUSTOMIZATION

def FeatureStdExtM
    : SubtargetFeature<"m", "HasStdExtM", "true",
                       "'M' (Integer Multiplication and Division)">;
def HasStdExtM : Predicate<"Subtarget->hasStdExtM()">,
                           AssemblerPredicate<(all_of FeatureStdExtM),
                           "'M' (Integer Multiplication and Division)">;

def FeatureStdExtZmmul
    : SubtargetFeature<"zmmul", "HasStdExtZmmul", "true",
                       "'Zmmul' (Integer Multiplication)">;

def HasStdExtMOrZmmul
    : Predicate<"Subtarget->hasStdExtM() || Subtarget->hasStdExtZmmul()">,
                AssemblerPredicate<(any_of FeatureStdExtM, FeatureStdExtZmmul),
                                   "'M' (Integer Multiplication and Division) or "
                                   "'Zmmul' (Integer Multiplication)">;

def FeatureStdExtA
    : SubtargetFeature<"a", "HasStdExtA", "true",
                       "'A' (Atomic Instructions)">;
def HasStdExtA : Predicate<"Subtarget->hasStdExtA()">,
                           AssemblerPredicate<(all_of FeatureStdExtA),
                           "'A' (Atomic Instructions)">;

def FeatureStdExtF
    : SubtargetFeature<"f", "HasStdExtF", "true",
                       "'F' (Single-Precision Floating-Point)",
                       [FeatureStdExtZicsr]>;
def HasStdExtF : Predicate<"Subtarget->hasStdExtF()">,
                           AssemblerPredicate<(all_of FeatureStdExtF),
                           "'F' (Single-Precision Floating-Point)">;

def FeatureStdExtD
    : SubtargetFeature<"d", "HasStdExtD", "true",
                       "'D' (Double-Precision Floating-Point)",
                       [FeatureStdExtF]>;
def HasStdExtD : Predicate<"Subtarget->hasStdExtD()">,
                           AssemblerPredicate<(all_of FeatureStdExtD),
                           "'D' (Double-Precision Floating-Point)">;

def FeatureStdExtH
    : SubtargetFeature<"h", "HasStdExtH", "true",
                       "'H' (Hypervisor)">;

def HasStdExtH : Predicate<"Subtarget->hasStdExtH()">,
                           AssemblerPredicate<(all_of FeatureStdExtH),
                           "'H' (Hypervisor)">;

def FeatureStdExtZihintpause
    : SubtargetFeature<"zihintpause", "HasStdExtZihintpause", "true",
                       "'Zihintpause' (Pause Hint)">;
def HasStdExtZihintpause : Predicate<"Subtarget->hasStdExtZihintpause()">,
                                     AssemblerPredicate<(all_of FeatureStdExtZihintpause),
                                     "'Zihintpause' (Pause Hint)">;

def FeatureStdExtZihintntl
    : SubtargetFeature<"zihintntl", "HasStdExtZihintntl", "true",
                       "'Zihintntl' (Non-Temporal Locality Hints)">;
def HasStdExtZihintntl : Predicate<"Subtarget->hasStdExtZihintntl()">,
                                    AssemblerPredicate<(all_of FeatureStdExtZihintntl),
                                    "'Zihintntl' (Non-Temporal Locality Hints)">;

def FeatureStdExtZifencei
    : SubtargetFeature<"zifencei", "HasStdExtZifencei", "true",
                       "'Zifencei' (fence.i)">;
def HasStdExtZifencei : Predicate<"Subtarget->hasStdExtZifencei()">,
                                   AssemblerPredicate<(all_of FeatureStdExtZifencei),
                                   "'Zifencei' (fence.i)">;

def FeatureStdExtZicntr
    : SubtargetFeature<"zicntr", "HasStdExtZicntr", "true",
                       "'Zicntr' (Base Counters and Timers)",
                       [FeatureStdExtZicsr]>;

def FeatureStdExtZihpm
    : SubtargetFeature<"zihpm", "HasStdExtZihpm", "true",
                       "'Zihpm' (Hardware Performance Counters)",
                       [FeatureStdExtZicsr]>;

def FeatureStdExtZfhmin
    : SubtargetFeature<"zfhmin", "HasStdExtZfhmin", "true",
                       "'Zfhmin' (Half-Precision Floating-Point Minimal)",
                       [FeatureStdExtF]>;
def HasStdExtZfhmin : Predicate<"Subtarget->hasStdExtZfhmin()">,
                             AssemblerPredicate<(all_of FeatureStdExtZfhmin),
                             "'Zfhmin' (Half-Precision Floating-Point Minimal)">;

def FeatureStdExtZfh
    : SubtargetFeature<"zfh", "HasStdExtZfh", "true",
                       "'Zfh' (Half-Precision Floating-Point)",
                       [FeatureStdExtF]>;
def HasStdExtZfh : Predicate<"Subtarget->hasStdExtZfh()">,
                             AssemblerPredicate<(all_of FeatureStdExtZfh),
                             "'Zfh' (Half-Precision Floating-Point)">;
def NoStdExtZfh : Predicate<"!Subtarget->hasStdExtZfh()">;

def HasStdExtZfhOrZfhmin
    : Predicate<"Subtarget->hasStdExtZfhOrZfhmin()">,
                AssemblerPredicate<(any_of FeatureStdExtZfh, FeatureStdExtZfhmin),
                                   "'Zfh' (Half-Precision Floating-Point) or "
                                   "'Zfhmin' (Half-Precision Floating-Point Minimal)">;

def FeatureStdExtZfinx
    : SubtargetFeature<"zfinx", "HasStdExtZfinx", "true",
                       "'Zfinx' (Float in Integer)",
                       [FeatureStdExtZicsr]>;
def HasStdExtZfinx : Predicate<"Subtarget->hasStdExtZfinx()">,
                               AssemblerPredicate<(all_of FeatureStdExtZfinx),
                               "'Zfinx' (Float in Integer)">;

def FeatureStdExtZdinx
    : SubtargetFeature<"zdinx", "HasStdExtZdinx", "true",
                       "'Zdinx' (Double in Integer)",
                       [FeatureStdExtZfinx]>;
def HasStdExtZdinx : Predicate<"Subtarget->hasStdExtZdinx()">,
                               AssemblerPredicate<(all_of FeatureStdExtZdinx),
                               "'Zdinx' (Double in Integer)">;

def FeatureStdExtZhinxmin
    : SubtargetFeature<"zhinxmin", "HasStdExtZhinxmin", "true",
                       "'Zhinxmin' (Half Float in Integer Minimal)",
                       [FeatureStdExtZfinx]>;
def HasStdExtZhinxmin : Predicate<"Subtarget->hasStdExtZhinxmin()">,
                                  AssemblerPredicate<(all_of FeatureStdExtZhinxmin),
                                  "'Zhinxmin' (Half Float in Integer Minimal)">;

def FeatureStdExtZhinx
    : SubtargetFeature<"zhinx", "HasStdExtZhinx", "true",
                       "'Zhinx' (Half Float in Integer)",
                       [FeatureStdExtZfinx]>;
def HasStdExtZhinx : Predicate<"Subtarget->hasStdExtZhinx()">,
                               AssemblerPredicate<(all_of FeatureStdExtZhinx),
                               "'Zhinx' (Half Float in Integer)">;
def NoStdExtZhinx : Predicate<"!Subtarget->hasStdExtZhinx()">;

def HasStdExtZhinxOrZhinxmin
    : Predicate<"Subtarget->hasStdExtZhinx() || Subtarget->hasStdExtZhinxmin()">,
                AssemblerPredicate<(any_of FeatureStdExtZhinx, FeatureStdExtZhinxmin),
                                   "'Zhinx' (Half Float in Integer) or "
                                   "'Zhinxmin' (Half Float in Integer Minimal)">;

def FeatureStdExtZfa
    : SubtargetFeature<"experimental-zfa", "HasStdExtZfa", "true",
                       "'Zfa' (Additional Floating-Point)",
                       [FeatureStdExtF]>;
def HasStdExtZfa : Predicate<"Subtarget->hasStdExtZfa()">,
                             AssemblerPredicate<(all_of FeatureStdExtZfa),
                             "'Zfa' (Additional Floating-Point)">;

def FeatureStdExtC
    : SubtargetFeature<"c", "HasStdExtC", "true",
                       "'C' (Compressed Instructions)">;
def HasStdExtC : Predicate<"Subtarget->hasStdExtC()">,
                           AssemblerPredicate<(all_of FeatureStdExtC),
                           "'C' (Compressed Instructions)">;

def FeatureStdExtZba
    : SubtargetFeature<"zba", "HasStdExtZba", "true",
                       "'Zba' (Address Generation Instructions)">;
def HasStdExtZba : Predicate<"Subtarget->hasStdExtZba()">,
                             AssemblerPredicate<(all_of FeatureStdExtZba),
                             "'Zba' (Address Generation Instructions)">;
def NotHasStdExtZba : Predicate<"!Subtarget->hasStdExtZba()">;

def FeatureStdExtZbb
    : SubtargetFeature<"zbb", "HasStdExtZbb", "true",
                       "'Zbb' (Basic Bit-Manipulation)">;
def HasStdExtZbb : Predicate<"Subtarget->hasStdExtZbb()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbb),
                             "'Zbb' (Basic Bit-Manipulation)">;

def FeatureStdExtZbc
    : SubtargetFeature<"zbc", "HasStdExtZbc", "true",
                       "'Zbc' (Carry-Less Multiplication)">;
def HasStdExtZbc : Predicate<"Subtarget->hasStdExtZbc()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbc),
                             "'Zbc' (Carry-Less Multiplication)">;

def FeatureStdExtZbs
    : SubtargetFeature<"zbs", "HasStdExtZbs", "true",
                       "'Zbs' (Single-Bit Instructions)">;
def HasStdExtZbs : Predicate<"Subtarget->hasStdExtZbs()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbs),
                             "'Zbs' (Single-Bit Instructions)">;

def FeatureStdExtZbkb
    : SubtargetFeature<"zbkb", "HasStdExtZbkb", "true",
                       "'Zbkb' (Bitmanip instructions for Cryptography)">;
def HasStdExtZbkb : Predicate<"Subtarget->hasStdExtZbkb()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbkb),
                             "'Zbkb' (Bitmanip instructions for Cryptography)">;

def FeatureStdExtZbkx
    : SubtargetFeature<"zbkx", "HasStdExtZbkx", "true",
                       "'Zbkx' (Crossbar permutation instructions)">;
def HasStdExtZbkx : Predicate<"Subtarget->hasStdExtZbkx()">,
                             AssemblerPredicate<(all_of FeatureStdExtZbkx),
                             "'Zbkx' (Crossbar permutation instructions)">;

def HasStdExtZbbOrZbkb
    : Predicate<"Subtarget->hasStdExtZbb() || Subtarget->hasStdExtZbkb()">,
                AssemblerPredicate<(any_of FeatureStdExtZbb, FeatureStdExtZbkb),
                                   "'Zbb' (Basic Bit-Manipulation) or "
                                   "'Zbkb' (Bitmanip instructions for Cryptography)">;

// The Carry-less multiply subextension for cryptography is a subset of basic
// carry-less multiply subextension. The former should be enabled if the latter
// is enabled.
def FeatureStdExtZbkc
    : SubtargetFeature<"zbkc", "HasStdExtZbkc", "true",
                       "'Zbkc' (Carry-less multiply instructions for "
                       "Cryptography)">;
def HasStdExtZbkc
    : Predicate<"Subtarget->hasStdExtZbkc()">,
                AssemblerPredicate<(all_of FeatureStdExtZbkc),
                "'Zbkc' (Carry-less multiply instructions for Cryptography)">;

def HasStdExtZbcOrZbkc
    : Predicate<"Subtarget->hasStdExtZbc() || Subtarget->hasStdExtZbkc()">,
                AssemblerPredicate<(any_of FeatureStdExtZbc, FeatureStdExtZbkc),
                                   "'Zbc' (Carry-Less Multiplication) or "
                                   "'Zbkc' (Carry-less multiply instructions "
                                   "for Cryptography)">;

def FeatureStdExtZknd
    : SubtargetFeature<"zknd", "HasStdExtZknd", "true",
                       "'Zknd' (NIST Suite: AES Decryption)">;
def HasStdExtZknd : Predicate<"Subtarget->hasStdExtZknd()">,
                              AssemblerPredicate<(all_of FeatureStdExtZknd),
                              "'Zknd' (NIST Suite: AES Decryption)">;

def FeatureStdExtZkne
    : SubtargetFeature<"zkne", "HasStdExtZkne", "true",
                       "'Zkne' (NIST Suite: AES Encryption)">;
def HasStdExtZkne : Predicate<"Subtarget->hasStdExtZkne()">,
                              AssemblerPredicate<(all_of FeatureStdExtZkne),
                              "'Zkne' (NIST Suite: AES Encryption)">;

// Some instructions belong to both Zknd and Zkne subextensions.
// They should be enabled if either has been specified.
def HasStdExtZkndOrZkne
    : Predicate<"Subtarget->hasStdExtZknd() || Subtarget->hasStdExtZkne()">,
                AssemblerPredicate<(any_of FeatureStdExtZknd, FeatureStdExtZkne),
                                   "'Zknd' (NIST Suite: AES Decryption) or "
                                   "'Zkne' (NIST Suite: AES Encryption)">;

def FeatureStdExtZknh
    : SubtargetFeature<"zknh", "HasStdExtZknh", "true",
                       "'Zknh' (NIST Suite: Hash Function Instructions)">;
def HasStdExtZknh : Predicate<"Subtarget->hasStdExtZknh()">,
                             AssemblerPredicate<(all_of FeatureStdExtZknh),
                             "'Zknh' (NIST Suite: Hash Function Instructions)">;

def FeatureStdExtZksed
    : SubtargetFeature<"zksed", "HasStdExtZksed", "true",
                       "'Zksed' (ShangMi Suite: SM4 Block Cipher Instructions)">;
def HasStdExtZksed : Predicate<"Subtarget->hasStdExtZksed()">,
                             AssemblerPredicate<(all_of FeatureStdExtZksed),
                             "'Zksed' (ShangMi Suite: SM4 Block Cipher Instructions)">;

def FeatureStdExtZksh
    : SubtargetFeature<"zksh", "HasStdExtZksh", "true",
                       "'Zksh' (ShangMi Suite: SM3 Hash Function Instructions)">;
def HasStdExtZksh : Predicate<"Subtarget->hasStdExtZksh()">,
                              AssemblerPredicate<(all_of FeatureStdExtZksh),
                              "'Zksh' (ShangMi Suite: SM3 Hash Function "
                              "Instructions)">;

def FeatureStdExtZkr
    : SubtargetFeature<"zkr", "HasStdExtZkr", "true",
                       "'Zkr' (Entropy Source Extension)">;
def HasStdExtZkr : Predicate<"Subtarget->hasStdExtZkr()">,
                             AssemblerPredicate<(all_of FeatureStdExtZkr),
                             "'Zkr' (Entropy Source Extension)">;

def FeatureStdExtZkn
    : SubtargetFeature<"zkn", "HasStdExtZkn", "true",
                       "'Zkn' (NIST Algorithm Suite)",
                       [FeatureStdExtZbkb,
                        FeatureStdExtZbkc,
                        FeatureStdExtZbkx,
                        FeatureStdExtZkne,
                        FeatureStdExtZknd,
                        FeatureStdExtZknh]>;

def FeatureStdExtZks
    : SubtargetFeature<"zks", "HasStdExtZks", "true",
                       "'Zks' (ShangMi Algorithm Suite)",
                       [FeatureStdExtZbkb,
                        FeatureStdExtZbkc,
                        FeatureStdExtZbkx,
                        FeatureStdExtZksed,
                        FeatureStdExtZksh]>;

def FeatureStdExtZkt
    : SubtargetFeature<"zkt", "HasStdExtZkt", "true",
                       "'Zkt' (Data Independent Execution Latency)">;

def FeatureStdExtZk
    : SubtargetFeature<"zk", "HasStdExtZk", "true",
                       "'Zk' (Standard scalar cryptography extension)",
                       [FeatureStdExtZkn,
                        FeatureStdExtZkr,
                        FeatureStdExtZkt]>;

def FeatureStdExtZca
    : SubtargetFeature<"zca", "HasStdExtZca", "true",
                       "'Zca' (part of the C extension, excluding compressed "
                       "floating point loads/stores)">;

def HasStdExtCOrZca
    : Predicate<"Subtarget->hasStdExtCOrZca()">,
                AssemblerPredicate<(any_of FeatureStdExtC, FeatureStdExtZca),
                                   "'C' (Compressed Instructions) or "
                                   "'Zca' (part of the C extension, excluding "
                                   "compressed floating point loads/stores)">;

def FeatureStdExtZcb
    : SubtargetFeature<"zcb", "HasStdExtZcb", "true",
                       "'Zcb' (Compressed basic bit manipulation instructions)",
                       [FeatureStdExtZca]>;
def HasStdExtZcb : Predicate<"Subtarget->hasStdExtZcb()">,
                             AssemblerPredicate<(all_of FeatureStdExtZcb),
                             "'Zcb' (Compressed basic bit manipulation instructions)">;

def FeatureStdExtZcd
    : SubtargetFeature<"zcd", "HasStdExtZcd", "true",
                       "'Zcd' (Compressed Double-Precision Floating-Point Instructions)",
                       [FeatureStdExtZca]>;

def HasStdExtCOrZcd
    : Predicate<"Subtarget->hasStdExtC() || Subtarget->hasStdExtZcd()">,
                AssemblerPredicate<(any_of FeatureStdExtC, FeatureStdExtZcd),
                                   "'C' (Compressed Instructions) or "
                                   "'Zcd' (Compressed Double-Precision Floating-Point Instructions)">;

def FeatureStdExtZcf
    : SubtargetFeature<"zcf", "HasStdExtZcf", "true",
                       "'Zcf' (Compressed Single-Precision Floating-Point Instructions)",
                       [FeatureStdExtZca]>;

def FeatureStdExtZcmp
    : SubtargetFeature<"zcmp", "HasStdExtZcmp", "true",
                       "'Zcmp' (sequenced instuctions for code-size reduction)",
                       [FeatureStdExtZca]>;
def HasStdExtZcmp : Predicate<"Subtarget->hasStdExtZcmp() && !Subtarget->hasStdExtC()">,
                               AssemblerPredicate<(all_of FeatureStdExtZcmp),
                               "'Zcmp' (sequenced instuctions for code-size reduction)">;

def FeatureStdExtZcmt
    : SubtargetFeature<"zcmt", "HasStdExtZcmt", "true",
                       "'Zcmt' (table jump instuctions for code-size reduction)",
                       [FeatureStdExtZca, FeatureStdExtZicsr]>;
def HasStdExtZcmt : Predicate<"Subtarget->hasStdExtZcmt()">,
                           AssemblerPredicate<(all_of FeatureStdExtZcmt),
                           "'Zcmt' (table jump instuctions for code-size reduction)">;

def FeatureStdExtZce
    : SubtargetFeature<"zce", "HasStdExtZce", "true",
                       "'Zce' (Compressed extensions for microcontrollers)",
                       [FeatureStdExtZca, FeatureStdExtZcb, FeatureStdExtZcmp,
                        FeatureStdExtZcmt]>;

def HasStdExtCOrZcfOrZce
    : Predicate<"Subtarget->hasStdExtC() || Subtarget->hasStdExtZcf() "
                "Subtarget->hasStdExtZce()">,
                AssemblerPredicate<(any_of FeatureStdExtC, FeatureStdExtZcf,
                                           FeatureStdExtZce),
                                   "'C' (Compressed Instructions) or "
                                   "'Zcf' (Compressed Single-Precision Floating-Point Instructions)">;

def FeatureNoRVCHints
    : SubtargetFeature<"no-rvc-hints", "EnableRVCHintInstrs", "false",
                       "Disable RVC Hint Instructions.">;
def HasRVCHints : Predicate<"Subtarget->enableRVCHintInstrs()">,
                  AssemblerPredicate<(all_of(not FeatureNoRVCHints)),
                                      "RVC Hint Instructions">;

def FeatureStdExtZvl32b : SubtargetFeature<"zvl32b", "ZvlLen", "32",
                                           "'Zvl' (Minimum Vector Length) 32">;

foreach i = { 6-16 } in {
  defvar I = !shl(1, i);
  def FeatureStdExtZvl#I#b :
      SubtargetFeature<"zvl"#I#"b", "ZvlLen", !cast<string>(I),
                       "'Zvl' (Minimum Vector Length) "#I,
                       [!cast<SubtargetFeature>("FeatureStdExtZvl"#!srl(I, 1)#"b")]>;
}

def FeatureStdExtZve32x
    : SubtargetFeature<"zve32x", "HasStdExtZve32x", "true",
                       "'Zve32x' (Vector Extensions for Embedded Processors "
                       "with maximal 32 EEW)",
                       [FeatureStdExtZicsr, FeatureStdExtZvl32b]>;

def FeatureStdExtZve32f
    : SubtargetFeature<"zve32f", "HasStdExtZve32f", "true",
                       "'Zve32f' (Vector Extensions for Embedded Processors "
                       "with maximal 32 EEW and F extension)",
                       [FeatureStdExtZve32x, FeatureStdExtF]>;

def FeatureStdExtZve64x
    : SubtargetFeature<"zve64x", "HasStdExtZve64x", "true",
                       "'Zve64x' (Vector Extensions for Embedded Processors "
                       "with maximal 64 EEW)",
                       [FeatureStdExtZve32x, FeatureStdExtZvl64b]>;

def FeatureStdExtZve64f
    : SubtargetFeature<"zve64f", "HasStdExtZve64f", "true",
                       "'Zve64f' (Vector Extensions for Embedded Processors "
                       "with maximal 64 EEW and F extension)",
                       [FeatureStdExtZve32f, FeatureStdExtZve64x]>;

def FeatureStdExtZve64d
    : SubtargetFeature<"zve64d", "HasStdExtZve64d", "true",
                       "'Zve64d' (Vector Extensions for Embedded Processors "
                       "with maximal 64 EEW, F and D extension)",
                       [FeatureStdExtZve64f, FeatureStdExtD]>;

def FeatureStdExtV
    : SubtargetFeature<"v", "HasStdExtV", "true",
                       "'V' (Vector Extension for Application Processors)",
                       [FeatureStdExtZvl128b, FeatureStdExtZve64d]>;

def HasVInstructions    : Predicate<"Subtarget->hasVInstructions()">,
      AssemblerPredicate<
          (any_of FeatureStdExtZve32x),
          "'V' (Vector Extension for Application Processors), 'Zve32x' or "
          "'Zve64x' (Vector Extensions for Embedded Processors)">;
def HasVInstructionsI64 : Predicate<"Subtarget->hasVInstructionsI64()">,
      AssemblerPredicate<
          (any_of FeatureStdExtZve64x),
          "'V' (Vector Extension for Application Processors) or 'Zve64x' "
          "(Vector Extensions for Embedded Processors)">;
def HasVInstructionsAnyF : Predicate<"Subtarget->hasVInstructionsAnyF()">,
      AssemblerPredicate<
          (any_of FeatureStdExtZve32f),
          "'V' (Vector Extension for Application Processors), 'Zve32f', "
          "'Zve64f' or 'Zve64d' (Vector Extensions for Embedded Processors)">;

def HasVInstructionsF64 : Predicate<"Subtarget->hasVInstructionsF64()">;

def HasVInstructionsFullMultiply : Predicate<"Subtarget->hasVInstructionsFullMultiply()">;

def FeatureStdExtZfbfmin
    : SubtargetFeature<"experimental-zfbfmin", "HasStdExtZfbfmin", "true",
                       "'Zfbfmin' (Scalar BF16 Converts)",
                       [FeatureStdExtF]>;
def HasStdExtZfbfmin : Predicate<"Subtarget->hasStdExtZfbfmin()">,
                                 AssemblerPredicate<(all_of FeatureStdExtZfbfmin),
                                 "'Zfbfmin' (Scalar BF16 Converts)">;

def FeatureStdExtZvfbfmin
    : SubtargetFeature<"experimental-zvfbfmin", "HasStdExtZvfbfmin", "true",
                       "'Zvbfmin' (Vector BF16 Converts)",
                       [FeatureStdExtZve32f, FeatureStdExtZfbfmin]>;
def HasStdExtZvfbfmin : Predicate<"Subtarget->hasStdExtZvfbfmin()">,
                                  AssemblerPredicate<(all_of FeatureStdExtZvfbfmin),
                                  "'Zvfbfmin' (Vector BF16 Converts)">;

def FeatureStdExtZvfbfwma
    : SubtargetFeature<"experimental-zvfbfwma", "HasStdExtZvfbfwma", "true",
                       "'Zvfbfwma' (Vector BF16 widening mul-add)",
                       [FeatureStdExtZvfbfmin]>;
def HasStdExtZvfbfwma : Predicate<"Subtarget->hasStdExtZvfbfwma()">,
                                  AssemblerPredicate<(all_of FeatureStdExtZvfbfwma),
                                  "'Zvfbfwma' (Vector BF16 widening mul-add)">;

def HasVInstructionsBF16 : Predicate<"Subtarget->hasVInstructionsBF16()">;

def FeatureStdExtZvfh
    : SubtargetFeature<"zvfh", "HasStdExtZvfh", "true",
                       "'Zvfh' (Vector Half-Precision Floating-Point)",
                       [FeatureStdExtZve32f, FeatureStdExtZfhmin]>;

def HasVInstructionsF16 : Predicate<"Subtarget->hasVInstructionsF16()">;

def HasStdExtZfhOrZvfh
    : Predicate<"Subtarget->hasStdExtZfh() || Subtarget->hasStdExtZvfh()">,
                AssemblerPredicate<(any_of FeatureStdExtZfh, FeatureStdExtZvfh),
                                   "'Zfh' (Half-Precision Floating-Point) or "
                                   "'Zvfh' (Vector Half-Precision Floating-Point)">;

def FeatureStdExtZicbom
    : SubtargetFeature<"zicbom", "HasStdExtZicbom", "true",
                       "'Zicbom' (Cache-Block Management Instructions)">;
def HasStdExtZicbom : Predicate<"Subtarget->hasStdExtZicbom()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicbom),
                                "'Zicbom' (Cache-Block Management Instructions)">;

def FeatureStdExtZicboz
    : SubtargetFeature<"zicboz", "HasStdExtZicboz", "true",
                       "'Zicboz' (Cache-Block Zero Instructions)">;
def HasStdExtZicboz : Predicate<"Subtarget->hasStdExtZicboz()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicboz),
                                "'Zicboz' (Cache-Block Zero Instructions)">;

def FeatureStdExtZicbop
    : SubtargetFeature<"zicbop", "HasStdExtZicbop", "true",
                       "'Zicbop' (Cache-Block Prefetch Instructions)">;
def HasStdExtZicbop : Predicate<"Subtarget->hasStdExtZicbop()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicbop),
                                "'Zicbop' (Cache-Block Prefetch Instructions)">;

def FeatureStdExtSvnapot
    : SubtargetFeature<"svnapot", "HasStdExtSvnapot", "true",
                       "'Svnapot' (NAPOT Translation Contiguity)">;

def FeatureStdExtSvpbmt
    : SubtargetFeature<"svpbmt", "HasStdExtSvpbmt", "true",
                       "'Svpbmt' (Page-Based Memory Types)">;

def FeatureStdExtSvinval
    : SubtargetFeature<"svinval", "HasStdExtSvinval", "true",
                       "'Svinval' (Fine-Grained Address-Translation Cache Invalidation)">;
def HasStdExtSvinval : Predicate<"Subtarget->hasStdExtSvinval()">,
                                AssemblerPredicate<(all_of FeatureStdExtSvinval),
                                "'Svinval' (Fine-Grained Address-Translation Cache Invalidation)">;

def FeatureStdExtZtso
    : SubtargetFeature<"experimental-ztso", "HasStdExtZtso", "true",
                       "'Ztso' (Memory Model - Total Store Order)">;
def HasStdExtZtso : Predicate<"Subtarget->hasStdExtZTso()">,
                              AssemblerPredicate<(all_of FeatureStdExtZtso),
                              "'Ztso' (Memory Model - Total Store Order)">;
def NotHasStdExtZtso : Predicate<"!Subtarget->hasStdExtZtso()">;

def FeatureStdExtZawrs : SubtargetFeature<"zawrs", "HasStdExtZawrs", "true",
                                          "'Zawrs' (Wait on Reservation Set)">;
def HasStdExtZawrs : Predicate<"Subtarget->hasStdExtZawrs()">,
                               AssemblerPredicate<(all_of FeatureStdExtZawrs),
                               "'Zawrs' (Wait on Reservation Set)">;

def FeatureStdExtZvkb
    : SubtargetFeature<"experimental-zvkb", "HasStdExtZvkb", "true",
                       "'Zvkb' (Vector Bitmanip instructions for Cryptography.)">;
def HasStdExtZvkb : Predicate<"Subtarget->hasStdExtZvkb()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvkb),
                              "'Zvkb' (Vector Bitmanip instructions for Cryptography.)">;

def FeatureStdExtZvbb
    : SubtargetFeature<"experimental-zvbb", "HasStdExtZvbb", "true",
                       "'Zvbb' (Vector basic bit-manipulation instructions.)",
                       [FeatureStdExtZvkb]>;
def HasStdExtZvbb : Predicate<"Subtarget->hasStdExtZvbb()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvbb),
                              "'Zvbb' (Vector basic bit-manipulation instructions.)">;

def FeatureStdExtZvbc
    : SubtargetFeature<"experimental-zvbc", "HasStdExtZvbc", "true",
                       "'Zvbc' (Vector Carryless Multiplication)">;
def HasStdExtZvbc : Predicate<"Subtarget->hasStdExtZvbc()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvbc),
                              "'Zvbc' (Vector Carryless Multiplication)">;

def FeatureStdExtZvkg
    : SubtargetFeature<"experimental-zvkg", "HasStdExtZvkg", "true",
                       "'Zvkg' (Vector GCM instructions for Cryptography)">;
def HasStdExtZvkg : Predicate<"Subtarget->hasStdExtZvkg()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvkg),
                              "'Zvkg' (Vector GCM instructions for Cryptography)">;

def FeatureStdExtZvkned
    : SubtargetFeature<"experimental-zvkned", "HasStdExtZvkned", "true",
                       "'Zvkned' (Vector AES Encryption & Decryption (Single Round))">;
def HasStdExtZvkned : Predicate<"Subtarget->hasStdExtZvkned()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvkned),
                                "'Zvkned' (Vector AES Encryption & Decryption (Single Round))">;

def FeatureStdExtZvknha
    : SubtargetFeature<"experimental-zvknha", "HasStdExtZvknha", "true",
                       "'Zvknha' (Vector SHA-2 (SHA-256 only))">;

def FeatureStdExtZvknhb
    : SubtargetFeature<"experimental-zvknhb", "HasStdExtZvknhb", "true",
                       "'Zvknhb' (Vector SHA-2 (SHA-256 and SHA-512))",
                       [FeatureStdExtZvknha]>;
def HasStdExtZvknha : Predicate<"Subtarget->hasStdExtZvknha()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvknha),
                                "'Zvknha' (Vector SHA-2 (SHA-256 only))">;

def FeatureStdExtZvksed
    : SubtargetFeature<"experimental-zvksed", "HasStdExtZvksed", "true",
                       "'Zvksed' (SM4 Block Cipher Instructions)">;
def HasStdExtZvksed : Predicate<"Subtarget->hasStdExtZvksed()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvksed),
                                "'Zvksed' (SM4 Block Cipher Instructions)">;

def FeatureStdExtZvksh
    : SubtargetFeature<"experimental-zvksh", "HasStdExtZvksh", "true",
                       "'Zvksh' (SM3 Hash Function Instructions)">;
def HasStdExtZvksh : Predicate<"Subtarget->hasStdExtZvksh()">,
                               AssemblerPredicate<(all_of FeatureStdExtZvksh),
                               "'Zvksh' (SM3 Hash Function Instructions)">;

#ifdef SIFIVE_CUSTOMIZATION
def FeatureStdExtZvkb0p1
    : SubtargetFeature<"experimental-zvkb0p1", "HasStdExtZvkb0p1", "true",
                       "'Zvkb' (Vector Bitmanip instructions for Cryptography.)">;
def HasStdExtZvkb0p1 : Predicate<"Subtarget->hasStdExtZvkb0p1()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvkb0p1),
                              "'Zvkb' (Vector Bitmanip instructions for Cryptography.)">;
def HasStdExtZvkbOrZvkb0p1
    : Predicate<"Subtarget->hasStdExtZvkb() || Subtarget->hasStdExtZvkb0p1()">,
      AssemblerPredicate<(any_of FeatureStdExtZvkb, FeatureStdExtZvkb0p1),
                         "'Zvkb' (Vector Bitmanip instructions for Cryptography.)">;

def HasStdExtZvbcOrZvkb0p1
    : Predicate<"Subtarget->hasStdExtZvbc() || Subtarget->hasStdExtZvkb0p1()">,
      AssemblerPredicate<(any_of FeatureStdExtZvbc, FeatureStdExtZvkb0p1),
                         "'Zvkb' or 'Zvbc' (Vector Bitmanip instructions for Cryptography.)">;

def FeatureStdExtZvkg0p1
    : SubtargetFeature<"experimental-zvkg0p1", "HasStdExtZvkg0p1", "true",
                       "'Zvkg' (Vector GCM instructions for Cryptography.)">;
def HasStdExtZvkg0p1 : Predicate<"Subtarget->hasStdExtZvkg0p1()">,
                              AssemblerPredicate<(all_of FeatureStdExtZvkg0p1),
                              "'Zvkg' (Vector GCM instructions for Cryptography.)">;

def FeatureStdExtZvknha0p1
    : SubtargetFeature<"experimental-zvknha0p1", "HasStdExtZvknha0p1", "true",
                       "'Zvknha' (Vector SHA-2. (SHA-256 only))">;

def FeatureStdExtZvknhb0p1
    : SubtargetFeature<"experimental-zvknhb0p1", "HasStdExtZvknhb0p1", "true",
                       "'Zvknhb' (Vector SHA-2. (SHA-256 and SHA-512))",
                       [FeatureStdExtZvknha0p1]>;
def FeatureStdExtZvkns
    : SubtargetFeature<"experimental-zvkns", "HasStdExtZvkns", "true",
                       "'Zvkns' (Vector AES Encryption & Decryption (Single Round))">;
def HasStdExtZvknsOrZvkned
    : Predicate<"Subtarget->hasStdExtZvkns() || Subtarget->hasStdExtZvkned()">,
                AssemblerPredicate<(any_of FeatureStdExtZvkns, FeatureStdExtZvkned),
                "'Zvkns' or 'Zvkned' (Vector AES Encryption & Decryption (Single Round))">;

def FeatureStdExtZvksed0p1
    : SubtargetFeature<"experimental-zvksed0p1", "HasStdExtZvksed0p1", "true",
                       "'Zvksed' (SM4 Block Cipher Instructions.)">;

def FeatureStdExtZvksh0p1
    : SubtargetFeature<"experimental-zvksh0p1", "HasStdExtZvksh0p1", "true",
                       "'Zvksh' (SM3 Hash Function Instructions.)">;
#endif // SIFIVE_CUSTOMIZATION

def FeatureStdExtZvkt
    : SubtargetFeature<"experimental-zvkt", "HasStdExtZvkt", "true",
                       "'Zvkt' (Vector Data-Independent Execution Latency)">;

// Zvk short-hand extensions

def FeatureStdExtZvkn
    : SubtargetFeature<"experimental-zvkn", "HasStdExtZvkn", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvkned, Zvknhb, Zvkb and Zvkt.",
                       [FeatureStdExtZvkned, FeatureStdExtZvknhb,
                        FeatureStdExtZvkb, FeatureStdExtZvkt]>;

def FeatureStdExtZvknc
    : SubtargetFeature<"experimental-zvknc", "HasStdExtZvknc", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvkn and Zvbc.",
                       [FeatureStdExtZvkn, FeatureStdExtZvbc]>;

def FeatureStdExtZvkng
    : SubtargetFeature<"experimental-zvkng", "HasStdExtZvkng", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvkn and Zvkg.",
                       [FeatureStdExtZvkn, FeatureStdExtZvkg]>;

def FeatureStdExtZvks
    : SubtargetFeature<"experimental-zvks", "HasStdExtZvks", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvksed, Zvksh, Zvkb and Zvkt.",
                       [FeatureStdExtZvksed, FeatureStdExtZvksh,
                        FeatureStdExtZvkb, FeatureStdExtZvkt]>;

def FeatureStdExtZvksc
    : SubtargetFeature<"experimental-zvksc", "HasStdExtZvksc", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvks and Zvbc.",
                       [FeatureStdExtZvks, FeatureStdExtZvbc]>;

def FeatureStdExtZvksg
    : SubtargetFeature<"experimental-zvksg", "HasStdExtZvksg", "true",
                       "This extension is shorthand for the following set of "
                       "other extensions: Zvks and Zvkg.",
                       [FeatureStdExtZvks, FeatureStdExtZvkg]>;

def FeatureStdExtZicond
    : SubtargetFeature<"experimental-zicond", "HasStdExtZicond", "true",
                       "'Zicond' (Integer Conditional Operations)">;
def HasStdExtZicond : Predicate<"Subtarget->hasStdExtZicond()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicond),
                                "'Zicond' (Integer Conditional Operations)">;

def FeatureStdExtSmaia
    : SubtargetFeature<"experimental-smaia", "HasStdExtSmaia", "true",
                       "'Smaia' (Smaia encompasses all added CSRs and all "
                       "modifications to interrupt response behavior that the "
                       "AIA specifies for a hart, over all privilege levels.)",
                       []>;

def FeatureStdExtSsaia
    : SubtargetFeature<"experimental-ssaia", "HasStdExtSsaia", "true",
                       "'Ssaia' (Ssaia is essentially the same as Smaia except "
                       "excluding the machine-level CSRs and behavior not "
                       "directly visible to supervisor level.)", []>;

def HasHalfFPLoadStoreMove
    : Predicate<"Subtarget->hasHalfFPLoadStoreMove()">,
                AssemblerPredicate<(any_of FeatureStdExtZfh, FeatureStdExtZfhmin,
                                    FeatureStdExtZfbfmin),
                                    "'Zfh' (Half-Precision Floating-Point) or "
                                    "'Zfhmin' (Half-Precision Floating-Point Minimal) or "
                                    "'Zfbfmin' (Scalar BF16 Converts)">;

def FeatureStdExtZacas
    : SubtargetFeature<"experimental-zacas", "HasStdExtZacas", "true",
                       "'Zacas' (Atomic Compare-And-Swap Instructions)">;
def HasStdExtZacas : Predicate<"Subtarget->hasStdExtZacas()">,
                               AssemblerPredicate<(all_of FeatureStdExtZacas),
                               "'Zacas' (Atomic Compare-And-Swap Instructions)">;

//===----------------------------------------------------------------------===//
// Vendor extensions
//===----------------------------------------------------------------------===//

def FeatureVendorXVentanaCondOps
    : SubtargetFeature<"xventanacondops", "HasVendorXVentanaCondOps", "true",
                       "'XVentanaCondOps' (Ventana Conditional Ops)">;
def HasVendorXVentanaCondOps : Predicate<"Subtarget->hasVendorXVentanaCondOps()">,
                                AssemblerPredicate<(all_of FeatureVendorXVentanaCondOps),
                                "'XVentanaCondOps' (Ventana Conditional Ops)">;

def FeatureVendorXTHeadBa
    : SubtargetFeature<"xtheadba", "HasVendorXTHeadBa", "true",
                       "'xtheadba' (T-Head address calculation instructions)">;
def HasVendorXTHeadBa : Predicate<"Subtarget->hasVendorXTHeadBa()">,
                                  AssemblerPredicate<(all_of FeatureVendorXTHeadBa),
                                  "'xtheadba' (T-Head address calculation instructions)">;

def FeatureVendorXTHeadBb
    : SubtargetFeature<"xtheadbb", "HasVendorXTHeadBb", "true",
                       "'xtheadbb' (T-Head basic bit-manipulation instructions)">;
def HasVendorXTHeadBb : Predicate<"Subtarget->hasVendorXTHeadBb()">,
                                  AssemblerPredicate<(all_of FeatureVendorXTHeadBb),
                                  "'xtheadbb' (T-Head basic bit-manipulation instructions)">;

def FeatureVendorXTHeadBs
    : SubtargetFeature<"xtheadbs", "HasVendorXTHeadBs", "true",
                       "'xtheadbs' (T-Head single-bit instructions)">;
def HasVendorXTHeadBs : Predicate<"Subtarget->hasVendorXTHeadBs()">,
                                  AssemblerPredicate<(all_of FeatureVendorXTHeadBs),
                                  "'xtheadbs' (T-Head single-bit instructions)">;

def FeatureVendorXTHeadCondMov
    : SubtargetFeature<"xtheadcondmov", "HasVendorXTHeadCondMov", "true",
                       "'xtheadcondmov' (T-Head conditional move instructions)">;
def HasVendorXTHeadCondMov : Predicate<"Subtarget->hasVendorXTHeadCondMov()">,
                                       AssemblerPredicate<(all_of FeatureVendorXTHeadCondMov),
                                       "'xtheadcondmov' (T-Head conditional move instructions)">;

def FeatureVendorXTHeadCmo
    : SubtargetFeature<"xtheadcmo", "HasVendorXTHeadCmo", "true",
                       "'xtheadcmo' (T-Head cache management instructions)">;
def HasVendorXTHeadCmo : Predicate<"Subtarget->hasVendorXTHeadCmo()">,
                                   AssemblerPredicate<(all_of FeatureVendorXTHeadCmo),
                                   "'xtheadcmo' (T-Head cache management instructions)">;

def FeatureVendorXTHeadFMemIdx
    : SubtargetFeature<"xtheadfmemidx", "HasVendorXTHeadFMemIdx", "true",
                       "'xtheadfmemidx' (T-Head FP Indexed Memory Operations)",
                       [FeatureStdExtF]>;
def HasVendorXTHeadFMemIdx : Predicate<"Subtarget->hasVendorXTHeadFMemIdx()">,
                                       AssemblerPredicate<(all_of FeatureVendorXTHeadFMemIdx),
                                       "'xtheadfmemidx' (T-Head FP Indexed Memory Operations)">;

def FeatureVendorXTHeadMac
    : SubtargetFeature<"xtheadmac", "HasVendorXTHeadMac", "true",
                       "'xtheadmac' (T-Head Multiply-Accumulate Instructions)">;
def HasVendorXTHeadMac : Predicate<"Subtarget->hasVendorXTHeadMac()">,
                                   AssemblerPredicate<(all_of FeatureVendorXTHeadMac),
                                   "'xtheadmac' (T-Head Multiply-Accumulate Instructions)">;

def FeatureVendorXTHeadMemIdx
    : SubtargetFeature<"xtheadmemidx", "HasVendorXTHeadMemIdx", "true",
                       "'xtheadmemidx' (T-Head Indexed Memory Operations)">;
def HasVendorXTHeadMemIdx : Predicate<"Subtarget->hasVendorXTHeadMemIdx()">,
                                      AssemblerPredicate<(all_of FeatureVendorXTHeadMemIdx),
                                      "'xtheadmemidx' (T-Head Indexed Memory Operations)">;

def FeatureVendorXTHeadMemPair
    : SubtargetFeature<"xtheadmempair", "HasVendorXTHeadMemPair", "true",
                       "'xtheadmempair' (T-Head two-GPR Memory Operations)">;
def HasVendorXTHeadMemPair : Predicate<"Subtarget->hasVendorXTHeadMemPair()">,
                                    AssemblerPredicate<(all_of FeatureVendorXTHeadMemPair),
                                    "'xtheadmempair' (T-Head two-GPR Memory Operations)">;

def FeatureVendorXTHeadSync
    : SubtargetFeature<"xtheadsync", "HasVendorXTHeadSync", "true",
                       "'xtheadsync' (T-Head multicore synchronization instructions)">;
def HasVendorXTHeadSync : Predicate<"Subtarget->hasVendorXTHeadSync()">,
                                    AssemblerPredicate<(all_of FeatureVendorXTHeadSync),
                                    "'xtheadsync' (T-Head multicore synchronization instructions)">;

def FeatureVendorXTHeadVdot
    : SubtargetFeature<"xtheadvdot", "HasVendorXTHeadVdot", "true",
                       "'xtheadvdot' (T-Head Vector Extensions for Dot)",
                       [FeatureStdExtV]>;
def HasVendorXTHeadVdot : Predicate<"Subtarget->hasVendorXTHeadVdot()">,
                                    AssemblerPredicate<(all_of FeatureVendorXTHeadVdot),
                                    "'xtheadvdot' (T-Head Vector Extensions for Dot)">;

def FeatureVendorXSfvcp
    : SubtargetFeature<"xsfvcp", "HasVendorXSfvcp", "true",
                       "'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions)",
                       [FeatureStdExtZve32x]>;
def HasVendorXSfvcp : Predicate<"Subtarget->hasVendorXSfvcp()">,
                                AssemblerPredicate<(all_of FeatureVendorXSfvcp),
                                "'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions)">;

#ifdef SIFIVE_CUSTOMIZATION
def FeatureVendorXSfvqmaccqoq
    : SubtargetFeature<"xsfvqmaccqoq", "HasVendorXSfvqmaccqoq", "true",
                       "Xsfvqmaccqoq (SiFive Custom int8 Matrix-Multiply Instructions for DLEN >= 256)",
                       [FeatureStdExtZve32x, FeatureStdExtZvl256b]>;
def HasVendorXSfvqmaccqoq : Predicate<"Subtarget->hasVendorXSfvqmaccqoq()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfvqmaccqoq),
                         "'Xsfvqmaccqoq' (SiFive Custom int8 Matrix-Multiply Instructions for DLEN >= 256)">;
def FeatureVendorXSfvqmaccdod
    : SubtargetFeature<"xsfvqmaccdod", "HasVendorXSfvqmaccdod", "true",
                       "Xsfvqmaccdod (SiFive Custom int8 Matrix-Multiply Instructions for DLEN >= 128)",
                       [FeatureStdExtZve32x, FeatureStdExtZvl128b]>;
def HasVendorXSfvqmaccdod : Predicate<"Subtarget->hasVendorXSfvqmaccdod()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfvqmaccdod),
                         "'Xsfvqmaccdod' (SiFive Custom int8 Matrix-Multiply Instructions for DLEN >= 128)">;

def FeatureVendorXSfvfhbfmin
    : SubtargetFeature<"xsfvfhbfmin", "HasVendorXSfvfhbfmin", "true",
                       "Xsfvfhbfmin (SiFive custom minimal BF16 vector support)",
                       [FeatureStdExtZve32f]>;
def HasVendorXSfvfhbfmin : Predicate<"Subtarget->hasVendorXSfvfhbfmin()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfvfhbfmin),
                         "'Xsfvfhbfmin' (SiFive custom minimal BF16 vector support)">;

def FeatureVendorXSfvfwmaccqqq
    : SubtargetFeature<"xsfvfwmaccqqq", "HasVendorXSfvfwmaccqqq", "true",
                       "Xsfvfwmaccqqq (Support SiFive custom BF16 matrix arithmetic vector instructions)",
                       [FeatureStdExtZve32f, FeatureStdExtZvl256b]>;
def HasVendorXSfvfwmaccqqq : Predicate<"Subtarget->hasVendorXSfvfwmaccqqq()">,
                          AssemblerPredicate<(all_of FeatureVendorXSfvfwmaccqqq),
                          "'Xsfvfwmaccqqq' (SiFive custom BF16 matrix arithmetic vector instructions)">;

def FeatureVendorXSfvfnrclipxfqf
    : SubtargetFeature<"xsfvfnrclipxfqf", "HasVendorXSfvfnrclipxfqf", "true",
                       "Xsfvfnrclipxfqf (Support SiFive custom FP32-to-int8 ranged clip instructions)",
                       [FeatureStdExtZve32f]>;
def HasVendorXSfvfnrclipxfqf : Predicate<"Subtarget->hasVendorXSfvfnrclipxfqf()">,
                            AssemblerPredicate<(all_of FeatureVendorXSfvfnrclipxfqf),
                            "'Xsfvfnrclipxfqf' (SiFive custom FP32-to-int8 ranged clip instructions)">;
#endif // SIFIVE_CUSTOMIZATION
def FeatureVendorXSfcie
    : SubtargetFeature<"xsfcie", "HasVendorXSfcie", "true",
                       "'XSfcie' (SiFive Custom Instruction Extension SCIE.)">;
def HasVendorXSfcie : Predicate<"Subtarget->hasVendorXSfcie()">,
                        AssemblerPredicate<(all_of FeatureVendorXSfcie),
                        "'XSfcie' (SiFive Custom Instruction Extension SCIE.)">;

def FeatureVendorXCVbitmanip
    : SubtargetFeature<"xcvbitmanip", "HasVendorXCVbitmanip", "true",
                       "'XCVbitmanip' (CORE-V Bit Manipulation)">;
def HasVendorXCVbitmanip : Predicate<"Subtarget->hasVendorXCVbitmanip()">,
                                AssemblerPredicate<(all_of FeatureVendorXCVbitmanip),
                                "'XCVbitmanip' (CORE-V Bit Manipulation)">;

def FeatureVendorXCVmac
    : SubtargetFeature<"xcvmac", "HasVendorXCVmac", "true",
                       "'XCVmac' (CORE-V Multiply-Accumulate)">;
def HasVendorXCVmac : Predicate<"Subtarget->hasVendorXCVmac()">,
                                AssemblerPredicate<(all_of FeatureVendorXCVmac),
                                "'XCVmac' (CORE-V Multiply-Accumulate)">;

def FeatureVendorXCValu
    : SubtargetFeature<"xcvalu", "HasVendorXCValu", "true",
                       "'XCValu' (CORE-V ALU Operations)">;
def HasVendorXCValu : Predicate<"Subtarget->hasVendorXCValu()">,
                                AssemblerPredicate<(all_of FeatureVendorXCValu),
                                "'XCValu' (CORE-V ALU Operations)">;

def FeatureVendorXCVsimd
    : SubtargetFeature<"xcvsimd", "HasVendorXCvsimd", "true",
                       "'XCVsimd' (CORE-V SIMD ALU)">;
def HasVendorXCVsimd
    : Predicate<"Subtarget->hasVendorXCVsimd()">,
                AssemblerPredicate<(any_of FeatureVendorXCVsimd),
                "'XCVsimd' (CORE-V SIMD ALU)">;

def FeatureVendorXCVbi
    : SubtargetFeature<"xcvbi", "HasVendorXCVbi", "true",
                       "'XCVbi' (CORE-V Immediate Branching)">;
def HasVendorXCVbi : Predicate<"Subtarget->hasVendorXCVbi()">,
                               AssemblerPredicate<(all_of FeatureVendorXCVbi),
                               "'XCVbi' (CORE-V Immediate Branching)">;

//===----------------------------------------------------------------------===//
// LLVM specific features and extensions
//===----------------------------------------------------------------------===//

// Feature32Bit exists to mark CPUs that support RV32 to distinquish them from
// tuning CPU names.
def Feature32Bit
    : SubtargetFeature<"32bit", "IsRV32", "true", "Implements RV32">;

#ifdef SIFIVE_CUSTOMIZATION
def FeatureStdExtSscofpmf
    : SubtargetFeature<"sscofpmf", "HasStdExtSscofpmf", "true",
                       "'Sscofpmf' (Count Overflow and Mode-Based Filtering Architecture extension)">;
def HasStdExtSscofpmf : Predicate<"Subtarget->hasStdExtSscofpmf()">,
                                  AssemblerPredicate<(all_of FeatureStdExtSscofpmf),
                                  "'Sscofpmf' (Count Overflow and Mode-Based Filtering Architecture extension)">;

def FeatureStdExtZicclsm
    : SubtargetFeature<"zicclsm", "HasStdExtZicclsm", "true",
                       "Zicclsm (Misaligned loads and stores to main memory regions with both the cacheability and coherence PMAs.)">;

def FeatureStdExtZiccif
    : SubtargetFeature<"ziccif", "HasStdExtZiccif", "true",
                       "Ziccif (Main memory regions with both the cacheability and coherence PMAs must support instruction fetch.)">;

def FeatureStdExtZiccamoa
    : SubtargetFeature<"ziccamoa", "HasStdExtZiccamoa", "true",
                       "Ziccamoa (Main memory regions with both the cacheability and coherence PMAs must support AMOArithmetic.)">;

def FeatureStdExtZiccrse
    : SubtargetFeature<"ziccrse", "HasStdExtZiccrse", "true",
                       "Ziccrse (Main memory regions with both the cacheability and coherence PMAs must support RsrvEventual.)">;

def FeatureStdExtZa64rs
    : SubtargetFeature<"za64rs", "HasStdExtZa64rs", "true",
                       "Za64rs (Reservation sets are contiguous, naturally aligned, and a maximum of 64 bytes.)">;

def FeatureStdExtZic64b
    : SubtargetFeature<"zic64b", "HasStdExtZic64b", "true",
                       "Zic64b (Cache blocks must be 64 bytes in size, naturally aligned in the address space.)">;

def FeatureRecodeDotprod
    : SubtargetFeature<"dotprod", "HasRecodeDotprod", "true",
                       "dotprod (A corresponding feature for NEON dotprod.)">;

def FeatureRecodeFullfp16
    : SubtargetFeature<"fullfp16", "HasRecodeFullfp16", "true",
                       "fullfp16 (A corresponding feature for NEON fullfp16.)",
                       [FeatureStdExtZfh, FeatureStdExtZvfh]>;

def HasStdExtZvknhaOrZvknhb  : Predicate<"Subtarget->hasStdExtZvknha() || Subtarget->hasStdExtZvknhb()">,
                                AssemblerPredicate<(any_of FeatureStdExtZvknha, FeatureStdExtZvknhb),
                                "'Zvknha' (Vector SHA-2. (SHA-256 only)) or "
                                "'Zvknhb' (Vector SHA-2. (SHA-256 and SHA-512))">;

#endif // SIFIVE_CUSTOMIZATION

def Feature64Bit
    : SubtargetFeature<"64bit", "IsRV64", "true", "Implements RV64">;
def IsRV64 : Predicate<"Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of Feature64Bit),
                                         "RV64I Base Instruction Set">;
def IsRV32 : Predicate<"!Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of (not Feature64Bit)),
                                          "RV32I Base Instruction Set">;

defvar RV32 = DefaultMode;
def RV64           : HwMode<"+64bit", [IsRV64]>;

def FeatureRVE
    : SubtargetFeature<"e", "IsRVE", "true",
                       "Implements RV{32,64}E (provides 16 rather than 32 GPRs)">;
def IsRVE : Predicate<"Subtarget->isRVE()">,
                        AssemblerPredicate<(all_of FeatureRVE)>;

def FeatureRelax
    : SubtargetFeature<"relax", "EnableLinkerRelax", "true",
                       "Enable Linker relaxation.">;

foreach i = {1-31} in
  def FeatureReserveX#i :
      SubtargetFeature<"reserve-x"#i, "UserReservedRegister[RISCV::X"#i#"]",
                       "true", "Reserve X"#i>;

def FeatureSaveRestore : SubtargetFeature<"save-restore", "EnableSaveRestore",
                                          "true", "Enable save/restore.">;

def FeatureTrailingSeqCstFence : SubtargetFeature<"seq-cst-trailing-fence",
                                          "EnableSeqCstTrailingFence",
                                          "true",
                                          "Enable trailing fence for seq-cst store.">;

def FeatureUnalignedScalarMem
   : SubtargetFeature<"unaligned-scalar-mem", "EnableUnalignedScalarMem",
                      "true", "Has reasonably performant unaligned scalar "
                      "loads and stores">;

def FeatureUnalignedVectorMem
   : SubtargetFeature<"unaligned-vector-mem", "EnableUnalignedVectorMem",
                      "true", "Has reasonably performant unaligned vector "
                      "loads and stores">;

def TuneNoOptimizedZeroStrideLoad
   : SubtargetFeature<"no-optimized-zero-stride-load", "HasOptimizedZeroStrideLoad",
                      "false", "Hasn't optimized (perform fewer memory operations)"
                      "zero-stride vector load">;

// Some vector hardware implementations do not process all VLEN bits in parallel
// and instead split over multiple cycles. DLEN refers to the datapath width
// that can be done in parallel.
def TuneDLenFactor2
   : SubtargetFeature<"dlen-factor-2", "DLenFactor2", "true",
                      "Vector unit DLEN(data path width) is half of VLEN">;

def TuneLUIADDIFusion
    : SubtargetFeature<"lui-addi-fusion", "HasLUIADDIFusion",
                       "true", "Enable LUI+ADDI macrofusion">;

def TuneNoDefaultUnroll
    : SubtargetFeature<"no-default-unroll", "EnableDefaultUnroll", "false",
                       "Disable default unroll preference.">;

// SiFive 7 is able to fuse integer ALU operations with a preceding branch
// instruction.
def TuneShortForwardBranchOpt
    : SubtargetFeature<"short-forward-branch-opt", "HasShortForwardBranchOpt",
                       "true", "Enable short forward branch optimization">;
def HasShortForwardBranchOpt : Predicate<"Subtarget->hasShortForwardBranchOpt()">;
def NoShortForwardBranchOpt : Predicate<"!Subtarget->hasShortForwardBranchOpt()">;

#ifdef SIFIVE_CUSTOMIZATION
def TuneCMOVBranchOpt
    : SubtargetFeature<"cmov-branch-opt", "HasCMOVBranchOpt",
                       "true", "Enable CMOV branch optimization">;
def CanUseCMOVBranchOpt : Predicate<"Subtarget->canUseCMOVBranchOpt()">;
def NoCMOVBranchOpt : Predicate<"!Subtarget->hasCMOVBranchOpt()">;

def TuneJumpIsCheap
    : SubtargetFeature<"jump-is-cheap", "SetJumpIsCheap",
                       "true", "Set jump is cheap.">;

def TuneFuseLUILoad
    : SubtargetFeature<"fuse-lui-load", "HasFuseLUILoad",
                       "true", "CPU fuses LUI/AUIPC with load">;

def TuneFuseIndexedLoad
    : SubtargetFeature<"fuse-indexed-load", "HasFuseIndexedLoad",
                       "true", "CPU fuses indexed load">;
def TuneFuseZbaLoad
    : SubtargetFeature<"fuse-zba-load", "HasFuseZbaLoad",
                       "true", "CPU fuses Zba instructions with load",
                       [TuneFuseIndexedLoad]>;

def TuneFuseArithEqZ
    : SubtargetFeature<"fuse-arith-eqz", "HasFuseArithEqZ",
                       "true", "CPU fuses arithmetic with seqz/snez">;

def TuneFuseBFX
    : SubtargetFeature<"fuse-bitfield-extract", "HasFuseBFX",
                       "true", "CPU fuses SLLI with SRLI/SRAI to extract "
                       "bitfield">;
def HasFuseBFX : Predicate<"Subtarget->hasFuseBFX()">;

// P670 requires a S2V transfer buffer to move scalars into vectors.
// FIXME: Forming .vx/.vf can reduce register pressure.
def TuneDontSinkSplatOperands
    : SubtargetFeature<"dont-sink-splat-operands", "DontSinkSplatOperands",
                       "true", "Don't sink splat operands to enable .vx or .vf "
                       "instructions">;

def TunePseudoLIsimm32
    : SubtargetFeature<"pseudo-li-simm32", "UsePseudoLIsimm32",
                       "true", "Enable treating LUI+ADDI(W) as a single macro instruction in some passes">;

def TuneDLen128b : SubtargetFeature<"dlen128b", "DLen", "128",
                                    "'DLen' (SiFive datapath length) 128">;

def TuneDLen256b : SubtargetFeature<"dlen256b", "DLen", "256",
                                    "'DLen' (SiFive datapath length) 256">;

def TuneOptimizedVRGatherVCompress
    : SubtargetFeature<"optimized-vrgather-vrcompress",
                       "HasOptimizedVRGatherVRCompress", "true",
                       "VRGather and VRCompress are optimized for single DLEN">;

def TuneSiFive6 : SubtargetFeature<"sifive6", "RISCVProcFamily", "SiFive6",
                                   "SiFive 6-Series processors",
                                   [TuneNoDefaultUnroll,
                                    TuneShortForwardBranchOpt,
                                    TuneJumpIsCheap]>;
def TuneSiFive7 : SubtargetFeature<"sifive7", "RISCVProcFamily", "SiFive7",
                                   "SiFive 7-Series processors",
                                   [TuneNoDefaultUnroll,
                                    TuneShortForwardBranchOpt,
                                    TuneJumpIsCheap]>;
def TuneSiFiveP400 : SubtargetFeature<"sifive-p400", "RISCVProcFamily", "SiFiveP400",
                                      "SiFive P400-Series processors",
                                      [TuneNoDefaultUnroll,
                                       TuneCMOVBranchOpt,
                                       TuneJumpIsCheap,
                                       TuneLUIADDIFusion,
                                       TuneDontSinkSplatOperands]>;
def TuneSiFiveP500 : SubtargetFeature<"sifive-p500", "RISCVProcFamily", "SiFiveP500",
                                      "SiFive P500-Series processors",
                                      [TuneNoDefaultUnroll,
                                       TuneCMOVBranchOpt,
                                       TuneJumpIsCheap,
                                       TuneLUIADDIFusion,
                                       TuneDontSinkSplatOperands]>;
def TuneSiFiveP600 : SubtargetFeature<"sifive-p600", "RISCVProcFamily", "SiFiveP600",
                                      "SiFive P600-Series processors",
                                      [TuneNoDefaultUnroll,
                                       TuneCMOVBranchOpt,
                                       TuneJumpIsCheap,
                                       TuneLUIADDIFusion,
                                       TuneDontSinkSplatOperands]>;

def TuneSiFiveP800 : SubtargetFeature<"sifive-p800", "RISCVProcFamily", "SiFiveP800",
                                      "SiFive P800-Series processors",
                                      [TuneNoDefaultUnroll,
                                       TuneCMOVBranchOpt,
                                       TuneJumpIsCheap,
                                       TuneLUIADDIFusion,
                                       TuneDontSinkSplatOperands]>;
#endif // SIFIVE_CUSTOMIZATION

// Assume that lock-free native-width atomics are available, even if the target
// and operating system combination would not usually provide them. The user
// is responsible for providing any necessary __sync implementations. Code
// built with this feature is not ABI-compatible with code built without this
// feature, if atomic variables are exposed across the ABI boundary.
def FeatureForcedAtomics : SubtargetFeature<
    "forced-atomics", "HasForcedAtomics", "true",
    "Assume that lock-free native-width atomics are available">;
def HasAtomicLdSt
    : Predicate<"Subtarget->hasStdExtA() || Subtarget->hasForcedAtomics()">;

def FeatureTaggedGlobals : SubtargetFeature<"tagged-globals",
    "AllowTaggedGlobals",
    "true", "Use an instruction sequence for taking the address of a global "
    "that allows a memory tag in the upper address bits">;
