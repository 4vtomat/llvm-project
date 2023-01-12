//===-- RISCV.td - Describe the RISCV Target Machine -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// RISC-V subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

def FeatureStdExtZicsr
    : SubtargetFeature<"zicsr", "HasStdExtZicsr", "true",
                       "'Zicsr' (Control and Status Register (CSR) Instructions)">;
def HasStdExtZicsr : Predicate<"Subtarget->hasStdExtZicsr()">,
                             AssemblerPredicate<(all_of FeatureStdExtZicsr),
                             "'Zicsr' (Control and Status Register (CSR) Instructions)">;

def FeatureStdExtZifencei
    : SubtargetFeature<"zifencei", "HasStdExtZifencei", "true",
                       "'Zifencei' (Instruction-Fetch Fence)">;
def HasStdExtZifencei : Predicate<"Subtarget->hasStdExtZifencei()">,
                             AssemblerPredicate<(all_of FeatureStdExtZifencei),
                             "'Zifencei' (Instruction-Fetch Fence)">;

#ifdef SIFIVE_CUSTOMIZATION
def FeatureStdExtZicntr
    : SubtargetFeature<"zicntr", "HasStdExtZicntr", "true",
                       "'Zicntr' (Base Counters and Timers)",
                       [FeatureStdExtZicsr]>;
def HasStdExtZicntr : Predicate<"Subtarget->hasStdExtZicntr()">,
                                AssemblerPredicate<(all_of FeatureStdExtZicntr),
                                "'Zicntr' (Base Counters and Timers)">;

def FeatureStdExtZihpm
    : SubtargetFeature<"zihpm", "HasStdExtZihpm", "true",
                       "'Zihpm' (Hardware Performance Counters)",
                       [FeatureStdExtZicsr]>;
def HasStdExtZihpm : Predicate<"Subtarget->hasStdExtZihpm()">,
                                AssemblerPredicate<(all_of FeatureStdExtZihpm),
                                "'Zihpm' (Hardware Performance Counters)">;

def FeatureStdExtSs
    : SubtargetFeature<"ss", "HasStdExtSs", "true",
                       "'Ss' (Supervisor Architecture)",
                       []>;

def FeatureStdExtSvbare
    : SubtargetFeature<"svbare", "HasStdExtSvbare", "true",
                       "'Svbare'",
                       []>;

def FeatureStdExtSsptead
    : SubtargetFeature<"ssptead", "HasStdExtSsptead", "true",
                       "'Ssptead' (Page-fault exceptions are raised when a page "
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

def FeatureStdExtZihintpause
    : SubtargetFeature<"zihintpause", "HasStdExtZihintpause", "true",
                       "'zihintpause' (Pause Hint)">;
def HasStdExtZihintpause : Predicate<"Subtarget->hasStdExtZihintpause()">,
                                     AssemblerPredicate<(all_of FeatureStdExtZihintpause),
                                     "'Zihintpause' (Pause Hint)">;

def FeatureStdExtZihintntl
    : SubtargetFeature<"experimental-zihintntl", "HasStdExtZihintntl", "true",
                       "'zihintntl' (Non-Temporal Locality Hints)">;
def HasStdExtZihintntl : Predicate<"Subtarget->hasStdExtZihintntl()">,
                                    AssemblerPredicate<(all_of FeatureStdExtZihintntl),
                                    "'Zihintntl' (Non-Temporal Locality Hints)">;

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
    : Predicate<"Subtarget->hasStdExtZfh() || Subtarget->hasStdExtZfhmin()">,
                AssemblerPredicate<(any_of FeatureStdExtZfh, FeatureStdExtZfhmin),
                                   "'Zfh' (Half-Precision Floating-Point) or "
                                   "'Zfhmin' (Half-Precision Floating-Point Minimal)">;

def FeatureStdExtZfinx
    : SubtargetFeature<"zfinx", "HasStdExtZfinx", "true",
                       "'Zfinx' (Float in Integer)">;
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

def HasStdExtZhinxOrZhinxmin
    : Predicate<"Subtarget->hasStdExtZhinx() || Subtarget->hasStdExtZhinxmin()">,
                AssemblerPredicate<(any_of FeatureStdExtZhinx, FeatureStdExtZhinxmin),
                                   "'Zhinx' (Half Float in Integer) or "
                                   "'Zhinxmin' (Half Float in Integer Minimal)">;

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

def FeatureExtZca
    : SubtargetFeature<"experimental-zca", "HasStdExtZca", "true",
                       "'Zca' (part of the C extension, excluding compressed "
                       "floating point loads/stores)">;

def HasStdExtCOrZca
    : Predicate<"Subtarget->hasStdExtC() || Subtarget->hasStdExtZca()">,
                AssemblerPredicate<(any_of FeatureStdExtC, FeatureExtZca),
                                   "'C' (Compressed Instructions) or "
                                   "'Zca' (part of the C extension, excluding "
                                   "compressed floating point loads/stores)">;

def FeatureExtZcd
    : SubtargetFeature<"experimental-zcd", "HasStdExtZcd", "true",
                       "'Zcd' (Compressed Double-Precision Floating-Point Instructions)">;

def HasStdExtCOrZcd
    : Predicate<"Subtarget->hasStdExtC() || Subtarget->hasStdExtZcd()">,
                AssemblerPredicate<(any_of FeatureStdExtC, FeatureExtZcd),
                                   "'C' (Compressed Instructions) or "
                                   "'Zcd' (Compressed Double-Precision Floating-Point Instructions)">;

def FeatureExtZcf
    : SubtargetFeature<"experimental-zcf", "HasStdExtZcf", "true",
                       "'Zcf' (Compressed Single-Precision Floating-Point Instructions)">;

def HasStdExtCOrZcf
    : Predicate<"Subtarget->hasStdExtC() || Subtarget->hasStdExtZcf()">,
                AssemblerPredicate<(any_of FeatureStdExtC, FeatureExtZcf),
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

foreach i = { 6-15 } in {
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
                       [FeatureStdExtZicsr, FeatureStdExtZvl32b]>; // SIFIVE

def FeatureStdExtZve32f
    : SubtargetFeature<"zve32f", "HasStdExtZve32f", "true",
                       "'Zve32f' (Vector Extensions for Embedded Processors "
                       "with maximal 32 EEW and F extension)",
                       [FeatureStdExtZve32x]>;

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
                       [FeatureStdExtZve64f]>;

def FeatureStdExtV
    : SubtargetFeature<"v", "HasStdExtV", "true",
                       "'V' (Vector Extension for Application Processors)",
                       [FeatureStdExtZvl128b, FeatureStdExtZve64d,
                        FeatureStdExtF, FeatureStdExtD]>;

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

def FeatureStdExtZvfh
    : SubtargetFeature<"experimental-zvfh", "HasStdExtZvfh", "true",
                       "'Zvfh' (Vector Half-Precision Floating-Point)",
                       [FeatureStdExtZve32f]>;

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

def FeatureStdExtZawrs
    : SubtargetFeature<"experimental-zawrs", "HasStdExtZawrs", "true",
                       "'Zawrs' (Wait on Reservation Set)">;
def HasStdExtZawrs : Predicate<"Subtarget->hasStdExtZawrs()">,
                               AssemblerPredicate<(all_of FeatureStdExtZawrs),
                               "'Zawrs' (Wait on Reservation Set)">;

//===----------------------------------------------------------------------===//
// Vendor extensions
//===----------------------------------------------------------------------===//

def FeatureVendorXVentanaCondOps
    : SubtargetFeature<"xventanacondops", "HasVendorXVentanaCondOps", "true",
                       "'XVentanaCondOps' (Ventana Conditional Ops)">;
def HasVendorXVentanaCondOps : Predicate<"Subtarget->hasVendorXVentanaCondOps()">,
                                AssemblerPredicate<(all_of FeatureVendorXVentanaCondOps),
                                "'XVentanaCondOps' (Ventana Conditional Ops)">;

def FeatureVendorXTHeadVdot
    : SubtargetFeature<"xtheadvdot", "HasVendorXTHeadVdot", "true",
                       "'xtheadvdot' (T-Head Vector Extensions for Dot)",
                       [FeatureStdExtV]>;
def HasVendorXTHeadVdot : Predicate<"Subtarget->hasVendorXTHeadVdot()">,
                                    AssemblerPredicate<(all_of FeatureVendorXTHeadVdot),
                                    "'xtheadvdot' (T-Head Vector Extensions for Dot)">;

//===----------------------------------------------------------------------===//
// LLVM specific features and extensions
//===----------------------------------------------------------------------===//

// Feature32Bit exists to mark CPUs that support RV32 to distinquish them from
// tuning CPU names.
def Feature32Bit
    : SubtargetFeature<"32bit", "HasRV32", "true", "Implements RV32">;

#ifdef SIFIVE_CUSTOMIZATION
def FeatureStdExtSscofpmf
    : SubtargetFeature<"sscofpmf", "HasStdExtSscofpmf", "true",
                       "'Sscofpmf' (Count Overflow and Mode-Based Filtering Architecture extension)">;
def HasStdExtSscofpmf : Predicate<"Subtarget->hasStdExtSscofpmf()">,
                                  AssemblerPredicate<(all_of FeatureStdExtSscofpmf),
                                  "'Sscofpmf' (Count Overflow and Mode-Based Filtering Architecture extension)">;

def FeatureStdExtXsfvqmaccqoq
    : SubtargetFeature<"xsfvqmaccqoq", "HasExtXsfvqmaccqoq", "true",
                       "Xsfvqmaccqoq (SiFive Custom int8 Matrix-Multiply Instructions for DLEN >= 256)",
                       [FeatureStdExtZve32x, FeatureStdExtZvl256b]>;
def HasExtXsfvqmaccqoq : Predicate<"Subtarget->hasExtXsfvqmaccqoq()">,
                         AssemblerPredicate<(all_of FeatureStdExtXsfvqmaccqoq),
                         "'Xsfvqmaccqoq' (SiFive Custom int8 Matrix-Multiply Instructions for DLEN >= 256)">;
def FeatureStdExtXsfvqmaccdod
    : SubtargetFeature<"xsfvqmaccdod", "HasExtXsfvqmaccdod", "true",
                       "Xsfvqmaccdod (SiFive Custom int8 Matrix-Multiply Instructions for DLEN >= 128)",
                       [FeatureStdExtZve32x, FeatureStdExtZvl128b]>;
def HasExtXsfvqmaccdod : Predicate<"Subtarget->hasExtXsfvqmaccdod()">,
                         AssemblerPredicate<(all_of FeatureStdExtXsfvqmaccdod),
                         "'Xsfvqmaccdod' (SiFive Custom int8 Matrix-Multiply Instructions for DLEN >= 128)">;

def FeatureStdExtXsfvfhbfmin
    : SubtargetFeature<"xsfvfhbfmin", "HasExtXsfvfhbfmin", "true",
                       "Xsfvfhbfmin (SiFive custom minimal BF16 vector support)",
                       [FeatureStdExtZve32f]>;
def HasExtXsfvfhbfmin : Predicate<"Subtarget->hasExtXsfvfhbfmin()">,
                         AssemblerPredicate<(all_of FeatureStdExtXsfvfhbfmin),
                         "'Xsfvfhbfmin' (SiFive custom minimal BF16 vector support)">;

def FeatureStdExtXsfvfwmaccqqq
    : SubtargetFeature<"xsfvfwmaccqqq", "HasExtXsfvfwmaccqqq", "true",
                       "Xsfvfwmaccqqq (Support SiFive custom BF16 matrix arithmetic vector instructions)",
                       [FeatureStdExtZve32f, FeatureStdExtZvl256b]>;
def HasExtXsfvfwmaccqqq : Predicate<"Subtarget->hasExtXsfvfwmaccqqq()">,
                          AssemblerPredicate<(all_of FeatureStdExtXsfvfwmaccqqq),
                          "'Xsfvfwmaccqqq' (SiFive custom BF16 matrix arithmetic vector instructions)">;

def FeatureStdExtXsfvfnrclipxfqf
    : SubtargetFeature<"xsfvfnrclipxfqf", "HasExtXsfvfnrclipxfqf", "true",
                       "Xsfvfnrclipxfqf (Support SiFive custom FP32-to-int8 ranged clip instructions)",
                       [FeatureStdExtZve32f]>;
def HasExtXsfvfnrclipxfqf : Predicate<"Subtarget->hasExtXsfvfnrclipxfqf()">,
                            AssemblerPredicate<(all_of FeatureStdExtXsfvfnrclipxfqf),
                            "'Xsfvfnrclipxfqf' (SiFive custom FP32-to-int8 ranged clip instructions)">;

def FeatureStdExtXsfvcp
    : SubtargetFeature<"xsfvcp", "HasExtXsfvcp", "true",
                       "Xsfvcp (Support SiFive custom vector coprocessor interface instructions)",
                       []>; // FIXME: Need which implication?
def HasExtXsfvcp : Predicate<"Subtarget->hasExtXsfvcp()">,
                             AssemblerPredicate<(all_of FeatureStdExtXsfvcp),
                             "'Xsfvcp' (SiFive custom vector coprocessor interface instructions)">;

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

def FeatureStdExtZvkb
    : SubtargetFeature<"experimental-zvkb", "HasStdExtZvkb", "true",
                       "'Zvkb' (Vector Bitmanip instructions for Cryptography.)">;
def HasStdExtZvkb : Predicate<"Subtarget->hasStdExtZvkb()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvkb),
                                "'Zvkb' (Vector Bitmanip instructions for Cryptography.)">;

def FeatureStdExtZvkg
    : SubtargetFeature<"experimental-zvkg", "HasStdExtZvkg", "true",
                       "'Zvkg' (Vector GCM instructions for Cryptography.)">;
def HasStdExtZvkg : Predicate<"Subtarget->hasStdExtZvkg()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvkg),
                                "'Zvkg' (Vector GCM instructions for Cryptography.)">;

def FeatureStdExtZvknha
    : SubtargetFeature<"experimental-zvknha", "HasStdExtZvknha", "true",
                       "'Zvknha' (Vector SHA-2. (SHA-256 only))">;

def FeatureStdExtZvknhb
    : SubtargetFeature<"experimental-zvknhb", "HasStdExtZvknhb", "true",
                       "'Zvknhb' (Vector SHA-2. (SHA-256 and SHA-512))",
		       [FeatureStdExtZvknha]>;
def HasStdExtZvknhaOrZvknhb  : Predicate<"Subtarget->hasStdExtZvknha() || Subtarget->hasStdExtZvknhb()">,
                                AssemblerPredicate<(any_of FeatureStdExtZvknha, FeatureStdExtZvknhb),
                                "'Zvknha' (Vector SHA-2. (SHA-256 only)) or "
                                "'Zvknhb' (Vector SHA-2. (SHA-256 and SHA-512))">;

def FeatureStdExtZvkns
    : SubtargetFeature<"experimental-zvkns", "HasStdExtZvkns", "true",
                       "'Zvkns' (Vector AES Encryption & Decryption (Single Round))">;
def HasStdExtZvkns : Predicate<"Subtarget->hasStdExtZvkns()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvkns),
                                "'Zvkns' (Vector AES Encryption & Decryption (Single Round))">;

def FeatureStdExtZvksed
    : SubtargetFeature<"experimental-zvksed", "HasStdExtZvksed", "true",
                       "'Zvksed' (SM4 Block Cipher Instructions.)">;
def HasStdExtZvksed : Predicate<"Subtarget->hasStdExtZvksed()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvksed),
                                "'Zvksed' (SM4 Block Cipher Instructions.)">;

def FeatureStdExtZvksh
    : SubtargetFeature<"experimental-zvksh", "HasStdExtZvksh", "true",
                       "'Zvksh' (SM3 Hash Function Instructions.)">;
def HasStdExtZvksh : Predicate<"Subtarget->hasStdExtZvksh()">,
                                AssemblerPredicate<(all_of FeatureStdExtZvksh),
                                "'Zvksh' (SM3 Hash Function Instructions.)">;
#endif // SIFIVE_CUSTOMIZATION

def Feature64Bit
    : SubtargetFeature<"64bit", "HasRV64", "true", "Implements RV64">;
def IsRV64 : Predicate<"Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of Feature64Bit),
                                         "RV64I Base Instruction Set">;
def IsRV32 : Predicate<"!Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of (not Feature64Bit)),
                                          "RV32I Base Instruction Set">;

defvar RV32 = DefaultMode;
def RV64           : HwMode<"+64bit">;

def FeatureRV32E
    : SubtargetFeature<"e", "IsRV32E", "true",
                       "Implements RV32E (provides 16 rather than 32 GPRs)">;
def IsRV32E : Predicate<"Subtarget->isRV32E()">,
                        AssemblerPredicate<(all_of FeatureRV32E)>;

def FeatureRelax
    : SubtargetFeature<"relax", "EnableLinkerRelax", "true",
                       "Enable Linker relaxation.">;

foreach i = {1-31} in
  def FeatureReserveX#i :
      SubtargetFeature<"reserve-x"#i, "UserReservedRegister[RISCV::X"#i#"]",
                       "true", "Reserve X"#i>;

def FeatureSaveRestore : SubtargetFeature<"save-restore", "EnableSaveRestore",
                                          "true", "Enable save/restore.">;

def FeatureUnalignedScalarMem
   : SubtargetFeature<"unaligned-scalar-mem", "EnableUnalignedScalarMem",
                      "true", "Has reasonably performant unaligned scalar "
                      "loads and stores">;

def TuneNoOptimizedZeroStrideLoad
   : SubtargetFeature<"no-optimized-zero-stride-load", "HasOptimizedZeroStrideLoad",
                      "false", "Hasn't optimized (perform fewer memory operations)"
                      "zero-stride vector load">;

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

//===----------------------------------------------------------------------===//
// Named operands for CSR instructions.
//===----------------------------------------------------------------------===//

include "RISCVSystemOperands.td"

//===----------------------------------------------------------------------===//
// Registers, calling conventions, instruction descriptions.
//===----------------------------------------------------------------------===//

include "RISCVSchedule.td"
include "RISCVRegisterInfo.td"
include "RISCVCallingConv.td"
include "RISCVInstrInfo.td"
include "GISel/RISCVRegisterBanks.td"
include "RISCVSchedRocket.td"
include "RISCVSchedSiFive7.td"
include "RISCVSchedSyntacoreSCR1.td"
#ifdef SIFIVE_CUSTOMIZATION
include "RISCVSchedSiFive6.td"
include "RISCVSchedSiFive7N.td"
include "SiFive_RISCVSchedSiFiveP400.td"
include "SiFive_RISCVSchedSiFiveP500.td"
include "SiFive_RISCVSchedSiFiveP600.td"
#endif // SIFIVE_CUSTOMIZATION

//===----------------------------------------------------------------------===//
// RISC-V processors supported.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"generic-rv32", NoSchedModel, [Feature32Bit]>;
def : ProcessorModel<"generic-rv64", NoSchedModel, [Feature64Bit]>;
// Support generic for compatibility with other targets. The triple will be used
// to change to the appropriate rv32/rv64 version.
def : ProcessorModel<"generic", NoSchedModel, []>;

def : ProcessorModel<"rocket-rv32", RocketModel, [Feature32Bit]>;
def : ProcessorModel<"rocket-rv64", RocketModel, [Feature64Bit]>;
def : ProcessorModel<"rocket", RocketModel, []>;

def : ProcessorModel<"sifive-7-series", SiFive7Model, [],
                     [TuneSiFive7]>;

#ifdef SIFIVE_CUSTOMIZATION

def : ProcessorModel<"sifive-6-series", SiFive6Model, [],
                     [TuneSiFive6]>;

def : ProcessorModel<"sifive-7m-series", SiFive7Model, [],
                     [TuneSiFive7, TunePseudoLIsimm32]>;
def : ProcessorModel<"sifive-7n-series", SiFive7NModel, [],
                     [TuneSiFive7, TunePseudoLIsimm32]>;
def : ProcessorModel<"sifive-7o-series", SiFive7NModel, [],
                     [TuneSiFive7, TunePseudoLIsimm32]>;

def : ProcessorModel<"sifive-p400-series", SiFiveP400Model, [],
                     [TuneSiFiveP400]>;

def : ProcessorModel<"sifive-8-series", SiFiveP500Model, [],
                     [TuneSiFiveP500]>;
def : ProcessorModel<"sifive-p500-series", SiFiveP500Model, [],
                     [TuneSiFiveP500]>;

def : ProcessorModel<"sifive-p600-series", SiFiveP600Model, [],
                     [TuneSiFiveP600]>;
#endif // SIFIVE_CUSTOMIZATION

def : ProcessorModel<"sifive-e20", RocketModel, [Feature32Bit,
                                                 FeatureStdExtM,
                                                 FeatureStdExtC]>;

def : ProcessorModel<"sifive-e21", RocketModel, [Feature32Bit,
                                                 FeatureStdExtM,
                                                 FeatureStdExtA,
                                                 FeatureStdExtC]>;

def : ProcessorModel<"sifive-e24", RocketModel, [Feature32Bit,
                                                 FeatureStdExtM,
                                                 FeatureStdExtA,
                                                 FeatureStdExtF,
                                                 FeatureStdExtC]>;

def : ProcessorModel<"sifive-e31", RocketModel, [Feature32Bit,
                                                 FeatureStdExtM,
                                                 FeatureStdExtA,
                                                 FeatureStdExtC]>;

def : ProcessorModel<"sifive-e34", RocketModel, [Feature32Bit,
                                                 FeatureStdExtM,
                                                 FeatureStdExtA,
                                                 FeatureStdExtF,
                                                 FeatureStdExtC]>;

def : ProcessorModel<"sifive-e76", SiFive7Model, [Feature32Bit,
                                                  FeatureStdExtM,
                                                  FeatureStdExtA,
                                                  FeatureStdExtF,
                                                  FeatureStdExtC],
                     [TuneSiFive7]>;

#ifdef SIFIVE_CUSTOMIZATION
def : ProcessorModel<"sifive-e66", SiFive6Model, [Feature32Bit,
                                                  FeatureStdExtM,
                                                  FeatureStdExtA,
                                                  FeatureStdExtF,
                                                  FeatureStdExtC,
                                                  FeatureStdExtZba,
                                                  FeatureStdExtZbb],
                     [TuneSiFive6]>;

def : ProcessorModel<"sifive-e76m", SiFive7Model, [Feature32Bit,
                                                   FeatureStdExtM,
                                                   FeatureStdExtA,
                                                   FeatureStdExtF,
                                                   FeatureStdExtC,
                                                   FeatureStdExtZba,
                                                   FeatureStdExtZbb],
                     [TuneSiFive7]>;

def : ProcessorModel<"sifive-e76n", SiFive7NModel, [Feature32Bit,
                                                    FeatureStdExtM,
                                                    FeatureStdExtA,
                                                    FeatureStdExtF,
                                                    FeatureStdExtC,
                                                    FeatureStdExtZba,
                                                    FeatureStdExtZbb],
                     [TuneSiFive7, TunePseudoLIsimm32]>;
#endif // SIFIVE_CUSTOMIZATION

def : ProcessorModel<"sifive-s21", RocketModel, [Feature64Bit,
                                                 FeatureStdExtM,
                                                 FeatureStdExtA,
                                                 FeatureStdExtC]>;

def : ProcessorModel<"sifive-s51", RocketModel, [Feature64Bit,
                                                 FeatureStdExtM,
                                                 FeatureStdExtA,
                                                 FeatureStdExtC]>;

def : ProcessorModel<"sifive-s54", RocketModel, [Feature64Bit,
                                                 FeatureStdExtM,
                                                 FeatureStdExtA,
                                                 FeatureStdExtF,
                                                 FeatureStdExtD,
                                                 FeatureStdExtC]>;

def : ProcessorModel<"sifive-s76", SiFive7Model, [Feature64Bit,
                                                  FeatureStdExtM,
                                                  FeatureStdExtA,
                                                  FeatureStdExtF,
                                                  FeatureStdExtD,
                                                  FeatureStdExtC],
                     [TuneSiFive7]>;

#ifdef SIFIVE_CUSTOMIZATION
def : ProcessorModel<"sifive-s66", SiFive6Model, [Feature64Bit,
                                                  FeatureStdExtM,
                                                  FeatureStdExtA,
                                                  FeatureStdExtF,
                                                  FeatureStdExtD,
                                                  FeatureStdExtC,
                                                  FeatureStdExtZba,
                                                  FeatureStdExtZbb],
                     [TuneSiFive6]>;


def : ProcessorModel<"sifive-s76m", SiFive7Model, [Feature64Bit,
                                                   FeatureStdExtM,
                                                   FeatureStdExtA,
                                                   FeatureStdExtF,
                                                   FeatureStdExtD,
                                                   FeatureStdExtC,
                                                   FeatureStdExtZba,
                                                   FeatureStdExtZbb],
                     [TuneSiFive7]>;

def : ProcessorModel<"sifive-s76n", SiFive7NModel, [Feature64Bit,
                                                    FeatureStdExtM,
                                                    FeatureStdExtA,
                                                    FeatureStdExtF,
                                                    FeatureStdExtD,
                                                    FeatureStdExtC,
                                                    FeatureStdExtZba,
                                                    FeatureStdExtZbb],
                     [TuneSiFive7, TunePseudoLIsimm32]>;
#endif // SIFIVE_CUSTOMIZATION

def : ProcessorModel<"sifive-u54", RocketModel, [Feature64Bit,
                                                 FeatureStdExtM,
                                                 FeatureStdExtA,
                                                 FeatureStdExtF,
                                                 FeatureStdExtD,
                                                 FeatureStdExtC]>;

def : ProcessorModel<"sifive-u74", SiFive7Model, [Feature64Bit,
                                                  FeatureStdExtM,
                                                  FeatureStdExtA,
                                                  FeatureStdExtF,
                                                  FeatureStdExtD,
                                                  FeatureStdExtC],
                     [TuneSiFive7]>;

def : ProcessorModel<"syntacore-scr1-base", SyntacoreSCR1Model,
                     [Feature32Bit, FeatureStdExtC],
                     [TuneNoDefaultUnroll]>;

def : ProcessorModel<"syntacore-scr1-max", SyntacoreSCR1Model,
                     [Feature32Bit, FeatureStdExtM, FeatureStdExtC],
                     [TuneNoDefaultUnroll]>;

#ifdef SIFIVE_CUSTOMIZATION
def : ProcessorModel<"sifive-u64", SiFive6Model, [Feature64Bit,
                                                  FeatureStdExtM,
                                                  FeatureStdExtA,
                                                  FeatureStdExtF,
                                                  FeatureStdExtD,
                                                  FeatureStdExtC,
                                                  FeatureStdExtZba,
                                                  FeatureStdExtZbb],
                     [TuneSiFive6]>;

def : ProcessorModel<"sifive-u74m", SiFive7Model, [Feature64Bit,
                                                   FeatureStdExtM,
                                                   FeatureStdExtA,
                                                   FeatureStdExtF,
                                                   FeatureStdExtD,
                                                   FeatureStdExtC,
                                                   FeatureStdExtZba,
                                                   FeatureStdExtZbb],
                     [TuneSiFive7]>;

def : ProcessorModel<"sifive-u74n", SiFive7NModel, [Feature64Bit,
                                                    FeatureStdExtM,
                                                    FeatureStdExtA,
                                                    FeatureStdExtF,
                                                    FeatureStdExtD,
                                                    FeatureStdExtC,
                                                    FeatureStdExtZba,
                                                    FeatureStdExtZbb],
                     [TuneSiFive7, TunePseudoLIsimm32]>;

def : ProcessorModel<"sifive-p270", SiFive7Model, [Feature64Bit,
                                                   FeatureStdExtM,
                                                   FeatureStdExtA,
                                                   FeatureStdExtF,
                                                   FeatureStdExtD,
                                                   FeatureStdExtC,
                                                   FeatureStdExtV,
                                                   FeatureStdExtZvl256b,
                                                   FeatureStdExtZfh,
                                                   FeatureStdExtZvfh,
                                                   FeatureStdExtZba,
                                                   FeatureStdExtZbb],
                     [TuneSiFive7, TuneDLen128b]>;

def : ProcessorModel<"sifive-p270n", SiFive7NModel, [Feature64Bit,
                                                     FeatureStdExtM,
                                                     FeatureStdExtA,
                                                     FeatureStdExtF,
                                                     FeatureStdExtD,
                                                     FeatureStdExtC,
                                                     FeatureStdExtV,
                                                     FeatureStdExtZvl256b,
                                                     FeatureStdExtZfh,
                                                     FeatureStdExtZvfh,
                                                     FeatureStdExtZba,
                                                     FeatureStdExtZbb],
                     [TuneSiFive7, TuneDLen128b, TunePseudoLIsimm32]>;

def : ProcessorModel<"sifive-x280", SiFive7Model, [Feature64Bit,
                                                   FeatureStdExtM,
                                                   FeatureStdExtA,
                                                   FeatureStdExtF,
                                                   FeatureStdExtD,
                                                   FeatureStdExtC,
                                                   FeatureStdExtV,
                                                   FeatureStdExtZvl512b,
                                                   FeatureStdExtZfh,
                                                   FeatureStdExtZvfh,
                                                   FeatureStdExtZba,
                                                   FeatureStdExtZbb,
                                                   FeatureStdExtXsfvqmaccqoq,
                                                   FeatureStdExtXsfvfhbfmin],
                     [TuneSiFive7, TuneDLen256b]>;

def : ProcessorModel<"sifive-x280n", SiFive7NModel, [Feature64Bit,
                                                     FeatureStdExtM,
                                                     FeatureStdExtA,
                                                     FeatureStdExtF,
                                                     FeatureStdExtD,
                                                     FeatureStdExtC,
                                                     FeatureStdExtV,
                                                     FeatureStdExtZvl512b,
                                                     FeatureStdExtZfh,
                                                     FeatureStdExtZvfh,
                                                     FeatureStdExtZba,
                                                     FeatureStdExtZbb,
                                                     FeatureStdExtXsfvfhbfmin,
                                                     FeatureStdExtXsfvqmaccqoq,
                                                     FeatureStdExtXsfvfnrclipxfqf,
                                                     FeatureStdExtXsfvfwmaccqqq],
                     [TuneSiFive7, TuneDLen256b, TunePseudoLIsimm32]>;

def : ProcessorModel<"sifive-x280o", SiFive7NModel, [Feature64Bit,
                                                     FeatureStdExtM,
                                                     FeatureStdExtA,
                                                     FeatureStdExtF,
                                                     FeatureStdExtD,
                                                     FeatureStdExtC,
                                                     FeatureStdExtV,
                                                     FeatureStdExtZvl512b,
                                                     FeatureStdExtZfh,
                                                     FeatureStdExtZvfh,
                                                     FeatureStdExtZba,
                                                     FeatureStdExtZbb,
                                                     FeatureStdExtXsfvfhbfmin,
                                                     FeatureStdExtXsfvqmaccqoq,
                                                     FeatureStdExtXsfvfnrclipxfqf,
                                                     FeatureStdExtXsfvfwmaccqqq],
                     [TuneSiFive7, TuneDLen256b, TunePseudoLIsimm32]>;

// FIXME: This needs its own scheduler model
def : ProcessorModel<"sifive-p450", SiFiveP400Model, [Feature64Bit,
                                                      FeatureStdExtM,
                                                      FeatureStdExtA,
                                                      FeatureStdExtF,
                                                      FeatureStdExtD,
                                                      FeatureStdExtC,
                                                      FeatureStdExtZba,
                                                      FeatureStdExtZbb,
                                                      FeatureStdExtZbs,
                                                      FeatureStdExtZfhmin],
                     [TuneSiFiveP400]>;

// FIXME: This needs its own scheduler model
def : ProcessorModel<"sifive-p470", SiFiveP400Model, [Feature64Bit,
                                                      FeatureStdExtM,
                                                      FeatureStdExtA,
                                                      FeatureStdExtF,
                                                      FeatureStdExtD,
                                                      FeatureStdExtC,
                                                      FeatureStdExtZba,
                                                      FeatureStdExtZbb,
                                                      FeatureStdExtZbs,
                                                      FeatureStdExtZfhmin,
                                                      FeatureStdExtV],
                     [TuneSiFiveP400]>;

def : ProcessorModel<"sifive-p550", SiFiveP500Model, [Feature64Bit,
                                                      FeatureStdExtM,
                                                      FeatureStdExtA,
                                                      FeatureStdExtF,
                                                      FeatureStdExtD,
                                                      FeatureStdExtC,
                                                      FeatureStdExtZba,
                                                      FeatureStdExtZbb],
                     [TuneSiFiveP500]>;

def : ProcessorModel<"sifive-p650", SiFiveP600Model, [Feature64Bit,
                                                      FeatureStdExtM,
                                                      FeatureStdExtA,
                                                      FeatureStdExtF,
                                                      FeatureStdExtD,
                                                      FeatureStdExtC,
                                                      FeatureStdExtZba,
                                                      FeatureStdExtZbb,
                                                      FeatureStdExtZbs,
                                                      FeatureStdExtZfhmin],
                     [TuneSiFiveP600]>;
def : ProcessorModel<"sifive-p670", SiFiveP600Model, [Feature64Bit,
                                                      FeatureStdExtM,
                                                      FeatureStdExtA,
                                                      FeatureStdExtF,
                                                      FeatureStdExtD,
                                                      FeatureStdExtC,
                                                      FeatureStdExtZba,
                                                      FeatureStdExtZbb,
                                                      FeatureStdExtZbs,
                                                      FeatureStdExtZfhmin,
                                                      FeatureStdExtV],
                     [TuneSiFiveP600]>;
#endif // SIFIVE_CUSTOMIZATION

//===----------------------------------------------------------------------===//
// Define the RISC-V target.
//===----------------------------------------------------------------------===//

def RISCVInstrInfo : InstrInfo {
  let guessInstructionProperties = 0;
}

def RISCVAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
  let AllowDuplicateRegisterNames = 1;
}

def RISCVAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def RISCV : Target {
  let InstructionSet = RISCVInstrInfo;
  let AssemblyParsers = [RISCVAsmParser];
  let AssemblyWriters = [RISCVAsmWriter];
  let AllowRegisterRenaming = 1;
}
