//===-- SiFive_RISCVFeatures.td - SiFive Custom Extensions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Vector Extensions

def FeatureStdExtZvfofp8min
    : RISCVExperimentalExtension<0, 2, "OFP8 conversion extension Zvfofp8min",
                                 [FeatureStdExtZve32f]>;
def HasStdExtZvfofp8min
    : Predicate<"Subtarget->hasStdExtZvfofp8min()">,
      AssemblerPredicate<(all_of FeatureStdExtZvfofp8min),
                          "'Zvfofp8min' (OFP8 conversion extension Zvfofp8min)">;
def HasStdExtZvfbfminOrZvfofp8min
    : Predicate<"Subtarget->hasStdExtZvfbfmin() || Subtarget->hasStdExtZvfofp8min()">,
      AssemblerPredicate<(any_of FeatureStdExtZvfbfmin, FeatureStdExtZvfofp8min),
                         "'Zvfbfmin' or 'Zvfofp8min'">;

def FeatureStdExtZvfofp4min
    : RISCVExperimentalExtension<0, 1, "OFP4 conversion extension Zvfofp4min",
                                 [FeatureStdExtZve32f]>;
def HasStdExtZvfofp4min
    : Predicate<"Subtarget->hasStdExtZvfofp4min()">,
      AssemblerPredicate<(all_of FeatureStdExtZvfofp4min),
                          "'Zvfofp4min' (OFP4 conversion extension Zvfofp4min)">;

// Vector Cryptography and Bitmanip Extensions

def FeatureStdExtZvkb0p1
    : SubtargetFeature<"zvkb0p1", "HasStdExtZvkb0p1", "true",
                       "'Zvkb' (Vector Bit-manipulation used in Cryptography)">;
def HasStdExtZvkb0p1 : Predicate<"Subtarget->hasStdExtZvkb0p1()">,
                       AssemblerPredicate<(all_of FeatureStdExtZvkb0p1),
                       "'Zvkb' (Vector Bit-manipulation used in Cryptography)">;
def HasStdExtZvkbOrZvkb0p1
    : Predicate<"Subtarget->hasStdExtZvkb() || Subtarget->hasStdExtZvkb0p1()">,
      AssemblerPredicate<(any_of FeatureStdExtZvkb, FeatureStdExtZvkb0p1),
                         "'Zvkb' (Vector Bit-manipulation used in Cryptography)">;

def HasStdExtZvbcOrZvkb0p1
    : Predicate<"Subtarget->hasStdExtZvbc() || Subtarget->hasStdExtZvkb0p1()">,
      AssemblerPredicate<(any_of FeatureStdExtZvbc, FeatureStdExtZvkb0p1),
                         "'Zvbc' (Vector Carryless Multiplication)">;

def FeatureStdExtZvkg0p1
    : SubtargetFeature<"zvkg0p1", "HasStdExtZvkg0p1", "true",
                       "'Zvkg' (Vector GCM instructions for Cryptography)">;
def HasStdExtZvkg0p1 : Predicate<"Subtarget->hasStdExtZvkg0p1()">,
                       AssemblerPredicate<(all_of FeatureStdExtZvkg0p1),
                       "'Zvkg' (Vector GCM instructions for Cryptography)">;

def FeatureStdExtZvknha0p1
    : SubtargetFeature<"zvknha0p1", "HasStdExtZvknha0p1", "true",
                       "'Zvknha' (Vector SHA-2. (SHA-256 only))">;

def FeatureStdExtZvknhb0p1
    : SubtargetFeature<"zvknhb0p1", "HasStdExtZvknhb0p1", "true",
                       "'Zvknhb' (Vector SHA-2. (SHA-256 and SHA-512))",
                       [FeatureStdExtZvknha0p1]>;
def FeatureStdExtZvkns
    : RISCVExtension<0, 1,
                     "Vector AES Encryption & Decryption (Single Round)">;
def HasStdExtZvknsOrZvkned
    : Predicate<"Subtarget->hasStdExtZvkns() || Subtarget->hasStdExtZvkned()">,
      AssemblerPredicate<(any_of FeatureStdExtZvkns, FeatureStdExtZvkned),
          "'Zvkned' (Vector AES Encryption & Decryption (Single Round))">;

def FeatureStdExtZvksed0p1
    : SubtargetFeature<"zvksed0p1", "HasStdExtZvksed0p1", "true",
                       "'Zvksed' (SM4 Block Cipher Instructions)">;

def FeatureStdExtZvksh0p1
    : SubtargetFeature<"zvksh0p1", "HasStdExtZvksh0p1", "true",
                       "'Zvksh' (SM3 Hash Function Instructions)">;

// Pointer masking extension

def FeatureStdExtZjid
    : RISCVExperimentalExtension<0, 0,
                                 "Instruction/Data Cache Synchronization">;

def HasStdExtZjid : Predicate<"Subtarget->hasStdExtZjid()">,
                    AssemblerPredicate<(all_of FeatureStdExtZjid),
                                        "'Zjid' (Instruction/Data Cache Synchronization)">;

// Supervisor extensions

let AdditionalVersions = [[1, 12]] in
def FeatureStdExtSs
    : RISCVExtension<1, 13,
                     "Supervisor Architecture">;

def FeatureStdExtSmwg
    : RISCVExperimentalExtension<0, 3,
                                 "The Smwg extension adds the mlwid CSR, which is "
                                 "an M-mode read-write CSR, whose least-significant bits "
                                 "set the WID to be used by lower-privilege modes.">;

def FeatureStdExtSmwgd
    : RISCVExperimentalExtension<0, 3,
                                 "The Smwgd extension adds the mwiddeleg M-mode "
                                 "read-write CSR. The mwiddeleg register represents a set "
                                 "of WIDs as a bit vector with WID i represented by bit i "
                                 "of the register.">;

def FeatureStdExtSswg
    : RISCVExperimentalExtension<0, 3,
                                 "The Sswg extension adds the [H]S-mode read-write "
                                 "slwid CSR, which sets the WID used for modes lower than "
                                 "[H]S-mode.">;

// SiFive extensions

def FeatureVendorXSfpgflushdlone
    : RISCVExtension<0, 1,
                     "Cache Flush/Power Down Instructions">;
def HasVendorXSfpgflushdlone
    : Predicate<"Subtarget->hasVendorXSfpgflushdlone()">,
      AssemblerPredicate<(all_of FeatureVendorXSfpgflushdlone),
                         "Cache Flush/Power Down Instructions">;

def FeatureVendorXSfsci
    : RISCVExtension<1, 0,
                     "SiFive Custom Scalar Coprocessor Interface Instructions">;
def HasVendorXSfsci : Predicate<"Subtarget->hasVendorXSfsci()">,
                      AssemblerPredicate<(all_of FeatureVendorXSfsci),
                          "'XSfsci' (SiFive Custom Scalar Coprocessor Interface Instructions)">;

def FeatureVendorXSfpmpmt
    : RISCVExtension<0, 1,
                     "SiFive PMP-based Memory Types Extension">;

def FeatureVendorXSfvfbfexp16e
    : RISCVExtension<0, 5,
                     "SiFive Vector Floating-Point Exponential Function Instruction, BFloat16">;
def HasVendorXSfvfbfexp16e : Predicate<"Subtarget->hasVendorXSfvfbfexp16e()">;

def FeatureVendorXSfvfexp16e
    : RISCVExtension<0, 5,
                     "SiFive Vector Floating-Point Exponential Function Instruction, Half Precision",
                     [FeatureStdExtZvfh]>;
def HasVendorXSfvfexp16e : Predicate<"Subtarget->hasVendorXSfvfexp16e()">;

def FeatureVendorXSfvfexp32e
    : RISCVExtension<0, 5,
                     "SiFive Vector Floating-Point Exponential Function Instruction, Single Precision",
                     [FeatureStdExtZve32f]>;
def HasVendorXSfvfexp32e : Predicate<"Subtarget->hasVendorXSfvfexp32e()">;

def HasVendorXSfvfexpAnyFloat : Predicate<"Subtarget->hasVendorXSfvfexp16e() || Subtarget->hasVendorXSfvfexp32e()">;
def HasVendorXSfvfexpAny : Predicate<"Subtarget->hasVendorXSfvfbfexp16e() || Subtarget->hasVendorXSfvfexp16e() || Subtarget->hasVendorXSfvfexp32e()">,
                           AssemblerPredicate<(any_of FeatureVendorXSfvfbfexp16e, FeatureVendorXSfvfexp16e, FeatureVendorXSfvfexp32e),
                           "'Xsfvfbfexp16e/Xsfvfexp16e/Xsfvfexp32e' (SiFive Vector Floating-Point Exponential Function Instruction)">;

def FeatureVendorXSfvfexpa
    : RISCVExtension<0, 2,
                     "SiFive Vector Floating-Point Exponential Approximation Instruction",
                     [FeatureStdExtZve32f]>;
def HasVendorXSfvfexpa : Predicate<"Subtarget->hasVendorXSfvfexpa()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfvfexpa),
                         "'Xsfvfexpa' (SiFive Vector Floating-Point Exponential Approximation Instruction)">;

def FeatureVendorXSfvfexpa64e
    : RISCVExtension<0, 2,
                     "SiFive Vector Floating-Point Exponential Approximation Instruction with Double-Precision",
                     [FeatureVendorXSfvfexpa, FeatureStdExtZve64d]>;
def HasVendorXSfvfexpa64e : Predicate<"Subtarget->hasVendorXSfvfexpa64e()">;

def FeatureVendorXSfvfhbfmin
    : RISCVExtension<0, 1,
                     "SiFive custom minimal BF16 vector support",
                     [FeatureStdExtZvfbfmin]>;
def HasVendorXSfvfhbfmin : Predicate<"Subtarget->hasVendorXSfvfhbfmin()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfvfhbfmin),
                         "'Xsfvfhbfmin' (SiFive custom minimal BF16 vector support)">;

def FeatureVendorXSfvqdotq
    : RISCVExtension<0, 1,
                     "SiFive Vector Quad-Widening 4D Dot Product Instructions",
                     [FeatureStdExtZve32x]>;
def HasVendorXSfvqdotq : Predicate<"Subtarget->hasVendorXSfvqdotq()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfvqdotq),
                         "'Xsfvqdotq' (SiFive Vector Quad-Widening 4D Dot Product Instructions)">;

def FeatureVendorXSfvfbfa
    : RISCVExtension<0, 1,
                     "SiFive custom additional BF16 vector compute support",
                     [FeatureStdExtZve32f, FeatureStdExtZfbfmin]>;
def HasVendorXSfvfbfa
    : Predicate<"Subtarget->hasVendorXSfvfbfa()">,
      AssemblerPredicate<(all_of FeatureVendorXSfvfbfa),
                         "'XSfvfbfa' (SiFive custom additional BF16 vector compute support)">;
