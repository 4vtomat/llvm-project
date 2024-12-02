//===------ RISCVProfiles.td - RISC-V Profiles -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Profile Featuyre Lists
//===----------------------------------------------------------------------===//

// RVI Profile Family

defvar RVI20U32Features = [Feature32Bit, FeatureStdExtI];
defvar RVI20U64Features = [Feature64Bit, FeatureStdExtI];

// RVA Profile Family

defvar RVA20U64BaseFeatures = [Feature64Bit,
                               FeatureStdExtI,
                               FeatureStdExtM,
                               FeatureStdExtA,
                               FeatureStdExtF,
                               FeatureStdExtD,
                               FeatureStdExtC,
                               FeatureStdExtZicntr,
                               FeatureStdExtZiccif,
                               FeatureStdExtZiccrse,
                               FeatureStdExtZiccamoa,
                               FeatureStdExtZicclsm];
defvar RVA20U64Features = !listconcat(RVA20U64BaseFeatures,
                                      [FeatureStdExtZa128rs]);

defvar RVA20S64BaseFeatures = [FeatureStdExtZifencei,
                               FeatureStdExtSvbare,
                               FeatureStdExtSvade,
                               FeatureStdExtSsccptr,
                               FeatureStdExtSstvecd,
                               FeatureStdExtSstvala];
defvar RVA20S64Features = !listconcat(RVA20U64Features,
                                      RVA20S64BaseFeatures);

defvar RVA22U64Features = !listconcat(RVA20U64BaseFeatures,
                                      [FeatureStdExtZa64rs,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtB,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZfhmin,
                                       FeatureStdExtZkt]);

defvar RVA22S64BaseFeatures = !listconcat(RVA20S64BaseFeatures,
                                          [FeatureStdExtSscounterenw,
                                           FeatureStdExtSvpbmt,
                                           FeatureStdExtSvinval]);
defvar RVA22S64Features = !listconcat(RVA22U64Features,
                                      RVA22S64BaseFeatures);

defvar RVA23U64Features = !listconcat(RVA22U64Features,
                                      [FeatureStdExtV,
                                       FeatureStdExtZvfhmin,
                                       FeatureStdExtZvbb,
                                       FeatureStdExtZvkt,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZicond,
                                       FeatureStdExtZimop,
                                       FeatureStdExtZcmop,
                                       FeatureStdExtZcb,
                                       FeatureStdExtZfa,
                                       FeatureStdExtZawrs,
                                       FeatureStdExtSupm]);

defvar RVA23S64BaseFeatures = !listconcat(RVA22S64BaseFeatures,
                                          [FeatureStdExtSvnapot,
                                           FeatureStdExtSstc,
                                           FeatureStdExtSscofpmf,
                                           FeatureStdExtSsnpm,
                                           FeatureStdExtSsu64xl,
                                           FeatureStdExtSha]);
defvar RVA23S64Features = !listconcat(RVA23U64Features,
                                      RVA23S64BaseFeatures);

// RVB Profile Family

defvar RVB23U64Features = !listconcat(RVA20U64BaseFeatures,
                                      [FeatureStdExtZihpm,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtB,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZkt,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZicond,
                                       FeatureStdExtZimop,
                                       FeatureStdExtZcmop,
                                       FeatureStdExtZcb,
                                       FeatureStdExtZfa,
                                       FeatureStdExtZawrs]);

defvar RVB23S64Features = !listconcat(RVB23U64Features,
                                      [FeatureStdExtZifencei,
                                       FeatureStdExtSvnapot,
                                       FeatureStdExtSvbare,
                                       FeatureStdExtSvade,
                                       FeatureStdExtSsccptr,
                                       FeatureStdExtSstvecd,
                                       FeatureStdExtSstvala,
                                       FeatureStdExtSscounterenw,
                                       FeatureStdExtSvpbmt,
                                       FeatureStdExtSvinval,
                                       FeatureStdExtSstc,
                                       FeatureStdExtSscofpmf,
                                       FeatureStdExtSsu64xl]);

// RVM Profile Family

defvar RVM23U32Features = [Feature32Bit,
                           FeatureStdExtI,
                           FeatureStdExtM,
                           FeatureStdExtB,
                           FeatureStdExtZicond,
                           FeatureStdExtZihintpause,
                           FeatureStdExtZihintntl,
                           FeatureStdExtZce,
                           FeatureStdExtZicbop,
                           FeatureStdExtZimop,
                           FeatureStdExtZcmop];

#ifdef SIFIVE_CUSTOMIZATION
defvar SFA23Features = !listconcat(RVA23S64Features,
                                   [FeatureStdExtZicfilp,
                                    FeatureStdExtZicfiss,
                                    FeatureStdExtZkr,
                                    FeatureStdExtSsqosid,
                                    FeatureStdExtSsaia]);

defvar SFX23Features = !listconcat(SFA23Features,
                                   [FeatureStdExtZfh,
                                    FeatureStdExtZfbfmin,
                                    FeatureStdExtZvfh,
                                    FeatureStdExtZvfbfmin,
                                    FeatureStdExtZvfbfwma,
                                    FeatureVendorXSfvfnrclipxfqf,
                                    FeatureVendorXSfvfexpa]);

defvar SFP23Features = !listconcat(SFA23Features,
                                   [FeatureStdExtZfh,
                                    FeatureStdExtZfbfmin,
                                    FeatureStdExtZvfh,
                                    FeatureStdExtZvfbfmin,
                                    FeatureStdExtZvfbfwma,
                                    FeatureVendorXSfvqdotq]);

defvar SFE23Features = !listconcat(RVI20U32Features,
                                   [FeatureStdExtM,
                                    FeatureStdExtC,
                                    FeatureStdExtZba,
                                    FeatureStdExtZbb,
                                    FeatureStdExtZbs,
                                    FeatureStdExtZaamo,
                                    FeatureStdExtZicsr,
                                    FeatureStdExtZifencei,
                                    FeatureStdExtZicntr,
                                    FeatureStdExtZihintntl,
                                    FeatureStdExtZicond,
                                    FeatureStdExtZicbop,
                                    FeatureStdExtZimop,
                                    FeatureStdExtZcmop,
                                    FeatureStdExtZcb,
                                    FeatureStdExtZkt]);

defvar SFS23Features = !listconcat(RVI20U64Features,
                                   [FeatureStdExtM,
                                    FeatureStdExtC,
                                    FeatureStdExtZba,
                                    FeatureStdExtZbb,
                                    FeatureStdExtZbs,
                                    FeatureStdExtZaamo,
                                    FeatureStdExtZicsr,
                                    FeatureStdExtZifencei,
                                    FeatureStdExtZicntr,
                                    FeatureStdExtZihintntl,
                                    FeatureStdExtZicond,
                                    FeatureStdExtZicbop,
                                    FeatureStdExtZimop,
                                    FeatureStdExtZcmop,
                                    FeatureStdExtZcb,
                                    FeatureStdExtZkt]);

defvar SFB23Features = !listconcat(RVB23S64Features,
                                   [FeatureStdExtZfhmin,
                                    FeatureStdExtZicfilp,
                                    FeatureStdExtZicfiss,
                                    FeatureStdExtSscofpmf]);
#endif // SIFIVE_CUSTOMIZATION

//===----------------------------------------------------------------------===//
// Profile Definitions for ISA String
//===----------------------------------------------------------------------===//

class RISCVProfile<string name, list<SubtargetFeature> features>
    : SubtargetFeature<name, "Is" # NAME, "true",
                       "RISC-V " # name # " profile", features> {
  // Indicates if the profile is not yet ratified, so should be treated as
  // experimental.
  bit Experimental = false;
}
class RISCVExperimentalProfile<string name, list<SubtargetFeature> features>
    : RISCVProfile<"experimental-"#name, features> {
  let Experimental = true;
}

def RVI20U32 : RISCVProfile<"rvi20u32", RVI20U32Features>;
def RVI20U64 : RISCVProfile<"rvi20u64", RVI20U64Features>;
def RVA20U64 : RISCVProfile<"rva20u64", RVA20U64Features>;
def RVA20S64 : RISCVProfile<"rva20s64", RVA20S64Features>;
def RVA22U64 : RISCVProfile<"rva22u64", RVA22U64Features>;
def RVA22S64 : RISCVProfile<"rva22s64", RVA22S64Features>;
def RVA23U64 : RISCVProfile<"rva23u64", RVA23U64Features>;
def RVA23S64 : RISCVProfile<"rva23s64", RVA23S64Features>;
def RVB23U64 : RISCVProfile<"rvb23u64", RVB23U64Features>;
def RVB23S64 : RISCVProfile<"rvb23s64", RVB23S64Features>;
def RVM23U32 : RISCVExperimentalProfile<"rvm23u32", RVM23U32Features>;

#ifdef SIFIVE_CUSTOMIZATION
def SFA23 : RISCVProfile<"sfa23", SFA23Features>;
def SFX23 : RISCVProfile<"sfx23", SFX23Features>;
def SFP23 : RISCVProfile<"sfp23", SFP23Features>;
def SFE23 : RISCVProfile<"sfe23", SFE23Features>;
def SFS23 : RISCVProfile<"sfs23", SFS23Features>;
def SFB23 : RISCVProfile<"sfb23", SFB23Features>;
#endif // SIFIVE_CUSTOMIZATION
