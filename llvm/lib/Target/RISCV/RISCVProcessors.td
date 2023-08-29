//===-- RISCVProcessors.td - RISC-V Processors -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V processors supported.
//===----------------------------------------------------------------------===//

class RISCVProcessorModel<string n,
                          SchedMachineModel m,
                          list<SubtargetFeature> f,
                          list<SubtargetFeature> tunef = [],
                          string default_march = "">
      :  ProcessorModel<n, m, f, tunef> {
  string DefaultMarch = default_march;
}

class RISCVTuneProcessorModel<string n,
                              SchedMachineModel m,
                              list<SubtargetFeature> tunef = [],
                              list<SubtargetFeature> f = []>
      : ProcessorModel<n, m, f,tunef>;

def GENERIC_RV32 : RISCVProcessorModel<"generic-rv32",
                                       NoSchedModel,
                                       [Feature32Bit]>;
def GENERIC_RV64 : RISCVProcessorModel<"generic-rv64",
                                       NoSchedModel,
                                       [Feature64Bit]>;
// Support generic for compatibility with other targets. The triple will be used
// to change to the appropriate rv32/rv64 version.
def : ProcessorModel<"generic", NoSchedModel, []>;

def ROCKET_RV32 : RISCVProcessorModel<"rocket-rv32",
                                      RocketModel,
                                      [Feature32Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZicsr]>;
def ROCKET_RV64 : RISCVProcessorModel<"rocket-rv64",
                                      RocketModel,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZicsr]>;
def ROCKET : RISCVTuneProcessorModel<"rocket",
                                     RocketModel>;

def SIFIVE_7 : RISCVTuneProcessorModel<"sifive-7-series",
                                       SiFive7Model,
                                       [TuneSiFive7]>;

#ifdef SIFIVE_CUSTOMIZATION
def SIFIVE_6 : RISCVTuneProcessorModel<"sifive-6-series", SiFive6Model,
                                       [TuneSiFive6], []>;
def SIFIVE_7M : RISCVTuneProcessorModel<"sifive-7m-series", SiFive7Model,
                                        [TuneSiFive7, TunePseudoLIsimm32], []>;
def SIFIVE_7N : RISCVTuneProcessorModel<"sifive-7n-series", SiFive7NModel,
                                        [TuneSiFive7, TunePseudoLIsimm32], []>;
def SIFIVE_7O : RISCVTuneProcessorModel<"sifive-7o-series", SiFive7NModel,
                                        [TuneSiFive7, TunePseudoLIsimm32], []>;
def SIFIVE_P400 : RISCVTuneProcessorModel<"sifive-p400-series",
                                          SiFiveP400Model, [TuneSiFiveP400],
                                          []>;
def SIFIVE_8 : RISCVTuneProcessorModel<"sifive-8-series", SiFiveP500Model,
                                       [TuneSiFiveP500], []>;
def SIFIVE_P500 : RISCVTuneProcessorModel<"sifive-p500-series",
                                          SiFiveP500Model, [TuneSiFiveP500],
                                          []>;
def SIFIVE_P600 : RISCVTuneProcessorModel<"sifive-p600-series",
                                          SiFiveP600Model, [TuneSiFiveP600],
                                          []>;
#endif // SIFIVE_CUSTOMIZATION

def SIFIVE_E20 : RISCVProcessorModel<"sifive-e20",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtC]>;

def SIFIVE_E21 : RISCVProcessorModel<"sifive-e21",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_E24 : RISCVProcessorModel<"sifive-e24",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC]>;

def SIFIVE_E31 : RISCVProcessorModel<"sifive-e31",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_E34 : RISCVProcessorModel<"sifive-e34",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC]>;

def SIFIVE_E76 : RISCVProcessorModel<"sifive-e76",
                                     SiFive7Model,
                                     [Feature32Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC],
                                     [TuneSiFive7]>;

#ifdef SIFIVE_CUSTOMIZATION
def SIFIVE_E61 : RISCVProcessorModel<"sifive-e61", SiFive6Model,
                                      [Feature32Bit,
                                       FeatureStdExtZicsr,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      [TuneSiFive6]>;
def SIFIVE_E76M : RISCVProcessorModel<"sifive-e76m", SiFive7Model,
                                      [Feature32Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      [TuneSiFive7]>;
def SIFIVE_E76N : RISCVProcessorModel<"sifive-e76n", SiFive7NModel,
                                      [Feature32Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      [TuneSiFive7, TunePseudoLIsimm32]>;
#endif // SIFIVE_CUSTOMIZATION

def SIFIVE_S21 : RISCVProcessorModel<"sifive-s21",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_S51 : RISCVProcessorModel<"sifive-s51",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_S54 : RISCVProcessorModel<"sifive-s54",
                                      RocketModel,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC]>;

#ifdef SIFIVE_CUSTOMIZATION
def SIFIVE_S61 : RISCVProcessorModel<"sifive-s61", SiFive6Model,
                                     [Feature64Bit,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC,
                                      FeatureStdExtZba,
                                      FeatureStdExtZbb],
                                     [TuneSiFive6]>;
#endif // SIFIVE_CUSTOMIZATION

def SIFIVE_S76 : RISCVProcessorModel<"sifive-s76",
                                     SiFive7Model,
                                     [Feature64Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC,
                                      FeatureStdExtZihintpause,
                                      FeatureVendorXSfcie],
                                     [TuneSiFive7]>;

#ifdef SIFIVE_CUSTOMIZATION
def SIFIVE_S76M : RISCVProcessorModel<"sifive-s76m", SiFive7Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      [TuneSiFive7]>;

def SIFIVE_S76N : RISCVProcessorModel<"sifive-s76n", SiFive7NModel,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      [TuneSiFive7, TunePseudoLIsimm32]>;
#endif // SIFIVE_CUSTOMIZATION

def SIFIVE_U54 : RISCVProcessorModel<"sifive-u54",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC]>;

#ifdef SIFIVE_CUSTOMIZATION
def SIFIVE_U64 : RISCVProcessorModel<"sifive-u64", SiFive6Model,
                                     [Feature64Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC,
                                      FeatureStdExtZba,
                                      FeatureStdExtZbb],
                                     [TuneSiFive6]>;

def SIFIVE_U64R : RISCVProcessorModel<"sifive-u64r", SiFive6Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb,
                                       FeatureStdExtZbs,
                                       FeatureStdExtZcb,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZfhmin,
                                       FeatureStdExtZicntr,
                                       FeatureStdExtZicsr,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZkt],
                                      [TuneSiFive6]>;
#endif // SIFIVE_CUSTOMIZATION

def SIFIVE_U74 : RISCVProcessorModel<"sifive-u74",
                                     SiFive7Model,
                                     [Feature64Bit,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC],
                                     [TuneSiFive7]>;

#ifdef SIFIVE_CUSTOMIZATION
def SIFIVE_U74M : RISCVProcessorModel<"sifive-u74m", SiFive7Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      [TuneSiFive7]>;

def SIFIVE_U74N : RISCVProcessorModel<"sifive-u74n", SiFive7NModel,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      [TuneSiFive7, TunePseudoLIsimm32]>;

def SIFIVE_U74R : RISCVProcessorModel<"sifive-u74r", SiFive7NModel,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb,
                                       FeatureStdExtZbs,
                                       FeatureStdExtZcb,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZfhmin,
                                       FeatureStdExtZicntr,
                                       FeatureStdExtZicsr,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZkt],
                                      [TuneSiFive7, TunePseudoLIsimm32]>;

def SIFIVE_P270 :
  RISCVProcessorModel<"sifive-p270", SiFive7Model,
                      [Feature64Bit,
                       FeatureStdExtZifencei,
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

def SIFIVE_P270N :
  RISCVProcessorModel<"sifive-p270n", SiFive7NModel,
                      [Feature64Bit,
                       FeatureStdExtZifencei,
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
#endif // SIFIVE_CUSTOMIZATION

def SIFIVE_X280 : RISCVProcessorModel<"sifive-x280", SiFive7Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
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
                                       FeatureVendorXSfvqmaccqoq,   // SIFIVE
                                       FeatureVendorXSfvfhbfmin],   // SIFIVE
                                      [TuneSiFive7, // SIFIVE
                                       TuneDLenFactor2, // SIFIVE
                                       TuneDLen256b]>; // SIFIVE

#ifdef SIFIVE_CUSTOMIZATION
def SIFIVE_X280N :
  RISCVProcessorModel<"sifive-x280n", SiFive7NModel,
                      [Feature64Bit,
                       FeatureStdExtZifencei,
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
                       FeatureVendorXSfvfhbfmin,
                       FeatureVendorXSfvqmaccqoq,
                       FeatureVendorXSfvfnrclipxfqf,
                       FeatureVendorXSfvfwmaccqqq],
                      [TuneSiFive7,
                       TuneDLenFactor2,
                       TuneDLen256b,
                       TunePseudoLIsimm32]>;

def SIFIVE_X280O :
  RISCVProcessorModel<"sifive-x280o", SiFive7NModel,
                      [Feature64Bit,
                       FeatureStdExtZifencei,
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
                       FeatureVendorXSfvfhbfmin,
                       FeatureVendorXSfvqmaccqoq,
                       FeatureVendorXSfvfnrclipxfqf,
                       FeatureVendorXSfvfwmaccqqq],
                      [TuneSiFive7,
                       TuneDLenFactor2,
                       TuneDLen256b,
                       TunePseudoLIsimm32,
                       TuneOptimizedVRGatherVCompress]>;

// FIXME: This needs its own scheduler model
def SIFIVE_P450 : RISCVProcessorModel<"sifive-p450", SiFiveP400Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtXsifivecdiscarddlone,
                                       FeatureStdExtXsifivecflushdlone,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb,
                                       FeatureStdExtZbs,
                                       FeatureStdExtZfhmin],
                                      [TuneSiFiveP400]>;

// FIXME: This needs its own scheduler model
def SIFIVE_P470 : RISCVProcessorModel<"sifive-p470", SiFiveP400Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtXsifivecdiscarddlone,
                                       FeatureStdExtXsifivecflushdlone,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb,
                                       FeatureStdExtZbs,
                                       FeatureStdExtZfhmin,
                                       FeatureStdExtV,
                                       FeatureStdExtZvl128b,
                                       FeatureStdExtZvbb,
                                       FeatureStdExtZvknc,
                                       FeatureStdExtZvkng,
                                       FeatureStdExtZvksc,
                                       FeatureStdExtZvksg],
                                      [TuneSiFiveP400, TuneDLen128b]>;

def SIFIVE_P550 : RISCVProcessorModel<"sifive-p550", SiFiveP500Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      [TuneSiFiveP500]>;

def SIFIVE_P650 : RISCVProcessorModel<"sifive-p650", SiFiveP600Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtXsifivecdiscarddlone,
                                       FeatureStdExtXsifivecflushdlone,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb,
                                       FeatureStdExtZbs,
                                       FeatureStdExtZfhmin],
                                      [TuneSiFiveP600]>;

def SIFIVE_P670 : RISCVProcessorModel<"sifive-p670", SiFiveP600Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtXsifivecdiscarddlone,
                                       FeatureStdExtXsifivecflushdlone,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb,
                                       FeatureStdExtZbs,
                                       FeatureStdExtZfhmin,
                                       FeatureStdExtV,
                                       FeatureStdExtZvl128b,
                                       FeatureStdExtZvbb,
                                       FeatureStdExtZvknc,
                                       FeatureStdExtZvkng,
                                       FeatureStdExtZvksc,
                                       FeatureStdExtZvksg],
                                      [TuneSiFiveP600, TuneDLen128b]>;

def SIFIVE_P870 : RISCVProcessorModel<"sifive-p870", SiFiveP800Model,
                                      [Feature64Bit,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtXsifivecdiscarddlone,
                                       FeatureStdExtXsifivecflushdlone,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb,
                                       FeatureStdExtZbs,
                                       FeatureStdExtZfhmin,
                                       FeatureStdExtV,
                                       FeatureStdExtZvl128b,
                                       FeatureStdExtZvbb,
                                       FeatureStdExtZvknc,
                                       FeatureStdExtZvkng,
                                       FeatureStdExtZvksc,
                                       FeatureStdExtZvksg],
                                      [TuneSiFiveP800, TuneDLen128b]>;
#endif // SIFIVE_CUSTOMIZATION

def SYNTACORE_SCR1_BASE : RISCVProcessorModel<"syntacore-scr1-base",
                                              SyntacoreSCR1Model,
                                              [Feature32Bit,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll]>;

def SYNTACORE_SCR1_MAX : RISCVProcessorModel<"syntacore-scr1-max",
                                             SyntacoreSCR1Model,
                                             [Feature32Bit,
                                              FeatureStdExtZicsr,
                                              FeatureStdExtZifencei,
                                              FeatureStdExtM,
                                              FeatureStdExtC],
                                             [TuneNoDefaultUnroll]>;
