// RUN: llvm-tblgen -gen-riscv-isa-info-json -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

class RISCVExtension<string name, int major, int minor, string desc,
                     list<SubtargetFeature> implies = [],
                     string fieldname = !subst("Feature", "Has", NAME),
                     string value = "true">
    : SubtargetFeature<name, fieldname, value, desc, implies> {
  int MajorVersion = major;
  int MinorVersion = minor;
  bit Experimental = false;

  list<int> AdditionalVersions = []; // SIFIVE
}

class RISCVExperimentalExtension<string name, int major, int minor, string desc,
                                 list<RISCVExtension> implies = [],
                                 string fieldname = !subst("Feature", "Has", NAME),
                                 string value = "true">
    : RISCVExtension<"experimental-"#name, major, minor, desc, implies,
                     fieldname, value> {
  let Experimental = true;
}

def FeatureStdExtI
    : RISCVExtension<"i", 2, 1,
                     "'I' (Base Integer Instruction Set)">;

def FeatureStdExtZicsr
    : RISCVExtension<"zicsr", 2, 0,
                     "'zicsr' (CSRs)">;

def FeatureStdExtZifencei
    : RISCVExtension<"zifencei", 2, 0,
                     "'Zifencei' (fence.i)">;

def FeatureStdExtF
    : RISCVExtension<"f", 2, 2,
                     "'F' (Single-Precision Floating-Point)",
                     [FeatureStdExtZicsr]>;

def FeatureStdExtZidummy
    : RISCVExperimentalExtension<"zidummy", 0, 1,
                                 "Dummy">;

def FeaturesStdExtZidummya
    : RISCVExperimentalExtension<"zidummya", 0, 1,
                                 "Dummy A">;

def FeaturesStdExtZidummyb
    : RISCVExperimentalExtension<"zidummyb", 0, 1,
                                 "Dummy B",
                                 [FeaturesStdExtZidummya]>;

def FeaturesStdExtZidummyc
    : RISCVExperimentalExtension<"zidummyc", 0, 1,
                                 "Dummy C",
                                 [FeaturesStdExtZidummya, FeaturesStdExtZidummyb]>;

def Feature32Bit
    : SubtargetFeature<"32bit", "IsRV32", "true", "Implements RV32">;
def Feature64Bit
    : SubtargetFeature<"64bit", "IsRV64", "true", "Implements RV64">;

// Dummy feature that isn't an extension.
def FeatureDummy
    : SubtargetFeature<"dummy", "Dummy", "true", "Dummy">;


// CHECK:     {
// CHECK-NEXT:  "supported_extensions": {
// CHECK-NEXT:    "f": {
// CHECK-NEXT:      "major_version": 2,
// CHECK-NEXT:      "minor_version": 2,
// CHECK-NEXT:      "experimental": false
// CHECK-NEXT:    },
// CHECK-NEXT:    "i": {
// CHECK-NEXT:      "major_version": 2,
// CHECK-NEXT:      "minor_version": 1,
// CHECK-NEXT:      "experimental": false
// CHECK-NEXT:    },
// CHECK-NEXT:    "zicsr": {
// CHECK-NEXT:      "major_version": 2,
// CHECK-NEXT:      "minor_version": 0,
// CHECK-NEXT:      "experimental": false
// CHECK-NEXT:    },
// CHECK-NEXT:    "zidummy": {
// CHECK-NEXT:      "major_version": 0,
// CHECK-NEXT:      "minor_version": 1,
// CHECK-NEXT:      "experimental": true
// CHECK-NEXT:    },
// CHECK-NEXT:    "zidummya": {
// CHECK-NEXT:      "major_version": 0,
// CHECK-NEXT:      "minor_version": 1,
// CHECK-NEXT:      "experimental": true
// CHECK-NEXT:    },
// CHECK-NEXT:    "zidummyb": {
// CHECK-NEXT:      "major_version": 0,
// CHECK-NEXT:      "minor_version": 1,
// CHECK-NEXT:      "experimental": true
// CHECK-NEXT:    },
// CHECK-NEXT:    "zidummyc": {
// CHECK-NEXT:      "major_version": 0,
// CHECK-NEXT:      "minor_version": 1,
// CHECK-NEXT:      "experimental": true
// CHECK-NEXT:    },
// CHECK-NEXT:    "zifencei": {
// CHECK-NEXT:      "major_version": 2,
// CHECK-NEXT:      "minor_version": 0,
// CHECK-NEXT:      "experimental": false
// CHECK-NEXT:    }
// CHECK-NEXT:  },
// CHECK-NEXT:  "implied_extensions": {
// CHECK-NEXT:    "f": ["zicsr"],
// CHECK-NEXT:    "zidummyb": ["zidummya"],
// CHECK-NEXT:    "zidummyc": ["zidummya", "zidummyb"]
// CHECK-NEXT:  }
// CHECK-NEXT:}
