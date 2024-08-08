// RUN: llvm-tblgen -gen-riscv-profile-json -DSIFIVE_CUSTOMIZATION -I %p/../../include %s | FileCheck %s

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

def Feature64Bit
    : SubtargetFeature<"64bit", "IsRV64", "true", "Implements RV64">;

def IsRV64 : Predicate<"Subtarget->is64Bit()">,
             AssemblerPredicate<(all_of Feature64Bit),
                                "RV64I Base Instruction Set">;

def FeatureStdExtI
    : RISCVExtension<"i", 2, 1,
                     "'I' (Base Integer Instruction Set)">;

def FeatureStdExtF
    : RISCVExtension<"f", 2, 2,
                     "'F' (Single-Precision Floating-Point)">;

defvar RVI20U64Features = [Feature64Bit, FeatureStdExtI];

#ifdef SIFIVE_CUSTOMIZATION
defvar SFDUMMYFeatures = [Feature64Bit, FeatureStdExtI, FeatureStdExtF];
#endif

def RVI20U64 : RISCVProfile<"rvi20u64", RVI20U64Features>;

#ifdef SIFIVE_CUSTOMIZATION
def SFDUMMY : RISCVExperimentalProfile<"sfdummy", SFDUMMYFeatures>;
#endif

// CHECK:     {
// CHECK-NEXT:  "supported_profiles": {
// CHECK-NEXT:    "rvi20u64": {
// CHECK-NEXT:      "march": "rv64i2p1"
// CHECK-NEXT:    },
// CHECK-NEXT:    "sfdummy": {
// CHECK-NEXT:      "march": "rv64i2p1_f2p2",
// CHECK-NEXT:      "experimental": true
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:}
