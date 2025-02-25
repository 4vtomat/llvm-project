//===- LoopVectorizationLegality.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides loop vectorization legality analysis. Original code
// resided in LoopVectorize.cpp for a long time.
//
// At this point, it is implemented as a utility class, not as an analysis
// pass. It should be easy to create an analysis pass around it if there
// is a need (but D45420 needs to happen first).
//

#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"
#if SIFIVE_CUSTOMIZATION
#include "SiFive_VPlanCostModel.h"
#include "llvm/ADT/Statistic.h"
#endif // SIFIVE_CUSTOMIZATION
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#if SIFIVE_CUSTOMIZATION
#include "llvm/Analysis/LoopIterator.h"
#endif // SIFIVE_CUSTOMIZATION
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"

using namespace llvm;
using namespace PatternMatch;

#define LV_NAME "loop-vectorize"
#define DEBUG_TYPE LV_NAME

#if SIFIVE_CUSTOMIZATION
STATISTIC(
    NumOfUncountableLoopsAnalyzedForVectorization,
    "Number of uncountable loops inspected for uncountable loop opportunity");
STATISTIC(NumOfUncountableLoopsVectorizable,
          "Number of vectorizable uncountable loops");
STATISTIC(NumOfUncountableLoopsWithOneBlock,
          "Number of uncountable loops with one block");
STATISTIC(NumOfUncountableLoopsWithTwoBlocks,
          "Number of uncountable loops with two blocks");
STATISTIC(NumOfUncountableLoopsWithMoreThanTwoBlocks,
          "Number of uncountable loops with more than two blocks");
STATISTIC(NumOfUncountableLoopsSpeculationUnsafe,
          "Number of uncountable loops unsafe for speculation");
STATISTIC(NumOfUncountableLoopsNotEndingWithConditionalBranch,
          "Number of uncountable loops not ending with conditional branch");
STATISTIC(NumOfUncountableLoopsWithUnsupportedPHI,
          "Number of uncountable loops with unsupported phi");
STATISTIC(NumOfUncountableLoopsWithoutHeaderPHI,
          "Number of uncountable loops without a PHI in header");
STATISTIC(NumOfUncountableLoopsWithNonPtrIVs,
          "Number of uncountable loops with non-ptr induction variables");
STATISTIC(NumOfUncountableLoopsWithNonIVLiveOutValues,
          "Number of uncountable loops with non-IV live out values");
STATISTIC(NumberOfMonotonics, "Number of monotonics");

namespace UncountableLoopVectorization {
enum class Option {
  On = 0,       // Uncountable loop vectorization on
  Off,          // Uncountable loop vectorization off (default)
  AnalysisOnly, // Uncountable loop vectorizaton analysis only
  Stress // Uncountable loop vectorizaton stress mode to test scenarios that are
         // not enabled by default
};
} // namespace UncountableLoopVectorization

static cl::opt<UncountableLoopVectorization::Option>
    UncountableLoopVectorizationOption(
        "sifive-uncountable-loop-vectorization",
        cl::init(UncountableLoopVectorization::Option::On), cl::Hidden,
        cl::desc("Knobs for the uncountable loop vectorization pipeline."),
        cl::values(
            clEnumValN(UncountableLoopVectorization::Option::On, "on",
                       "Uncountable loop vectorization on"),
            clEnumValN(UncountableLoopVectorization::Option::Off, "off",
                       "Uncountable loop vectorization off (default)"),
            clEnumValN(UncountableLoopVectorization::Option::AnalysisOnly,
                       "analysis-only",
                       "Uncountable loop vectorizaton analysis only"),
            clEnumValN(UncountableLoopVectorization::Option::Stress, "stress",
                       "Uncountable loop vectorizaton stress mode to test "
                       "scenarios that are not enabled by default")));
#endif // SIFIVE_CUSTOMIZATION

static cl::opt<bool>
    EnableIfConversion("enable-if-conversion", cl::init(true), cl::Hidden,
                       cl::desc("Enable if-conversion during vectorization."));

static cl::opt<bool>
#if SIFIVE_CUSTOMIZATION
AllowStridedPointerIVs("lv-strided-pointer-ivs", cl::init(true), cl::Hidden,
#else
AllowStridedPointerIVs("lv-strided-pointer-ivs", cl::init(false), cl::Hidden,
#endif // SIFIVE_CUSTOMIZATION
                       cl::desc("Enable recognition of non-constant strided "
                                "pointer induction variables."));

namespace llvm {
#if SIFIVE_CUSTOMIZATION
// Don't allow fp reordering even if vectorization was enforced or width was
// specified. User need to use `#pragma clang fp reassociate(on)` within the
// loop body to allow reassociation
cl::opt<bool>
    HintsAllowReordering("hints-allow-reordering", cl::init(false), cl::Hidden,
#else
cl::opt<bool>
    HintsAllowReordering("hints-allow-reordering", cl::init(true), cl::Hidden,
#endif // SIFIVE_CUSTOMIZATION
                         cl::desc("Allow enabling loop hints to reorder "
                                  "FP operations during vectorization."));
} // namespace llvm

// TODO: Move size-based thresholds out of legality checking, make cost based
// decisions instead of hard thresholds.
static cl::opt<unsigned> VectorizeSCEVCheckThreshold(
    "vectorize-scev-check-threshold", cl::init(16), cl::Hidden,
    cl::desc("The maximum number of SCEV checks allowed."));

static cl::opt<unsigned> PragmaVectorizeSCEVCheckThreshold(
    "pragma-vectorize-scev-check-threshold", cl::init(128), cl::Hidden,
    cl::desc("The maximum number of SCEV checks allowed with a "
             "vectorize(enable) pragma"));

#if SIFIVE_CUSTOMIZATION
static cl::opt<bool>
    ForceVectorization("force-vectorization", cl::init(false), cl::Hidden,
                       cl::desc("Force vectorization regardless if "
                                "vectorization is profitable or not"));
#endif // SIFIVE_CUSTOMIZATION

static cl::opt<LoopVectorizeHints::ScalableForceKind>
    ForceScalableVectorization(
        "scalable-vectorization", cl::init(LoopVectorizeHints::SK_Unspecified),
        cl::Hidden,
        cl::desc("Control whether the compiler can use scalable vectors to "
                 "vectorize a loop"),
        cl::values(
            clEnumValN(LoopVectorizeHints::SK_FixedWidthOnly, "off",
                       "Scalable vectorization is disabled."),
            clEnumValN(
                LoopVectorizeHints::SK_PreferScalable, "preferred",
                "Scalable vectorization is available and favored when the "
                "cost is inconclusive."),
            clEnumValN(
                LoopVectorizeHints::SK_PreferScalable, "on",
                "Scalable vectorization is available and favored when the "
                "cost is inconclusive.")
#if SIFIVE_CUSTOMIZATION
            ,
            clEnumValN(LoopVectorizeHints::SK_ScalableOnly, "only",
                       "Scalable vectorization is the only option available")));
#else
            ));
#endif // SIFIVE_CUSTOMIZATION
#if SIFIVE_CUSTOMIZATION
static cl::opt<bool>
    EnableCSA("sifive-enable-csa", cl::init(true), cl::Hidden,
              cl::desc("Control whether CSA loop vectorization is enabled"));

static cl::opt<bool>
    EnableMonotonics("sifive-enable-monotonics", cl::init(true), cl::Hidden,
                     cl::desc("Control whether vectorization of loops with "
                              "monotonic variables is enabled"));
#endif

static cl::opt<bool> EnableHistogramVectorization(
    "enable-histogram-loop-vectorization", cl::init(false), cl::Hidden,
    cl::desc("Enables autovectorization of some loops containing histograms"));

/// Maximum vectorization interleave count.
static const unsigned MaxInterleaveFactor = 16;

namespace llvm {

bool LoopVectorizeHints::Hint::validate(unsigned Val) {
  switch (Kind) {
  case HK_WIDTH:
    return isPowerOf2_32(Val) && Val <= VectorizerParams::MaxVectorWidth;
  case HK_INTERLEAVE:
    return isPowerOf2_32(Val) && Val <= MaxInterleaveFactor;
  case HK_FORCE:
    return (Val <= 1);
  case HK_ISVECTORIZED:
  case HK_PREDICATE:
  case HK_SCALABLE:
    return (Val == 0 || Val == 1);
#if SIFIVE_CUSTOMIZATION
  case HK_LMUL_SEW:
    // This hint is not handlded by this validate function
    return false;
#endif // SIFIVE_CUSTOMIZATION
  }
  return false;
}

#if SIFIVE_CUSTOMIZATION
// Return true if vectorizer is able to support VLA vectorization for the loop
// \p L.
static bool allowVLAVectorizer(const TargetTransformInfo &TTI, const Loop &L) {
  // TODO: Support outer loop VLA vectorization
  if (!L.isInnermost())
    return false;

  // TODO: Add an override option, like -sifive-enable-vla-vectorizer.
  return TTI.useVLAVectorizer();
}
#endif // SIFIVE_CUSTOMIZATION

LoopVectorizeHints::LoopVectorizeHints(const Loop *L,
                                       bool InterleaveOnlyWhenForced,
                                       OptimizationRemarkEmitter &ORE,
#if SIFIVE_CUSTOMIZATION
                                       const bool ReportInvalid,
#endif // SIFIVE_CUSTOMIZATION
                                       const TargetTransformInfo *TTI)
    : Width("vectorize.width", VectorizerParams::VectorizationFactor, HK_WIDTH),
      Interleave("interleave.count", InterleaveOnlyWhenForced, HK_INTERLEAVE),
      Force("vectorize.enable", FK_Undefined, HK_FORCE),
      IsVectorized("isvectorized", 0, HK_ISVECTORIZED),
      Predicate("vectorize.predicate.enable", FK_Undefined, HK_PREDICATE),
      Scalable("vectorize.scalable.enable", SK_Unspecified, HK_SCALABLE),
#if SIFIVE_CUSTOMIZATION
      LmulSew("vectorize.lmul_sew", -1U, HK_LMUL_SEW),
#endif // SIFIVE_CUSTOMIZATION
      TheLoop(L), ORE(ORE) {
  // Populate values with existing loop metadata.
#if SIFIVE_CUSTOMIZATION
  getHintsFromMetadata(ReportInvalid);
#else
  getHintsFromMetadata();
#endif // SIFIVE_CUSTOMIZATION

  // force-vector-interleave overrides DisableInterleaving.
  if (VectorizerParams::isInterleaveForced())
    Interleave.Value = VectorizerParams::VectorizationInterleave;

#if SIFIVE_CUSTOMIZATION
  // Set undefined force hint to enabled if force-vectorization is true.
  if (ForceVectorization &&
      ((LoopVectorizeHints::ForceKind)Force.Value == FK_Undefined))
    Force.Value = FK_Enabled;
#endif // SIFIVE_CUSTOMIZATION

  // If the metadata doesn't explicitly specify whether to enable scalable
  // vectorization, then decide based on the following criteria (increasing
  // level of priority):
  //  - Target default
  //  - Metadata width
  //  - Force option (always overrides)
  if ((LoopVectorizeHints::ScalableForceKind)Scalable.Value == SK_Unspecified) {
    if (TTI)
      Scalable.Value = TTI->enableScalableVectorization() ? SK_PreferScalable
                                                          : SK_FixedWidthOnly;

    if (Width.Value)
      // If the width is set, but the metadata says nothing about the scalable
      // property, then assume it concerns only a fixed-width UserVF.
      // If width is not set, the flag takes precedence.
      Scalable.Value = SK_FixedWidthOnly;
  }

  // If the flag is set to force any use of scalable vectors, override the loop
  // hints.
  if (ForceScalableVectorization.getValue() !=
      LoopVectorizeHints::SK_Unspecified)
    Scalable.Value = ForceScalableVectorization.getValue();

  // Scalable vectorization is disabled if no preference is specified.
  if ((LoopVectorizeHints::ScalableForceKind)Scalable.Value == SK_Unspecified)
    Scalable.Value = SK_FixedWidthOnly;
#if SIFIVE_CUSTOMIZATION
  else if (ForceScalableVectorization == SK_ScalableOnly)
    // If the flag is set to disable any use of fixed vectors, override the
    // loop hint.
    Scalable.Value = SK_ScalableOnly;

  bool UseVLAVectorizer = TTI && allowVLAVectorizer(*TTI, *L);
  if (UseVLAVectorizer && ForceScalableVectorization == SK_Unspecified)
    Scalable.Value = SK_ScalableOnly;

  // Forced vector width from the metadata should be ignored if VLA is enabled
  // if it is suggesting a fixed vector width.
  if (UseVLAVectorizer && Width.Value) {
    Width.Value = VectorizerParams::DefaultVectorizationFactor;
    ORE.emit([&]() {
      return DiagnosticInfoOptimizationFailure(DEBUG_TYPE, "IgnoreUserVF",
                                               L->getStartLoc(), L->getHeader())
             << "ignoring user-specified vector width because RVV VLA "
                "vectorization was enabled. Consider using '#pragma clang rvv "
                "lmul_sew(LMUL, SEW)' instead";
    });
  }
#endif // SIFIVE_CUSTOMIZATION

  if (IsVectorized.Value != 1)
    // If the vectorization width and interleaving count are both 1 then
    // consider the loop to have been already vectorized because there's
    // nothing more that we can do.
    IsVectorized.Value =
        getWidth() == ElementCount::getFixed(1) && getInterleave() == 1;
  LLVM_DEBUG(if (InterleaveOnlyWhenForced && getInterleave() == 1) dbgs()
             << "LV: Interleaving disabled by the pass manager\n");
}

#if SIFIVE_CUSTOMIZATION
std::optional<int> LoopVectorizeHints::getLMULExp() const {
  if (LmulSew.Lmul == INT_MAX)
    return std::nullopt;
  return LmulSew.Lmul;
}

std::optional<unsigned> LoopVectorizeHints::getSEW() const {
  if (LmulSew.Value == -1U)
    return std::nullopt;
  return LmulSew.Value;
}

ElementCount LoopVectorizeHints::getWidth() const {
  // Since a lot of code depends on this function, build ElementCount from LMUL
  // and SEW here
  std::optional<int> LMULExp = getLMULExp();
  std::optional<unsigned> SEW = getSEW();
  if (LMULExp && SEW)
    return RVVPair::getElementCount(*LMULExp, *SEW);
  return ElementCount::get(
      Width.Value, ((ScalableForceKind)Scalable.Value == SK_PreferScalable ||
                    (ScalableForceKind)Scalable.Value == SK_ScalableOnly));
}
#endif // SIFIVE_CUSTOMIZATION

void LoopVectorizeHints::setAlreadyVectorized() {
  LLVMContext &Context = TheLoop->getHeader()->getContext();

  MDNode *IsVectorizedMD = MDNode::get(
      Context,
      {MDString::get(Context, "llvm.loop.isvectorized"),
       ConstantAsMetadata::get(ConstantInt::get(Context, APInt(32, 1)))});
  MDNode *LoopID = TheLoop->getLoopID();
  MDNode *NewLoopID =
      makePostTransformationMetadata(Context, LoopID,
                                     {Twine(Prefix(), "vectorize.").str(),
#if SIFIVE_CUSTOMIZATION
                                      Twine(Prefix(), "interleave.").str(),
                                      LoopMetaData::NoScevChecks},
#endif // SIFIVE_CUSTOMIZATION
                                     {IsVectorizedMD});
  TheLoop->setLoopID(NewLoopID);

  // Update internal cache.
  IsVectorized.Value = 1;
}

#if SIFIVE_CUSTOMIZATION
void LoopVectorizeHints::setRevectorizeWithoutStrideChecks() {
  LLVMContext &Context = TheLoop->getHeader()->getContext();

  MDNode *RevectorizeMD = MDNode::get(
      Context,
      {MDString::get(Context, LoopMetaData::NoScevChecks),
       ConstantAsMetadata::get(ConstantInt::get(Context, APInt(32, 1)))});
  MDNode *LoopID = TheLoop->getLoopID();
  MDNode *NewLoopID = makePostTransformationMetadata(
      Context, LoopID, std::nullopt, {RevectorizeMD});
  TheLoop->setLoopID(NewLoopID);
}

void LoopVectorizeHints::setVectorizeWithoutStrideChecks() {
  LLVMContext &Context = TheLoop->getHeader()->getContext();

  MDNode *RevectorizeMD = MDNode::get(
      Context,
      {MDString::get(Context, LoopMetaData::NoScevStrideChecks),
       ConstantAsMetadata::get(ConstantInt::get(Context, APInt(32, 1)))});
  MDNode *LoopID = TheLoop->getLoopID();
  MDNode *NewLoopID = makePostTransformationMetadata(
      Context, LoopID, std::nullopt, {RevectorizeMD});
  TheLoop->setLoopID(NewLoopID);
}
#endif // SIFIVE_CUSTOMIZATION

bool LoopVectorizeHints::allowVectorization(
    Function *F, Loop *L, bool VectorizeOnlyWhenForced) const {
  if (getForce() == LoopVectorizeHints::FK_Disabled) {
#if SIFIVE_CUSTOMIZATION
    if (ForceVectorization)
      LLVM_DEBUG(
          dbgs() << "LV: Not vectorizing, Respects #pragma vectorize(disable) "
                    "over option 'force-vectorization'\n");
#endif // SIFIVE_CUSTOMIZATION
    LLVM_DEBUG(dbgs() << "LV: Not vectorizing: #pragma vectorize disable.\n");
    emitRemarkWithHints();
    return false;
  }

  if (VectorizeOnlyWhenForced && getForce() != LoopVectorizeHints::FK_Enabled) {
    LLVM_DEBUG(dbgs() << "LV: Not vectorizing: No #pragma vectorize enable.\n");
    emitRemarkWithHints();
    return false;
  }

  if (getIsVectorized() == 1) {
    LLVM_DEBUG(dbgs() << "LV: Not vectorizing: Disabled/already vectorized.\n");
    // FIXME: Add interleave.disable metadata. This will allow
    // vectorize.disable to be used without disabling the pass and errors
    // to differentiate between disabled vectorization and a width of 1.
    ORE.emit([&]() {
      return OptimizationRemarkAnalysis(vectorizeAnalysisPassName(),
                                        "AllDisabled", L->getStartLoc(),
                                        L->getHeader())
             << "loop not vectorized: vectorization and interleaving are "
                "explicitly disabled, or the loop has already been "
                "vectorized";
    });
    return false;
  }

  return true;
}

void LoopVectorizeHints::emitRemarkWithHints() const {
  using namespace ore;

  ORE.emit([&]() {
    if (Force.Value == LoopVectorizeHints::FK_Disabled)
      return OptimizationRemarkMissed(LV_NAME, "MissedExplicitlyDisabled",
                                      TheLoop->getStartLoc(),
                                      TheLoop->getHeader())
             << "loop not vectorized: vectorization is explicitly disabled";

    OptimizationRemarkMissed R(LV_NAME, "MissedDetails", TheLoop->getStartLoc(),
                               TheLoop->getHeader());
    R << "loop not vectorized";
    if (Force.Value == LoopVectorizeHints::FK_Enabled) {
      R << " (Force=" << NV("Force", true);
      if (Width.Value != 0)
        R << ", Vector Width=" << NV("VectorWidth", getWidth());
      if (getInterleave() != 0)
        R << ", Interleave Count=" << NV("InterleaveCount", getInterleave());
      R << ")";
    }
    return R;
  });
}

const char *LoopVectorizeHints::vectorizeAnalysisPassName() const {
  if (getWidth() == ElementCount::getFixed(1))
    return LV_NAME;
  if (getForce() == LoopVectorizeHints::FK_Disabled)
    return LV_NAME;
  if (getForce() == LoopVectorizeHints::FK_Undefined && getWidth().isZero())
    return LV_NAME;
  return OptimizationRemarkAnalysis::AlwaysPrint;
}

bool LoopVectorizeHints::allowReordering() const {
  // Allow the vectorizer to change the order of operations if enabling
  // loop hints are provided
  ElementCount EC = getWidth();
  return HintsAllowReordering &&
         (getForce() == LoopVectorizeHints::FK_Enabled ||
          EC.getKnownMinValue() > 1);
}

#if SIFIVE_CUSTOMIZATION
void LoopVectorizeHints::getHintsFromMetadata(const bool ReportInvalid) {
#else
void LoopVectorizeHints::getHintsFromMetadata() {
#endif // SIFIVE_CUSTOMIZATION
  MDNode *LoopID = TheLoop->getLoopID();
  if (!LoopID)
    return;

  // First operand should refer to the loop id itself.
  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  for (const MDOperand &MDO : llvm::drop_begin(LoopID->operands())) {
    const MDString *S = nullptr;
    SmallVector<Metadata *, 4> Args;

    // The expected hint is either a MDString or a MDNode with the first
    // operand a MDString.
    if (const MDNode *MD = dyn_cast<MDNode>(MDO)) {
      if (!MD || MD->getNumOperands() == 0)
        continue;
      S = dyn_cast<MDString>(MD->getOperand(0));
      for (unsigned Idx = 1; Idx < MD->getNumOperands(); ++Idx)
        Args.push_back(MD->getOperand(Idx));
    } else {
      S = dyn_cast<MDString>(MDO);
      assert(Args.size() == 0 && "too many arguments for MDString");
    }

    if (!S)
      continue;

    // Check if the hint starts with the loop metadata prefix.
    StringRef Name = S->getString();
    if (Args.size() == 1)
#if SIFIVE_CUSTOMIZATION
      setHint(Name, Args[0], ReportInvalid);

    if (Args.size() == 2)
      setLmulSewHint(Name, Args, ReportInvalid);
#else
      setHint(Name, Args[0]);
#endif // SIFIVE_CUSTOMIZATION
  }
}

#if SIFIVE_CUSTOMIZATION
void LoopVectorizeHints::setLmulSewHint(StringRef Name,
                                        ArrayRef<Metadata *> Args,
                                        const bool ReportInvalid) {
  if (!Name.starts_with(Prefix()))
    return;

  Name = Name.substr(Prefix().size(), StringRef::npos);
  if (Name != LmulSew.Name)
    return;

  auto *C = mdconst::dyn_extract<ConstantInt>(Args[0]);
  if (!C) {
    LLVM_DEBUG(dbgs() << "LV: `llvm.loop.vectorize.lmul_sew` metadata "
                         "suppose to have ConstantInt as a first operand\n");
    return;
  }
  int LMUL = C->getSExtValue();

  C = mdconst::dyn_extract<ConstantInt>(Args[1]);
  if (!C) {
    LLVM_DEBUG(dbgs() << "LV: `llvm.loop.vectorize.lmul_sew` metadata "
                         "suppose to have ConstantInt as a second operand\n");
    return;
  }
  unsigned SEW = C->getZExtValue();

  bool UnsupportedPair = false;
  if (SEW >= 64 && LMUL < 0)
    UnsupportedPair = true;
  else if (SEW >= 32 && LMUL < -1)
    UnsupportedPair = true;
  else if (SEW >= 16 && LMUL < -2)
    UnsupportedPair = true;

  if (UnsupportedPair) {
    Force.Value = FK_Undefined;
    LLVM_DEBUG(dbgs() << "LV: ignore 'lmul_sew' clause and assume "
                         "vectorization is not forced\n");
    if (ReportInvalid) {
      ORE.emit([&]() {
        return DiagnosticInfoOptimizationFailure(DEBUG_TYPE, "IgnoreUserVF",
                                                 TheLoop->getStartLoc(),
                                                 TheLoop->getHeader())
               << "ignoring user-specified '#pragma clang rvv "
                  "lmul_sew' specified pair is not valid";
      });
    }
    return;
  }
  LmulSew.Value = SEW;
  LmulSew.Lmul = LMUL;
}

void LoopVectorizeHints::setHint(StringRef Name, Metadata *Arg,
                                 const bool ReportInvalid) {
#else
void LoopVectorizeHints::setHint(StringRef Name, Metadata *Arg) {
#endif // SIFIVE_CUSTOMIZATION
  if (!Name.starts_with(Prefix()))
    return;
  Name = Name.substr(Prefix().size(), StringRef::npos);

  const ConstantInt *C = mdconst::dyn_extract<ConstantInt>(Arg);
  if (!C)
    return;
  unsigned Val = C->getZExtValue();

  Hint *Hints[] = {&Width,        &Interleave, &Force,
                   &IsVectorized, &Predicate,  &Scalable};
  for (auto *H : Hints) {
    if (Name == H->Name) {
#if SIFIVE_CUSTOMIZATION
      if (H->validate(Val)) {
        H->Value = Val;
      } else {
        LLVM_DEBUG(dbgs() << "LV: ignoring invalid hint '" << Name << "'\n");
        if (ReportInvalid) {
          ORE.emit([&]() {
            StringRef HintName = "<unknown>";
            switch (H->Kind) {
            case HK_WIDTH:
              HintName = "vectorize_width";
              break;
            case HK_INTERLEAVE:
              HintName = "interleave_count";
              break;
            default:
              break;
            }
            return OptimizationRemarkAnalysis(
                       vectorizeAnalysisPassName(), "loop remark",
                       TheLoop->getStartLoc(), TheLoop->getHeader())
                   << "loop remark: ignoring invalid " << HintName << " value";
          });
        }
      }
#else
      if (H->validate(Val))
        H->Value = Val;
      else
        LLVM_DEBUG(dbgs() << "LV: ignoring invalid hint '" << Name << "'\n");
#endif // SIFIVE_CUSTOMIZATION
      break;
    }
  }
}

// Return true if the inner loop \p Lp is uniform with regard to the outer loop
// \p OuterLp (i.e., if the outer loop is vectorized, all the vector lanes
// executing the inner loop will execute the same iterations). This check is
// very constrained for now but it will be relaxed in the future. \p Lp is
// considered uniform if it meets all the following conditions:
//   1) it has a canonical IV (starting from 0 and with stride 1),
//   2) its latch terminator is a conditional branch and,
//   3) its latch condition is a compare instruction whose operands are the
//      canonical IV and an OuterLp invariant.
// This check doesn't take into account the uniformity of other conditions not
// related to the loop latch because they don't affect the loop uniformity.
//
// NOTE: We decided to keep all these checks and its associated documentation
// together so that we can easily have a picture of the current supported loop
// nests. However, some of the current checks don't depend on \p OuterLp and
// would be redundantly executed for each \p Lp if we invoked this function for
// different candidate outer loops. This is not the case for now because we
// don't currently have the infrastructure to evaluate multiple candidate outer
// loops and \p OuterLp will be a fixed parameter while we only support explicit
// outer loop vectorization. It's also very likely that these checks go away
// before introducing the aforementioned infrastructure. However, if this is not
// the case, we should move the \p OuterLp independent checks to a separate
// function that is only executed once for each \p Lp.
static bool isUniformLoop(Loop *Lp, Loop *OuterLp) {
  assert(Lp->getLoopLatch() && "Expected loop with a single latch.");

  // If Lp is the outer loop, it's uniform by definition.
  if (Lp == OuterLp)
    return true;
  assert(OuterLp->contains(Lp) && "OuterLp must contain Lp.");

  // 1.
  PHINode *IV = Lp->getCanonicalInductionVariable();
  if (!IV) {
    LLVM_DEBUG(dbgs() << "LV: Canonical IV not found.\n");
    return false;
  }

  // 2.
  BasicBlock *Latch = Lp->getLoopLatch();
  auto *LatchBr = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LatchBr || LatchBr->isUnconditional()) {
    LLVM_DEBUG(dbgs() << "LV: Unsupported loop latch branch.\n");
    return false;
  }

  // 3.
  auto *LatchCmp = dyn_cast<CmpInst>(LatchBr->getCondition());
  if (!LatchCmp) {
    LLVM_DEBUG(
        dbgs() << "LV: Loop latch condition is not a compare instruction.\n");
    return false;
  }

  Value *CondOp0 = LatchCmp->getOperand(0);
  Value *CondOp1 = LatchCmp->getOperand(1);
  Value *IVUpdate = IV->getIncomingValueForBlock(Latch);
  if (!(CondOp0 == IVUpdate && OuterLp->isLoopInvariant(CondOp1)) &&
      !(CondOp1 == IVUpdate && OuterLp->isLoopInvariant(CondOp0))) {
    LLVM_DEBUG(dbgs() << "LV: Loop latch condition is not uniform.\n");
    return false;
  }

  return true;
}

// Return true if \p Lp and all its nested loops are uniform with regard to \p
// OuterLp.
static bool isUniformLoopNest(Loop *Lp, Loop *OuterLp) {
  if (!isUniformLoop(Lp, OuterLp))
    return false;

  // Check if nested loops are uniform.
  for (Loop *SubLp : *Lp)
    if (!isUniformLoopNest(SubLp, OuterLp))
      return false;

  return true;
}

static Type *convertPointerToIntegerType(const DataLayout &DL, Type *Ty) {
  if (Ty->isPointerTy())
    return DL.getIntPtrType(Ty);

  // It is possible that char's or short's overflow when we ask for the loop's
  // trip count, work around this by changing the type size.
  if (Ty->getScalarSizeInBits() < 32)
    return Type::getInt32Ty(Ty->getContext());

  return Ty;
}

static Type *getWiderType(const DataLayout &DL, Type *Ty0, Type *Ty1) {
  Ty0 = convertPointerToIntegerType(DL, Ty0);
  Ty1 = convertPointerToIntegerType(DL, Ty1);
  if (Ty0->getScalarSizeInBits() > Ty1->getScalarSizeInBits())
    return Ty0;
  return Ty1;
}

/// Check that the instruction has outside loop users and is not an
/// identified reduction variable.
static bool hasOutsideLoopUser(const Loop *TheLoop, Instruction *Inst,
                               SmallPtrSetImpl<Value *> &AllowedExit) {
  // Reductions, Inductions and non-header phis are allowed to have exit users. All
  // other instructions must not have external users.
  if (!AllowedExit.count(Inst))
    // Check that all of the users of the loop are inside the BB.
    for (User *U : Inst->users()) {
      Instruction *UI = cast<Instruction>(U);
      // This user may be a reduction exit value.
      if (!TheLoop->contains(UI)) {
        LLVM_DEBUG(dbgs() << "LV: Found an outside user for : " << *UI << '\n');
        return true;
      }
    }
  return false;
}

/// Returns true if A and B have same pointer operands or same SCEVs addresses
static bool storeToSameAddress(ScalarEvolution *SE, StoreInst *A,
                               StoreInst *B) {
  // Compare store
  if (A == B)
    return true;

  // Otherwise Compare pointers
  Value *APtr = A->getPointerOperand();
  Value *BPtr = B->getPointerOperand();
  if (APtr == BPtr)
    return true;

  // Otherwise compare address SCEVs
  return SE->getSCEV(APtr) == SE->getSCEV(BPtr);
}

int LoopVectorizationLegality::isConsecutivePtr(Type *AccessTy,
                                                Value *Ptr) const {
  // FIXME: Currently, the set of symbolic strides is sometimes queried before
  // it's collected.  This happens from canVectorizeWithIfConvert, when the
  // pointer is checked to reference consecutive elements suitable for a
  // masked access.
  const auto &Strides =
    LAI ? LAI->getSymbolicStrides() : DenseMap<Value *, const SCEV *>();

  bool CanAddPredicate = !llvm::shouldOptimizeForSize(
      TheLoop->getHeader(), PSI, BFI, PGSOQueryType::IRPass);
  int Stride = getPtrStride(PSE, AccessTy, Ptr, TheLoop, Strides,
                            CanAddPredicate, false).value_or(0);
  if (Stride == 1 || Stride == -1)
    return Stride;
  return 0;
}

#if SIFIVE_CUSTOMIZATION
std::optional<int64_t>
LoopVectorizationLegality::isConsecutiveOrUnknownPtr(Type *AccessTy,
                                                     Value *Ptr) const {
  const auto &Strides =
      LAI ? LAI->getSymbolicStrides() : DenseMap<Value *, const SCEV *>();

  Function *F = TheLoop->getHeader()->getParent();
  bool OptForSize = F->hasOptSize() ||
                    llvm::shouldOptimizeForSize(TheLoop->getHeader(), PSI, BFI,
                                                PGSOQueryType::IRPass);
  bool CanAddPredicate = !OptForSize;
  std::optional<int64_t> Stride = getPtrStride(PSE, AccessTy, Ptr, TheLoop,
                                               Strides, CanAddPredicate, false);
  if (!Stride.has_value())
    return std::nullopt;
  if (Stride.value() == 1 || Stride.value() == -1)
    return Stride;
  return 0;
}
#endif // SIFIVE_CUSTOMIZATION

bool LoopVectorizationLegality::isInvariant(Value *V) const {
  return LAI->isInvariant(V);
}

namespace {
/// A rewriter to build the SCEVs for each of the VF lanes in the expected
/// vectorized loop, which can then be compared to detect their uniformity. This
/// is done by replacing the AddRec SCEVs of the original scalar loop (TheLoop)
/// with new AddRecs where the step is multiplied by StepMultiplier and Offset *
/// Step is added. Also checks if all sub-expressions are analyzable w.r.t.
/// uniformity.
class SCEVAddRecForUniformityRewriter
    : public SCEVRewriteVisitor<SCEVAddRecForUniformityRewriter> {
  /// Multiplier to be applied to the step of AddRecs in TheLoop.
  unsigned StepMultiplier;

  /// Offset to be added to the AddRecs in TheLoop.
  unsigned Offset;

  /// Loop for which to rewrite AddRecsFor.
  Loop *TheLoop;

  /// Is any sub-expressions not analyzable w.r.t. uniformity?
  bool CannotAnalyze = false;

  bool canAnalyze() const { return !CannotAnalyze; }

public:
  SCEVAddRecForUniformityRewriter(ScalarEvolution &SE, unsigned StepMultiplier,
                                  unsigned Offset, Loop *TheLoop)
      : SCEVRewriteVisitor(SE), StepMultiplier(StepMultiplier), Offset(Offset),
        TheLoop(TheLoop) {}

  const SCEV *visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    assert(Expr->getLoop() == TheLoop &&
           "addrec outside of TheLoop must be invariant and should have been "
           "handled earlier");
    // Build a new AddRec by multiplying the step by StepMultiplier and
    // incrementing the start by Offset * step.
    Type *Ty = Expr->getType();
    const SCEV *Step = Expr->getStepRecurrence(SE);
    if (!SE.isLoopInvariant(Step, TheLoop)) {
      CannotAnalyze = true;
      return Expr;
    }
    const SCEV *NewStep =
        SE.getMulExpr(Step, SE.getConstant(Ty, StepMultiplier));
    const SCEV *ScaledOffset = SE.getMulExpr(Step, SE.getConstant(Ty, Offset));
    const SCEV *NewStart = SE.getAddExpr(Expr->getStart(), ScaledOffset);
    return SE.getAddRecExpr(NewStart, NewStep, TheLoop, SCEV::FlagAnyWrap);
  }

  const SCEV *visit(const SCEV *S) {
    if (CannotAnalyze || SE.isLoopInvariant(S, TheLoop))
      return S;
    return SCEVRewriteVisitor<SCEVAddRecForUniformityRewriter>::visit(S);
  }

  const SCEV *visitUnknown(const SCEVUnknown *S) {
    if (SE.isLoopInvariant(S, TheLoop))
      return S;
    // The value could vary across iterations.
    CannotAnalyze = true;
    return S;
  }

  const SCEV *visitCouldNotCompute(const SCEVCouldNotCompute *S) {
    // Could not analyze the expression.
    CannotAnalyze = true;
    return S;
  }

  static const SCEV *rewrite(const SCEV *S, ScalarEvolution &SE,
                             unsigned StepMultiplier, unsigned Offset,
                             Loop *TheLoop) {
    /// Bail out if the expression does not contain an UDiv expression.
    /// Uniform values which are not loop invariant require operations to strip
    /// out the lowest bits. For now just look for UDivs and use it to avoid
    /// re-writing UDIV-free expressions for other lanes to limit compile time.
    if (!SCEVExprContains(S,
                          [](const SCEV *S) { return isa<SCEVUDivExpr>(S); }))
      return SE.getCouldNotCompute();

    SCEVAddRecForUniformityRewriter Rewriter(SE, StepMultiplier, Offset,
                                             TheLoop);
    const SCEV *Result = Rewriter.visit(S);

    if (Rewriter.canAnalyze())
      return Result;
    return SE.getCouldNotCompute();
  }
};

} // namespace

bool LoopVectorizationLegality::isUniform(Value *V, ElementCount VF) const {
  if (isInvariant(V))
    return true;
  if (VF.isScalable())
    return false;
  if (VF.isScalar())
    return true;

  // Since we rely on SCEV for uniformity, if the type is not SCEVable, it is
  // never considered uniform.
  auto *SE = PSE.getSE();
  if (!SE->isSCEVable(V->getType()))
    return false;
  const SCEV *S = SE->getSCEV(V);

  // Rewrite AddRecs in TheLoop to step by VF and check if the expression for
  // lane 0 matches the expressions for all other lanes.
  unsigned FixedVF = VF.getKnownMinValue();
  const SCEV *FirstLaneExpr =
      SCEVAddRecForUniformityRewriter::rewrite(S, *SE, FixedVF, 0, TheLoop);
  if (isa<SCEVCouldNotCompute>(FirstLaneExpr))
    return false;

  // Make sure the expressions for lanes FixedVF-1..1 match the expression for
  // lane 0. We check lanes in reverse order for compile-time, as frequently
  // checking the last lane is sufficient to rule out uniformity.
  return all_of(reverse(seq<unsigned>(1, FixedVF)), [&](unsigned I) {
    const SCEV *IthLaneExpr =
        SCEVAddRecForUniformityRewriter::rewrite(S, *SE, FixedVF, I, TheLoop);
    return FirstLaneExpr == IthLaneExpr;
  });
}

bool LoopVectorizationLegality::isUniformMemOp(Instruction &I,
                                               ElementCount VF) const {
  Value *Ptr = getLoadStorePointerOperand(&I);
  if (!Ptr)
    return false;
  // Note: There's nothing inherent which prevents predicated loads and
  // stores from being uniform.  The current lowering simply doesn't handle
  // it; in particular, the cost model distinguishes scatter/gather from
  // scalar w/predication, and we currently rely on the scalar path.
  return isUniform(Ptr, VF) && !blockNeedsPredication(I.getParent());
}

bool LoopVectorizationLegality::canVectorizeOuterLoop() {
  assert(!TheLoop->isInnermost() && "We are not vectorizing an outer loop.");
  // Store the result and return it at the end instead of exiting early, in case
  // allowExtraAnalysis is used to report multiple reasons for not vectorizing.
  bool Result = true;
  bool DoExtraAnalysis = ORE->allowExtraAnalysis(DEBUG_TYPE);

  for (BasicBlock *BB : TheLoop->blocks()) {
    // Check whether the BB terminator is a BranchInst. Any other terminator is
    // not supported yet.
    auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Br) {
      reportVectorizationFailure("Unsupported basic block terminator",
          "loop control flow is not understood by vectorizer",
          "CFGNotUnderstood", ORE, TheLoop);
      if (DoExtraAnalysis)
        Result = false;
      else
        return false;
    }

    // Check whether the BranchInst is a supported one. Only unconditional
    // branches, conditional branches with an outer loop invariant condition or
    // backedges are supported.
    // FIXME: We skip these checks when VPlan predication is enabled as we
    // want to allow divergent branches. This whole check will be removed
    // once VPlan predication is on by default.
    if (Br && Br->isConditional() &&
        !TheLoop->isLoopInvariant(Br->getCondition()) &&
        !LI->isLoopHeader(Br->getSuccessor(0)) &&
        !LI->isLoopHeader(Br->getSuccessor(1))) {
      reportVectorizationFailure("Unsupported conditional branch",
          "loop control flow is not understood by vectorizer",
          "CFGNotUnderstood", ORE, TheLoop);
      if (DoExtraAnalysis)
        Result = false;
      else
        return false;
    }
  }

  // Check whether inner loops are uniform. At this point, we only support
  // simple outer loops scenarios with uniform nested loops.
  if (!isUniformLoopNest(TheLoop /*loop nest*/,
                         TheLoop /*context outer loop*/)) {
    reportVectorizationFailure("Outer loop contains divergent loops",
        "loop control flow is not understood by vectorizer",
        "CFGNotUnderstood", ORE, TheLoop);
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // Check whether we are able to set up outer loop induction.
  if (!setupOuterLoopInductions()) {
    reportVectorizationFailure("Unsupported outer loop Phi(s)",
                               "UnsupportedPhi", ORE, TheLoop);
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  return Result;
}

void LoopVectorizationLegality::addInductionPhi(
    PHINode *Phi, const InductionDescriptor &ID,
    SmallPtrSetImpl<Value *> &AllowedExit) {
  Inductions[Phi] = ID;

  // In case this induction also comes with casts that we know we can ignore
  // in the vectorized loop body, record them here. All casts could be recorded
  // here for ignoring, but suffices to record only the first (as it is the
  // only one that may bw used outside the cast sequence).
  const SmallVectorImpl<Instruction *> &Casts = ID.getCastInsts();
  if (!Casts.empty())
    InductionCastsToIgnore.insert(*Casts.begin());

  Type *PhiTy = Phi->getType();
  const DataLayout &DL = Phi->getDataLayout();

  // Get the widest type.
  if (!PhiTy->isFloatingPointTy()) {
    if (!WidestIndTy)
      WidestIndTy = convertPointerToIntegerType(DL, PhiTy);
    else
      WidestIndTy = getWiderType(DL, PhiTy, WidestIndTy);
  }

  // Int inductions are special because we only allow one IV.
  if (ID.getKind() == InductionDescriptor::IK_IntInduction &&
      ID.getConstIntStepValue() && ID.getConstIntStepValue()->isOne() &&
      isa<Constant>(ID.getStartValue()) &&
      cast<Constant>(ID.getStartValue())->isNullValue()) {

    // Use the phi node with the widest type as induction. Use the last
    // one if there are multiple (no good reason for doing this other
    // than it is expedient). We've checked that it begins at zero and
    // steps by one, so this is a canonical induction variable.
    if (!PrimaryInduction || PhiTy == WidestIndTy)
      PrimaryInduction = Phi;
  }

  // Both the PHI node itself, and the "post-increment" value feeding
  // back into the PHI node may have external users.
  // We can allow those uses, except if the SCEVs we have for them rely
  // on predicates that only hold within the loop, since allowing the exit
  // currently means re-using this SCEV outside the loop (see PR33706 for more
  // details).
  if (PSE.getPredicate().isAlwaysTrue()) {
    AllowedExit.insert(Phi);
    AllowedExit.insert(Phi->getIncomingValueForBlock(TheLoop->getLoopLatch()));
  }

  LLVM_DEBUG(dbgs() << "LV: Found an induction variable.\n");
}

#if SIFIVE_CUSTOMIZATION
void LoopVectorizationLegality::addMonotonic(const MonotonicDescriptor &MD) {
  for (PHINode *P : MD.getPhis())
    MonotonicPhis[P] = MD;
}
#endif // SIFIVE_CUSTOMIZATION

bool LoopVectorizationLegality::setupOuterLoopInductions() {
  BasicBlock *Header = TheLoop->getHeader();

  // Returns true if a given Phi is a supported induction.
  auto IsSupportedPhi = [&](PHINode &Phi) -> bool {
    InductionDescriptor ID;
    if (InductionDescriptor::isInductionPHI(&Phi, TheLoop, PSE, ID) &&
        ID.getKind() == InductionDescriptor::IK_IntInduction) {
      addInductionPhi(&Phi, ID, AllowedExit);
      return true;
    }
    // Bail out for any Phi in the outer loop header that is not a supported
    // induction.
    LLVM_DEBUG(
        dbgs() << "LV: Found unsupported PHI for outer loop vectorization.\n");
    return false;
  };

  return llvm::all_of(Header->phis(), IsSupportedPhi);
}

/// Checks if a function is scalarizable according to the TLI, in
/// the sense that it should be vectorized and then expanded in
/// multiple scalar calls. This is represented in the
/// TLI via mappings that do not specify a vector name, as in the
/// following example:
///
///    const VecDesc VecIntrinsics[] = {
///      {"llvm.phx.abs.i32", "", 4}
///    };
static bool isTLIScalarize(const TargetLibraryInfo &TLI, const CallInst &CI) {
  const StringRef ScalarName = CI.getCalledFunction()->getName();
  bool Scalarize = TLI.isFunctionVectorizable(ScalarName);
  // Check that all known VFs are not associated to a vector
  // function, i.e. the vector name is emty.
  if (Scalarize) {
    ElementCount WidestFixedVF, WidestScalableVF;
    TLI.getWidestVF(ScalarName, WidestFixedVF, WidestScalableVF);
    for (ElementCount VF = ElementCount::getFixed(2);
         ElementCount::isKnownLE(VF, WidestFixedVF); VF *= 2)
      Scalarize &= !TLI.isFunctionVectorizable(ScalarName, VF);
    for (ElementCount VF = ElementCount::getScalable(1);
         ElementCount::isKnownLE(VF, WidestScalableVF); VF *= 2)
      Scalarize &= !TLI.isFunctionVectorizable(ScalarName, VF);
    assert((WidestScalableVF.isZero() || !Scalarize) &&
           "Caller may decide to scalarize a variant using a scalable VF");
  }
  return Scalarize;
}

bool LoopVectorizationLegality::canVectorizeInstrs() {
  BasicBlock *Header = TheLoop->getHeader();

  // For each block in the loop.
  for (BasicBlock *BB : TheLoop->blocks()) {
    // Scan the instructions in the block and look for hazards.
    for (Instruction &I : *BB) {
      if (auto *Phi = dyn_cast<PHINode>(&I)) {
        Type *PhiTy = Phi->getType();
        // Check that this PHI type is allowed.
        if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
            !PhiTy->isPointerTy()) {
          reportVectorizationFailure("Found a non-int non-pointer PHI",
                                     "loop control flow is not understood by vectorizer",
                                     "CFGNotUnderstood", ORE, TheLoop);
          return false;
        }

        // If this PHINode is not in the header block, then we know that we
        // can convert it to select during if-conversion. No need to check if
        // the PHIs in this block are induction or reduction variables.
        if (BB != Header) {
          // Non-header phi nodes that have outside uses can be vectorized. Add
          // them to the list of allowed exits.
          // Unsafe cyclic dependencies with header phis are identified during
          // legalization for reduction, induction and fixed order
          // recurrences.
          AllowedExit.insert(&I);
          continue;
        }

        // We only allow if-converted PHIs with exactly two incoming values.
        if (Phi->getNumIncomingValues() != 2) {
          reportVectorizationFailure("Found an invalid PHI",
              "loop control flow is not understood by vectorizer",
              "CFGNotUnderstood", ORE, TheLoop, Phi);
          return false;
        }

        RecurrenceDescriptor RedDes;
        if (RecurrenceDescriptor::isReductionPHI(Phi, TheLoop, RedDes, DB, AC,
                                                 DT, PSE.getSE())) {
          Requirements->addExactFPMathInst(RedDes.getExactFPMathInst());
          AllowedExit.insert(RedDes.getLoopExitInstr());
          Reductions[Phi] = RedDes;
          continue;
        }

        // We prevent matching non-constant strided pointer IVS to preserve
        // historical vectorizer behavior after a generalization of the
        // IVDescriptor code.  The intent is to remove this check, but we
        // have to fix issues around code quality for such loops first.
        auto IsDisallowedStridedPointerInduction =
            [](const InductionDescriptor &ID) {
              if (AllowStridedPointerIVs)
                return false;
              return ID.getKind() == InductionDescriptor::IK_PtrInduction &&
                     ID.getConstIntStepValue() == nullptr;
            };

        // TODO: Instead of recording the AllowedExit, it would be good to
        // record the complementary set: NotAllowedExit. These include (but may
        // not be limited to):
        // 1. Reduction phis as they represent the one-before-last value, which
        // is not available when vectorized
        // 2. Induction phis and increment when SCEV predicates cannot be used
        // outside the loop - see addInductionPhi
        // 3. Non-Phis with outside uses when SCEV predicates cannot be used
        // outside the loop - see call to hasOutsideLoopUser in the non-phi
        // handling below
        // 4. FixedOrderRecurrence phis that can possibly be handled by
        // extraction.
        // By recording these, we can then reason about ways to vectorize each
        // of these NotAllowedExit.
        InductionDescriptor ID;
        if (InductionDescriptor::isInductionPHI(Phi, TheLoop, PSE, ID) &&
            !IsDisallowedStridedPointerInduction(ID)) {
          addInductionPhi(Phi, ID, AllowedExit);
          Requirements->addExactFPMathInst(ID.getExactFPMathInst());
          continue;
        }

        if (RecurrenceDescriptor::isFixedOrderRecurrence(Phi, TheLoop, DT)) {
          AllowedExit.insert(Phi);
          FixedOrderRecurrences.insert(Phi);
          continue;
        }

        // As a last resort, coerce the PHI to a AddRec expression
        // and re-try classifying it a an induction PHI.
        if (InductionDescriptor::isInductionPHI(Phi, TheLoop, PSE, ID, true) &&
            !IsDisallowedStridedPointerInduction(ID)) {
          addInductionPhi(Phi, ID, AllowedExit);
          continue;
        }
#if SIFIVE_CUSTOMIZATION
        if (useVLAVectorizer() &&
            ((TTI->enableCSAVectorization() && EnableCSA) ||
             ForceVectorization)) {
          CSADescriptor CSADesc =
              CSADescriptor::createCSADescriptor(Phi, TheLoop);
          if (CSADesc.isValidCSA()) {
            LLVM_DEBUG(dbgs()
                       << "LV: found legal CSA opportunity" << *Phi << "\n");
            CSAs.insert({Phi, CSADesc});
            continue;
          }
        }
        if (EnableMonotonics && useVLAVectorizer() &&
                   TTI->enableMonotonicsVectorization()) {
          if (auto MD =
                  MonotonicDescriptor::isMonotonicPHI(Phi, TheLoop, PSE)) {
            ++NumberOfMonotonics;
            addMonotonic(MD);
            continue;
          }
        }
#endif // SIFIVE_CUSTOMIZATION

        reportVectorizationFailure("Found an unidentified PHI",
            "value that could not be identified as "
            "reduction is used outside the loop",
            "NonReductionValueUsedOutsideLoop", ORE, TheLoop, Phi);
        return false;
      } // end of PHI handling

      // We handle calls that:
      //   * Are debug info intrinsics.
      //   * Have a mapping to an IR intrinsic.
      //   * Have a vector version available.
      auto *CI = dyn_cast<CallInst>(&I);

      if (CI && !getVectorIntrinsicIDForCall(CI, TLI) &&
          !isa<DbgInfoIntrinsic>(CI) &&
          !(CI->getCalledFunction() && TLI &&
            (!VFDatabase::getMappings(*CI).empty() ||
             isTLIScalarize(*TLI, *CI)))) {
        // If the call is a recognized math libary call, it is likely that
        // we can vectorize it given loosened floating-point constraints.
        LibFunc Func;
        bool IsMathLibCall =
            TLI && CI->getCalledFunction() &&
            CI->getType()->isFloatingPointTy() &&
            TLI->getLibFunc(CI->getCalledFunction()->getName(), Func) &&
            TLI->hasOptimizedCodeGen(Func);

        if (IsMathLibCall) {
          // TODO: Ideally, we should not use clang-specific language here,
          // but it's hard to provide meaningful yet generic advice.
          // Also, should this be guarded by allowExtraAnalysis() and/or be part
          // of the returned info from isFunctionVectorizable()?
          reportVectorizationFailure(
              "Found a non-intrinsic callsite",
              "library call cannot be vectorized. "
              "Try compiling with -fno-math-errno, -ffast-math, "
              "or similar flags",
              "CantVectorizeLibcall", ORE, TheLoop, CI);
        } else {
          reportVectorizationFailure("Found a non-intrinsic callsite",
                                     "call instruction cannot be vectorized",
                                     "CantVectorizeLibcall", ORE, TheLoop, CI);
        }
        return false;
      }

      // Some intrinsics have scalar arguments and should be same in order for
      // them to be vectorized (i.e. loop invariant).
      if (CI) {
        auto *SE = PSE.getSE();
        Intrinsic::ID IntrinID = getVectorIntrinsicIDForCall(CI, TLI);
        for (unsigned Idx = 0; Idx < CI->arg_size(); ++Idx)
          if (isVectorIntrinsicWithScalarOpAtArg(IntrinID, Idx, TTI)) {
            if (!SE->isLoopInvariant(PSE.getSCEV(CI->getOperand(Idx)),
                                     TheLoop)) {
              reportVectorizationFailure("Found unvectorizable intrinsic",
                  "intrinsic instruction cannot be vectorized",
                  "CantVectorizeIntrinsic", ORE, TheLoop, CI);
              return false;
            }
          }
      }

      // If we found a vectorized variant of a function, note that so LV can
      // make better decisions about maximum VF.
      if (CI && !VFDatabase::getMappings(*CI).empty())
        VecCallVariantsFound = true;

      // Check that the instruction return type is vectorizable.
      // We can't vectorize casts from vector type to scalar type.
      // Also, we can't vectorize extractelement instructions.
      if ((!VectorType::isValidElementType(I.getType()) &&
           !I.getType()->isVoidTy()) ||
          (isa<CastInst>(I) &&
           !VectorType::isValidElementType(I.getOperand(0)->getType())) ||
          isa<ExtractElementInst>(I)) {
        reportVectorizationFailure("Found unvectorizable type",
            "instruction return type cannot be vectorized",
            "CantVectorizeInstructionReturnType", ORE, TheLoop, &I);
        return false;
      }

      // Check that the stored type is vectorizable.
      if (auto *ST = dyn_cast<StoreInst>(&I)) {
        Type *T = ST->getValueOperand()->getType();
        if (!VectorType::isValidElementType(T)) {
          reportVectorizationFailure("Store instruction cannot be vectorized",
                                     "CantVectorizeStore", ORE, TheLoop, ST);
          return false;
        }

        // For nontemporal stores, check that a nontemporal vector version is
        // supported on the target.
        if (ST->getMetadata(LLVMContext::MD_nontemporal)) {
          // Arbitrarily try a vector of 2 elements.
          auto *VecTy = FixedVectorType::get(T, /*NumElts=*/2);
          assert(VecTy && "did not find vectorized version of stored type");
          if (!TTI->isLegalNTStore(VecTy, ST->getAlign())) {
            reportVectorizationFailure(
                "nontemporal store instruction cannot be vectorized",
                "CantVectorizeNontemporalStore", ORE, TheLoop, ST);
            return false;
          }
        }

      } else if (auto *LD = dyn_cast<LoadInst>(&I)) {
        if (LD->getMetadata(LLVMContext::MD_nontemporal)) {
          // For nontemporal loads, check that a nontemporal vector version is
          // supported on the target (arbitrarily try a vector of 2 elements).
          auto *VecTy = FixedVectorType::get(I.getType(), /*NumElts=*/2);
          assert(VecTy && "did not find vectorized version of load type");
          if (!TTI->isLegalNTLoad(VecTy, LD->getAlign())) {
            reportVectorizationFailure(
                "nontemporal load instruction cannot be vectorized",
                "CantVectorizeNontemporalLoad", ORE, TheLoop, LD);
            return false;
          }
        }

        // FP instructions can allow unsafe algebra, thus vectorizable by
        // non-IEEE-754 compliant SIMD units.
        // This applies to floating-point math operations and calls, not memory
        // operations, shuffles, or casts, as they don't change precision or
        // semantics.
      } else if (I.getType()->isFloatingPointTy() && (CI || I.isBinaryOp()) &&
                 !I.isFast()) {
        LLVM_DEBUG(dbgs() << "LV: Found FP op with unsafe algebra.\n");
        Hints->setPotentiallyUnsafe();
      }

      // Reduction instructions are allowed to have exit users.
      // All other instructions must not have external users.
      if (hasOutsideLoopUser(TheLoop, &I, AllowedExit)) {
        // We can safely vectorize loops where instructions within the loop are
        // used outside the loop only if the SCEV predicates within the loop is
        // same as outside the loop. Allowing the exit means reusing the SCEV
        // outside the loop.
        if (PSE.getPredicate().isAlwaysTrue()) {
          AllowedExit.insert(&I);
          continue;
        }
        reportVectorizationFailure("Value cannot be used outside the loop",
                                   "ValueUsedOutsideLoop", ORE, TheLoop, &I);
        return false;
      }
    } // next instr.
  }

  if (!PrimaryInduction) {
    if (Inductions.empty()) {
      reportVectorizationFailure("Did not find one integer induction var",
          "loop induction variable could not be identified",
          "NoInductionVariable", ORE, TheLoop);
      return false;
    }
    if (!WidestIndTy) {
      reportVectorizationFailure("Did not find one integer induction var",
          "integer loop induction variable could not be identified",
          "NoIntegerInductionVariable", ORE, TheLoop);
      return false;
    }
    LLVM_DEBUG(dbgs() << "LV: Did not find one integer induction var.\n");
  }

  // Now we know the widest induction type, check if our found induction
  // is the same size. If it's not, unset it here and InnerLoopVectorizer
  // will create another.
  if (PrimaryInduction && WidestIndTy != PrimaryInduction->getType())
    PrimaryInduction = nullptr;

  return true;
}

/// Find histogram operations that match high-level code in loops:
/// \code
/// buckets[indices[i]]+=step;
/// \endcode
///
/// It matches a pattern starting from \p HSt, which Stores to the 'buckets'
/// array the computed histogram. It uses a BinOp to sum all counts, storing
/// them using a loop-variant index Load from the 'indices' input array.
///
/// On successful matches it updates the STATISTIC 'HistogramsDetected',
/// regardless of hardware support. When there is support, it additionally
/// stores the BinOp/Load pairs in \p HistogramCounts, as well the pointers
/// used to update histogram in \p HistogramPtrs.
static bool findHistogram(LoadInst *LI, StoreInst *HSt, Loop *TheLoop,
                          const PredicatedScalarEvolution &PSE,
                          SmallVectorImpl<HistogramInfo> &Histograms) {

  // Store value must come from a Binary Operation.
  Instruction *HPtrInstr = nullptr;
  BinaryOperator *HBinOp = nullptr;
  if (!match(HSt, m_Store(m_BinOp(HBinOp), m_Instruction(HPtrInstr))))
    return false;

  // BinOp must be an Add or a Sub modifying the bucket value by a
  // loop invariant amount.
  // FIXME: We assume the loop invariant term is on the RHS.
  //        Fine for an immediate/constant, but maybe not a generic value?
  Value *HIncVal = nullptr;
  if (!match(HBinOp, m_Add(m_Load(m_Specific(HPtrInstr)), m_Value(HIncVal))) &&
      !match(HBinOp, m_Sub(m_Load(m_Specific(HPtrInstr)), m_Value(HIncVal))))
    return false;

  // Make sure the increment value is loop invariant.
  if (!TheLoop->isLoopInvariant(HIncVal))
    return false;

  // The address to store is calculated through a GEP Instruction.
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(HPtrInstr);
  if (!GEP)
    return false;

  // Restrict address calculation to constant indices except for the last term.
  Value *HIdx = nullptr;
  for (Value *Index : GEP->indices()) {
    if (HIdx)
      return false;
    if (!isa<ConstantInt>(Index))
      HIdx = Index;
  }

  if (!HIdx)
    return false;

  // Check that the index is calculated by loading from another array. Ignore
  // any extensions.
  // FIXME: Support indices from other sources than a linear load from memory?
  //        We're currently trying to match an operation looping over an array
  //        of indices, but there could be additional levels of indirection
  //        in place, or possibly some additional calculation to form the index
  //        from the loaded data.
  Value *VPtrVal;
  if (!match(HIdx, m_ZExtOrSExtOrSelf(m_Load(m_Value(VPtrVal)))))
    return false;

  // Make sure the index address varies in this loop, not an outer loop.
  const auto *AR = dyn_cast<SCEVAddRecExpr>(PSE.getSE()->getSCEV(VPtrVal));
  if (!AR || AR->getLoop() != TheLoop)
    return false;

  // Ensure we'll have the same mask by checking that all parts of the histogram
  // (gather load, update, scatter store) are in the same block.
  LoadInst *IndexedLoad = cast<LoadInst>(HBinOp->getOperand(0));
  BasicBlock *LdBB = IndexedLoad->getParent();
  if (LdBB != HBinOp->getParent() || LdBB != HSt->getParent())
    return false;

  LLVM_DEBUG(dbgs() << "LV: Found histogram for: " << *HSt << "\n");

  // Store the operations that make up the histogram.
  Histograms.emplace_back(IndexedLoad, HBinOp, HSt);
  return true;
}

bool LoopVectorizationLegality::canVectorizeIndirectUnsafeDependences() {
  // For now, we only support an IndirectUnsafe dependency that calculates
  // a histogram
  if (!EnableHistogramVectorization)
    return false;

  // Find a single IndirectUnsafe dependency.
  const MemoryDepChecker::Dependence *IUDep = nullptr;
  const MemoryDepChecker &DepChecker = LAI->getDepChecker();
  const auto *Deps = DepChecker.getDependences();
  // If there were too many dependences, LAA abandons recording them. We can't
  // proceed safely if we don't know what the dependences are.
  if (!Deps)
    return false;

  for (const MemoryDepChecker::Dependence &Dep : *Deps) {
    // Ignore dependencies that are either known to be safe or can be
    // checked at runtime.
    if (MemoryDepChecker::Dependence::isSafeForVectorization(Dep.Type) !=
        MemoryDepChecker::VectorizationSafetyStatus::Unsafe)
      continue;

    // We're only interested in IndirectUnsafe dependencies here, where the
    // address might come from a load from memory. We also only want to handle
    // one such dependency, at least for now.
    if (Dep.Type != MemoryDepChecker::Dependence::IndirectUnsafe || IUDep)
      return false;

    IUDep = &Dep;
  }
  if (!IUDep)
    return false;

  // For now only normal loads and stores are supported.
  LoadInst *LI = dyn_cast<LoadInst>(IUDep->getSource(DepChecker));
  StoreInst *SI = dyn_cast<StoreInst>(IUDep->getDestination(DepChecker));

  if (!LI || !SI)
    return false;

  LLVM_DEBUG(dbgs() << "LV: Checking for a histogram on: " << *SI << "\n");
  return findHistogram(LI, SI, TheLoop, LAI->getPSE(), Histograms);
}

bool LoopVectorizationLegality::canVectorizeMemory() {
  LAI = &LAIs.getInfo(*TheLoop);
  const OptimizationRemarkAnalysis *LAR = LAI->getReport();
  if (LAR) {
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(Hints->vectorizeAnalysisPassName(),
                                        "loop not vectorized: ", *LAR);
    });
  }

  if (!LAI->canVectorizeMemory())
    return canVectorizeIndirectUnsafeDependences();

  if (LAI->hasLoadStoreDependenceInvolvingLoopInvariantAddress()) {
    reportVectorizationFailure("We don't allow storing to uniform addresses",
                               "write to a loop invariant address could not "
                               "be vectorized",
                               "CantVectorizeStoreToLoopInvariantAddress", ORE,
                               TheLoop);
    return false;
  }

  // We can vectorize stores to invariant address when final reduction value is
  // guaranteed to be stored at the end of the loop. Also, if decision to
  // vectorize loop is made, runtime checks are added so as to make sure that
  // invariant address won't alias with any other objects.
  if (!LAI->getStoresToInvariantAddresses().empty()) {
    // For each invariant address, check if last stored value is unconditional
    // and the address is not calculated inside the loop.
    for (StoreInst *SI : LAI->getStoresToInvariantAddresses()) {
      if (!isInvariantStoreOfReduction(SI))
        continue;

      if (blockNeedsPredication(SI->getParent())) {
        reportVectorizationFailure(
            "We don't allow storing to uniform addresses",
            "write of conditional recurring variant value to a loop "
            "invariant address could not be vectorized",
            "CantVectorizeStoreToLoopInvariantAddress", ORE, TheLoop);
        return false;
      }

      // Invariant address should be defined outside of loop. LICM pass usually
      // makes sure it happens, but in rare cases it does not, we do not want
      // to overcomplicate vectorization to support this case.
      if (Instruction *Ptr = dyn_cast<Instruction>(SI->getPointerOperand())) {
        if (TheLoop->contains(Ptr)) {
          reportVectorizationFailure(
              "Invariant address is calculated inside the loop",
              "write to a loop invariant address could not "
              "be vectorized",
              "CantVectorizeStoreToLoopInvariantAddress", ORE, TheLoop);
          return false;
        }
      }
    }

    if (LAI->hasStoreStoreDependenceInvolvingLoopInvariantAddress()) {
      // For each invariant address, check its last stored value is the result
      // of one of our reductions.
      //
      // We do not check if dependence with loads exists because that is already
      // checked via hasLoadStoreDependenceInvolvingLoopInvariantAddress.
      ScalarEvolution *SE = PSE.getSE();
      SmallVector<StoreInst *, 4> UnhandledStores;
      for (StoreInst *SI : LAI->getStoresToInvariantAddresses()) {
        if (isInvariantStoreOfReduction(SI)) {
          // Earlier stores to this address are effectively deadcode.
          // With opaque pointers it is possible for one pointer to be used with
          // different sizes of stored values:
          //    store i32 0, ptr %x
          //    store i8 0, ptr %x
          // The latest store doesn't complitely overwrite the first one in the
          // example. That is why we have to make sure that types of stored
          // values are same.
          // TODO: Check that bitwidth of unhandled store is smaller then the
          // one that overwrites it and add a test.
          erase_if(UnhandledStores, [SE, SI](StoreInst *I) {
            return storeToSameAddress(SE, SI, I) &&
                   I->getValueOperand()->getType() ==
                       SI->getValueOperand()->getType();
          });
          continue;
        }
        UnhandledStores.push_back(SI);
      }

      bool IsOK = UnhandledStores.empty();
      // TODO: we should also validate against InvariantMemSets.
      if (!IsOK) {
        reportVectorizationFailure(
            "We don't allow storing to uniform addresses",
            "write to a loop invariant address could not "
            "be vectorized",
            "CantVectorizeStoreToLoopInvariantAddress", ORE, TheLoop);
        return false;
      }
    }
  }

  PSE.addPredicate(LAI->getPSE().getPredicate());
  return true;
}

bool LoopVectorizationLegality::canVectorizeFPMath(
    bool EnableStrictReductions) {

  // First check if there is any ExactFP math or if we allow reassociations
  if (!Requirements->getExactFPInst() || Hints->allowReordering())
    return true;

  // If the above is false, we have ExactFPMath & do not allow reordering.
  // If the EnableStrictReductions flag is set, first check if we have any
  // Exact FP induction vars, which we cannot vectorize.
  if (!EnableStrictReductions ||
      any_of(getInductionVars(), [&](auto &Induction) -> bool {
        InductionDescriptor IndDesc = Induction.second;
        return IndDesc.getExactFPMathInst();
      }))
    return false;

  // We can now only vectorize if all reductions with Exact FP math also
  // have the isOrdered flag set, which indicates that we can move the
  // reduction operations in-loop.
  return (all_of(getReductionVars(), [&](auto &Reduction) -> bool {
    const RecurrenceDescriptor &RdxDesc = Reduction.second;
    return !RdxDesc.hasExactFPMath() || RdxDesc.isOrdered();
  }));
}

bool LoopVectorizationLegality::isInvariantStoreOfReduction(StoreInst *SI) {
  return any_of(getReductionVars(), [&](auto &Reduction) -> bool {
    const RecurrenceDescriptor &RdxDesc = Reduction.second;
    return RdxDesc.IntermediateStore == SI;
  });
}

bool LoopVectorizationLegality::isInvariantAddressOfReduction(Value *V) {
  return any_of(getReductionVars(), [&](auto &Reduction) -> bool {
    const RecurrenceDescriptor &RdxDesc = Reduction.second;
    if (!RdxDesc.IntermediateStore)
      return false;

    ScalarEvolution *SE = PSE.getSE();
    Value *InvariantAddress = RdxDesc.IntermediateStore->getPointerOperand();
    return V == InvariantAddress ||
           SE->getSCEV(V) == SE->getSCEV(InvariantAddress);
  });
}

bool LoopVectorizationLegality::isInductionPhi(const Value *V) const {
  Value *In0 = const_cast<Value *>(V);
  PHINode *PN = dyn_cast_or_null<PHINode>(In0);
  if (!PN)
    return false;

  return Inductions.count(PN);
}

const InductionDescriptor *
LoopVectorizationLegality::getIntOrFpInductionDescriptor(PHINode *Phi) const {
  if (!isInductionPhi(Phi))
    return nullptr;
  auto &ID = getInductionVars().find(Phi)->second;
  if (ID.getKind() == InductionDescriptor::IK_IntInduction ||
      ID.getKind() == InductionDescriptor::IK_FpInduction)
    return &ID;
  return nullptr;
}

const InductionDescriptor *
LoopVectorizationLegality::getPointerInductionDescriptor(PHINode *Phi) const {
  if (!isInductionPhi(Phi))
    return nullptr;
  auto &ID = getInductionVars().find(Phi)->second;
  if (ID.getKind() == InductionDescriptor::IK_PtrInduction)
    return &ID;
  return nullptr;
}

bool LoopVectorizationLegality::isCastedInductionVariable(
    const Value *V) const {
  auto *Inst = dyn_cast<Instruction>(V);
  return (Inst && InductionCastsToIgnore.count(Inst));
}

bool LoopVectorizationLegality::isInductionVariable(const Value *V) const {
  return isInductionPhi(V) || isCastedInductionVariable(V);
}

bool LoopVectorizationLegality::isFixedOrderRecurrence(
    const PHINode *Phi) const {
  return FixedOrderRecurrences.count(Phi);
}

bool LoopVectorizationLegality::blockNeedsPredication(BasicBlock *BB) const {
  // When vectorizing early exits, create predicates for the latch block only.
  // The early exiting block must be a direct predecessor of the latch at the
  // moment.
  BasicBlock *Latch = TheLoop->getLoopLatch();
  if (hasUncountableEarlyExit()) {
    assert(
        is_contained(predecessors(Latch), getUncountableEarlyExitingBlock()) &&
        "Uncountable exiting block must be a direct predecessor of latch");
    return BB == Latch;
  }
  return LoopAccessInfo::blockNeedsPredication(BB, TheLoop, DT);
}

bool LoopVectorizationLegality::blockCanBePredicated(
    BasicBlock *BB, SmallPtrSetImpl<Value *> &SafePtrs,
    SmallPtrSetImpl<const Instruction *> &MaskedOp) const {
  for (Instruction &I : *BB) {
    // We can predicate blocks with calls to assume, as long as we drop them in
    // case we flatten the CFG via predication.
    if (match(&I, m_Intrinsic<Intrinsic::assume>())) {
      MaskedOp.insert(&I);
      continue;
    }

    // Do not let llvm.experimental.noalias.scope.decl block the vectorization.
    // TODO: there might be cases that it should block the vectorization. Let's
    // ignore those for now.
    if (isa<NoAliasScopeDeclInst>(&I))
      continue;

    // We can allow masked calls if there's at least one vector variant, even
    // if we end up scalarizing due to the cost model calculations.
    // TODO: Allow other calls if they have appropriate attributes... readonly
    // and argmemonly?
    if (CallInst *CI = dyn_cast<CallInst>(&I))
      if (VFDatabase::hasMaskedVariant(*CI)) {
        MaskedOp.insert(CI);
        continue;
      }

    // Loads are handled via masking (or speculated if safe to do so.)
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (!SafePtrs.count(LI->getPointerOperand()))
        MaskedOp.insert(LI);
      continue;
    }

    // Predicated store requires some form of masking:
    // 1) masked store HW instruction,
    // 2) emulation via load-blend-store (only if safe and legal to do so,
    //    be aware on the race conditions), or
    // 3) element-by-element predicate check and scalar store.
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      MaskedOp.insert(SI);
      continue;
    }

    if (I.mayReadFromMemory() || I.mayWriteToMemory() || I.mayThrow())
      return false;
  }

  return true;
}

bool LoopVectorizationLegality::canVectorizeWithIfConvert() {
  if (!EnableIfConversion) {
    reportVectorizationFailure("If-conversion is disabled",
                               "IfConversionDisabled", ORE, TheLoop);
    return false;
  }

  assert(TheLoop->getNumBlocks() > 1 && "Single block loops are vectorizable");

  // A list of pointers which are known to be dereferenceable within scope of
  // the loop body for each iteration of the loop which executes.  That is,
  // the memory pointed to can be dereferenced (with the access size implied by
  // the value's type) unconditionally within the loop header without
  // introducing a new fault.
  SmallPtrSet<Value *, 8> SafePointers;

  // Collect safe addresses.
  for (BasicBlock *BB : TheLoop->blocks()) {
    if (!blockNeedsPredication(BB)) {
      for (Instruction &I : *BB)
        if (auto *Ptr = getLoadStorePointerOperand(&I))
          SafePointers.insert(Ptr);
      continue;
    }

    // For a block which requires predication, a address may be safe to access
    // in the loop w/o predication if we can prove dereferenceability facts
    // sufficient to ensure it'll never fault within the loop. For the moment,
    // we restrict this to loads; stores are more complicated due to
    // concurrency restrictions.
    ScalarEvolution &SE = *PSE.getSE();
    SmallVector<const SCEVPredicate *, 4> Predicates;
    for (Instruction &I : *BB) {
      LoadInst *LI = dyn_cast<LoadInst>(&I);
      // Pass the Predicates pointer to isDereferenceableAndAlignedInLoop so
      // that it will consider loops that need guarding by SCEV checks. The
      // vectoriser will generate these checks if we decide to vectorise.
      if (LI && !LI->getType()->isVectorTy() && !mustSuppressSpeculation(*LI) &&
          isDereferenceableAndAlignedInLoop(LI, TheLoop, SE, *DT, AC,
                                            &Predicates))
        SafePointers.insert(LI->getPointerOperand());
      Predicates.clear();
    }
  }

  // Collect the blocks that need predication.
  for (BasicBlock *BB : TheLoop->blocks()) {
    // We support only branches and switch statements as terminators inside the
    // loop.
    if (isa<SwitchInst>(BB->getTerminator())) {
      if (TheLoop->isLoopExiting(BB)) {
        reportVectorizationFailure("Loop contains an unsupported switch",
                                   "LoopContainsUnsupportedSwitch", ORE,
                                   TheLoop, BB->getTerminator());
        return false;
      }
    } else if (!isa<BranchInst>(BB->getTerminator())) {
      reportVectorizationFailure("Loop contains an unsupported terminator",
                                 "LoopContainsUnsupportedTerminator", ORE,
                                 TheLoop, BB->getTerminator());
      return false;
    }

    // We must be able to predicate all blocks that need to be predicated.
    if (blockNeedsPredication(BB) &&
        !blockCanBePredicated(BB, SafePointers, MaskedOp)) {
      reportVectorizationFailure(
          "Control flow cannot be substituted for a select", "NoCFGForSelect",
          ORE, TheLoop, BB->getTerminator());
      return false;
    }
  }

  // We can if-convert this loop.
  return true;
}

// Helper function to canVectorizeLoopNestCFG.
bool LoopVectorizationLegality::canVectorizeLoopCFG(Loop *Lp,
                                                    bool UseVPlanNativePath) {
  assert((UseVPlanNativePath || Lp->isInnermost()) &&
         "VPlan-native path is not enabled.");

  // TODO: ORE should be improved to show more accurate information when an
  // outer loop can't be vectorized because a nested loop is not understood or
  // legal. Something like: "outer_loop_location: loop not vectorized:
  // (inner_loop_location) loop control flow is not understood by vectorizer".

  // Store the result and return it at the end instead of exiting early, in case
  // allowExtraAnalysis is used to report multiple reasons for not vectorizing.
  bool Result = true;
  bool DoExtraAnalysis = ORE->allowExtraAnalysis(DEBUG_TYPE);

  // We must have a loop in canonical form. Loops with indirectbr in them cannot
  // be canonicalized.
  if (!Lp->getLoopPreheader()) {
    reportVectorizationFailure("Loop doesn't have a legal pre-header",
        "loop control flow is not understood by vectorizer",
        "CFGNotUnderstood", ORE, TheLoop);
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // We must have a single backedge.
  if (Lp->getNumBackEdges() != 1) {
    reportVectorizationFailure("The loop must have a single backedge",
        "loop control flow is not understood by vectorizer",
        "CFGNotUnderstood", ORE, TheLoop);
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  return Result;
}

bool LoopVectorizationLegality::canVectorizeLoopNestCFG(
    Loop *Lp, bool UseVPlanNativePath) {
  // Store the result and return it at the end instead of exiting early, in case
  // allowExtraAnalysis is used to report multiple reasons for not vectorizing.
  bool Result = true;
  bool DoExtraAnalysis = ORE->allowExtraAnalysis(DEBUG_TYPE);
  if (!canVectorizeLoopCFG(Lp, UseVPlanNativePath)) {
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // Recursively check whether the loop control flow of nested loops is
  // understood.
  for (Loop *SubLp : *Lp)
    if (!canVectorizeLoopNestCFG(SubLp, UseVPlanNativePath)) {
      if (DoExtraAnalysis)
        Result = false;
      else
        return false;
    }

  return Result;
}

bool LoopVectorizationLegality::isVectorizableEarlyExitLoop() {
  BasicBlock *LatchBB = TheLoop->getLoopLatch();
  if (!LatchBB) {
    reportVectorizationFailure("Loop does not have a latch",
                               "Cannot vectorize early exit loop",
                               "NoLatchEarlyExit", ORE, TheLoop);
    return false;
  }

  if (Reductions.size() || FixedOrderRecurrences.size()) {
    reportVectorizationFailure(
        "Found reductions or recurrences in early-exit loop",
        "Cannot vectorize early exit loop with reductions or recurrences",
        "RecurrencesInEarlyExitLoop", ORE, TheLoop);
    return false;
  }

  SmallVector<BasicBlock *, 8> ExitingBlocks;
  TheLoop->getExitingBlocks(ExitingBlocks);

  // Keep a record of all the exiting blocks.
  SmallVector<const SCEVPredicate *, 4> Predicates;
  for (BasicBlock *BB : ExitingBlocks) {
    const SCEV *EC =
        PSE.getSE()->getPredicatedExitCount(TheLoop, BB, &Predicates);
    if (isa<SCEVCouldNotCompute>(EC)) {
      UncountableExitingBlocks.push_back(BB);

      SmallVector<BasicBlock *, 2> Succs(successors(BB));
      if (Succs.size() != 2) {
        reportVectorizationFailure(
            "Early exiting block does not have exactly two successors",
            "Incorrect number of successors from early exiting block",
            "EarlyExitTooManySuccessors", ORE, TheLoop);
        return false;
      }

      BasicBlock *ExitBlock;
      if (!TheLoop->contains(Succs[0]))
        ExitBlock = Succs[0];
      else {
        assert(!TheLoop->contains(Succs[1]));
        ExitBlock = Succs[1];
      }
      UncountableExitBlocks.push_back(ExitBlock);
    } else
      CountableExitingBlocks.push_back(BB);
  }
  // We can safely ignore the predicates here because when vectorizing the loop
  // the PredicatatedScalarEvolution class will keep track of all predicates
  // for each exiting block anyway. This happens when calling
  // PSE.getSymbolicMaxBackedgeTakenCount() below.
  Predicates.clear();

  // We only support one uncountable early exit.
  if (getUncountableExitingBlocks().size() != 1) {
    reportVectorizationFailure(
        "Loop has too many uncountable exits",
        "Cannot vectorize early exit loop with more than one early exit",
        "TooManyUncountableEarlyExits", ORE, TheLoop);
    return false;
  }

  // The only supported early exit loops so far are ones where the early
  // exiting block is a unique predecessor of the latch block.
  BasicBlock *LatchPredBB = LatchBB->getUniquePredecessor();
  if (LatchPredBB != getUncountableEarlyExitingBlock()) {
    reportVectorizationFailure("Early exit is not the latch predecessor",
                               "Cannot vectorize early exit loop",
                               "EarlyExitNotLatchPredecessor", ORE, TheLoop);
    return false;
  }

  // The latch block must have a countable exit.
  if (isa<SCEVCouldNotCompute>(
          PSE.getSE()->getPredicatedExitCount(TheLoop, LatchBB, &Predicates))) {
    reportVectorizationFailure(
        "Cannot determine exact exit count for latch block",
        "Cannot vectorize early exit loop",
        "UnknownLatchExitCountEarlyExitLoop", ORE, TheLoop);
    return false;
  }
  assert(llvm::is_contained(CountableExitingBlocks, LatchBB) &&
         "Latch block not found in list of countable exits!");

  // Check to see if there are instructions that could potentially generate
  // exceptions or have side-effects.
  auto IsSafeOperation = [](Instruction *I) -> bool {
    switch (I->getOpcode()) {
    case Instruction::Load:
    case Instruction::Store:
    case Instruction::PHI:
    case Instruction::Br:
      // These are checked separately.
      return true;
    default:
      return isSafeToSpeculativelyExecute(I);
    }
  };

  for (auto *BB : TheLoop->blocks())
    for (auto &I : *BB) {
      if (I.mayWriteToMemory()) {
        // We don't support writes to memory.
        reportVectorizationFailure(
            "Writes to memory unsupported in early exit loops",
            "Cannot vectorize early exit loop with writes to memory",
            "WritesInEarlyExitLoop", ORE, TheLoop);
        return false;
      } else if (!IsSafeOperation(&I)) {
        reportVectorizationFailure("Early exit loop contains operations that "
                                   "cannot be speculatively executed",
                                   "UnsafeOperationsEarlyExitLoop", ORE,
                                   TheLoop);
        return false;
      }
    }

  // The vectoriser cannot handle loads that occur after the early exit block.
  assert(LatchBB->getUniquePredecessor() == getUncountableEarlyExitingBlock() &&
         "Expected latch predecessor to be the early exiting block");

  // TODO: Handle loops that may fault.
  Predicates.clear();
  if (!isDereferenceableReadOnlyLoop(TheLoop, PSE.getSE(), DT, AC,
                                     &Predicates)) {
    reportVectorizationFailure(
        "Loop may fault",
        "Cannot vectorize potentially faulting early exit loop",
        "PotentiallyFaultingEarlyExitLoop", ORE, TheLoop);
    return false;
  }

  [[maybe_unused]] const SCEV *SymbolicMaxBTC =
      PSE.getSymbolicMaxBackedgeTakenCount();
  // Since we have an exact exit count for the latch and the early exit
  // dominates the latch, then this should guarantee a computed SCEV value.
  assert(!isa<SCEVCouldNotCompute>(SymbolicMaxBTC) &&
         "Failed to get symbolic expression for backedge taken count");
  LLVM_DEBUG(dbgs() << "LV: Found an early exit loop with symbolic max "
                       "backedge taken count: "
                    << *SymbolicMaxBTC << '\n');
  return true;
}

bool LoopVectorizationLegality::canVectorize(bool UseVPlanNativePath) {
  // Store the result and return it at the end instead of exiting early, in case
  // allowExtraAnalysis is used to report multiple reasons for not vectorizing.
  bool Result = true;

  bool DoExtraAnalysis = ORE->allowExtraAnalysis(DEBUG_TYPE);
  // Check whether the loop-related control flow in the loop nest is expected by
  // vectorizer.
  if (!canVectorizeLoopNestCFG(TheLoop, UseVPlanNativePath)) {
    if (DoExtraAnalysis) {
      LLVM_DEBUG(dbgs() << "LV: legality check failed: loop nest");
      Result = false;
    } else {
      return false;
    }
  }

  // We need to have a loop header.
  LLVM_DEBUG(dbgs() << "LV: Found a loop: " << TheLoop->getHeader()->getName()
                    << '\n');

  // Specific checks for outer loops. We skip the remaining legal checks at this
  // point because they don't support outer loops.
  if (!TheLoop->isInnermost()) {
    assert(UseVPlanNativePath && "VPlan-native path is not enabled.");

    if (!canVectorizeOuterLoop()) {
      reportVectorizationFailure("Unsupported outer loop",
                                 "UnsupportedOuterLoop", ORE, TheLoop);
      // TODO: Implement DoExtraAnalysis when subsequent legal checks support
      // outer loops.
      return false;
    }

    LLVM_DEBUG(dbgs() << "LV: We can vectorize this outer loop!\n");
    return Result;
  }

  assert(TheLoop->isInnermost() && "Inner loop expected.");
  // Check if we can if-convert non-single-bb loops.
  unsigned NumBlocks = TheLoop->getNumBlocks();
  if (NumBlocks != 1 && !canVectorizeWithIfConvert()) {
    LLVM_DEBUG(dbgs() << "LV: Can't if-convert the loop.\n");
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // Check if we can vectorize the instructions and CFG in this loop.
  if (!canVectorizeInstrs()) {
    LLVM_DEBUG(dbgs() << "LV: Can't vectorize the instructions or CFG\n");
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }


#if SIFIVE_CUSTOMIZATION
  // Countable vs uncountable loops
  const SCEV *ExitCount = PSE.getBackedgeTakenCount();
  if (isa<SCEVCouldNotCompute>(ExitCount) && useVLAVectorizer()) {
    // TODO: Consider merging memory safety analysis with LAA.
    // There are false positives that SE categorizes countable loops as
    // uncountable.
    if (canVectorizeUncountableLoop(PSE)) {
      // Vectorizable uncountable loops still need to set up LAI.
      LLVM_DEBUG(dbgs() << "LV: Can vectorize an uncountable loop!\n");
      setVectorizableUncountable();
      // Set up analysis results
      LAI = &LAIs.getInfo(*TheLoop);
      return true;
    }

    // Non-vectorizable uncountable loops need to print the same diagnoistic
    // message as the countable pipeline.
    ORE->emit(OptimizationRemarkAnalysis(
                  Hints->vectorizeAnalysisPassName(), "loop not vectorized: ",
                  OptimizationRemarkAnalysis(
                      DEBUG_TYPE, "CantComputeNumberOfIterations",
                      TheLoop->getStartLoc(), TheLoop->getHeader()))
              << "could not determine number of loop iterations");
    LLVM_DEBUG(dbgs() << "LV: Can't vectorize non-uncountable-loop "
                         "uncountable loops yet\n");
    return false;
  }
  // Memory safety analysis has a separate pipeline for uncountable loops.
  // Skip the countable loop one.
#endif // SIFIVE_CUSTOMIZATION

  HasUncountableEarlyExit = false;

  if (isa<SCEVCouldNotCompute>(PSE.getBackedgeTakenCount())) {
    HasUncountableEarlyExit = true;
    if (!isVectorizableEarlyExitLoop()) {
      UncountableExitingBlocks.clear();
      HasUncountableEarlyExit = false;
      if (DoExtraAnalysis)
        Result = false;
      else
        return false;
    }
  }

  // Go over each instruction and look at memory deps.
  if (!canVectorizeMemory()) {
    LLVM_DEBUG(dbgs() << "LV: Can't vectorize due to memory conflicts\n");
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  if (Result) {
    LLVM_DEBUG(dbgs() << "LV: We can vectorize this loop"
                      << (LAI->getRuntimePointerChecking()->Need
                              ? " (with a runtime bound check)"
                              : "")
                      << "!\n");
  }

  unsigned SCEVThreshold = VectorizeSCEVCheckThreshold;
  if (Hints->getForce() == LoopVectorizeHints::FK_Enabled)
    SCEVThreshold = PragmaVectorizeSCEVCheckThreshold;

  if (PSE.getPredicate().getComplexity() > SCEVThreshold) {
    LLVM_DEBUG(dbgs() << "LV: Vectorization not profitable "
                         "due to SCEVThreshold");
    reportVectorizationFailure("Too many SCEV checks needed",
        "Too many SCEV assumptions need to be made and checked at runtime",
        "TooManySCEVRunTimeChecks", ORE, TheLoop);
    if (DoExtraAnalysis)
      Result = false;
    else
      return false;
  }

  // Okay! We've done all the tests. If any have failed, return false. Otherwise
  // we can vectorize, and at this point we don't have any other mem analysis
  // which may limit our maximum vectorization factor, so just return true with
  // no restrictions.
  return Result;
}

bool LoopVectorizationLegality::canFoldTailByMasking() const {

  LLVM_DEBUG(dbgs() << "LV: checking if tail can be folded by masking.\n");

  SmallPtrSet<const Value *, 8> ReductionLiveOuts;

  for (const auto &Reduction : getReductionVars())
    ReductionLiveOuts.insert(Reduction.second.getLoopExitInstr());

#if SIFIVE_CUSTOMIZATION
  // The limitations that LV has for masking loop body are not applicable to RVV
  // VLA vectorization as loop body is not masked.
  //
  // FIXME: This function shouldn't be called for RVV VLA in the first place.
  // However, there's coupling in CM and TTI that expect all operations to be
  // masked. More specifically, this function places all instructions into
  // `MaskedOp` container, and for unmasked operations TTI returns `Invalid`
  // cost
  // Need to modify TTI to return costs for unmasked operations and then not
  // call this function if we do RVV VLA vectorization.
  if (!useVLAVectorizer()) {
#endif // SIFIVE_CUSTOMIZATION
  // TODO: handle non-reduction outside users when tail is folded by masking.
  for (auto *AE : AllowedExit) {
    // Check that all users of allowed exit values are inside the loop or
    // are the live-out of a reduction.
    if (ReductionLiveOuts.count(AE))
      continue;
    for (User *U : AE->users()) {
      Instruction *UI = cast<Instruction>(U);
      if (TheLoop->contains(UI))
        continue;
      LLVM_DEBUG(
          dbgs()
          << "LV: Cannot fold tail by masking, loop has an outside user for "
          << *UI << "\n");
      return false;
    }
  }
#if SIFIVE_CUSTOMIZATION
  }
#endif // SIFIVE_CUSTOMIZATION

  for (const auto &Entry : getInductionVars()) {
    PHINode *OrigPhi = Entry.first;
    for (User *U : OrigPhi->users()) {
      auto *UI = cast<Instruction>(U);
      if (!TheLoop->contains(UI)) {
        LLVM_DEBUG(dbgs() << "LV: Cannot fold tail by masking, loop IV has an "
                             "outside user for "
                          << *UI << "\n");
        return false;
      }
    }
  }

  // The list of pointers that we can safely read and write to remains empty.
  SmallPtrSet<Value *, 8> SafePointers;

  // Check all blocks for predication, including those that ordinarily do not
  // need predication such as the header block.
  SmallPtrSet<const Instruction *, 8> TmpMaskedOp;
  for (BasicBlock *BB : TheLoop->blocks()) {
    if (!blockCanBePredicated(BB, SafePointers, TmpMaskedOp)) {
      LLVM_DEBUG(dbgs() << "LV: Cannot fold tail by masking.\n");
      return false;
    }
  }

  LLVM_DEBUG(dbgs() << "LV: can fold tail by masking.\n");

  return true;
}

#if SIFIVE_CUSTOMIZATION
bool LoopVectorizationLegality::useVLAVectorizer() const {
  return allowVLAVectorizer(*TTI, *TheLoop);
}

// Return: true - good for vectorization
//         false - bad for vectorization
bool LoopVectorizationLegality::isSpeculationSafe(
    PredicatedScalarEvolution &PSE) {
  ScalarEvolution *SE = PSE.getSE();

  SmallVector<BasicBlock *, 16> ExitingBlocks;
  TheLoop->getExitingBlocks(ExitingBlocks);

  // Find uncountable exiting blocks
  SmallVector<const BasicBlock *, 4> UncountableExitingBlocks;
  for (BasicBlock *ExitingBB : ExitingBlocks) {
    const SCEV *ExitCount =
        SE->getExitCount(TheLoop, ExitingBB, ScalarEvolution::SymbolicMaximum);
    if (isa<SCEVCouldNotCompute>(ExitCount))
      UncountableExitingBlocks.push_back(ExitingBB);
  }

  // Find header PHIs
  SmallPtrSet<PHINode *, 4> HeaderPhis;
  for (PHINode &PN : TheLoop->getHeader()->phis())
    HeaderPhis.insert(&PN);

  // Find the seeding set of speculative loads and stores
  for (const BasicBlock *UncountableExitingBlock : UncountableExitingBlocks) {
    // Mark loads and stores that the uncountable exit depends
    auto *BI = UncountableExitingBlock->getTerminator();
    // TODO: Support unconditional uncountable exiting blocks
    if (!isa<BranchInst>(BI) || !cast<BranchInst>(BI)->isConditional())
      return false;

    SmallVector<Value *, 4> Worklist;
    SmallPtrSet<Value *, 4> Visited;

    for (Value *Operand : BI->operands())
      Worklist.push_back(Operand);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      Visited.insert(V);

      // Stop at header phis
      if (auto *PHI = dyn_cast<PHINode>(V)) {
        if (HeaderPhis.contains(PHI))
          continue;
      }

      if (auto *I = dyn_cast<Instruction>(V)) {
        if (!TheLoop->contains(I))
          continue;

        switch (I->getOpcode()) {
        // Unsafe speculative instructions other than loads and stores
        // are not supported yet.
        case Instruction::Load:
        case Instruction::Store:
          if (I->getOpcode() == Instruction::Load) {
            SpeculativeLoads.insert(I);
          } else {
            SpeculativeStores.insert(I);
          }
          LLVM_FALLTHROUGH;
        default:
          for (Value *Operand : I->operands()) {
            if (Visited.contains(Operand))
              continue;
            Worklist.push_back(Operand);
          }
          break;
        }
      }
    }
  }

  LLVM_DEBUG(
      dbgs() << "Uncountable Loop: Seeding set of speculative loads:\n";
      for (auto *I : SpeculativeLoads) { I->dump(); }
      dbgs() << "Uncountable Loop: Seeding set of speculative stores:\n";
      for (auto *I : SpeculativeStores) { I->dump(); }
  );

  // Limit to speculative loads only first
  if (!SpeculativeStores.empty()) {
    LLVM_DEBUG(dbgs() << "Uncountable Loop: Have speculative store\n");
    return false;
  }

  // Limit to up two speculative loads first
  if (SpeculativeLoads.size() > 2) {
    LLVM_DEBUG(dbgs() << "Uncountable Loop: More than two speculative loads\n");
    return false;
  }

  // TODO: Find stores aliasing with speculative loads and stores and mark them
  // as speculative.

  // TODO: Before de-speculation is in place, assume all loads are
  // speculative.
  // TODO: Before de-speculation is in place, no store is supported.
  for (BasicBlock *BB : TheLoop->blocks()) {
    for (Instruction &I : *BB) {
      if (isSafeToSpeculativelyExecute(&I))
        continue;

      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!SpeculativeLoads.contains(LI)) {
          LLVM_DEBUG(dbgs() << "Uncountable Loop: Add an independent "
                               "speculative load: \n";
                     LI->dump(););
          SpeculativeLoads.insert(LI);
        }
        continue;
      }

      if (isa<StoreInst>(&I)) {
        LLVM_DEBUG(
            dbgs() << "Uncountable Loop: Have potential speculative store\n");
        return false;
      }

      if (!isa<PHINode, BranchInst>(&I)) {
        LLVM_DEBUG(
            dbgs() << "Uncountable Loop: Other speculation unsafe instructions\n";
            I.dump(););
        return false;
      }
    }
  }

  // All speculative loads should stride at step 1.
  for (Instruction *I : SpeculativeLoads) {
    auto *LI = cast<LoadInst>(I);
    if (!LI->isSimple()) {
      LLVM_DEBUG(dbgs() << "Uncountable Loop: Have complex loads\n");
      return false;
    }

    int Stride =
        isConsecutivePtr(getLoadStoreType(LI), getLoadStorePointerOperand(LI));
    if (Stride != 1) {
      LLVM_DEBUG(
          dbgs()
          << "Uncountable Loop: Have speculative load not striding at 1\n");
      return false;
    }
  }

  return true;
}

bool LoopVectorizationLegality::canVectorizeUncountableLoop(
    PredicatedScalarEvolution &PSE) {
  LLVM_DEBUG(dbgs() << "\nUncountable Loop: Inspecting an uncountable loop "
                       "for vectorization opportunity\n");
  NumOfUncountableLoopsAnalyzedForVectorization++;

  bool VectorizationDisabled = (UncountableLoopVectorizationOption !=
                                UncountableLoopVectorization::Option::Stress) &&
                               (UncountableLoopVectorizationOption ==
                                    UncountableLoopVectorization::Option::Off ||
                                !TTI->enableUncountableVectorization());

  // !!!BIG RED SWITCH!!!
  if (VectorizationDisabled) {
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(
                 Hints->vectorizeAnalysisPassName(),
                 "loop not vectorized: ", TheLoop->getStartLoc(),
                 TheLoop->getHeader())
             << "Uncountable loop vectorization is disabled";
    });
    LLVM_DEBUG(dbgs() << "\nUncountable Loop: Uncountable loop vectorization "
                         "is disabled\n");
    return false;
  }

  if (!TheLoop->isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "\nUncountable Loop: Loop is not in simplify form\n");
    return false;
  }

  if (!TheLoop->isLCSSAForm(*DT)) {
    LLVM_DEBUG(dbgs() << "\nUncountable Loop: Loop is not in LCSSA form\n");
    return false;
  }

  // Limit to single block to cover strlen-like loops first.
  // TODO: Expand to support multi-block uncountable loops.
  const unsigned NumBlocks = TheLoop->getNumBlocks();
  if (NumBlocks == 1)
    NumOfUncountableLoopsWithOneBlock++;
  else if (NumBlocks == 2)
    NumOfUncountableLoopsWithTwoBlocks++;
  else
    NumOfUncountableLoopsWithMoreThanTwoBlocks++;

  // Limit to two blocks to cover strlen-like & std::find loops first.
  if (NumBlocks > 2) {
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(
                 Hints->vectorizeAnalysisPassName(),
                 "loop not vectorized: ", TheLoop->getStartLoc(),
                 TheLoop->getHeader())
             << "Uncountable loop is not single block";
    });
    LLVM_DEBUG(dbgs() << "\nUncountable Loop: Loop is not single block\n");
    return false;
  }

  if (!TheLoop->getUniqueExitBlock()) {
    LLVM_DEBUG(dbgs() << "\nUncountable Loop: Loop doesn't have unique exit block\n");
    return false;
  }
  // Limit to conditional exit branches first
  // TODO: Support unconditional exit branches
  SmallVector<BasicBlock *, 8> ExitingBlocks;
  TheLoop->getExitingBlocks(ExitingBlocks);

  SmallVector<const SCEVPredicate *, 4> Predicates;
  for (BasicBlock *ExitingBB : ExitingBlocks) {
    auto *BI = dyn_cast<BranchInst>(ExitingBB->getTerminator());
    if (!BI || !BI->isConditional() || isa<ConstantInt>(BI->getCondition())) {
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(
                   Hints->vectorizeAnalysisPassName(),
                   "loop not vectorized: ", TheLoop->getStartLoc(),
                   TheLoop->getHeader())
               << "Uncountable loop does not exit with conditional branches";
      });

      LLVM_DEBUG(
          dbgs()
          << "\nUncountable Loop: Does not exit with conditional branches\n");
      NumOfUncountableLoopsNotEndingWithConditionalBranch++;
      return false;
    }
    const SCEV *EC =
        PSE.getSE()->getPredicatedExitCount(TheLoop, ExitingBB, &Predicates);
    if (isa<SCEVCouldNotCompute>(EC)) {
      CouldNotComputeExitingBlock = ExitingBB;
      UncountableExitingBlocks.push_back(ExitingBB);
    } else
      CountableExitingBlocks.push_back(ExitingBB);
  }

  // We only support one uncountable early exit.
  if (getUncountableExitingBlocks().size() != 1) {
    reportVectorizationFailure(
        "Loop has too many uncountable exits",
        "Cannot vectorize early exit loop with more than one early exit",
        "TooManyUncountableEarlyExits", ORE, TheLoop);
    return false;
  }

  if (!CouldNotComputeExitingBlock) {
    LLVM_DEBUG(
        dbgs() << "\nUncountable Loop: Does not have a speculative exit\n");
    return false;
  }

  // Handle countable loops with early exits
  if (!getCountableExitingBlocks().empty()) {
    BasicBlock *LatchBB = TheLoop->getLoopLatch();
    // make sure the early exit is not the latch
    if (CouldNotComputeExitingBlock == LatchBB) {
      LLVM_DEBUG(dbgs() << "\nUncountable Loop: Expect the latch to be "
                           "countable when there is an early exiting\n");
      return false;
    }
    for (BasicBlock *BB : successors(CouldNotComputeExitingBlock))
      if (BB != LatchBB) {
        UncountableExitBlocks.push_back(BB);
        break;
      }

    HasUncountableEarlyExit = true;
  }

  // Exclude integer induction variables first.
  // TODO: Support signed and unsigned induction variables.
  for (const std::pair<PHINode *, InductionDescriptor> &InductionEntry :
       getInductionVars()) {
    if (!InductionEntry.first->getType()->isPointerTy()) {
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(
                   Hints->vectorizeAnalysisPassName(),
                   "loop not vectorized: ", TheLoop->getStartLoc(),
                   TheLoop->getHeader())
               << "Uncountable loop has non-ptr induction variables";
      });
      LLVM_DEBUG(
          dbgs()
          << "\nUncountable Loop: Loop has non-ptr induction variables\n");
      NumOfUncountableLoopsWithNonPtrIVs++;
      return false;
    }
  }

  // 1) Up to one PHI
  // 2) All PHIs are IV
  size_t NumOfHeaderPhis = 0;
  for (PHINode &PN : TheLoop->getHeader()->phis()) {
    ++NumOfHeaderPhis;

    // TODO: Cover int and fp inductions and runtime constant steps.
    InductionDescriptor IndDesc;
    if (!InductionDescriptor::isInductionPHI(&PN, TheLoop, PSE, IndDesc) ||
        IndDesc.getKind() != InductionDescriptor::IK_PtrInduction ||
        !IndDesc.getConstIntStepValue()) {
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(
                   Hints->vectorizeAnalysisPassName(),
                   "loop not vectorized: ", TheLoop->getStartLoc(),
                   TheLoop->getHeader())
               << "Uncountable loop's ptr IV has non constant step";
      });
      LLVM_DEBUG(
          dbgs() << "\nUncountable Loop: Ptr IV with non constant step\n");
      NumOfUncountableLoopsWithUnsupportedPHI++;
      return false;
    }
  }

  // TODO: Remove this restriction
  if (NumOfHeaderPhis != 1) {
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(
                 Hints->vectorizeAnalysisPassName(),
                 "loop not vectorized: ", TheLoop->getStartLoc(),
                 TheLoop->getHeader())
             << "Uncountable loop's header does not have a single PHI";
    });
    LLVM_DEBUG(
        dbgs()
        << "\nUncountable Loop: Loop header does not have a single PHI\n");
    NumOfUncountableLoopsWithoutHeaderPHI++;
    return false;
  }

  if (!isSpeculationSafe(PSE)) {
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(
                 Hints->vectorizeAnalysisPassName(),
                 "loop not vectorized: ", TheLoop->getStartLoc(),
                 TheLoop->getHeader())
             << "Uncountable loop is unsafe for speculation";
    });
    LLVM_DEBUG(dbgs() << "\nUncountable Loop: Unsafe for speculation\n");
    NumOfUncountableLoopsSpeculationUnsafe++;
    return false;
  }

  SmallPtrSet<Value *, 4> LiveOutValues;
  for (BasicBlock *ExitingBB : ExitingBlocks) {
    for (BasicBlock *ExitBB : successors(ExitingBB))
      for (PHINode &ExitPhi : ExitBB->phis()) {
        Value *IncomingValue = ExitPhi.getIncomingValueForBlock(ExitingBB);
        if (auto *I = dyn_cast<Instruction>(IncomingValue);
            I && TheLoop->contains(I))
          LiveOutValues.insert(IncomingValue);
      }
  }

  // Limit liveouts to IVs or their next op only
  for (const Value *LiveOut : LiveOutValues) {
    if (!isInductionVariable(LiveOut)) {
      bool IsInductionNext =
          any_of(getInductionVars(), [&](const auto &InductionEntry) {
            PHINode *InductionEntryPHI = InductionEntry.first;
            return any_of(
                InductionEntryPHI->incoming_values(), [&](const Use &Incoming) {
                  return Incoming == LiveOut &&
                         TheLoop->contains(
                             InductionEntryPHI->getIncomingBlock(Incoming));
                });
          });

      if (!IsInductionNext) {
        ORE->emit([&]() {
          return OptimizationRemarkAnalysis(
                     Hints->vectorizeAnalysisPassName(),
                     "loop not vectorized: ", TheLoop->getStartLoc(),
                     TheLoop->getHeader())
                 << "Uncountable loop has unsupported liveouts";
        });
        LLVM_DEBUG(dbgs() << "\nUncountable Loop: Have unsupported liveouts\n");
        NumOfUncountableLoopsWithNonIVLiveOutValues++;
        return false;
      }
    }
  }

  // TODO: Profile manually unrolled uncountable loops
  // TODO: profile loop bounds

  NumOfUncountableLoopsVectorizable++;

  // clang-format off
  LLVM_DEBUG(
      dbgs() << "\nUncountable Loop: Found one vectorizable uncountable loop";
      dbgs() << "\n*** Uncountable Loop: BEGIN ***\n";
      if (Module *TheModule = TheLoop->getHeader()->getModule())
        if (Function *TheFunction = TheLoop->getHeader()->getParent())
          if (DISubprogram *SP = TheFunction->getSubprogram())
            if (auto SLoc = TheLoop->getStartLoc())
              dbgs() << "\nUncountable Loop: Loop on line "
                     << SLoc->getLine() << " of function "
                     << SP->getName() << " in file "
                     << TheModule->getSourceFileName() << "\n";
      TheLoop->dumpVerbose();
      dbgs() << "*** Uncountable Loop: Loop END ***\n";
  );
  // clang-format on

  return !(UncountableLoopVectorizationOption ==
           UncountableLoopVectorization::Option::AnalysisOnly);
}

class SCEVMonotonicStrideExpr final
    : public SCEVVisitor<SCEVMonotonicStrideExpr, bool> {
private:
  using RetVal = bool;
  using Base = SCEVVisitor<SCEVMonotonicStrideExpr, RetVal>;

  const LoopVectorizationLegality &LVL;
  SmallVector<Instruction *> Monotonics;

  template <typename SCEVT> bool visitExpr(SCEVT *S) {
    return all_of(S->operands(),
                  [&](const SCEV *Op) { return Base::visit(Op); });
  }

public:
  explicit SCEVMonotonicStrideExpr(const LoopVectorizationLegality &LVL)
      : LVL(LVL) {}

  ArrayRef<Instruction *> getMonotonics() const { return Monotonics; }

  bool visitUnknown(const SCEVUnknown *S) {
    if (auto *I = dyn_cast<Instruction>(S->getValue())) {
      if (LVL.isMonotonicPhi(I)) {
        if (!Monotonics.empty()) {
          LLVM_DEBUG(dbgs() << "LV: for now can only support single use of "
                               "monotonic within address computation\n");
          return false;
        }
        Monotonics.push_back(I);
      } else if (!LVL.isInvariant(I)) {
        LLVM_DEBUG(dbgs() << "LV: for now can only support invariant values "
                             "within address computation\n");
        return false;
      }
    }
    return true;
  }

  /// Don't know what to do with this expression. Assume unsafe.
  bool visitCouldNotCompute(const SCEVCouldNotCompute *S) { return false; }

  /// All other expressions are good and won't prevent SCEV expansion in the
  /// vector loop
  bool visitConstant(const SCEVConstant *S) { return true; }
  bool visitVScale(const SCEVVScale *S) { return true; }
  bool visitPtrToIntExpr(const SCEVPtrToIntExpr *S) { return visitExpr(S); }
  bool visitTruncateExpr(const SCEVTruncateExpr *S) { return visitExpr(S); }
  bool visitZeroExtendExpr(const SCEVZeroExtendExpr *S) { return visitExpr(S); }
  bool visitSignExtendExpr(const SCEVSignExtendExpr *S) { return visitExpr(S); }
  bool visitAddExpr(const SCEVAddExpr *S) { return visitExpr(S); }
  bool visitMulExpr(const SCEVMulExpr *S) { return visitExpr(S); }
  bool visitUDivExpr(const SCEVUDivExpr *S) { return visitExpr(S); }
  bool visitAddRecExpr(const SCEVAddRecExpr *S) { return visitExpr(S); }
  bool visitSMaxExpr(const SCEVSMaxExpr *S) { return visitExpr(S); }
  bool visitUMaxExpr(const SCEVUMaxExpr *S) { return visitExpr(S); }
  bool visitSMinExpr(const SCEVSMinExpr *S) { return visitExpr(S); }
  bool visitUMinExpr(const SCEVUMinExpr *S) { return visitExpr(S); }
  bool visitSequentialUMinExpr(const SCEVSequentialMinMaxExpr *S) {
    return visitExpr(S);
  }
};

/// Visit SCEVExpr and verifies that it can be expanded safely in the other
/// loop, i.e. it has not leaf node that is computed within the original loop.
class SCEVRuntimeStrideChecker final
    : public SCEVVisitor<SCEVRuntimeStrideChecker, bool> {
private:
  using RetVal = bool;
  using Base = SCEVVisitor<SCEVRuntimeStrideChecker, RetVal>;

  const LoopVectorizationLegality &LVL;

  template <typename SCEVT> bool visitExpr(SCEVT *S) {
    return all_of(S->operands(),
                  [&](const SCEV *Op) { return Base::visit(Op); });
  }

public:
  explicit SCEVRuntimeStrideChecker(const LoopVectorizationLegality &LVL)
      : LVL(LVL) {}

  /// SCEVUnknown contains pointer to the original Value that needs to be
  /// investigated for safe expansion in the new loop. At this point it's not
  /// possible to call `SCEVExpander.SafeToHoist` as we don't have proper
  /// InsertionPoint to pass to that function (vector skeleton for the vector
  /// loop). Thus simply check if value is NOT defined within the loop we're
  /// trying to vectorize
  bool visitUnknown(const SCEVUnknown *S) {
    if (auto *I = dyn_cast<Instruction>(S->getValue()))
      if (LVL.getLoop()->contains(I) && !LVL.isMonotonicPhi(I)) {
        LLVM_DEBUG(dbgs() << "SCEVUnknown = "; S->print(dbgs());
                   dbgs() << " is defined within the loop\n");
        return false;
      }
    return true;
  }

  /// Don't know what to do with this expression. Assume unsafe.
  bool visitCouldNotCompute(const SCEVCouldNotCompute *S) { return false; }

  /// All other expressions are good and won't prevent SCEV expansion in the
  /// vector loop
  bool visitConstant(const SCEVConstant *S) { return true; }
  bool visitVScale(const SCEVVScale *S) { return true; }
  bool visitPtrToIntExpr(const SCEVPtrToIntExpr *S) { return visitExpr(S); }
  bool visitTruncateExpr(const SCEVTruncateExpr *S) { return visitExpr(S); }
  bool visitZeroExtendExpr(const SCEVZeroExtendExpr *S) { return visitExpr(S); }
  bool visitSignExtendExpr(const SCEVSignExtendExpr *S) { return visitExpr(S); }
  bool visitAddExpr(const SCEVAddExpr *S) { return visitExpr(S); }
  bool visitMulExpr(const SCEVMulExpr *S) { return visitExpr(S); }
  bool visitUDivExpr(const SCEVUDivExpr *S) {
    if (!LVL.getScalarEvolution()->isKnownNonZero(S->getRHS()))
      return false;
    return visitExpr(S);
  }
  bool visitAddRecExpr(const SCEVAddRecExpr *S) { return visitExpr(S); }
  bool visitSMaxExpr(const SCEVSMaxExpr *S) { return visitExpr(S); }
  bool visitUMaxExpr(const SCEVUMaxExpr *S) { return visitExpr(S); }
  bool visitSMinExpr(const SCEVSMinExpr *S) { return visitExpr(S); }
  bool visitUMinExpr(const SCEVUMinExpr *S) { return visitExpr(S); }
  bool visitSequentialUMinExpr(const SCEVSequentialMinMaxExpr *S) {
    return visitExpr(S);
  }
};

/// Return true if runtime stride will be safe to expand using SCEVExpander in
/// the vector loop
bool LoopVectorizationLegality::isSafeStrideAccessInfo(
    const StrideAccessInfo &SAI) const {
  if (!SAI)
    return false;

  SCEVRuntimeStrideChecker StrideChecker(*this);
  return StrideChecker.visit(SAI.getSCEVStride());
}

StrideAccessInfo
LoopVectorizationLegality::computeStrideAccessInfo(Instruction *I) const {
  Value *Ptr = getLoadStorePointerOperand(I);
  auto *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  const DataLayout &DL = I->getModule()->getDataLayout();
  unsigned EltSize = DL.getTypeAllocSize(getLoadStoreType(I));
  if (!PtrTy)
    return StrideAccessInfo();

  const SCEV *SPtr = PSE.getSCEV(Ptr);
  if (StrideAccessInfo SAI = getSimpleSCEVStride(*I, PSE, /*InBytes=*/true))
    return SAI;

  SCEVMonotonicStrideExpr SMSE(*this);
  if (SMSE.visit(SPtr) && !SMSE.getMonotonics().empty()) {
    assert(SMSE.getMonotonics().size() == 1 &&
           "Currently address computation supports only one monotonic");
    Instruction *Phi = SMSE.getMonotonics().front();
    const MonotonicDescriptor &MD =
        MonotonicPhis.find(cast<PHINode>(Phi))->second;
    const SCEV *Step = MD.getStep();
    ScalarEvolution *SE = PSE.getSE();
    // Note: this stride is currently not used in generated vector code, but is
    // used to conclude if access is unit-strided
    // With the support of non-unit-strided monotonics, that SCEV will be
    // expanded in generated vector code
    const SCEV *StepInBytes =
        SE->getMulExpr(Step, SE->getConstant(Step->getType(), EltSize));
    return StrideAccessInfo(SPtr, StepInBytes, EltSize, /*InBytes=*/true,
                            /*IsStrideMonotonic=*/true);
  }
  return StrideAccessInfo();
}
#endif // SIFIVE_CUSTOMIZATION

void LoopVectorizationLegality::prepareToFoldTailByMasking() {
  // The list of pointers that we can safely read and write to remains empty.
  SmallPtrSet<Value *, 8> SafePointers;

  // Mark all blocks for predication, including those that ordinarily do not
  // need predication such as the header block.
  for (BasicBlock *BB : TheLoop->blocks()) {
    [[maybe_unused]] bool R = blockCanBePredicated(BB, SafePointers, MaskedOp);
    assert(R && "Must be able to predicate block when tail-folding.");
  }
}

} // namespace llvm
