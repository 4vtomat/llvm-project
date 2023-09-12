//===- SiFive_LiveInterval.cpp - Live Interval Representation -------------===//
//
// Copyright (c) 2023 SiFive, Inc. -- Proprietary and Confidential All
// Rights Reserved.
//
// NOTICE: All information contained herein is, and remains the property of
// SiFive, Inc. The intellectual and technical concepts contained herein are
// proprietary to SiFive, Inc. and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// This work may not be copied, modified, re-published, uploaded, executed, or
// distributed in any way, in any medium, whether in whole or in part, without
// prior written permission from SiFive, Inc.  The copyright notice above does
// not evidence any actual or intended publication or disclosure of this source
// code, which includes information that is confidential and/or proprietary,
// and is a trade secret, of SiFive, Inc.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValueLiveRange and ValueLiveInterval classes.
// Given some numbering of each the instructions(value) an interval [i, j) is
// said to be a live range for value v. In this implementation ranges can
// have holes, i.e. a range might look like [1,20), [50,65), [1000,1001).  Each
// individual segment is represented as an instance of ValueLiveRange::Segment,
// and the whole range is represented as an instance of ValueLiveRange.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SiFive_LiveInterval.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <utility>

using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// Implementation of various methods necessary for calculation of live ranges.
// The implementation of the methods abstracts from the concrete type of the
// segment collection.
//
// Implementation of the class follows the Template design pattern. The base
// class contains generic algorithms that call collection-specific methods,
// which are provided in concrete subclasses. In order to avoid virtual calls
// these methods are provided by means of C++ template instantiation.
// The base class calls the methods of the subclass through method impl(),
// which casts 'this' pointer to the type of the subclass.
//
//===----------------------------------------------------------------------===//

template <typename ImplT, typename IteratorT, typename CollectionT>
class CalcValueLiveRangeUtilBase {
protected:
  ValueLiveRange *LR;

protected:
  CalcValueLiveRangeUtilBase(ValueLiveRange *LR) : LR(LR) {}

public:
  using Segment = ValueLiveRange::Segment;
  using iterator = IteratorT;

  /// A counterpart of ValueLiveRange::createDeadDef: Make sure the range has a
  /// value defined at @p Def.
  /// If @p ForVNI is null, and there is no value defined at @p Def, a new
  /// value will be allocated using @p ValueSlotInfoAllocator.
  /// If @p ForVNI is null, the return value is the value defined at @p Def,
  /// either a pre-existing one, or the one newly created.
  /// If @p ForVNI is not null, then @p Def should be the location where
  /// @p ForVNI is defined. If the range does not have a value defined at
  /// @p Def, the value @p ForVNI will be used instead of allocating a new
  /// one. If the range already has a value defined at @p Def, it must be
  /// same as @p ForVNI. In either case, @p ForVNI will be the return value.
  ValueSlotInfo *createDeadDef(ValueSlotIndex Def,
                               ValueSlotInfo::Allocator *ValueSlotInfoAllocator,
                               ValueSlotInfo *ForVNI) {
    assert(!Def.isDead() && "Cannot define a value at the dead slot");
    assert((!ForVNI || ForVNI->Def == Def) &&
           "If ForVNI is specified, it must match Def");
    iterator I = impl().find(Def);
    if (I == LiveSegments().end()) {
      ValueSlotInfo *VNI =
          ForVNI ? ForVNI : LR->getNextValue(Def, *ValueSlotInfoAllocator);
      impl().insertAtEnd(Segment(Def, Def.getDeadSlot(), VNI));
      return VNI;
    }

    Segment *S = segmentAt(I);
    if (ValueSlotIndex::isSameInstr(Def, S->Start)) {
      assert((!ForVNI || ForVNI == S->ValNo) && "Value number mismatch");
      assert(S->ValNo->Def == S->Start && "Inconsistent existing value def");

      // It is possible to have both normal and early-clobber defs of the same
      // register on an instruction. It doesn't make a lot of sense, but it is
      // possible to specify in inline assembly.
      //
      // Just convert everything to early-clobber.
      Def = std::min(Def, S->Start);
      if (Def != S->Start)
        S->Start = S->ValNo->Def = Def;
      return S->ValNo;
    }
    assert(ValueSlotIndex::isEarlierInstr(Def, S->Start) &&
           "Already live at def");
    ValueSlotInfo *VNI =
        ForVNI ? ForVNI : LR->getNextValue(Def, *ValueSlotInfoAllocator);
    LiveSegments().insert(I, Segment(Def, Def.getDeadSlot(), VNI));
    return VNI;
  }

  ValueSlotInfo *extendInBlock(ValueSlotIndex StartIdx, ValueSlotIndex Use) {
    if (LiveSegments().empty())
      return nullptr;
    iterator I = impl().findInsertPos(Segment(Use.getPrevSlot(), Use, nullptr));
    if (I == LiveSegments().begin())
      return nullptr;
    --I;
    if (I->End < StartIdx)
      return nullptr;
    if (I->End < Use)
      extendSegmentEndTo(I, Use);
    return I->ValNo;
  }

  std::pair<ValueSlotInfo *, bool>
  extendInBlock(ArrayRef<ValueSlotIndex> Undefs, ValueSlotIndex StartIdx,
                ValueSlotIndex Use) {
    if (LiveSegments().empty())
      return std::make_pair(nullptr, false);
    ValueSlotIndex BeforeUse = Use.getPrevSlot();
    iterator I = impl().findInsertPos(Segment(BeforeUse, Use, nullptr));
    if (I == LiveSegments().begin())
      return std::make_pair(nullptr,
                            LR->isUndefIn(Undefs, StartIdx, BeforeUse));
    --I;
    if (I->End < StartIdx)
      return std::make_pair(nullptr,
                            LR->isUndefIn(Undefs, StartIdx, BeforeUse));
    if (I->End < Use) {
      if (LR->isUndefIn(Undefs, I->End, BeforeUse))
        return std::make_pair(nullptr, true);
      extendSegmentEndTo(I, Use);
    }
    return std::make_pair(I->ValNo, false);
  }

  /// This method is used when we want to extend the segment specified
  /// by I to end at the specified endpoint. To do this, we should
  /// merge and eliminate all segments that this will overlap
  /// with. The iterator is not invalidated.
  void extendSegmentEndTo(iterator I, ValueSlotIndex NewEnd) {
    assert(I != LiveSegments().end() && "Not a valid segment!");
    Segment *S = segmentAt(I);
    ValueSlotInfo *ValNo = I->ValNo;

    // Search for the first segment that we can't merge with.
    iterator MergeTo = std::next(I);
    for (; MergeTo != LiveSegments().end() && NewEnd >= MergeTo->End; ++MergeTo)
      assert(MergeTo->ValNo == ValNo && "Cannot merge with differing values!");

    // If NewEnd was in the middle of a segment, make sure to get its endpoint.
    S->End = std::max(NewEnd, std::prev(MergeTo)->End);

    // If the newly formed segment now touches the segment after it and if they
    // have the same value number, merge the two segments into one segment.
    if (MergeTo != LiveSegments().end() && MergeTo->Start <= I->End &&
        MergeTo->ValNo == ValNo) {
      S->End = MergeTo->End;
      ++MergeTo;
    }

    // Erase any dead segments.
    LiveSegments().erase(std::next(I), MergeTo);
  }

  /// This method is used when we want to extend the segment specified
  /// by I to start at the specified endpoint.  To do this, we should
  /// merge and eliminate all segments that this will overlap with.
  iterator extendSegmentStartTo(iterator I, ValueSlotIndex NewStart) {
    assert(I != LiveSegments().end() && "Not a valid segment!");
    Segment *S = segmentAt(I);
    ValueSlotInfo *ValNo = I->ValNo;

    // Search for the first segment that we can't merge with.
    iterator MergeTo = I;
    do {
      if (MergeTo == LiveSegments().begin()) {
        S->Start = NewStart;
        LiveSegments().erase(MergeTo, I);
        return I;
      }
      assert(MergeTo->ValNo == ValNo && "Cannot merge with differing values!");
      --MergeTo;
    } while (NewStart <= MergeTo->Start);

    // If we start in the middle of another segment, just delete a range and
    // extend that segment.
    if (MergeTo->End >= NewStart && MergeTo->ValNo == ValNo) {
      segmentAt(MergeTo)->End = S->End;
    } else {
      // Otherwise, extend the segment right after.
      ++MergeTo;
      Segment *MergeToSeg = segmentAt(MergeTo);
      MergeToSeg->Start = NewStart;
      MergeToSeg->End = S->End;
    }

    LiveSegments().erase(std::next(MergeTo), std::next(I));
    return MergeTo;
  }

  iterator addSegment(Segment S) {
    ValueSlotIndex Start = S.Start, End = S.End;
    iterator I = impl().findInsertPos(S);

    // If the inserted segment starts in the middle or right at the end of
    // another segment, just extend that segment to contain the segment of S.
    if (I != LiveSegments().begin()) {
      iterator B = std::prev(I);
      if (S.ValNo == B->ValNo) {
        if (B->Start <= Start && B->End >= Start) {
          extendSegmentEndTo(B, End);
          return B;
        }
      } else {
        // Check to make sure that we are not overlapping two live segments with
        // different ValNo's.
        assert(B->End <= Start &&
               "Cannot overlap two LiveSegments with differing ValID's"
               " (did you def the same reg twice in a MachineInstr?)");
      }
    }

    // Otherwise, if this segment ends in the middle of, or right next
    // to, another segment, merge it into that segment.
    if (I != LiveSegments().end()) {
      if (S.ValNo == I->ValNo) {
        if (I->Start <= End) {
          I = extendSegmentStartTo(I, Start);

          // If S is a complete superset of a segment, we may need to grow its
          // endpoint as well.
          if (End > I->End)
            extendSegmentEndTo(I, End);
          return I;
        }
      } else {
        // Check to make sure that we are not overlapping two live LiveSegments with
        // different ValNo's.
        assert(I->Start >= End &&
               "Cannot overlap two LiveSegments with differing ValID's");
      }
    }

    // Otherwise, this is just a new segment that doesn't interact with
    // anything.
    // Insert it.
    return LiveSegments().insert(I, S);
  }

private:
  ImplT &impl() { return *static_cast<ImplT *>(this); }

  CollectionT &LiveSegments() { return impl().segmentsColl(); }

  Segment *segmentAt(iterator I) { return const_cast<Segment *>(&(*I)); }
};

//===----------------------------------------------------------------------===//
//   Instantiation of the methods for calculation of live ranges
//   based on a segment vector.
//===----------------------------------------------------------------------===//

class CalcValueLiveRangeUtilVector;
using CalcValueLiveRangeUtilVectorBase =
    CalcValueLiveRangeUtilBase<CalcValueLiveRangeUtilVector,
                               ValueLiveRange::iterator,
                               ValueLiveRange::Segments>;

class CalcValueLiveRangeUtilVector : public CalcValueLiveRangeUtilVectorBase {
public:
  CalcValueLiveRangeUtilVector(ValueLiveRange *LR)
      : CalcValueLiveRangeUtilVectorBase(LR) {}

private:
  friend CalcValueLiveRangeUtilVectorBase;

  ValueLiveRange::Segments &segmentsColl() { return LR->LiveSegments; }

  void insertAtEnd(const Segment &S) { LR->LiveSegments.push_back(S); }

  iterator find(ValueSlotIndex Pos) { return LR->find(Pos); }

  iterator findInsertPos(Segment S) { return llvm::upper_bound(*LR, S.Start); }
};

//===----------------------------------------------------------------------===//
//   Instantiation of the methods for calculation of live ranges
//   based on a segment set.
//===----------------------------------------------------------------------===//

class CalcValueLiveRangeUtilSet;
using CalcValueLiveRangeUtilSetBase =
    CalcValueLiveRangeUtilBase<CalcValueLiveRangeUtilSet,
                               ValueLiveRange::SegmentSet::iterator,
                               ValueLiveRange::SegmentSet>;

class CalcValueLiveRangeUtilSet : public CalcValueLiveRangeUtilSetBase {
public:
  CalcValueLiveRangeUtilSet(ValueLiveRange *LR)
      : CalcValueLiveRangeUtilSetBase(LR) {}

private:
  friend CalcValueLiveRangeUtilSetBase;

  ValueLiveRange::SegmentSet &segmentsColl() { return *LR->segmentSet; }

  void insertAtEnd(const Segment &S) {
    LR->segmentSet->insert(LR->segmentSet->end(), S);
  }

  iterator find(ValueSlotIndex Pos) {
    iterator I =
        LR->segmentSet->upper_bound(Segment(Pos, Pos.getNextSlot(), nullptr));
    if (I == LR->segmentSet->begin())
      return I;
    iterator PrevI = std::prev(I);
    if (Pos < (*PrevI).End)
      return PrevI;
    return I;
  }

  iterator findInsertPos(Segment S) {
    iterator I = LR->segmentSet->upper_bound(S);
    if (I != LR->segmentSet->end() && !(S.Start < *I))
      ++I;
    return I;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//   ValueLiveRange methods
//===----------------------------------------------------------------------===//

ValueLiveRange::iterator ValueLiveRange::find(ValueSlotIndex Pos) {
  return llvm::partition_point(*this,
                               [&](const Segment &X) { return X.End <= Pos; });
}

ValueSlotInfo *
ValueLiveRange::createDeadDef(ValueSlotIndex Def,
                              ValueSlotInfo::Allocator &VNIAlloc) {
  // Use the segment set, if it is available.
  if (segmentSet != nullptr)
    return CalcValueLiveRangeUtilSet(this).createDeadDef(Def, &VNIAlloc,
                                                         nullptr);
  // Otherwise use the segment vector.
  return CalcValueLiveRangeUtilVector(this).createDeadDef(Def, &VNIAlloc,
                                                          nullptr);
}

ValueSlotInfo *ValueLiveRange::createDeadDef(ValueSlotInfo *VNI) {
  // Use the segment set, if it is available.
  if (segmentSet != nullptr)
    return CalcValueLiveRangeUtilSet(this).createDeadDef(VNI->Def, nullptr,
                                                         VNI);
  // Otherwise use the segment vector.
  return CalcValueLiveRangeUtilVector(this).createDeadDef(VNI->Def, nullptr,
                                                          VNI);
}

// overlapsFrom - Return true if the intersection of the two live ranges is
// not empty.
//
// An example for overlaps():
//
// 0: A = ...
// 4: B = ...
// 8: C = A + B ;; last use of A
//
// The live ranges should look like:
//
// A = [3, 11)
// B = [7, x)
// C = [11, y)
//
// A->overlaps(C) should return false since we want to be able to join
// A and C.
//
bool ValueLiveRange::overlapsFrom(const ValueLiveRange &other,
                                  const_iterator StartPos) const {
  assert(!empty() && "empty range");
  const_iterator i = begin();
  const_iterator ie = end();
  const_iterator j = StartPos;
  const_iterator je = other.end();

  assert((StartPos->Start <= i->Start || StartPos == other.begin()) &&
         StartPos != other.end() && "Bogus Start position hint!");

  if (i->Start < j->Start) {
    i = std::upper_bound(i, ie, j->Start);
    if (i != begin())
      --i;
  } else if (j->Start < i->Start) {
    ++StartPos;
    if (StartPos != other.end() && StartPos->Start <= i->Start) {
      assert(StartPos < other.end() && i < end());
      j = std::upper_bound(j, je, i->Start);
      if (j != other.begin())
        --j;
    }
  } else {
    return true;
  }

  if (j == je)
    return false;

  while (i != ie) {
    if (i->Start > j->Start) {
      std::swap(i, j);
      std::swap(ie, je);
    }

    if (i->End > j->Start)
      return true;
    ++i;
  }

  return false;
}

/// overlaps - Return true if the live range overlaps an interval specified
/// by [Start, End).
bool ValueLiveRange::overlaps(ValueSlotIndex Start, ValueSlotIndex End) const {
  assert(Start < End && "Invalid range");
  const_iterator I = lower_bound(*this, End);
  return I != begin() && (--I)->End > Start;
}

bool ValueLiveRange::covers(const ValueLiveRange &Other) const {
  if (empty())
    return Other.empty();

  const_iterator I = begin();
  for (const Segment &O : Other.LiveSegments) {
    I = advanceTo(I, O.Start);
    if (I == end() || I->Start > O.Start)
      return false;

    // Check adjacent live segments and see if we can get behind O.end.
    while (I->End < O.End) {
      const_iterator Last = I;
      // Get next segment and abort if it was not adjacent.
      ++I;
      if (I == end() || Last->End != I->Start)
        return false;
    }
  }
  return true;
}

/// ValNo is dead, remove it.  If it is the largest value number, just nuke it
/// (and any other deleted values neighboring it), otherwise mark it as ~1U so
/// it can be nuked later.
void ValueLiveRange::markValNoForDeletion(ValueSlotInfo *ValNo) {
  if (ValNo->Id == getNumValNums() - 1) {
    do {
      ValNos.pop_back();
    } while (!ValNos.empty() && ValNos.back()->isUnused());
  } else {
    ValNo->markUnused();
  }
}

/// RenumberValues - Renumber all values in order of appearance and delete the
/// remaining unused values.
void ValueLiveRange::RenumberValues() {
  SmallPtrSet<ValueSlotInfo *, 8> Seen;
  ValNos.clear();
  for (const Segment &S : LiveSegments) {
    ValueSlotInfo *VNI = S.ValNo;
    if (!Seen.insert(VNI).second)
      continue;
    assert(!VNI->isUnused() && "Unused ValNo used by live segment");
    VNI->Id = (unsigned)ValNos.size();
    ValNos.push_back(VNI);
  }
}

void ValueLiveRange::addSegmentToSet(Segment S) {
  CalcValueLiveRangeUtilSet(this).addSegment(S);
}

ValueLiveRange::iterator ValueLiveRange::addSegment(Segment S) {
  // Use the segment set, if it is available.
  if (segmentSet != nullptr) {
    addSegmentToSet(S);
    return end();
  }
  // Otherwise use the segment vector.
  return CalcValueLiveRangeUtilVector(this).addSegment(S);
}

void ValueLiveRange::append(const Segment S) {
  // Check that the segment belongs to the back of the list.
  assert(LiveSegments.empty() || LiveSegments.back().End <= S.Start);
  LiveSegments.push_back(S);
}

std::pair<ValueSlotInfo *, bool>
ValueLiveRange::extendInBlock(ArrayRef<ValueSlotIndex> Undefs,
                              ValueSlotIndex StartIdx, ValueSlotIndex Kill) {
  // Use the segment set, if it is available.
  if (segmentSet != nullptr)
    return CalcValueLiveRangeUtilSet(this).extendInBlock(Undefs, StartIdx,
                                                         Kill);
  // Otherwise use the segment vector.
  return CalcValueLiveRangeUtilVector(this).extendInBlock(Undefs, StartIdx,
                                                          Kill);
}

ValueSlotInfo *ValueLiveRange::extendInBlock(ValueSlotIndex StartIdx,
                                             ValueSlotIndex Kill) {
  // Use the segment set, if it is available.
  if (segmentSet != nullptr)
    return CalcValueLiveRangeUtilSet(this).extendInBlock(StartIdx, Kill);
  // Otherwise use the segment vector.
  return CalcValueLiveRangeUtilVector(this).extendInBlock(StartIdx, Kill);
}

/// Remove the specified segment from this range.  Note that the segment must
/// be in a single Segment in its entirety.
void ValueLiveRange::removeSegment(ValueSlotIndex Start, ValueSlotIndex End,
                                   bool RemoveDeadValNo) {
  // Find the Segment containing this span.
  iterator I = find(Start);
  assert(I != end() && "Segment is not in range!");
  assert(I->containsInterval(Start, End) &&
         "Segment is not entirely in range!");

  // If the span we are removing is at the start of the Segment, adjust it.
  ValueSlotInfo *ValNo = I->ValNo;
  if (I->Start == Start) {
    if (I->End == End) {
      LiveSegments.erase(I); // Removed the whole Segment.

      if (RemoveDeadValNo)
        removeValNoIfDead(ValNo);
    } else
      I->Start = End;
    return;
  }

  // Otherwise if the span we are removing is at the end of the Segment,
  // adjust the other way.
  if (I->End == End) {
    I->End = Start;
    return;
  }

  // Otherwise, we are splitting the Segment into two pieces.
  ValueSlotIndex OldEnd = I->End;
  I->End = Start; // Trim the old segment.

  // Insert the new one.
  LiveSegments.insert(std::next(I), Segment(End, OldEnd, ValNo));
}

ValueLiveRange::iterator ValueLiveRange::removeSegment(iterator I,
                                                       bool RemoveDeadValNo) {
  ValueSlotInfo *ValNo = I->ValNo;
  I = LiveSegments.erase(I);
  if (RemoveDeadValNo)
    removeValNoIfDead(ValNo);
  return I;
}

void ValueLiveRange::removeValNoIfDead(ValueSlotInfo *ValNo) {
  if (none_of(*this, [=](const Segment &S) { return S.ValNo == ValNo; }))
    markValNoForDeletion(ValNo);
}

/// removeValNo - Remove all the segments defined by the specified value#.
/// Also remove the value# from value# list.
void ValueLiveRange::removeValNo(ValueSlotInfo *ValNo) {
  if (empty())
    return;
  llvm::erase_if(LiveSegments,
                 [ValNo](const Segment &S) { return S.ValNo == ValNo; });
  // Now that ValNo is dead, remove it.
  markValNoForDeletion(ValNo);
}

void ValueLiveRange::join(ValueLiveRange &Other, const int *LHSValNoAssignments,
                          const int *RHSValNoAssignments,
                          SmallVectorImpl<ValueSlotInfo *> &NewValueSlotInfo) {
  verify();

  // Determine if any of our values are mapped.  This is uncommon, so we want
  // to avoid the range scan if not.
  bool MustMapCurValNos = false;
  unsigned NumVals = getNumValNums();
  unsigned NumNewVals = NewValueSlotInfo.size();
  for (unsigned i = 0; i != NumVals; ++i) {
    unsigned LHSValID = LHSValNoAssignments[i];
    if (i != LHSValID || (NewValueSlotInfo[LHSValID] &&
                          NewValueSlotInfo[LHSValID] != getValNumInfo(i))) {
      MustMapCurValNos = true;
      break;
    }
  }

  // If we have to apply a mapping to our base range assignment, rewrite it now.
  if (MustMapCurValNos && !empty()) {
    // Map the first live range.

    iterator OutIt = begin();
    OutIt->ValNo = NewValueSlotInfo[LHSValNoAssignments[OutIt->ValNo->Id]];
    for (iterator I = std::next(OutIt), E = end(); I != E; ++I) {
      ValueSlotInfo *nextValNo =
          NewValueSlotInfo[LHSValNoAssignments[I->ValNo->Id]];
      assert(nextValNo && "Huh?");

      // If this live range has the same value # as its immediate predecessor,
      // and if they are neighbors, remove one Segment.  This happens when we
      // have [0,4:0)[4,7:1) and map 0/1 onto the same value #.
      if (OutIt->ValNo == nextValNo && OutIt->End == I->Start) {
        OutIt->End = I->End;
      } else {
        // Didn't merge. Move OutIt to the next segment,
        ++OutIt;
        OutIt->ValNo = nextValNo;
        if (OutIt != I) {
          OutIt->Start = I->Start;
          OutIt->End = I->End;
        }
      }
    }
    // If we merge some segments, chop off the end.
    ++OutIt;
    LiveSegments.erase(OutIt, end());
  }

  // Rewrite Other values before changing the ValueSlotInfo ids.
  // This can leave Other in an invalid state because we're not coalescing
  // touching segments that now have identical values. That's OK since Other is
  // not supposed to be valid after calling join();
  for (Segment &S : Other.LiveSegments)
    S.ValNo = NewValueSlotInfo[RHSValNoAssignments[S.ValNo->Id]];

  // Update val# info. Renumber them and make sure they all belong to this
  // ValueLiveRange now. Also remove dead val#'s.
  unsigned NumValNos = 0;
  for (unsigned i = 0; i < NumNewVals; ++i) {
    ValueSlotInfo *VNI = NewValueSlotInfo[i];
    if (VNI) {
      if (NumValNos >= NumVals)
        ValNos.push_back(VNI);
      else
        ValNos[NumValNos] = VNI;
      VNI->Id = NumValNos++; // Renumber val#.
    }
  }
  if (NumNewVals < NumVals)
    ValNos.resize(NumNewVals); // shrinkify

  // Okay, now insert the RHS live segments into the LHS.
  ValueLiveRangeUpdater Updater(this);
  for (Segment &S : Other.LiveSegments)
    Updater.add(S);
}

/// Merge all of the segments in RHS into this live range as the specified
/// value number.  The segments in RHS are allowed to overlap with segments in
/// the current range, but only if the overlapping segments have the
/// specified value number.
void ValueLiveRange::MergeSegmentsInAsValue(const ValueLiveRange &RHS,
                                            ValueSlotInfo *LHSValNo) {
  ValueLiveRangeUpdater Updater(this);
  for (const Segment &S : RHS.LiveSegments)
    Updater.add(S.Start, S.End, LHSValNo);
}

/// MergeValueInAsValue - Merge all of the live segments of a specific val#
/// in RHS into this live range as the specified value number.
/// The segments in RHS are allowed to overlap with segments in the
/// current range, it will replace the value numbers of the overlaped
/// segments with the specified value number.
void ValueLiveRange::MergeValueInAsValue(const ValueLiveRange &RHS,
                                         const ValueSlotInfo *RHSValNo,
                                         ValueSlotInfo *LHSValNo) {
  ValueLiveRangeUpdater Updater(this);
  for (const Segment &S : RHS.LiveSegments)
    if (S.ValNo == RHSValNo)
      Updater.add(S.Start, S.End, LHSValNo);
}

/// MergeValueNumberInto - This method is called when two value nubmers
/// are found to be equivalent.  This eliminates V1, replacing all
/// segments with the V1 value number with the V2 value number.  This can
/// cause merging of V1/V2 values numbers and compaction of the value space.
ValueSlotInfo *ValueLiveRange::MergeValueNumberInto(ValueSlotInfo *V1,
                                                    ValueSlotInfo *V2) {
  assert(V1 != V2 && "Identical value#'s are always equivalent!");

  // This code actually merges the (numerically) larger value number into the
  // smaller value number, which is likely to allow us to compactify the value
  // space.  The only thing we have to be careful of is to preserve the
  // instruction that defines the result value.

  // Make sure V2 is smaller than V1.
  if (V1->Id < V2->Id) {
    V1->copyFrom(*V2);
    std::swap(V1, V2);
  }

  // Merge V1 segments into V2.
  for (iterator I = begin(); I != end();) {
    iterator S = I++;
    if (S->ValNo != V1)
      continue; // Not a V1 Segment.

    // Okay, we found a V1 live range.  If it had a previous, touching, V2 live
    // range, extend it.
    if (S != begin()) {
      iterator Prev = S - 1;
      if (Prev->ValNo == V2 && Prev->End == S->Start) {
        Prev->End = S->End;

        // Erase this live-range.
        LiveSegments.erase(S);
        I = Prev + 1;
        S = Prev;
      }
    }

    // Okay, now we have a V1 or V2 live range that is maximally merged forward.
    // Ensure that it is a V2 live-range.
    S->ValNo = V2;

    // If we can merge it into later V2 segments, do so now.  We ignore any
    // following V1 segments, as they will be merged in subsequent iterations
    // of the loop.
    if (I != end()) {
      if (I->Start == S->End && I->ValNo == V2) {
        S->End = I->End;
        LiveSegments.erase(I);
        I = S + 1;
      }
    }
  }

  // Now that V1 is dead, remove it.
  markValNoForDeletion(V1);

  return V2;
}

void ValueLiveRange::flushSegmentSet() {
  assert(segmentSet != nullptr && "segment set must have been created");
  assert(
      LiveSegments.empty() &&
      "segment set can be used only initially before switching to the array");
  LiveSegments.append(segmentSet->begin(), segmentSet->end());
  segmentSet = nullptr;
  verify();
}

bool ValueLiveRange::isLiveAtIndexes(ArrayRef<ValueSlotIndex> Slots) const {
  ArrayRef<ValueSlotIndex>::iterator SlotI = Slots.begin();
  ArrayRef<ValueSlotIndex>::iterator SlotE = Slots.end();

  // If there are no regmask slots, we have nothing to search.
  if (SlotI == SlotE)
    return false;

  // Start our search at the first segment that ends after the first slot.
  const_iterator SegmentI = find(*SlotI);
  const_iterator SegmentE = end();

  // If there are no segments that end after the first slot, we're done.
  if (SegmentI == SegmentE)
    return false;

  // Look for each slot in the live range.
  for (; SlotI != SlotE; ++SlotI) {
    // Go to the next segment that ends after the current slot.
    // The slot may be within a hole in the range.
    SegmentI = advanceTo(SegmentI, *SlotI);
    if (SegmentI == SegmentE)
      return false;

    // If this segment contains the slot, we're done.
    if (SegmentI->contains(*SlotI))
      return true;
    // Otherwise, look for the next slot.
  }

  // We didn't find a segment containing any of the slots.
  return false;
}

unsigned ValueLiveInterval::getSize() const {
  unsigned Sum = 0;
  for (const Segment &S : LiveSegments)
    Sum += S.Start.distance(S.End);
  return Sum;
}

raw_ostream &llvm::operator<<(raw_ostream &OS,
                              const ValueLiveRange::Segment &S) {
  return OS << '[' << S.Start.getIndex() << ',' << S.End.getIndex() << ':'
            << S.ValNo->Id << ')';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ValueLiveRange::Segment::dump() const {
  dbgs() << *this << '\n';
}
#endif

void ValueLiveRange::print(raw_ostream &OS) const {
  if (empty())
    OS << "EMPTY";
  else {
    for (const Segment &S : LiveSegments) {
      OS << S;
      assert(S.ValNo == getValNumInfo(S.ValNo->Id) && "Bad ValueSlotInfo");
    }
  }

  // Print value number info.
  if (getNumValNums()) {
    OS << ' ';
    unsigned vnum = 0;
    for (const_vni_iterator i = vni_begin(), e = vni_end(); i != e;
         ++i, ++vnum) {
      const ValueSlotInfo *vni = *i;
      if (vnum)
        OS << ' ';
      OS << vnum << '@';
      if (vni->isUnused()) {
        OS << 'x';
      } else {
        OS << vni->Def.getIndex();
        if (vni->isPHIDef())
          OS << "-phi";
      }
    }
  }
  OS << "\n";
}

void ValueLiveInterval::print(raw_ostream &OS) const {
  OS << *V << ' ';
  super::print(OS);
  OS << "  weight:" << Weight;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ValueLiveRange::dump() const { dbgs() << *this << '\n'; }

LLVM_DUMP_METHOD void ValueLiveInterval::dump() const {
  dbgs() << *this << '\n';
}
#endif

#ifndef NDEBUG
void ValueLiveRange::verify() const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    assert(I->Start.isValid());
    assert(I->End.isValid());
    assert(I->Start < I->End);
    assert(I->ValNo != nullptr);
    assert(I->ValNo->Id < ValNos.size());
    assert(I->ValNo == ValNos[I->ValNo->Id]);
    if (std::next(I) != E) {
      assert(I->End <= std::next(I)->Start);
      if (I->End == std::next(I)->Start)
        assert(I->ValNo != std::next(I)->ValNo);
    }
  }
}

#endif

//===----------------------------------------------------------------------===//
//                           ValueLiveRangeUpdater class
//===----------------------------------------------------------------------===//
//
// The ValueLiveRangeUpdater class always maintains these invariants:
//
// - When LastStart is invalid and the iterators are invalid.
//   This is the initial state, and the state created by flush().
//   In this state, isDirty() returns false.
//
// Otherwise, LiveSegments are kept in three separate areas:
//
// 1. [begin; WriteI) at the front of LR.
// 2. [ReadI; end) at the back of LR.
//
// - LR.begin() <= WriteI <= ReadI <= LR.end().
// - Segments in all three areas are fully ordered and coalesced.
// - Segments in area 1 precede and can't coalesce with segments in area 2.
// - No coalescing is possible where there are no overlapping segments.
//
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void ValueLiveRangeUpdater::print(raw_ostream &OS) const {
  if (!isDirty()) {
    if (LR)
      OS << "Clean updater: " << *LR << '\n';
    else
      OS << "Null updater.\n";
    return;
  }
  assert(LR && "Can't have null LR in dirty updater.");
  OS << " updater with gap = " << (ReadI - WriteI)
     << ", last start = " << LastStart.getIndex() << ":\n  Area 1:";
  for (const auto &S : make_range(LR->begin(), WriteI))
    OS << ' ' << S;
  OS << "\n  Area 2:";
  for (const auto &S : make_range(ReadI, LR->end()))
    OS << ' ' << S;
  OS << '\n';
}

LLVM_DUMP_METHOD void ValueLiveRangeUpdater::dump() const { print(errs()); }
#endif

// Determine if A and B should be coalesced.
static inline bool coalescable(const ValueLiveRange::Segment &A,
                               const ValueLiveRange::Segment &B) {
  assert(A.Start <= B.Start && "Unordered live segments.");
  if (A.End == B.Start)
    return A.ValNo == B.ValNo;
  if (A.End < B.Start)
    return false;
  assert(A.ValNo == B.ValNo && "Cannot overlap different values");
  return true;
}

void ValueLiveRangeUpdater::add(ValueLiveRange::Segment Seg) {
  assert(LR && "Cannot add to a null destination");

  // Fall back to the regular add method if the live range
  // is using the segment set instead of the segment vector.
  if (LR->segmentSet != nullptr) {
    LR->addSegmentToSet(Seg);
    return;
  }

  // Flush the state if Start moves backwards.
  if (!LastStart.isValid() || LastStart > Seg.Start) {
    if (isDirty())
      flush();
    WriteI = ReadI = LR->begin();
  }

  // Remember start for next time.
  LastStart = Seg.Start;

  // Advance ReadI until it ends after Seg.Start.
  ValueLiveRange::iterator E = LR->end();
  if (ReadI != E && ReadI->End <= Seg.Start) {
    // Advance ReadI.
    if (ReadI == WriteI)
      ReadI = WriteI = LR->find(Seg.Start);
    else
      while (ReadI != E && ReadI->End <= Seg.Start)
        *WriteI++ = *ReadI++;
  }

  assert(ReadI == E || ReadI->End > Seg.Start);

  // Check if the ReadI segment begins early.
  if (ReadI != E && ReadI->Start <= Seg.Start) {
    assert(ReadI->ValNo == Seg.ValNo && "Cannot overlap different values");
    // Bail if Seg is completely contained in ReadI.
    if (ReadI->End >= Seg.End)
      return;
    // Coalesce into Seg.
    Seg.Start = ReadI->Start;
    ++ReadI;
  }

  // Coalesce as much as possible from ReadI into Seg.
  while (ReadI != E && coalescable(Seg, *ReadI)) {
    Seg.End = std::max(Seg.End, ReadI->End);
    ++ReadI;
  }

  // Try coalescing Seg into WriteI[-1].
  if (WriteI != LR->begin() && coalescable(WriteI[-1], Seg)) {
    WriteI[-1].End = std::max(WriteI[-1].End, Seg.End);
    return;
  }

  // Seg doesn't coalesce with anything, and needs to be inserted somewhere.
  if (WriteI != ReadI) {
    *WriteI++ = Seg;
    return;
  }

  // Finally, append to LR.
  if (WriteI == E) {
    LR->LiveSegments.push_back(Seg);
    WriteI = ReadI = LR->end();
  }
}

void ValueLiveRangeUpdater::flush() {
  if (!isDirty())
    return;
  // Clear the dirty state.
  LastStart = ValueSlotIndex();

  assert(LR && "Cannot add to a null destination");

  LR->LiveSegments.erase(WriteI, ReadI);
  LR->verify();
}
