//===- llvm/CodeGen/LiveInterval.h - Interval representation ----*- C++ -*-===//
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
// This file implements the ValueLiveRange and ValueLiveInterval classes.  Given
// some numbering of each value defined by an instruction/argument as an
// interval [i, j) is said to be a live range for value v.  In this
// implementation ranges can have holes, i.e. a range might look like
// [1,20), [50,65), [1000,1001).  Each individual segment is represented as an
// instance of ValueLiveRange::Segment, and the whole range is represented as
// an instance of ValueLiveRange.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LIVEINTERVAL_H
#define LLVM_ANALYSIS_LIVEINTERVAL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/SiFive_ValueSlotIndexes.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <tuple>
#include <utility>

namespace llvm {

class raw_ostream;

/// ValueSlotInfo - Value Number Information.
/// This class holds information about a machine level values, including
/// definition and use points.
///
class ValueSlotInfo {
public:
  using Allocator = BumpPtrAllocator;

  /// The ID number of this value.
  unsigned Id;

  /// The index of the defining instruction.
  ValueSlotIndex Def;

  /// ValueSlotInfo constructor.
  ValueSlotInfo(unsigned i, ValueSlotIndex d) : Id(i), Def(d) {}

  /// ValueSlotInfo constructor, copies values from orig, except for the value
  /// number.
  ValueSlotInfo(unsigned i, const ValueSlotInfo &orig) : Id(i), Def(orig.Def) {}

  /// Copy from the parameter into this ValueSlotInfo.
  void copyFrom(ValueSlotInfo &src) { Def = src.Def; }

  /// Returns true if this value is defined by a PHI instruction (or was,
  /// PHI instructions may have been eliminated).
  /// PHI-defs begin at a block boundary, all other defs begin at register or
  /// EC slots.
  bool isPHIDef() const { return Def.isBlock(); }

  /// Returns true if this value is unused.
  bool isUnused() const { return !Def.isValid(); }

  /// Mark this value as unused.
  void markUnused() { Def = ValueSlotIndex(); }
};

/// Result of a ValueLiveRange query. This class hides the implementation
/// details of live ranges, and it should be used as the primary interface for
/// examining live ranges around instructions.
class ValueLiveRangeQueryResult {
  ValueSlotInfo *const EarlyVal;
  ValueSlotInfo *const LateVal;
  const ValueSlotIndex EndPoint;
  const bool Kill;

public:
  ValueLiveRangeQueryResult(ValueSlotInfo *EarlyVal, ValueSlotInfo *LateVal,
                            ValueSlotIndex EndPoint, bool Kill)
      : EarlyVal(EarlyVal), LateVal(LateVal), EndPoint(EndPoint), Kill(Kill) {}

  /// Return the value that is live-in to the instruction. This is the value
  /// that will be read by the instruction's use operands. Return NULL if no
  /// value is live-in.
  ValueSlotInfo *valueIn() const { return EarlyVal; }

  /// Return true if the live-in value is killed by this instruction. This
  /// means that either the live range ends at the instruction, or it changes
  /// value.
  bool isKill() const { return Kill; }

  /// Return true if this instruction has a dead def.
  bool isDeadDef() const { return EndPoint.isDead(); }

  /// Return the value leaving the instruction, if any. This can be a
  /// live-through value, or a live def. A dead def returns NULL.
  ValueSlotInfo *valueOut() const { return isDeadDef() ? nullptr : LateVal; }

  /// Returns the value alive at the end of the instruction, if any. This can
  /// be a live-through value, a live def or a dead def.
  ValueSlotInfo *valueOutOrDead() const { return LateVal; }

  /// Return the value defined by this instruction, if any. This includes
  /// dead defs, it is the value created by the instruction's def operands.
  ValueSlotInfo *valueDefined() const {
    return EarlyVal == LateVal ? nullptr : LateVal;
  }

  /// Return the end point of the last live range segment to interact with
  /// the instruction, if any.
  ///
  /// The end point is an invalid ValueSlotIndex only if the live range doesn't
  /// intersect the instruction at all.
  ///
  /// The end point may be at or past the end of the instruction's basic
  /// block. That means the value was live out of the block.
  ValueSlotIndex endPoint() const { return EndPoint; }
};

/// This class represents the liveness of a register, stack slot, etc.
/// It manages an ordered list of Segment objects.
/// The Segments are organized in a static single assignment form: At places
/// where a new value is defined or different values reach a CFG join a new
/// segment with a new value number is used.
class ValueLiveRange {
public:
  /// This represents a simple continuous liveness interval for a value.
  /// The Start point is inclusive, the End point exclusive. These intervals
  /// are rendered as [Start,End).
  struct Segment {
    ValueSlotIndex Start;           // Start point of the interval (inclusive)
    ValueSlotIndex End;             // End point of the interval (exclusive)
    ValueSlotInfo *ValNo = nullptr; // identifier for the value contained in
                                    // this segment.

    Segment() = default;

    Segment(ValueSlotIndex S, ValueSlotIndex E, ValueSlotInfo *V)
        : Start(S), End(E), ValNo(V) {
      assert(S <= E && "Cannot add backwards segment");
    }

    /// Return true if the index is covered by this segment.
    bool contains(ValueSlotIndex I) const { return Start <= I && I < End; }

    /// Return true if the given interval, [S, E), is covered by this segment.
    bool containsInterval(ValueSlotIndex S, ValueSlotIndex E) const {
      assert((S < E) && "Backwards interval?");
      return (Start <= S && S < End) && (Start < E && E <= End);
    }

    bool operator<(const Segment &Other) const {
      return std::tie(Start, End) < std::tie(Other.Start, Other.End);
    }
    bool operator==(const Segment &Other) const {
      return Start == Other.Start && End == Other.End;
    }

    bool operator!=(const Segment &Other) const { return !(*this == Other); }

    void dump() const;
  };

  using Segments = SmallVector<Segment, 2>;
  using ValueSlotInfoList = SmallVector<ValueSlotInfo *, 2>;

  Segments LiveSegments;        // the liveness segments
  ValueSlotInfoList ValNos; // value#'s

  // The segment set is used temporarily to accelerate initial computation
  // of live ranges of physical registers in computeRegUnitRange.
  // After that the set is flushed to the segment vector and deleted.
  using SegmentSet = std::set<Segment>;
  std::unique_ptr<SegmentSet> segmentSet;

  using iterator = Segments::iterator;
  using const_iterator = Segments::const_iterator;

  iterator begin() { return LiveSegments.begin(); }
  iterator end() { return LiveSegments.end(); }

  const_iterator begin() const { return LiveSegments.begin(); }
  const_iterator end() const { return LiveSegments.end(); }

  using vni_iterator = ValueSlotInfoList::iterator;
  using const_vni_iterator = ValueSlotInfoList::const_iterator;

  vni_iterator vni_begin() { return ValNos.begin(); }
  vni_iterator vni_end() { return ValNos.end(); }

  const_vni_iterator vni_begin() const { return ValNos.begin(); }
  const_vni_iterator vni_end() const { return ValNos.end(); }

  iterator_range<vni_iterator> vnis() {
    return make_range(vni_begin(), vni_end());
  }

  iterator_range<const_vni_iterator> vnis() const {
    return make_range(vni_begin(), vni_end());
  }

  /// Constructs a new ValueLiveRange object.
  ValueLiveRange(bool UseSegmentSet = false)
      : segmentSet(UseSegmentSet ? std::make_unique<SegmentSet>() : nullptr) {}

  /// Constructs a new ValueLiveRange object by copying segments and ValNos from
  /// another ValueLiveRange.
  ValueLiveRange(const ValueLiveRange &Other, BumpPtrAllocator &Allocator) {
    assert(
        Other.segmentSet == nullptr &&
        "Copying of ValueLiveRanges with active SegmentSets is not supported");
    assign(Other, Allocator);
  }

  /// Copies values numbers and live segments from \p Other into this range.
  void assign(const ValueLiveRange &Other, BumpPtrAllocator &Allocator) {
    if (this == &Other)
      return;

    assert(
        Other.segmentSet == nullptr &&
        "Copying of ValueLiveRanges with active SegmentSets is not supported");
    // Duplicate ValNos.
    for (const ValueSlotInfo *VNI : Other.ValNos)
      createValueCopy(VNI, Allocator);
    // Now we can copy segments and remap their ValNos.
    for (const Segment &S : Other.LiveSegments)
      LiveSegments.push_back(Segment(S.Start, S.End, ValNos[S.ValNo->Id]));
  }

  /// advanceTo - Advance the specified iterator to point to the Segment
  /// containing the specified position, or end() if the position is past the
  /// end of the range.  If no Segment contains this position, but the
  /// position is in a hole, this method returns an iterator pointing to the
  /// Segment immediately after the hole.
  iterator advanceTo(iterator I, ValueSlotIndex Pos) {
    assert(I != end());
    if (Pos >= endIndex())
      return end();
    while (I->End <= Pos)
      ++I;
    return I;
  }

  const_iterator advanceTo(const_iterator I, ValueSlotIndex Pos) const {
    assert(I != end());
    if (Pos >= endIndex())
      return end();
    while (I->End <= Pos)
      ++I;
    return I;
  }

  /// find - Return an iterator pointing to the first segment that ends after
  /// Pos, or end(). This is the same as advanceTo(begin(), Pos), but faster
  /// when searching large ranges.
  ///
  /// If Pos is contained in a Segment, that segment is returned.
  /// If Pos is in a hole, the following Segment is returned.
  /// If Pos is beyond endIndex, end() is returned.
  iterator find(ValueSlotIndex Pos);

  const_iterator find(ValueSlotIndex Pos) const {
    return const_cast<ValueLiveRange *>(this)->find(Pos);
  }

  void clear() {
    ValNos.clear();
    LiveSegments.clear();
  }

  size_t size() const { return LiveSegments.size(); }

  bool hasAtLeastOneValue() const { return !ValNos.empty(); }

  bool containsOneValue() const { return ValNos.size() == 1; }

  unsigned getNumValNums() const { return (unsigned)ValNos.size(); }

  /// getValNumInfo - Returns pointer to the specified val#.
  ///
  inline ValueSlotInfo *getValNumInfo(unsigned ValNo) { return ValNos[ValNo]; }
  inline const ValueSlotInfo *getValNumInfo(unsigned ValNo) const {
    return ValNos[ValNo];
  }

  /// containsValue - Returns true if VNI belongs to this range.
  bool containsValue(const ValueSlotInfo *VNI) const {
    return VNI && VNI->Id < getNumValNums() && VNI == getValNumInfo(VNI->Id);
  }

  /// getNextValue - Create a new value number and return it.
  /// @p Def is the index of instruction that defines the value number.
  ValueSlotInfo *
  getNextValue(ValueSlotIndex Def,
               ValueSlotInfo::Allocator &ValueSlotInfoAllocator) {
    ValueSlotInfo *VNI = new (ValueSlotInfoAllocator)
        ValueSlotInfo((unsigned)ValNos.size(), Def);
    ValNos.push_back(VNI);
    return VNI;
  }

  /// createDeadDef - Make sure the range has a value defined at Def.
  /// If one already exists, return it. Otherwise allocate a new value and
  /// add liveness for a dead def.
  ValueSlotInfo *createDeadDef(ValueSlotIndex Def,
                               ValueSlotInfo::Allocator &VNIAlloc);

  /// Create a def of value @p VNI. Return @p VNI. If there already exists
  /// a definition at VNI->Def, the value defined there must be @p VNI.
  ValueSlotInfo *createDeadDef(ValueSlotInfo *VNI);

  /// Create a copy of the given value. The new value will be identical except
  /// for the Value number.
  ValueSlotInfo *
  createValueCopy(const ValueSlotInfo *orig,
                  ValueSlotInfo::Allocator &ValueSlotInfoAllocator) {
    ValueSlotInfo *VNI = new (ValueSlotInfoAllocator)
        ValueSlotInfo((unsigned)ValNos.size(), *orig);
    ValNos.push_back(VNI);
    return VNI;
  }

  /// RenumberValues - Renumber all values in order of appearance and remove
  /// unused values.
  void RenumberValues();

  /// MergeValueNumberInto - This method is called when two value numbers
  /// are found to be equivalent.  This eliminates V1, replacing all
  /// segments with the V1 value number with the V2 value number.  This can
  /// cause merging of V1/V2 values numbers and compaction of the value space.
  ValueSlotInfo *MergeValueNumberInto(ValueSlotInfo *V1, ValueSlotInfo *V2);

  /// Merge all of the live segments of a specific val# in RHS into this live
  /// range as the specified value number. The segments in RHS are allowed
  /// to overlap with segments in the current range, it will replace the
  /// value numbers of the overlaped live segments with the specified value
  /// number.
  void MergeSegmentsInAsValue(const ValueLiveRange &RHS,
                              ValueSlotInfo *LHSValNo);

  /// MergeValueInAsValue - Merge all of the segments of a specific val#
  /// in RHS into this live range as the specified value number.
  /// The segments in RHS are allowed to overlap with segments in the
  /// current range, but only if the overlapping segments have the
  /// specified value number.
  void MergeValueInAsValue(const ValueLiveRange &RHS,
                           const ValueSlotInfo *RHSValNo,
                           ValueSlotInfo *LHSValNo);

  bool empty() const { return LiveSegments.empty(); }

  /// beginIndex - Return the lowest numbered slot covered.
  ValueSlotIndex beginIndex() const {
    assert(!empty() && "Call to beginIndex() on empty range.");
    return LiveSegments.front().Start;
  }

  /// endNumber - return the maximum point of the range of the whole,
  /// exclusive.
  ValueSlotIndex endIndex() const {
    assert(!empty() && "Call to endIndex() on empty range.");
    return LiveSegments.back().End;
  }

  bool expiredAt(ValueSlotIndex index) const { return index >= endIndex(); }

  bool liveAt(ValueSlotIndex index) const {
    const_iterator r = find(index);
    return r != end() && r->Start <= index;
  }

  /// Return the segment that contains the specified index, or null if there
  /// is none.
  const Segment *getSegmentContaining(ValueSlotIndex Idx) const {
    const_iterator I = FindSegmentContaining(Idx);
    return I == end() ? nullptr : &*I;
  }

  /// Return the live segment that contains the specified index, or null if
  /// there is none.
  Segment *getSegmentContaining(ValueSlotIndex Idx) {
    iterator I = FindSegmentContaining(Idx);
    return I == end() ? nullptr : &*I;
  }

  /// getValueSlotInfoAt - Return the ValueSlotInfo that is live at Idx, or
  /// NULL.
  ValueSlotInfo *getValueSlotInfoAt(ValueSlotIndex Idx) const {
    const_iterator I = FindSegmentContaining(Idx);
    return I == end() ? nullptr : I->ValNo;
  }

  /// getValueSlotInfoBefore - Return the ValueSlotInfo that is live up to but
  /// not necessarilly including Idx, or NULL. Use this to find the reaching def
  /// used by an instruction at this ValueSlotIndex position.
  ValueSlotInfo *getValueSlotInfoBefore(ValueSlotIndex Idx) const {
    const_iterator I = FindSegmentContaining(Idx.getPrevSlot());
    return I == end() ? nullptr : I->ValNo;
  }

  /// Return an iterator to the segment that contains the specified index, or
  /// end() if there is none.
  iterator FindSegmentContaining(ValueSlotIndex Idx) {
    iterator I = find(Idx);
    return I != end() && I->Start <= Idx ? I : end();
  }

  const_iterator FindSegmentContaining(ValueSlotIndex Idx) const {
    const_iterator I = find(Idx);
    return I != end() && I->Start <= Idx ? I : end();
  }

  /// overlaps - Return true if the intersection of the two live ranges is
  /// not empty.
  bool overlaps(const ValueLiveRange &other) const {
    if (other.empty())
      return false;
    return overlapsFrom(other, other.begin());
  }

  /// overlaps - Return true if the live range overlaps an interval specified
  /// by [Start, End).
  bool overlaps(ValueSlotIndex Start, ValueSlotIndex End) const;

  /// overlapsFrom - Return true if the intersection of the two live ranges
  /// is not empty.  The specified iterator is a hint that we can begin
  /// scanning the Other range starting at I.
  bool overlapsFrom(const ValueLiveRange &Other, const_iterator StartPos) const;

  /// Returns true if all segments of the @p Other live range are completely
  /// covered by this live range.
  /// Adjacent live ranges do not affect the covering:the liverange
  /// [1,5](5,10] covers (3,7].
  bool covers(const ValueLiveRange &Other) const;

  /// Add the specified Segment to this range, merging segments as
  /// appropriate.  This returns an iterator to the inserted segment (which
  /// may have grown since it was inserted).
  iterator addSegment(Segment S);

  /// Attempt to extend a value defined after @p StartIdx to include @p Use.
  /// Both @p StartIdx and @p Use should be in the same basic block.
  /// The return value is a pair: the first element is ValueSlotInfo of the
  /// value that was extended (possibly nullptr), the second is a boolean value
  /// indicating whether an "undef" was encountered.
  /// If this range is live before @p Use in the basic block that starts at
  /// @p StartIdx, and there is no intervening "undef", extend it to be live
  /// up to @p Use, and return the pair {value, false}. If there is no
  /// segment before @p Use and there is no "undef" between @p StartIdx and
  /// @p Use, return {nullptr, false}. If there is an "undef" before @p Use,
  /// return {nullptr, true}.
  std::pair<ValueSlotInfo *, bool>
  extendInBlock(ArrayRef<ValueSlotIndex> Undefs, ValueSlotIndex StartIdx,
                ValueSlotIndex Kill);

  /// Simplified version of the above "extendInBlock", which assumes that
  /// no register lanes are undefined by <def,read-undef> operands.
  /// If this range is live before @p Use in the basic block that starts
  /// at @p StartIdx, extend it to be live up to @p Use, and return the
  /// value. If there is no segment before @p Use, return nullptr.
  ValueSlotInfo *extendInBlock(ValueSlotIndex StartIdx, ValueSlotIndex Kill);

  /// join - Join two live ranges (this, and other) together.  This applies
  /// mappings to the value numbers in the LHS/RHS ranges as specified.  If
  /// the ranges are not joinable, this aborts.
  void join(ValueLiveRange &Other, const int *ValNoAssignments,
            const int *RHSValNoAssignments,
            SmallVectorImpl<ValueSlotInfo *> &NewValueSlotInfo);

  /// True iff this segment is a single segment that lies between the
  /// specified boundaries, exclusively. Vregs live across a backedge are not
  /// considered local. The boundaries are expected to lie within an extended
  /// basic block, so vregs that are not live out should contain no holes.
  bool isLocal(ValueSlotIndex Start, ValueSlotIndex End) const {
    return beginIndex() > Start.getBaseIndex() &&
           endIndex() < End.getBoundaryIndex();
  }

  /// Remove the specified segment from this range.  Note that the segment
  /// must be a single Segment in its entirety.
  void removeSegment(ValueSlotIndex Start, ValueSlotIndex End,
                     bool RemoveDeadValNo = false);

  void removeSegment(Segment S, bool RemoveDeadValNo = false) {
    removeSegment(S.Start, S.End, RemoveDeadValNo);
  }

  /// Remove segment pointed to by iterator @p I from this range.
  iterator removeSegment(iterator I, bool RemoveDeadValNo = false);

  /// Mark \p ValNo for deletion if no segments in this range use it.
  void removeValNoIfDead(ValueSlotInfo *ValNo);

  /// Query Liveness at Idx.
  /// The sub-instruction slot of Idx doesn't matter, only the instruction
  /// it refers to is considered.
  ValueLiveRangeQueryResult Query(ValueSlotIndex Idx) const {
    // Find the segment that enters the instruction.
    const_iterator I = find(Idx.getBaseIndex());
    const_iterator E = end();
    if (I == E)
      return ValueLiveRangeQueryResult(nullptr, nullptr, ValueSlotIndex(),
                                       false);

    // Is this an instruction live-in segment?
    // If Idx is the start index of a basic block, include live-in segments
    // that start at Idx.getBaseIndex().
    ValueSlotInfo *EarlyVal = nullptr;
    ValueSlotInfo *LateVal = nullptr;
    ValueSlotIndex EndPoint;
    bool Kill = false;
    if (I->Start <= Idx.getBaseIndex()) {
      EarlyVal = I->ValNo;
      EndPoint = I->End;
      // Move to the potentially live-out segment.
      if (ValueSlotIndex::isSameInstr(Idx, I->End)) {
        Kill = true;
        if (++I == E)
          return ValueLiveRangeQueryResult(EarlyVal, LateVal, EndPoint, Kill);
      }
      // Special case: A PHIDef value can have its def in the middle of a
      // segment if the value happens to be live out of the layout
      // predecessor.
      // Such a value is not live-in.
      if (EarlyVal->Def == Idx.getBaseIndex())
        EarlyVal = nullptr;
    }
    // I now points to the segment that may be live-through, or defined by
    // this instr. Ignore segments starting afteS the current instr.
    if (!ValueSlotIndex::isEarlierInstr(Idx, I->Start)) {
      LateVal = I->ValNo;
      EndPoint = I->End;
    }
    return ValueLiveRangeQueryResult(EarlyVal, LateVal, EndPoint, Kill);
  }

  /// removeValNo - Remove all the segments defined by the specified value#.
  /// Also remove the value# from value# list.
  void removeValNo(ValueSlotInfo *ValNo);

  // Returns true if any segment in the live range contains any of the
  // provided slot indexes.  Slots which occur in holes between
  // segments will not cause the function to return true.
  bool isLiveAtIndexes(ArrayRef<ValueSlotIndex> Slots) const;

  bool operator<(const ValueLiveRange &other) const {
    const ValueSlotIndex &thisIndex = beginIndex();
    const ValueSlotIndex &otherIndex = other.beginIndex();
    return thisIndex < otherIndex;
  }

  /// Returns true if there is an explicit "undef" between @p Begin
  /// @p End.
  bool isUndefIn(ArrayRef<ValueSlotIndex> Undefs, ValueSlotIndex Begin,
                 ValueSlotIndex End) const {
    return llvm::any_of(Undefs, [Begin, End](ValueSlotIndex Idx) -> bool {
      return Begin <= Idx && Idx < End;
    });
  }

  /// Flush segment set into the regular segment vector.
  /// The method is to be called after the live range
  /// has been created, if use of the segment set was
  /// activated in the constructor of the live range.
  void flushSegmentSet();

  /// Stores indexes from the input index sequence R at which this
  /// ValueLiveRange is live to the output O iterator. R is a range of
  /// _ascending sorted_ _random_ access iterators to the input indexes. Indexes
  /// stored at O are ascending sorted so it can be used directly in the
  /// subsequent search (for example for subranges). Returns true if found at
  /// least one index.
  template <typename Range, typename OutputIt>
  bool findIndexesLiveAt(Range &&R, OutputIt O) const {
    assert(llvm::is_sorted(R));
    auto Idx = R.begin(), EndIdx = R.end();
    auto Seg = LiveSegments.begin(), EndSeg = LiveSegments.end();
    bool Found = false;
    while (Idx != EndIdx && Seg != EndSeg) {
      // if the Seg is lower find first segment that is above Idx using binary
      // search
      // FIXME: Temp fix before update from trunk is merge to fix a C++ bug.
      if (Seg->End <= *Idx) {
        Seg = std::upper_bound(++Seg, EndSeg, *Idx, [=](auto V, const auto &S) {
          return V < S.End;
        });
        if (Seg == EndSeg)
          break;
      }
      auto NotLessStart = std::lower_bound(Idx, EndIdx, Seg->Start);
      if (NotLessStart == EndIdx)
        break;
      auto NotLessEnd = std::lower_bound(NotLessStart, EndIdx, Seg->End);
      if (NotLessEnd != NotLessStart) {
        Found = true;
        O = std::copy(NotLessStart, NotLessEnd, O);
      }
      Idx = NotLessEnd;
      ++Seg;
    }
    return Found;
  }

  void print(raw_ostream &OS) const;
  void dump() const;

  /// Walk the range and assert if any invariants fail to hold.
  ///
  /// Note that this is a no-op when asserts are disabled.
#ifdef NDEBUG
  void verify() const {}
#else
void verify() const;
#endif

protected:
  /// Append a segment to the list of segments.
  void append(const ValueLiveRange::Segment S);

private:
  friend class ValueLiveRangeUpdater;
  void addSegmentToSet(Segment S);
  void markValNoForDeletion(ValueSlotInfo *V);
};

inline raw_ostream &operator<<(raw_ostream &OS, const ValueLiveRange &LR) {
  LR.print(OS);
  return OS;
}

/// ValueLiveInterval - This class represents the liveness of a register,
/// or stack slot.
class ValueLiveInterval : public ValueLiveRange {
public:
  using super = ValueLiveRange;

private:
  Value *V;           // the register or stack slot of this interval.
  float Weight = 0.0; // weight of this interval

public:
  Value *reg() { return V; }
  float weight() const { return Weight; }
  void incrementWeight(float Inc) { Weight += Inc; }
  void setWeight(float Value) { Weight = Value; }
  void setValue(Value *Val) { V = Val; }

  ValueLiveInterval(Value *V, float Weight) : V(V), Weight(Weight) {}
  ValueLiveInterval() {}

  /// getSize - Returns the sum of sizes of all the ValueLiveRange's.
  ///
  unsigned getSize() const;

  bool operator<(const ValueLiveInterval &other) const {
    const ValueSlotIndex &thisIndex = beginIndex();
    const ValueSlotIndex &otherIndex = other.beginIndex();
    return std::tie(thisIndex, V) < std::tie(otherIndex, other.V);
  }

  void print(raw_ostream &OS) const;
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const ValueLiveInterval &LI) {
  LI.print(OS);
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const ValueLiveRange::Segment &S);

inline bool operator<(ValueSlotIndex V, const ValueLiveRange::Segment &S) {
  return V < S.Start;
}

inline bool operator<(const ValueLiveRange::Segment &S, ValueSlotIndex V) {
  return S.Start < V;
}

/// Helper class for performant ValueLiveRange bulk updates.
///
/// Calling ValueLiveRange::addSegment() repeatedly can be expensive on large
/// live ranges because segments after the insertion point may need to be
/// shifted. The ValueLiveRangeUpdater class can defer the shifting when adding
/// many segments in order.
///
/// The ValueLiveRange will be in an invalid state until flush() is called.
class ValueLiveRangeUpdater {
  ValueLiveRange *LR;
  ValueSlotIndex LastStart;
  ValueLiveRange::iterator WriteI;
  ValueLiveRange::iterator ReadI;

public:
  /// Create a ValueLiveRangeUpdater for adding segments to LR.
  /// LR will temporarily be in an invalid state until flush() is called.
  ValueLiveRangeUpdater(ValueLiveRange *lr = nullptr) : LR(lr) {}

  ~ValueLiveRangeUpdater() { flush(); }

  /// Add a segment to LR and coalesce when possible, just like
  /// LR.addSegment(). Segments should be added in increasing start order for
  /// best performance.
  void add(ValueLiveRange::Segment);

  void add(ValueSlotIndex Start, ValueSlotIndex End, ValueSlotInfo *VNI) {
    add(ValueLiveRange::Segment(Start, End, VNI));
  }

  /// Return true if the LR is currently in an invalid state, and flush()
  /// needs to be called.
  bool isDirty() const { return LastStart.isValid(); }

  /// Flush the updater state to LR so it is valid and contains all added
  /// segments.
  void flush();

  /// Select a different destination live range.
  void setDest(ValueLiveRange *lr) {
    if (LR != lr && isDirty())
      flush();
    LR = lr;
  }

  /// Get the current destination live range.
  ValueLiveRange *getDest() const { return LR; }

  void dump() const;
  void print(raw_ostream &) const;
};

inline raw_ostream &operator<<(raw_ostream &OS,
                               const ValueLiveRangeUpdater &X) {
  X.print(OS);
  return OS;
}

} // end namespace llvm

#endif // LLVM_ANALYSIS_LIVEINTERVAL_H
