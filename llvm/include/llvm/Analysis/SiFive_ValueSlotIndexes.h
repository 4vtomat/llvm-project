//===- llvm/Analysis/SiFive_ValueSlotIndexes.h ----------------------------===//
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
// This file implements ValueSlotIndex and related classes. The purpose of
// ValueSlotIndex is to describe a position at which a register can become live,
// or cease to be live.
//
// ValueSlotIndex is mostly a proxy for entries of the ValueSlotIndexList, a
// class which is held is LiveIntervals and provides the real numbering. This
// allows LiveIntervals to perform largely transparent renumbering.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_VALUESLOTINDEXES_H
#define LLVM_ANALYSIS_VALUESLOTINDEXES_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/ilist.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Allocator.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>

namespace llvm {

class raw_ostream;

/// This class represents an entry in the slot index list held in the
/// LiveValues pass.
class IndexListEntry : public ilist_node<IndexListEntry> {
  Value *V;
  unsigned Index;

public:
  IndexListEntry() = default;
  IndexListEntry(Value *V, unsigned Index) : V(V), Index(Index) {}

  Value *getVal() const { return V; }
  void setVal(Value *V) { this->V = V; }

  unsigned getIndex() const { return Index; }
  void setIndex(unsigned Index) { this->Index = Index; }
};

template <>
struct ilist_alloc_traits<IndexListEntry>
    : public ilist_noalloc_traits<IndexListEntry> {};

/// ValueSlotIndex - An opaque wrapper around value indexes.
class ValueSlotIndex {
public:
  enum Slot {
    /// Basic block boundary.  Used for live ranges entering and leaving a
    /// block without being live in the layout neighbor.  Also used as the
    /// def slot of PHI-defs.
    Slot_Block,

    /// Early-clobber register use/def slot.  A live range defined at
    /// Slot_EarlyClobber interferes with normal live ranges killed at
    /// Slot_Register.  Also used as the kill slot for live ranges tied to an
    /// early-clobber def. TODO: see if we need this or not.
    Slot_EarlyClobber,

    /// Normal register use/def slot.  Normal instructions kill and define
    /// register live ranges at this slot.
    Slot_Register,

    /// Dead def kill point.  Kill slot for a live range that is defined by
    /// the same instruction (Slot_Register or Slot_EarlyClobber), but isn't
    /// used anywhere.
    Slot_Dead,

    Slot_Count
  };

private:
  PointerIntPair<IndexListEntry *, 2, unsigned> lie;

  IndexListEntry *listEntry() const {
    assert(isValid() && "Attempt to compare reserved index.");
    return lie.getPointer();
  }

public:
  unsigned getIndex() const {
    return listEntry()->getIndex(); // | getSlot();
  }

  /// Returns the slot for this ValueSlotIndex.
  Slot getSlot() const { return static_cast<Slot>(lie.getInt()); }

  /// Construct an invalid index.
  ValueSlotIndex() = default;

  // Creates a ValueSlotIndex from an IndexListEntry and a slot. Generally
  // should not be used. This method is only public to facilitate writing
  // certain unit tests.
  ValueSlotIndex(IndexListEntry *entry, unsigned slot) : lie(entry, slot) {
    assert(isValid() && "Not a valid Value Slot Index");
  }

  // Construct a new slot index from the given one, and set the slot.
  ValueSlotIndex(const ValueSlotIndex &li, Slot s)
      : lie(li.listEntry(), unsigned(s)) {
    assert(lie.getPointer() != nullptr &&
           "Attempt to construct index with 0 pointer.");
  }

  /// Returns true if this is a valid index. Invalid indices do
  /// not point into an index table, and cannot be compared.
  bool isValid() const { return lie.getPointer(); }

  /// Return true for a valid index.
  explicit operator bool() const { return isValid(); }

  /// Print this index to the given raw_ostream.
  void print(raw_ostream &os) const;

  /// Dump this index to stderr.
  void dump() const;

  /// Compare two ValueSlotIndex objects for equality.
  bool operator==(ValueSlotIndex other) const { return lie == other.lie; }
  /// Compare two ValueSlotIndex objects for inequality.
  bool operator!=(ValueSlotIndex other) const { return lie != other.lie; }

  /// Compare two ValueSlotIndex objects. Return true if the first index
  /// is strictly lower than the second.
  bool operator<(ValueSlotIndex other) const {
    return getIndex() < other.getIndex();
  }
  /// Compare two ValueSlotIndex objects. Return true if the first index
  /// is lower than, or equal to, the second.
  bool operator<=(ValueSlotIndex other) const {
    return getIndex() <= other.getIndex();
  }

  /// Compare two ValueSlotIndex objects. Return true if the first index
  /// is greater than the second.
  bool operator>(ValueSlotIndex other) const {
    return getIndex() > other.getIndex();
  }

  /// Compare two ValueSlotIndex objects. Return true if the first index
  /// is greater than, or equal to, the second.
  bool operator>=(ValueSlotIndex other) const {
    return getIndex() >= other.getIndex();
  }

  /// isSameInstr - Return true if A and B refer to the same value.
  static bool isSameInstr(ValueSlotIndex A, ValueSlotIndex B) {
    return A.lie.getPointer() == B.lie.getPointer();
  }

  /// isEarlierInstr - Return true if A refers to an value earlier than
  /// B. This is equivalent to A < B && !isSameInstr(A, B).
  static bool isEarlierInstr(ValueSlotIndex A, ValueSlotIndex B) {
    return A.getIndex() < B.getIndex();
  }

  /// Return true if A refers to the same value as B or an earlier one.
  /// This is equivalent to !isEarlierInstr(B, A).
  static bool isEarlierEqualInstr(ValueSlotIndex A, ValueSlotIndex B) {
    return !isEarlierInstr(B, A);
  }

  /// Return the distance from this index to the given one.
  int distance(ValueSlotIndex other) const {
    return other.getIndex() - getIndex();
  }

  /// isBlock - Returns true if this is a block boundary slot.
  bool isBlock() const { return getSlot() == Slot_Block; }

  /// isEarlyClobber - Returns true if this is an early-clobber slot.
  bool isEarlyClobber() const { return getSlot() == Slot_EarlyClobber; }

  /// isRegister - Returns true if this is a normal register use/def slot.
  /// Note that early-clobber slots may also be used for uses and defs.
  bool isRegister() const { return getSlot() == Slot_Register; }

  /// isDead - Returns true if this is a dead def kill slot.
  bool isDead() const { return getSlot() == Slot_Dead; }

  /// Returns the base index for associated with this index. The base index
  /// is the one associated with the Slot_Block slot for the value
  /// pointed to by this index.
  ValueSlotIndex getBaseIndex() const {
    return ValueSlotIndex(listEntry(), Slot_Block);
  }

  /// Returns the boundary index for associated with this index. The boundary
  /// index is the one associated with the Slot_Block slot for the value
  /// pointed to by this index.
  ValueSlotIndex getBoundaryIndex() const {
    return ValueSlotIndex(listEntry(), Slot_Dead);
  }

  /// Returns the register use/def slot in the current value for a
  /// normal or early-clobber def.
  ValueSlotIndex getRegSlot(bool EC = false) const {
    return ValueSlotIndex(listEntry(), EC ? Slot_EarlyClobber : Slot_Register);
  }

  /// Returns the dead def kill slot for the current value.
  ValueSlotIndex getDeadSlot() const {
    return ValueSlotIndex(listEntry(), Slot_Dead);
  }

  /// Returns the next slot in the index list. This could be either the
  /// next slot for the value pointed to by this index or, if this
  /// index is a STORE, the first slot for the next value instance.
  /// WARNING: This method is considerably more expensive than the methods
  /// that return specific slots (getUseIndex(), etc). If you can - please
  /// use one of those methods.
  ValueSlotIndex getNextSlot() const {
    Slot S = getSlot();
    if (S == Slot_Dead) {
      return ValueSlotIndex(&*++listEntry()->getIterator(), Slot_Block);
    }
    return ValueSlotIndex(listEntry(), S + 1);
  }

  /// Returns the next index. This is the index corresponding to the this
  /// index's slot, but for the next value.
  ValueSlotIndex getNextIndex() const {
    return ValueSlotIndex(&*++listEntry()->getIterator(), getSlot());
  }

  /// Returns the previous slot in the index list. This could be either the
  /// previous slot for the value pointed to by this index or, if this
  /// index is a Slot_Block, the last slot for the previous value instance.
  /// WARNING: This method is considerably more expensive than the methods
  /// that return specific slots (getUseIndex(), etc). If you can - please
  /// use one of those methods.
  ValueSlotIndex getPrevSlot() const {
    Slot S = getSlot();
    if (S == Slot_Block) {
      return ValueSlotIndex(&*--listEntry()->getIterator(), Slot_Dead);
    }
    return ValueSlotIndex(listEntry(), S - 1);
  }

  /// Returns the previous index. This is the index corresponding to this
  /// index's slot, but for the previous value instance.
  ValueSlotIndex getPrevIndex() const {
    return ValueSlotIndex(&*--listEntry()->getIterator(), getSlot());
  }
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_VALUESLOTINDEXES_H
