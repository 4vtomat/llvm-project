//===- Multilib.cpp - Multilib Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Multilib.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <string>

using namespace clang;
using namespace driver;
using namespace llvm::sys;

#if SIFIVE_CUSTOMIZATION
/// normalize Segment to "/foo/bar" or "".
static void normalizePathSegment(std::string &Segment) {
  StringRef seg = Segment;

  // Prune trailing "/" or "./"
  while (true) {
    StringRef last = path::filename(seg);
    if (last != ".")
      break;
    seg = path::parent_path(seg);
  }

  if (seg.empty() || seg == "/") {
    Segment.clear();
    return;
  }

  // Add leading '/'
  if (seg.front() != '/') {
    Segment = "/" + seg.str();
  } else {
    Segment = std::string(seg);
  }
}
#endif // SIFIVE_CUSTOMIZATION

Multilib::Multilib(StringRef GCCSuffix, StringRef OSSuffix,
                   StringRef IncludeSuffix, const flags_list &Flags)
    : GCCSuffix(GCCSuffix), OSSuffix(OSSuffix), IncludeSuffix(IncludeSuffix),
      Flags(Flags) {
#if SIFIVE_CUSTOMIZATION
  normalizePathSegment(this->GCCSuffix);
  normalizePathSegment(this->OSSuffix);
  normalizePathSegment(this->IncludeSuffix);
#else
  assert(GCCSuffix.empty() ||
         (StringRef(GCCSuffix).front() == '/' && GCCSuffix.size() > 1));
  assert(OSSuffix.empty() ||
         (StringRef(OSSuffix).front() == '/' && OSSuffix.size() > 1));
  assert(IncludeSuffix.empty() ||
         (StringRef(IncludeSuffix).front() == '/' && IncludeSuffix.size() > 1));
#endif // SIFIVE_CUSTOMIZATION
}

LLVM_DUMP_METHOD void Multilib::dump() const {
  print(llvm::errs());
}

void Multilib::print(raw_ostream &OS) const {
#if SIFIVE_CUSTOMIZATION
  assert(GCCSuffix.empty() || (StringRef(GCCSuffix).front() == '/'));
#endif // SIFIVE_CUSTOMIZATION
  if (GCCSuffix.empty())
    OS << ".";
  else {
    OS << StringRef(GCCSuffix).drop_front();
  }
  OS << ";";
  for (StringRef Flag : Flags) {
    if (Flag.front() == '+')
      OS << "@" << Flag.substr(1);
  }
}

#if SIFIVE_CUSTOMIZATION
bool Multilib::isValid() const {
  llvm::StringMap<int> FlagSet;
  for (unsigned I = 0, N = Flags.size(); I != N; ++I) {
    StringRef Flag(Flags[I]);
    llvm::StringMap<int>::iterator SI = FlagSet.find(Flag.substr(1));

    assert(StringRef(Flag).front() == '+' || StringRef(Flag).front() == '-');

    if (SI == FlagSet.end())
      FlagSet[Flag.substr(1)] = I;
    else if (Flags[I] != Flags[SI->getValue()])
      return false;
  }
  return true;
}
#endif // SIFIVE_CUSTOMIZATION

bool Multilib::operator==(const Multilib &Other) const {
  // Check whether the flags sets match
  // allowing for the match to be order invariant
  llvm::StringSet<> MyFlags;
  for (const auto &Flag : Flags)
    MyFlags.insert(Flag);

  for (const auto &Flag : Other.Flags)
    if (!MyFlags.contains(Flag))
      return false;

  if (osSuffix() != Other.osSuffix())
    return false;

  if (gccSuffix() != Other.gccSuffix())
    return false;

  if (includeSuffix() != Other.includeSuffix())
    return false;

  return true;
}

raw_ostream &clang::driver::operator<<(raw_ostream &OS, const Multilib &M) {
  M.print(OS);
  return OS;
}

#if SIFIVE_CUSTOMIZATION
// TODO: convert the use of Either to MultilibSetBuilder usage model
//       and eliminate these two functions.
static Multilib compose(const Multilib &Base, const Multilib &New) {
  SmallString<128> GCCSuffix;
  llvm::sys::path::append(GCCSuffix, "/", Base.gccSuffix(), New.gccSuffix());
  SmallString<128> OSSuffix;
  llvm::sys::path::append(OSSuffix, "/", Base.osSuffix(), New.osSuffix());
  SmallString<128> IncludeSuffix;
  llvm::sys::path::append(IncludeSuffix, "/", Base.includeSuffix(),
                          New.includeSuffix());

  Multilib Composed(GCCSuffix, OSSuffix, IncludeSuffix);

  Multilib::flags_list &Flags = Composed.flags();

  Flags.insert(Flags.end(), Base.flags().begin(), Base.flags().end());
  Flags.insert(Flags.end(), New.flags().begin(), New.flags().end());

  return Composed;
}

MultilibSet &MultilibSet::Either(ArrayRef<Multilib> MultilibSegments) {
  multilib_list Composed;

  if (Multilibs.empty())
    Multilibs.insert(Multilibs.end(), MultilibSegments.begin(),
                     MultilibSegments.end());
  else {
    for (const auto &New : MultilibSegments) {
      for (const auto &Base : *this) {
        Multilib MO = compose(Base, New);
        if (MO.isValid())
          Composed.push_back(MO);
      }
    }

    Multilibs = Composed;
  }

  return *this;
}
#endif // SIFIVE_CUSTOMIZATION

MultilibSet &MultilibSet::FilterOut(FilterCallback F) {
  llvm::erase_if(Multilibs, F);
  return *this;
}

void MultilibSet::push_back(const Multilib &M) { Multilibs.push_back(M); }

MultilibSet::multilib_list
MultilibSet::select(const Multilib::flags_list &Flags) const {
  llvm::StringSet<> FlagSet;
  for (const auto &Flag : Flags)
    FlagSet.insert(Flag);

  multilib_list Result;
  llvm::copy_if(Multilibs, std::back_inserter(Result),
                [&FlagSet](const Multilib &M) {
                  for (const std::string &F : M.flags())
                    if (!FlagSet.contains(F))
                      return false;
                  return true;
                });
  return Result;
}

bool MultilibSet::select(const Multilib::flags_list &Flags,
                         Multilib &Selected) const {
  multilib_list Result = select(Flags);
  if (Result.empty())
    return false;
  Selected = Result.back();
  return true;
}

LLVM_DUMP_METHOD void MultilibSet::dump() const {
  print(llvm::errs());
}

void MultilibSet::print(raw_ostream &OS) const {
  for (const auto &M : *this)
    OS << M << "\n";
}

raw_ostream &clang::driver::operator<<(raw_ostream &OS, const MultilibSet &MS) {
  MS.print(OS);
  return OS;
}
