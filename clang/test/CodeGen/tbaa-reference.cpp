<<<<<<< HEAD
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECKA,OLD-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes -pointer-tbaa %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECKA,OLD-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECKB,NEW-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -pointer-tbaa -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECKB,NEW-PATH
||||||| 864902e9b4d8
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECK,OLD-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes -pointer-tbaa %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECK,OLD-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECK,NEW-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -pointer-tbaa -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECK,NEW-PATH
=======
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECK,OLD-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes -pointer-tbaa %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECK,OLD-PATH-POINTER
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECK,NEW-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -pointer-tbaa -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECK,NEW-PATH-POINTER
>>>>>>> fe042904829b83a61c1f4bc904f8f9e5b6da891e
//
// Check that we generate correct TBAA information for reference accesses.

struct S;

struct B {
  S &s;
  B(S &s);
  S &get();
};

B::B(S &s) : s(s) {
// CHECKA-LABEL: _ZN1BC2ER1S
// CHECKB-LABEL: _ZN1BC2ER1S
// Check initialization of the reference parameter.
<<<<<<< HEAD
// CHECKA: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer:!.*]]
// CHECKB: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer1:!.*]]
// CHECKB: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer2:!.*]]
||||||| 864902e9b4d8
// CHECK: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer:!.*]]
=======
// CHECK: store ptr {{.*}}, ptr %s.addr, align 8, !tbaa [[TAG_S_PTR:!.*]]
>>>>>>> fe042904829b83a61c1f4bc904f8f9e5b6da891e

// Check loading of the reference parameter.
<<<<<<< HEAD
// CHECKA: load ptr, ptr {{.*}}, !tbaa [[TAG_pointer]]
// CHECKB: load ptr, ptr {{.*}}, align 8
// CHECKB: load ptr, ptr {{.*}}, !tbaa [[TAG_pointer2]]
||||||| 864902e9b4d8
// CHECK: load ptr, ptr {{.*}}, !tbaa [[TAG_pointer]]
=======
// CHECK: load ptr, ptr {{.*}}, !tbaa [[TAG_S_PTR:!.*]]
>>>>>>> fe042904829b83a61c1f4bc904f8f9e5b6da891e

// Check initialization of the reference member.
<<<<<<< HEAD
// CHECKA: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer]]
// CHECKB: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer2]]
||||||| 864902e9b4d8
// CHECK: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer]]
=======
// CHECK: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_S_PTR]]
>>>>>>> fe042904829b83a61c1f4bc904f8f9e5b6da891e
}

S &B::get() {
// CHECKA-LABEL: _ZN1B3getEv
// CHECKB-LABEL: _ZN1B3getEv
// Check that we access the reference as a structure member.
// CHECKA: load ptr, ptr {{.*}}, !tbaa [[TAG_B_s:!.*]]
// CHECKB: load ptr, ptr {{.*}}, align 8
// CHECKB: load ptr, ptr {{.*}}, !tbaa [[TAG_pointer3:!.*]]
  return s;
}

// OLD-PATH-DAG: [[TAG_S_PTR]] = !{[[TYPE_pointer:!.*]], [[TYPE_pointer]], i64 0}
// OLD-PATH-DAG: [[TAG_B_s]] = !{[[TYPE_B:!.*]], [[TYPE_pointer]], i64 0}
//
// OLD-PATH-DAG: [[TYPE_B]] = !{!"_ZTS1B", [[TYPE_pointer]], i64 0}
// OLD-PATH-DAG: [[TYPE_pointer]] = !{!"any pointer", [[TYPE_char:!.*]], i64 0}
// OLD-PATH-DAG: [[TYPE_char]] = !{!"omnipotent char", {{!.*}}, i64 0}

<<<<<<< HEAD
// NEW-PATH-DAG: [[TAG_pointer1]] = !{[[TYPE_pointer1:!.*]], [[TYPE_pointer1]], i64 0, i64 8}
// NEW-PATH-DAG: [[TAG_pointer2]] = !{[[TYPE_pointer2:!.*]], [[TYPE_pointer2]], i64 0, i64 8}
// NEW-PATH-DAG: [[TAG_pointer3]] = !{[[TYPE_pointer3:!.*]], [[TYPE_pointer2]], i64 0, i64 8}
||||||| 864902e9b4d8
// NEW-PATH-DAG: [[TAG_pointer]] = !{[[TYPE_pointer:!.*]], [[TYPE_pointer]], i64 0, i64 8}
// NEW-PATH-DAG: [[TAG_B_s]] = !{[[TYPE_B:!.*]], [[TYPE_pointer]], i64 0, i64 8}
=======
// OLD-PATH-POINTER-DAG: [[TAG_S_PTR]] = !{[[TYPE_S_PTR:!.*]], [[TYPE_S_PTR]], i64 0}
// OLD-PATH-POINTER-DAG: [[TAG_B_s]] = !{[[TYPE_B:!.*]], [[TYPE_S_PTR:!.*]], i64 0}
//
// OLD-PATH-POINTER-DAG: [[TYPE_B]] = !{!"_ZTS1B", [[TYPE_S_PTR:!.*]], i64 0}
// OLD-PATH-POINTER-DAG: [[TYPE_pointer:!.*]] = !{!"any pointer", [[TYPE_char:!.*]], i64 0}
// OLD-PATH-POINTER-DAG: [[TYPE_char]] = !{!"omnipotent char", {{!.*}}, i64 0}
// OLD-PATH-POINTER-DAG: [[TYPE_S_PTR]] = !{!"p1 _ZTS1S", [[TYPE_pointer]], i64 0}

// NEW-PATH-DAG: [[TAG_S_PTR]] = !{[[TYPE_pointer:!.*]], [[TYPE_pointer]], i64 0, i64 8}
// NEW-PATH-DAG: [[TAG_B_s]] = !{[[TYPE_B:!.*]], [[TYPE_pointer]], i64 0, i64 8}
>>>>>>> fe042904829b83a61c1f4bc904f8f9e5b6da891e
//
// NEW-PATH-DAG: [[TYPE_pointer1]] = !{[[TYPE_pointer2]], i64 8, !"p1 struct _ZTS1B"}
// NEW-PATH-DAG: [[TYPE_pointer2]] = !{[[TYPE_char:!.*]], i64 8, !"any pointer"}
// NEW-PATH-DAG: [[TYPE_char]] = !{{{!.*}}, i64 1, !"omnipotent char"}
<<<<<<< HEAD
// NEW-PATH-DAG: [[TYPE_pointer3]] = !{[[TYPE_char]], i64 8, !"_ZTS1B", [[TYPE_pointer2]], i64 0, i64 8}
||||||| 864902e9b4d8
=======

// NEW-PATH-POINTER-DAG: [[TAG_S_PTR]] = !{[[TYPE_S_PTR:!.*]], [[TYPE_S_PTR]], i64 0, i64 8}
// NEW-PATH-POINTER-DAG: [[TAG_B_s]] = !{[[TYPE_B:!.*]], [[TYPE_S_PTR]], i64 0, i64 8}
//
// NEW-PATH-POINTER-DAG: [[TYPE_B]] = !{[[TYPE_char:!.*]], i64 8, !"_ZTS1B", [[TYPE_S_PTR]], i64 0, i64 8}
// NEW-PATH-POINTER-DAG: [[TYPE_S_PTR]] = !{[[TYPE_pointer:!.+]], i64 8, !"p1 _ZTS1S"}
// NEW-PATH-POINTER-DAG: [[TYPE_pointer]] = !{[[TYPE_char:!.*]], i64 8, !"any pointer"}
// NEW-PATH-POINTER-DAG: [[TYPE_char]] = !{{{!.*}}, i64 1, !"omnipotent char"}
>>>>>>> fe042904829b83a61c1f4bc904f8f9e5b6da891e
