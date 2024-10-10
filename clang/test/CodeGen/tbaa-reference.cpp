<<<<<<< HEAD
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECKA,OLD-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECKB,NEW-PATH
=======
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECK,OLD-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes -pointer-tbaa %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECK,OLD-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECK,NEW-PATH
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes %s -pointer-tbaa -emit-llvm -new-struct-path-tbaa -o - | FileCheck %s -check-prefixes=CHECK,NEW-PATH
>>>>>>> d8a656ffaf735ed689856daa5dc13a9274358072
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
// CHECKA: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer:!.*]]
// CHECKB: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer1:!.*]]
// CHECKB: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer2:!.*]]

// Check loading of the reference parameter.
// CHECKA: load ptr, ptr {{.*}}, !tbaa [[TAG_pointer]]
// CHECKB: load ptr, ptr {{.*}}, align 8
// CHECKB: load ptr, ptr {{.*}}, !tbaa [[TAG_pointer2]]

// Check initialization of the reference member.
// CHECKA: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer]]
// CHECKB: store ptr {{.*}}, ptr {{.*}}, !tbaa [[TAG_pointer2]]
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

// OLD-PATH-DAG: [[TAG_pointer]] = !{[[TYPE_pointer:!.*]], [[TYPE_pointer]], i64 0}
// OLD-PATH-DAG: [[TAG_B_s]] = !{[[TYPE_B:!.*]], [[TYPE_pointer]], i64 0}
//
// OLD-PATH-DAG: [[TYPE_B]] = !{!"_ZTS1B", [[TYPE_pointer]], i64 0}
// OLD-PATH-DAG: [[TYPE_pointer]] = !{!"any pointer", [[TYPE_char:!.*]], i64 0}
// OLD-PATH-DAG: [[TYPE_char]] = !{!"omnipotent char", {{!.*}}, i64 0}

// NEW-PATH-DAG: [[TAG_pointer1]] = !{[[TYPE_pointer1:!.*]], [[TYPE_pointer1]], i64 0, i64 8}
// NEW-PATH-DAG: [[TAG_pointer2]] = !{[[TYPE_pointer2:!.*]], [[TYPE_pointer2]], i64 0, i64 8}
// NEW-PATH-DAG: [[TAG_pointer3]] = !{[[TYPE_pointer3:!.*]], [[TYPE_pointer2]], i64 0, i64 8}
//
// NEW-PATH-DAG: [[TYPE_pointer1]] = !{[[TYPE_pointer2]], i64 8, !"p1 struct _ZTS1B"}
// NEW-PATH-DAG: [[TYPE_pointer2]] = !{[[TYPE_char:!.*]], i64 8, !"any pointer"}
// NEW-PATH-DAG: [[TYPE_char]] = !{{{!.*}}, i64 1, !"omnipotent char"}
// NEW-PATH-DAG: [[TYPE_pointer3]] = !{[[TYPE_char]], i64 8, !"_ZTS1B", [[TYPE_pointer2]], i64 0, i64 8}
