//==- RISCVSchedBullet.td - Bullet Scheduling Classes --*- tablegen -*-=//
//
// Bullet Scheduling Classes that can be used by Bullet based processors to
// define their SchedModel.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// c is true if mx has the worst case behavior compared to LMULs in MxList.
/// On the Bullet, the worst case LMUL is the Largest LMUL
/// and the worst case sew is the smallest SEW for that LMUL.
class BulletIsWorstCaseMX<string mx, list<string> MxList> {
  string LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

/// c is true if mx and sew have the worst case behavior compared to LMULs in
/// MxList. On the Bullet, the worst case LMUL is the Largest LMUL
/// and the worst case sew is the smallest SEW for that LMUL.
class BulletIsWorstCaseMXSEW<string mx, int sew, list<string> MxList, bit isF = 0> {
  string LLMUL = LargestLMUL<MxList>.r;
  int SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

/// Number of DLEN parts = (LMUL * VLEN) / DLEN.
/// Since DLEN = VLEN / 2, Num DLEN parts = 2 * LMUL.
class BulletGetCyclesDefault<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 2,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

class BulletGetCyclesNarrowing<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 8,
    !eq(mx, "M4") : 16,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

class BulletGetCyclesVMask<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 1,
    !eq(mx, "M2") : 1,
    !eq(mx, "M4") : 1,
    !eq(mx, "M8") : 2,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

/// VLDM and VSTM can't read/write more than 2 DLENs of data.
/// 2 DLENs when LMUL=8. 1 DLEN for all other DLENs
class BulletGetMaskLoadStoreCycles<string mx> {
  int c = !cond(
    !eq(mx, "M8")  : 2,
    true : 1
  );
}

// Cycles for nf=2 segmented loads and stores are calculated using the
// formula (2 * VLEN * LMUL) / DLEN = 4 * LMUL
class BulletGetCyclesSegmentedSeg2<string mx> {
  int c = !cond(
    !eq(mx, "M1") :  4,
    !eq(mx, "M2") :  8,
    !eq(mx, "M4") :  16,
    !eq(mx, "M8") :  32,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

// Cycles for segmented loads and stores are calculated using the
// formula vl * ceil((SEW * nf) / DLEN), where SEW * nf is the segment size.
class BulletGetCyclesSegmented<string mx, int sew, int nf, int VLEN> {
  // (VLEN * LMUL) / SEW
  defvar VLUpperBound  = !cond(
    !eq(mx, "M1") : !div(VLEN, sew),
    !eq(mx, "M2") : !div(!mul(VLEN, 2), sew),
    !eq(mx, "M4") : !div(!mul(VLEN, 4), sew),
    !eq(mx, "M8") : !div(!mul(VLEN, 8), sew),
    !eq(mx, "MF2") : !div(!div(VLEN, 2), sew),
    !eq(mx, "MF4") : !div(!div(VLEN, 4), sew),
    !eq(mx, "MF8") : !div(!div(VLEN, 8), sew),
  );
  // We can calculate ceil(a/b) using (a + b - 1) / b.
  defvar a = !mul(sew, nf);
  defvar b = !div(VLEN, 2); // DLEN
  int c = !mul(VLUpperBound, !div(!sub(!add(a, b), 1), b));
}

class BulletGetCyclesOnePerElement<string mx, int sew, int VLEN> {
  // c = ceil(VLEN / SEW) * LMUL
  // Note: c >= 1 since the smallest VLEN is 512 / 8 = 8, and the
  // largest division performed on VLEN is in MF8 case with division
  // by 8. Therefore, there is no need to ceil the result.
  defvar a = !div(VLEN, sew);
  int c = !cond(
    !eq(mx, "M1")  : a,
    !eq(mx, "M2")  : !mul(a, 2),
    !eq(mx, "M4")  : !mul(a, 4),
    !eq(mx, "M8")  : !mul(a, 8),
    !eq(mx, "MF2") : !div(a, 2),
    !eq(mx, "MF4") : !div(a, 4),
    !eq(mx, "MF8") : !div(a, 8)
  );
}

class BulletGetDivOrSqrtFactor<int sew> {
  int c = !cond(
    !eq(sew, 16) : 15,
    !eq(sew, 32) : 28,
    !eq(sew, 64) : 57
  );
}

/// Cycles for reductions take approximately VL*SEW/DLEN + 5(4 + log(DLEN/SEW))
/// cycles.
class BulletGetReductionCycles<string mx, int sew, int VLEN> {
  // VLUpperBound*SEW/DLEN is equivalent to 2*LMUL since
  // VLUpperBound=(VLEN*LMUL)/SEW.
  defvar DLEN = !div(VLEN, 2);
  defvar TwoTimesLMUL = !cond(
    !eq(mx, "M1") : 2,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
  int c = !add(
    TwoTimesLMUL,
    !mul(5, !add(4, !logtwo(!div(DLEN, sew))))
  );
}

/// Cycles for ordered reductions take approximatley 6*VL cycles
class BulletGetOrderedReductionCycles<string mx, int sew, int VLEN> {
  // (VLEN * LMUL) / SEW
  defvar VLUpperBound  = !cond(
    !eq(mx, "M1") : !div(VLEN, sew),
    !eq(mx, "M2") : !div(!mul(VLEN, 2), sew),
    !eq(mx, "M4") : !div(!mul(VLEN, 4), sew),
    !eq(mx, "M8") : !div(!mul(VLEN, 8), sew),
    !eq(mx, "MF2") : !div(!div(VLEN, 2), sew),
    !eq(mx, "MF4") : !div(!div(VLEN, 4), sew),
    !eq(mx, "MF8") : !div(!div(VLEN, 8), sew),
  );
  int c = !mul(6, VLUpperBound);
}

// ALU, ALU_B, IMUL, LOAD, F2I, SFB bypass to ALU, ALU_B, Branch
class BulletAnyToGPRBypass<SchedRead read, int cycles = 2>
  : ReadAdvance<read, cycles, [WriteIALU, WriteIALU32,
                               WriteShiftImm, WriteShiftImm32,
                               WriteShiftReg, WriteShiftReg32,
                               WriteSHXADD, WriteSHXADD32,
                               WriteRotateImm, WriteRotateImm32,
                               WriteRotateReg, WriteRotateReg32,
                               WriteSingleBit, WriteSingleBitImm,
                               WriteBEXT, WriteBEXTI,
                               WriteCLZ, WriteCLZ32, WriteCTZ,
                               WriteCTZ32, WriteCPOP, WriteCPOP32,
                               WriteREV8, WriteORCB, WriteSFB,
                               WriteIMul, WriteIMul32, WriteLDB,
                               WriteLDH, WriteLDW, WriteLDW, WriteLDD,
                               WriteFMovF16ToI16, WriteFMovF32ToI32,
                               WriteFMovF64ToI64, WriteFCvtF16ToI32,
                               WriteFCvtF16ToI64, WriteFCvtF32ToI32,
                               WriteFCvtF32ToI64, WriteFCvtF64ToI32,
                               WriteFCvtF64ToI64]>;
// F2F, I2F to FMA, FALU, FDIV, F2F
class BulletAnyToFPRBypass<SchedRead read, int cycles = 1>
  : ReadAdvance<read, cycles, [WriteFCvtI32ToF16, WriteFCvtI32ToF32,
                               WriteFCvtI32ToF64, WriteFCvtI64ToF16,
                               WriteFCvtI64ToF32, WriteFCvtI64ToF64,
                               WriteFCvtF16ToF32, WriteFCvtF16ToF64,
                               WriteFCvtF32ToF16, WriteFCvtF32ToF64,
                               WriteFCvtF64ToF16, WriteFCvtF64ToF32,
                               WriteFMovI16ToF16, WriteFMovI32ToF32,
                               WriteFMovI64ToF64]>;

// The Bullet microarchitecture has three pipelines: A, B, V.
// Pipe A can handle memory, integer alu and vector operations.
// Pipe B can handle integer alu, control flow, integer multiply and divide,
// and floating point computation.
// The V pipeline is modeled by the VCQ, VA, VL, and VS resources.
// numVALUs describes the number of ALUs that make up the vector arithmetic
// sequencer.
multiclass BulletProcResources<int numVALUs = 1> {
  def PipeA : ProcResource<1>;
  def PipeB : ProcResource<1>;
  def Mem   : ProcResource<1>; // Memory
  def IDiv  : ProcResource<1>; // Int Division
  def FDiv  : ProcResource<1>; // FP Division/Sqrt
  def VA    : ProcResource<numVALUs>; // Arithmetic sequencer
  def VL    : ProcResource<1>; // Load sequencer
  def VS    : ProcResource<1>; // Store sequencer
  // The VCQ accepts instructions from the the A Pipe and holds them until the
  // vector unit is ready to dequeue them. The unit dequeues up to one instruction
  // per cycle, in order, as soon as the sequencer for that type of instruction is
  // avaliable. This resource is meant to be used for 1 cycle by all vector
  // instructions, to model that only one vector instruction may be dequed at a
  // time. The actual dequeueing into the sequencer is modeled by the VA, VL, and
  // VS sequencer resources below. Each of them will only accept a single
  // instruction at a time and remain busy for the number of cycles associated
  // with that instruction.
  def VCQ   : ProcResource<1>; // Vector Command Queue
}
