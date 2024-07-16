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

class BulletGetCyclesOnePerElement<string mx, int sew, int VLEN> {
  // c = ceil(VLEN / SEW) * LMUL
  // TODO: fix ceil to account for VLEN != 512
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

// Cycles for segmented loads and stores are calculated using the
// formula vl * ceil((SEW * nf) / DLEN), where SEW * nf is the segment size.
class BulletGetCyclesSegmented<string mx, int sew, int nf, int VLEN> {
  defvar VLUpperBound = BulletGetCyclesOnePerElement<mx, sew, VLEN>.c;
  // We can calculate ceil(a/b) using (a + b - 1) / b.
  defvar a = !mul(sew, nf);
  defvar b = !div(VLEN, 2); // DLEN
  int c = !mul(VLUpperBound, !div(!sub(!add(a, b), 1), b));
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
  defvar VLUpperBound = BulletGetCyclesOnePerElement<mx, sew, VLEN>.c;
  int c = !mul(6, VLUpperBound);
}

class isSingleDLEN<string mx> {
  bit c = !or(!eq(mx, "MF2"), !or(!eq(mx, "MF4"), !eq(mx, "MF8")));
}

class BulletGetCyclesVRGatherVV<string mx, int sew, int VLEN,
                              bit hasFastLargeGather,
                              bit hasFastGather> {
  assert !or(!not(hasFastLargeGather), hasFastGather),
         "hasFastLargeGather must imply hasFastGather";

  // if (hasFastGather && isSingleDLEN(mx))
  //   c = 1;
  //  else if (hasFastLargeGather && (log2(SEW/8) + log2(LMUL) <= log2(DLEN / 32))
  //   c = LMUL * 2 * ceil(vl * SEW / DLEN);
  //  else
  //   c = vl;

  defvar y = !logtwo(!div(sew, 8));
  defvar x = !cond(
    !eq(mx, "M1") : y,
    !eq(mx, "M2") : !add(y, 1),
    !eq(mx, "M4") : !add(y, 2),
    !eq(mx, "M8") : !add(y, 3),
    // Give isSingleDLEN(mx) cases a garbage value to avoid build failures,
    // even though x will go unused.
    true : 1
  );
  // LMUL * 2 * ceil(vl * SEW / DLEN) = LMUL * 2 * ceil(2 * LMUL)
  defvar z = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 16,
    !eq(mx, "M4") : 64,
    !eq(mx, "M8") : 256,
    // Give isSingleDLEN(mx) cases a garbage value to avoid build failures,
    // even though z will go unused.
    true : 1 
  );
  defvar VLUpperBound = BulletGetCyclesOnePerElement<mx, sew, VLEN>.c;
  bit IsSingleDLEN = isSingleDLEN<mx>.c;

  int c = !cond(
    !and(hasFastGather, IsSingleDLEN) : 1,
    !and(hasFastLargeGather, !le(x, !logtwo(!div(VLEN, 64)))) : z,
    true: VLUpperBound
  );
}

class BulletGetCyclesVCompress<string mx, int sew, int VLEN,
                               bit hasFastGather> {

  // if (hasFastGather && isSingleDLEN(mx))
  //   c = 1
  // else
  //   c = vl
  defvar VLUpperBound = BulletGetCyclesOnePerElement<mx, sew, VLEN>.c;
  bit IsSingleDLEN = isSingleDLEN<mx>.c;

  int c = !if(!and(hasFastGather, IsSingleDLEN),
              1,
              VLUpperBound);
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
                               WriteREV8, WriteORCB, WriteIMinMax, WriteSFB,
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
// The V pipeline is modeled by the VCQ, one (VA1) or two (VA1OrVA2)
// arithmetic sequencers, VL, and VS.
multiclass BulletProcResources<bit extraVALU = false> {
  def PipeA : ProcResource<1>;
  def PipeB : ProcResource<1>;
  def IDiv  : ProcResource<1>; // Int Division
  def FDiv  : ProcResource<1>; // FP Division/Sqrt
  // Bullet CPUs can have either one or two vector arithmetic sequencers
  // The VALUs are not symmetric. The first vector arithmetic sequencer can
  // take any vector arithmetic instruction, but the second takes only a
  // subset of vector arithmetic instructions. Instructions that execute on
  // either arithmetic sequencer should use VA1OrVA2. Instructions that
  // execute only on the first sequencer should use VA1.
  if extraVALU then {
    def VA1 : ProcResource<1>;
    def VA2 : ProcResource<1>;
  } else {
    def VA : ProcResource<1>;
  }
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

multiclass BulletWriteResBase<
    int VLEN, ProcResourceKind PipeA, ProcResourceKind PipeB,
    ProcResourceKind PipeAB, ProcResourceKind IDiv,
    ProcResourceKind FDiv, ProcResourceKind VA1, ProcResourceKind VA1OrVA2,
    ProcResourceKind VL, ProcResourceKind VS, ProcResourceKind VCQ,
    bit hasSlowFP64Vec = false,
    bit hasFastGather = false,
    bit hasFastLargeGather = false> {

  // Branching
  let Latency = 3 in {
    def : WriteRes<WriteJmp, [PipeB]>;
    def : WriteRes<WriteJal, [PipeB]>;
    def : WriteRes<WriteJalr, [PipeB]>;
  }
  
  //Short forward branch
  def : WriteRes<WriteSFB, [PipeA, PipeB]> {
    let Latency = 3;
    let NumMicroOps = 2;
  }
  
  // Integer arithmetic and logic
  let Latency = 3 in {
    def : WriteRes<WriteIALU, [PipeAB]>;
    def : WriteRes<WriteIALU32, [PipeAB]>;
    def : WriteRes<WriteShiftImm, [PipeAB]>;
    def : WriteRes<WriteShiftImm32, [PipeAB]>;
    def : WriteRes<WriteShiftReg, [PipeAB]>;
    def : WriteRes<WriteShiftReg32, [PipeAB]>;
  }
  
  // Integer multiplication
  let Latency = 3 in {
    def : WriteRes<WriteIMul, [PipeB]>;
    def : WriteRes<WriteIMul32, [PipeB]>;
  }
  
  // Integer division
  def : WriteRes<WriteIDiv, [PipeB, IDiv]> {
    let Latency = 66;
    let ReleaseAtCycles = [1, 65];
  }
  def : WriteRes<WriteIDiv32,  [PipeB, IDiv]> {
    let Latency = 34;
    let ReleaseAtCycles = [1, 33];
  }

  // Integer remainder
  def : WriteRes<WriteIRem, [PipeB, IDiv]> {
    let Latency = 66;
    let ReleaseAtCycles = [1, 65];
  }
  def : WriteRes<WriteIRem32,  [PipeB, IDiv]> {
    let Latency = 34;
    let ReleaseAtCycles = [1, 33];
  }

  // Bitmanip
  let Latency = 3 in {
    // Rotates are in the late-B ALU.
    def : WriteRes<WriteRotateImm, [PipeB]>;
    def : WriteRes<WriteRotateImm32, [PipeB]>;
    def : WriteRes<WriteRotateReg, [PipeB]>;
    def : WriteRes<WriteRotateReg32, [PipeB]>;
    
    // clz[w]/ctz[w] are in the late-B ALU.
    def : WriteRes<WriteCLZ, [PipeB]>;
    def : WriteRes<WriteCLZ32, [PipeB]>;
    def : WriteRes<WriteCTZ, [PipeB]>;
    def : WriteRes<WriteCTZ32, [PipeB]>;
    
    // cpop[w] look exactly like multiply.
    def : WriteRes<WriteCPOP, [PipeB]>;
    def : WriteRes<WriteCPOP32, [PipeB]>;
    
    // orc.b is in the late-B ALU.
    def : WriteRes<WriteORCB, [PipeB]>;

    // min/max are in the late-B ALU
    def : WriteRes<WriteIMinMax, [PipeB]>;
    
    // rev8 is in the late-A and late-B ALUs.
    def : WriteRes<WriteREV8, [PipeAB]>;
    
    // shNadd[.uw] is on the early-B and late-B ALUs.
    def : WriteRes<WriteSHXADD, [PipeB]>;
    def : WriteRes<WriteSHXADD32, [PipeB]>;
  }
  
  // Single-bit instructions
  // BEXT[I] instruction is available on all ALUs and the other instructions
  // are only available on the SiFive7NB pipe.
  let Latency = 3 in {
    def : WriteRes<WriteSingleBit, [PipeB]>;
    def : WriteRes<WriteSingleBitImm, [PipeB]>;
    def : WriteRes<WriteBEXT, [PipeAB]>;
    def : WriteRes<WriteBEXTI, [PipeAB]>;
  }
  
  // Memory
  def : WriteRes<WriteSTB, [PipeA]>;
  def : WriteRes<WriteSTH, [PipeA]>;
  def : WriteRes<WriteFST16, [PipeA]>;
  
  def : WriteRes<WriteSTW, [PipeA]>;
  def : WriteRes<WriteSTD, [PipeA]>;
  def : WriteRes<WriteFST32, [PipeA]>;
  def : WriteRes<WriteFST64, [PipeA]>;
  
  
  let Latency = 3 in {
    def : WriteRes<WriteLDB, [PipeA]>;
    def : WriteRes<WriteLDH, [PipeA]>;
    def : WriteRes<WriteLDW, [PipeA]>;
    def : WriteRes<WriteLDD, [PipeA]>;
  }
  
  let Latency = 3 in {
    def : WriteRes<WriteFLD16, [PipeA]>;
    def : WriteRes<WriteFLD32, [PipeA]>;
    def : WriteRes<WriteFLD64, [PipeA]>;
  }
  
  // Atomic memory
  def : WriteRes<WriteAtomicSTW, [PipeA]>;
  def : WriteRes<WriteAtomicSTD, [PipeA]>;
  
  let Latency = 3 in {
    def : WriteRes<WriteAtomicW, [PipeA]>;
    def : WriteRes<WriteAtomicD, [PipeA]>;
    def : WriteRes<WriteAtomicLDW, [PipeA]>;
    def : WriteRes<WriteAtomicLDD, [PipeA]>;
  }
  
  // Half Precision
  // FAdd16, FMul16, and FMA16 are not included in this multiclass and should be
  // defined by the SchedModel.
  let Latency = 3 in {
    def : WriteRes<WriteFSGNJ16, [PipeB]>;
    def : WriteRes<WriteFMinMax16, [PipeB]>;
  }
  let Latency = 14, ReleaseAtCycles = [1, 13] in {
    def : WriteRes<WriteFDiv16, [PipeB, FDiv]>;
    def : WriteRes<WriteFSqrt16, [PipeB, FDiv]>;
  }
  
  // Single precision
  // FAdd32, FMul32, and FMA32 are not included in this multiclass and should be
  // defined by the SchedModel
  let Latency = 3 in {
    def : WriteRes<WriteFSGNJ32, [PipeB]>;
    def : WriteRes<WriteFMinMax32, [PipeB]>;
  }
  let Latency = 27, ReleaseAtCycles = [1, 26] in {
    def : WriteRes<WriteFDiv32, [PipeB, FDiv]>;
    def : WriteRes<WriteFSqrt32, [PipeB, FDiv]>;
  }
  
  // Double precision.
  // FAdd64, FMul64, and FMA64 are not included in this multiclass and should be
  // defined by the SchedModel
  let Latency = 3 in {
    def : WriteRes<WriteFSGNJ64, [PipeB]>;
    def : WriteRes<WriteFMinMax64, [PipeB]>;
  }
  
  let Latency = 56, ReleaseAtCycles = [1, 55] in {
    def : WriteRes<WriteFDiv64, [PipeB, FDiv]>;
    def : WriteRes<WriteFSqrt64, [PipeB, FDiv]>;
  }
  
  // Conversions
  let Latency = 3 in {
    def : WriteRes<WriteFCvtI32ToF16, [PipeB]>;
    def : WriteRes<WriteFCvtI32ToF32, [PipeB]>;
    def : WriteRes<WriteFCvtI32ToF64, [PipeB]>;
    def : WriteRes<WriteFCvtI64ToF16, [PipeB]>;
    def : WriteRes<WriteFCvtI64ToF32, [PipeB]>;
    def : WriteRes<WriteFCvtI64ToF64, [PipeB]>;
    def : WriteRes<WriteFCvtF16ToI32, [PipeB]>;
    def : WriteRes<WriteFCvtF16ToI64, [PipeB]>;
    def : WriteRes<WriteFCvtF16ToF32, [PipeB]>;
    def : WriteRes<WriteFCvtF16ToF64, [PipeB]>;
    def : WriteRes<WriteFCvtF32ToI32, [PipeB]>;
    def : WriteRes<WriteFCvtF32ToI64, [PipeB]>;
    def : WriteRes<WriteFCvtF32ToF16, [PipeB]>;
    def : WriteRes<WriteFCvtF32ToF64, [PipeB]>;
    def : WriteRes<WriteFCvtF64ToI32, [PipeB]>;
    def : WriteRes<WriteFCvtF64ToI64, [PipeB]>;
    def : WriteRes<WriteFCvtF64ToF16, [PipeB]>;
    def : WriteRes<WriteFCvtF64ToF32, [PipeB]>;
    
    def : WriteRes<WriteFClass16, [PipeB]>;
    def : WriteRes<WriteFClass32, [PipeB]>;
    def : WriteRes<WriteFClass64, [PipeB]>;
    def : WriteRes<WriteFCmp16, [PipeB]>;
    def : WriteRes<WriteFCmp32, [PipeB]>;
    def : WriteRes<WriteFCmp64, [PipeB]>;
    def : WriteRes<WriteFMovI16ToF16, [PipeA]>;
    def : WriteRes<WriteFMovF16ToI16, [PipeA]>;
    def : WriteRes<WriteFMovI32ToF32, [PipeA]>;
    def : WriteRes<WriteFMovF32ToI32, [PipeA]>;
    def : WriteRes<WriteFMovI64ToF64, [PipeA]>;
    def : WriteRes<WriteFMovF64ToI64, [PipeA]>;
  }
  
  // 6. Configuration-Setting Instructions
  let Latency = 3 in {
    def : WriteRes<WriteVSETVLI, [PipeA]>;
    def : WriteRes<WriteVSETIVLI, [PipeA]>;
    def : WriteRes<WriteVSETVL, [PipeA]>;
  }
  
  // 7. Vector Loads and Stores
  // Unit-stride loads and stores can operate at the full bandwidth of the memory
  // pipe. The memory pipe is DLEN bits wide on x280.
  foreach mx = SchedMxList in {
    defvar Cycles = BulletGetCyclesDefault<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVLDE",    [VCQ, VL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDFF",   [VCQ, VL], mx, IsWorstCase>;
    }
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in
      defm "" : LMULWriteResMX<"WriteVSTE",    [VCQ, VS], mx, IsWorstCase>;
  }
  
  foreach mx = SchedMxList in {
    defvar Cycles = BulletGetMaskLoadStoreCycles<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in
      defm "" : LMULWriteResMX<"WriteVLDM",    [VCQ, VL], mx, IsWorstCase>;
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in
      defm "" : LMULWriteResMX<"WriteVSTM",    [VCQ, VS], mx, IsWorstCase>;
  }

  // Strided loads and stores operate at one element per cycle and should be
  // scheduled accordingly. Indexed loads and stores operate at one element per
  // cycle, and they stall the machine until all addresses have been generated,
  // so they cannot be scheduled. Indexed and strided loads and stores have LMUL
  // specific suffixes, but since SEW is already encoded in the name of the
  // resource, we do not need to use LMULSEWXXX constructors. However, we do
  // use the SEW from the name to determine the number of Cycles.

  foreach mx = SchedMxList in {
    defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
    defvar Cycles = BulletGetCyclesOnePerElement<mx, 8, VLEN>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    defm "" : LMULWriteResMXVariant<"WriteVLDS8",  VLDSX0Pred, [VCQ, VL],
                                         4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                         [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
    let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVLDUX8", [VCQ, VL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX8", [VCQ, VL], mx, IsWorstCase>;
    }
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSTS8",  [VCQ, VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTUX8", [VCQ, VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX8", [VCQ, VS], mx, IsWorstCase>;
    }
  }
  foreach mx = ["MF4", "MF2", "M1", "M2", "M4", "M8"] in {
    defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
    defvar Cycles = BulletGetCyclesOnePerElement<mx, 16, VLEN>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    defm "" : LMULWriteResMXVariant<"WriteVLDS16",  VLDSX0Pred, [VCQ, VL],
                                         4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                         [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
    let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVLDUX16", [VCQ, VL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX16", [VCQ, VL], mx, IsWorstCase>;
    }
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSTS16",  [VCQ, VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTUX16", [VCQ, VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX16", [VCQ, VS], mx, IsWorstCase>;
    }
  }
  foreach mx = ["MF2", "M1", "M2", "M4", "M8"] in {
    defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
    defvar Cycles = BulletGetCyclesOnePerElement<mx, 32, VLEN>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    defm "" : LMULWriteResMXVariant<"WriteVLDS32",  VLDSX0Pred, [VCQ, VL],
                                         4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                         [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
    let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVLDUX32", [VCQ, VL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX32", [VCQ, VL], mx, IsWorstCase>;
    }
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSTS32",  [VCQ, VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTUX32", [VCQ, VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX32", [VCQ, VS], mx, IsWorstCase>;
    }
  }
  foreach mx = ["M1", "M2", "M4", "M8"] in {
    defvar VLDSX0Cycles = BulletGetCyclesDefault<mx>.c;
    defvar Cycles = BulletGetCyclesOnePerElement<mx, 64, VLEN>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    defm "" : LMULWriteResMXVariant<"WriteVLDS64",  VLDSX0Pred, [VCQ, VL],
                                         4, [0, 1], [1, !add(1, VLDSX0Cycles)], !add(3, Cycles),
                                         [0, 1], [1, !add(1, Cycles)], mx, IsWorstCase>;
    let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVLDUX64", [VCQ, VL], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX64", [VCQ, VL], mx, IsWorstCase>;
    }
    let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSTS64",  [VCQ, VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTUX64", [VCQ, VS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX64", [VCQ, VS], mx, IsWorstCase>;
    }
  }

  // VLD*R is LMUL aware
  let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 2)] in
    def : WriteRes<WriteVLD1R,  [VCQ, VL]>;
  let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 4)] in
    def : WriteRes<WriteVLD2R,  [VCQ, VL]>;
  let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 8)] in
    def : WriteRes<WriteVLD4R,  [VCQ, VL]>;
  let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 16)] in
    def : WriteRes<WriteVLD8R,  [VCQ, VL]>;
  // VST*R is LMUL aware
  let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 2)] in
    def : WriteRes<WriteVST1R,   [VCQ, VS]>;
  let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 4)] in
    def : WriteRes<WriteVST2R,   [VCQ, VS]>;
  let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 8)] in
    def : WriteRes<WriteVST4R,   [VCQ, VS]>;
  let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 16)] in
    def : WriteRes<WriteVST8R,   [VCQ, VS]>;
  
  // Segmented Loads and Stores
  // Unit-stride segmented loads and stores are effectively converted into strided
  // segment loads and stores. Strided segment loads and stores operate at up to
  // one segment per cycle if the segment fits within one aligned memory beat.
  // Indexed segment loads and stores operate at the same rate as strided ones,
  // but they stall the machine until all addresses have been generated.
  foreach mx = SchedMxList in {
    foreach eew = [8, 16, 32, 64] in {
      defvar Cycles = BulletGetCyclesSegmentedSeg2<mx>.c;
      defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
      // Does not chain so set latency high
      let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
        defm "" : LMULWriteResMX<"WriteVLSEG2e" # eew,   [VCQ, VL], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF2e" # eew, [VCQ, VL], mx, IsWorstCase>;
      }
      let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in
      defm "" : LMULWriteResMX<"WriteVSSEG2e" # eew,   [VCQ, VS], mx, IsWorstCase>;
      foreach nf=3-8 in {
        defvar Cycles = BulletGetCyclesSegmented<mx, eew, nf, VLEN>.c;
        defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
        // Does not chain so set latency high
        let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
          defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" # eew,   [VCQ, VL], mx, IsWorstCase>;
          defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" # eew, [VCQ, VL], mx, IsWorstCase>;
        }
        let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" # eew,   [VCQ, VS], mx, IsWorstCase>;
      }
    }
  }
  foreach mx = SchedMxList in {
    foreach nf=2-8 in {
      foreach eew = [8, 16, 32, 64] in {
        defvar Cycles = BulletGetCyclesSegmented<mx, eew, nf, VLEN>.c;
        defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
        // Does not chain so set latency high
        let Latency = !add(3, Cycles), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
          defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" # eew,  [VCQ, VL], mx, IsWorstCase>;
          defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" # eew, [VCQ, VL], mx, IsWorstCase>;
          defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" # eew, [VCQ, VL], mx, IsWorstCase>;
        }
        let Latency = 1, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
          defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" # eew,  [VCQ, VS], mx, IsWorstCase>;
          defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" # eew, [VCQ, VS], mx, IsWorstCase>;
          defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" # eew, [VCQ, VS], mx, IsWorstCase>;
        }
      }
    }
  }
  
  // 11. Vector Integer Arithmetic Instructions
  foreach mx = SchedMxList in {
    defvar Cycles = BulletGetCyclesDefault<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVIALUV",     [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIALUX",     [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIALUI",     [VCQ, VA1OrVA2], mx, IsWorstCase>;
      // vmadc requires mask
      defm "" : LMULWriteResMX<"WriteVICALUV",    [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUX",    [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUI",    [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVShiftV",    [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVShiftX",    [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVShiftI",    [VCQ, VA1OrVA2], mx, IsWorstCase>;
      // min max requires merge
      defm "" : LMULWriteResMX<"WriteVIMinMaxV",  [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMinMaxX",  [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMergeV",   [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMergeX",   [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMergeI",   [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMovV",     [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMovX",     [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMovI",     [VCQ, VA1OrVA2], mx, IsWorstCase>;
      // VZEXT and VSEXT reuse the slide instruction's permutation network
      defm "" : LMULWriteResMX<"WriteVExtV",      [VCQ, VA1], mx, IsWorstCase>;
    }
    let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVIMulV",     [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMulX",     [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMulAddV",  [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMulAddX",  [VCQ, VA1OrVA2], mx, IsWorstCase>;
    }
    // Mask results can't chain.
    let Latency = !add(Cycles, 3), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVICmpV",     [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICmpX",     [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICmpI",     [VCQ, VA1], mx, IsWorstCase>;
    }
  }
  foreach mx = SchedMxList in {
    foreach sew = SchedSEWSet<mx>.val in {
      // One bit at a time, but up to four elements can be processed at a time.
      // Add 3 to number of elements to ensure the group formed by remainder
      // elements are accounted for.
      defvar Cycles =
          !mul(sew, !div(!add(3, BulletGetCyclesOnePerElement<mx, sew, VLEN>.c), 4));
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
      let Latency = Cycles, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [VCQ, VA1], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [VCQ, VA1], mx, sew, IsWorstCase>;
      }
    }
  }
  // Widening
  foreach mx = SchedMxListW in {
    defvar Cycles = BulletGetCyclesDefault<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxListW>.c;
    let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVIWALUV",    [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIWALUX",    [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIWALUI",    [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIWMulV",    [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIWMulX",    [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIWMulAddV", [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIWMulAddX", [VCQ, VA1OrVA2], mx, IsWorstCase>;
    }
  }
  // Narrowing
  foreach mx = SchedMxListW in {
    defvar Cycles = BulletGetCyclesNarrowing<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxListW>.c;
    let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVNShiftV",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVNShiftX",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVNShiftI",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
    }
  }
  
  // 12. Vector Fixed-Point Arithmetic Instructions
  foreach mx = SchedMxList in {
    defvar Cycles = BulletGetCyclesDefault<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSALUV",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSALUX",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSALUI",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVAALUV",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVAALUX",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSMulV",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSMulX",   [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSShiftV", [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSShiftX", [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSShiftI", [VCQ, VA1OrVA2], mx, IsWorstCase>;
    }
  }
  // Narrowing
  foreach mx = SchedMxListW in {
    defvar Cycles = BulletGetCyclesNarrowing<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxListW>.c;
    let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVNClipV",  [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVNClipX",  [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVNClipI",  [VCQ, VA1OrVA2], mx, IsWorstCase>;
    }
  }
  
  // 13. Vector Floating-Point Instructions
  foreach mx = SchedMxListF in {
    foreach sew = SchedSEWSet<mx, isF=1>.val in {
      defvar Cycles = !if(!and(hasSlowFP64Vec, !eq(sew, 64)),
                          BulletGetCyclesOnePerElement<mx, sew, VLEN>.c,
                          BulletGetCyclesDefault<mx>.c);
      defvar Lat8 = !if(!and(hasSlowFP64Vec, !eq(sew, 64)), Cycles, 8);
      defvar VA = !if(!and(hasSlowFP64Vec, !eq(sew, 64)), VA1, VA1OrVA2);
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
      let Latency = Lat8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV",  [VCQ, VA], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF",  [VCQ, VA], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV",  [VCQ, VA], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF",  [VCQ, VA], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [VCQ, VA], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [VCQ, VA], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV",   [VCQ, VA1], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [VCQ, VA1], mx, sew, IsWorstCase>;
      }
      defvar Lat4 = !if(!and(hasSlowFP64Vec, !eq(sew, 64)), Cycles, 4);
      let Latency = Lat4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [VCQ, VA], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [VCQ, VA], mx, sew, IsWorstCase>;
        // minmax requires merge
        defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV", [VCQ, VA1], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF", [VCQ, VA1], mx, sew, IsWorstCase>;
      }
    }
  }
  foreach mx = SchedMxList in {
    defvar Cycles = BulletGetCyclesDefault<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVFCvtFToIV",  [VCQ, VA1], mx, IsWorstCase>;
    }
    let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVFClassV",    [VCQ, VA1OrVA2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVFMergeV",    [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVFMovV",      [VCQ, VA1OrVA2], mx, IsWorstCase>;
    }
    // Mask results can't chain.
    let Latency = !add(Cycles, 3), AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      // fcmp requires mask
      defm "" : LMULWriteResMX<"WriteVFCmpV",      [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVFCmpF",      [VCQ, VA1], mx, IsWorstCase>;
    }
  }
  foreach mx = SchedMxListF in {
    foreach sew = SchedSEWSet<mx, isF=1>.val in {
      defvar Cycles = !mul(BulletGetDivOrSqrtFactor<sew>.c,
                           !div(BulletGetCyclesOnePerElement<mx, sew, VLEN>.c, 4));
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
      let Latency = Cycles, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [VCQ, VA1], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV",  [VCQ, VA1], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF",  [VCQ, VA1], mx, sew, IsWorstCase>;
      }
    }
  }
  
  // Widening
  foreach mx = SchedMxListW in {
    foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
      defvar Cycles = !if(!and(hasSlowFP64Vec, !eq(sew, 32)),
                          BulletGetCyclesOnePerElement<mx, sew, VLEN>.c,
                          BulletGetCyclesDefault<mx>.c);
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;
      let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV", [VCQ, VA1], mx, sew, IsWorstCase>;
    }
  }
  foreach mx = SchedMxListFW in {
    foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
      defvar Cycles = BulletGetCyclesDefault<mx>.c;
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
      let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV", [VCQ, VA1OrVA2], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF", [VCQ, VA1OrVA2], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV", [VCQ, VA1OrVA2], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF", [VCQ, VA1OrVA2], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [VCQ, VA1OrVA2], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [VCQ, VA1OrVA2], mx, sew, IsWorstCase>;
      }
      defvar CvtCycles = !if(!and(hasSlowFP64Vec, !eq(sew, 32)),
                          BulletGetCyclesOnePerElement<mx, sew, VLEN>.c,
                          BulletGetCyclesDefault<mx>.c);
      let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, CvtCycles)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV", [VCQ, VA1], mx, sew, IsWorstCase>;
    }
    defvar Cycles = BulletGetCyclesDefault<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxListFW>.c;
    let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in
    defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [VCQ, VA1], mx, IsWorstCase>;
  }
  // Narrowing
  foreach mx = SchedMxListW in {
    defvar Cycles = BulletGetCyclesNarrowing<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxListW>.c;
    let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      // TODO: Once these are SEW aware, this should be one cycle per element
      // for SEW=32.
      defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [VCQ, VA1], mx, IsWorstCase>;
    }
  }
  foreach mx = SchedMxListFW in {
    foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
      defvar Cycles = !if(!and(hasSlowFP64Vec, !eq(sew, 32)),
                          BulletGetCyclesOnePerElement<mx, sew, VLEN>.c,
                          BulletGetCyclesNarrowing<mx>.c);
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
      let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [VCQ, VA1], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [VCQ, VA1], mx, sew, IsWorstCase>;
      }
    }
  }
  
  // 14. Vector Reduction Operations
  foreach mx = SchedMxList in {
    foreach sew = SchedSEWSet<mx>.val in {
      defvar Cycles = BulletGetReductionCycles<mx, sew, VLEN>.c;
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
      let Latency = Cycles, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From", [VCQ, VA1],
                                       mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [VCQ, VA1],
                                       mx, sew, IsWorstCase>;
      }
    }
  }
  
  foreach mx = SchedMxListWRed in {
    foreach sew = SchedSEWSet<mx, 0, 1>.val in {
      defvar Cycles = BulletGetReductionCycles<mx, sew, VLEN>.c;
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;
      let Latency = Cycles, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [VCQ, VA1],
                                     mx, sew, IsWorstCase>;
    }
  }
  
  foreach mx = SchedMxListF in {
    foreach sew = SchedSEWSet<mx, 1>.val in {
      defvar RedCycles = BulletGetReductionCycles<mx, sew, VLEN>.c;
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
      let Latency = RedCycles, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, RedCycles)] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [VCQ, VA1],
                                       mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From", [VCQ, VA1],
                                       mx, sew, IsWorstCase>;
      }
      defvar OrdRedCycles = BulletGetOrderedReductionCycles<mx, sew, VLEN>.c;
      let Latency = OrdRedCycles, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, OrdRedCycles)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [VCQ, VA1],
                                     mx, sew, IsWorstCase>;
    }
  }
  
  foreach mx = SchedMxListFWRed in {
    foreach sew = SchedSEWSet<mx, 1, 1>.val in {
      defvar RedCycles = BulletGetReductionCycles<mx, sew, VLEN>.c;
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;
      let Latency = RedCycles, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, RedCycles)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From", [VCQ, VA1],
                                     mx, sew, IsWorstCase>;
      defvar OrdRedCycles = BulletGetOrderedReductionCycles<mx, sew, VLEN>.c;
      let Latency = OrdRedCycles, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, OrdRedCycles)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [VCQ, VA1],
                                     mx, sew, IsWorstCase>;
    }
  }
  
  // 15. Vector Mask Instructions
  // Simple mask logical
  foreach mx = SchedMxList in {
    defvar Cycles = BulletGetCyclesVMask<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in
      defm "" : LMULWriteResMX<"WriteVMALUV", [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVMSFSV", [VCQ, VA1], mx, IsWorstCase>;
  }
  // Simple mask logical used in series
  foreach mx = SchedMxList in {
    defvar Cycles = BulletGetCyclesDefault<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVIotaV", [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIdxV", [VCQ, VA1], mx, IsWorstCase>;
    }
  }
  // Mask reduction
  foreach mx = SchedMxList in {
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 11, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 11)] in {
      defm "" : LMULWriteResMX<"WriteVMFFSV", [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVMPopV", [VCQ, VA1], mx, IsWorstCase>;
    }
  }

  // 16. Vector Permutation Instructions
  let Latency = 11, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 11)] in {
    def : WriteRes<WriteVMovXS, [VCQ, VA1]>;
    def : WriteRes<WriteVMovFS, [VCQ, VA1]>;
  }
  let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 1)] in {
    def : WriteRes<WriteVMovSX, [VCQ, VA1OrVA2]>;
    def : WriteRes<WriteVMovSF, [VCQ, VA1OrVA2]>;
  }
  foreach mx = SchedMxList in {
    defvar Cycles = BulletGetCyclesDefault<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 8, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVRGatherVX", [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVRGatherVI", [VCQ, VA1], mx, IsWorstCase>;
    }
  }
  foreach mx = SchedMxList in {
    foreach sew = SchedSEWSet<mx>.val in {
      defvar IsWorstCase = BulletIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
      defvar IsSingleDLEN = isSingleDLEN<mx>.c;

      defvar GatherVVCycles =
        BulletGetCyclesVRGatherVV<mx, sew, VLEN, hasFastLargeGather,
                                hasFastGather>.c;
      // 7 + DLEN/ SEW
      defvar SlowGatherLat = !add(7, !div(!div(VLEN, 2), sew));
      defvar GatherVVLat = !if(!or(!and(hasFastGather, IsSingleDLEN),
                                   hasFastLargeGather),
                              !add(3, GatherVVCycles), SlowGatherLat);

      let Latency = GatherVVLat, AcquireAtCycles = [0, 1],
          ReleaseAtCycles = [1, !add(5, GatherVVCycles)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [VCQ, VA1], mx, sew, IsWorstCase>;

      // VRGatherEI16VV is not improved by fastGather or fastLargeGather.
      defvar GatherEI16VVCycles = BulletGetCyclesOnePerElement<mx, sew, VLEN>.c;
      let Latency = SlowGatherLat, AcquireAtCycles = [0, 1],
          ReleaseAtCycles = [1, !add(5, GatherEI16VVCycles)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [VCQ, VA1], mx, sew, IsWorstCase>;

      defvar CompressCycles = BulletGetCyclesVCompress<mx, sew, VLEN, hasFastGather>.c;
      defvar CompressLat = !if(!and(hasFastGather, IsSingleDLEN),
                               4,
                               !add(7, CompressCycles)); // 7 + VL
      let Latency = CompressLat, AcquireAtCycles = [0, 1],
          ReleaseAtCycles = [1, !add(8, CompressCycles)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [VCQ, VA1], mx, sew, IsWorstCase>;
    }
  }
  
  foreach mx = SchedMxList in {
    defvar Cycles = BulletGetCyclesDefault<mx>.c;
    defvar IsWorstCase = BulletIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVSlideUpX",     [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSlideDownX",   [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSlideI",   [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVISlide1X",  [VCQ, VA1], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVFSlide1F",  [VCQ, VA1], mx, IsWorstCase>;
    }
  }
  
  // VMov*V is LMUL Aware
  let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 2)] in
    def : WriteRes<WriteVMov1V,     [VCQ, VA1OrVA2]>;
  let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 4)] in
    def : WriteRes<WriteVMov2V,     [VCQ, VA1OrVA2]>;
  let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 8)] in
    def : WriteRes<WriteVMov4V,     [VCQ, VA1OrVA2]>;
  let Latency = 4, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, !add(1, 16)] in
    def : WriteRes<WriteVMov8V,     [VCQ, VA1OrVA2]>;
  
  // Others
  def : WriteRes<WriteCSR, [PipeB]>;
  def : WriteRes<WriteNop, []>;
  let Latency = 3 in
    def : WriteRes<WriteRdVLENB, [PipeB]>;

  // FIXME: This could be better modeled by looking at the regclasses of the operands.
  def : InstRW<[WriteIALU, ReadIALU], (instrs COPY)>;
}

multiclass BulletReadAdvance {
  def ReadMemBaseOffset0 : SchedReadAdvance<1, [WriteLDB, WriteLDH, WriteLDW, WriteLDD]>;
  defvar ReadMemBaseOffset0 = !cast<SchedReadAdvance>(NAME #ReadMemBaseOffset0);

  def MemBaseOffset0PredVar : SchedVar<MemBaseOffset0Pred, [ReadMemBaseOffset0]>;
  def MemBaseOffset0NoPredVar : SchedVar<NoSchedPred, [NoReadAdvance]>;
  defvar MemBaseOffset0PredVar = !cast<SchedVar>(NAME #MemBaseOffset0PredVar);
  defvar MemBaseOffset0NoPredVar = !cast<SchedVar>(NAME #MemBaseOffset0NoPredVar);

  // Bypass the AG stage of a dependendent memory op after a load if the offset is
  // zero.
  def ReadMemBaseVariant : SchedReadVariant<[MemBaseOffset0PredVar, MemBaseOffset0NoPredVar]>;
  defvar ReadMemBaseVariant = !cast<SchedReadVariant>(NAME # ReadMemBaseVariant);
  def : SchedAlias<ReadMemBase, ReadMemBaseVariant>;

  def : BulletAnyToGPRBypass<ReadJmp>;
  def : BulletAnyToGPRBypass<ReadJalr>;
  def : ReadAdvance<ReadCSR, 0>;
  def : BulletAnyToGPRBypass<ReadStoreData>;
  def : BulletAnyToGPRBypass<ReadIALU>;
  def : BulletAnyToGPRBypass<ReadIALU32>;
  def : BulletAnyToGPRBypass<ReadShiftImm>;
  def : BulletAnyToGPRBypass<ReadShiftImm32>;
  def : BulletAnyToGPRBypass<ReadShiftReg>;
  def : BulletAnyToGPRBypass<ReadShiftReg32>;
  def : ReadAdvance<ReadIDiv, 0>;
  def : ReadAdvance<ReadIDiv32, 0>;
  def : ReadAdvance<ReadIRem, 0>;
  def : ReadAdvance<ReadIRem32, 0>;
  def : ReadAdvance<ReadIMul, 0>;
  def : ReadAdvance<ReadIMul32, 0>;
  def : ReadAdvance<ReadAtomicWA, 0>;
  def : ReadAdvance<ReadAtomicWD, 0>;
  def : ReadAdvance<ReadAtomicDA, 0>;
  def : ReadAdvance<ReadAtomicDD, 0>;
  def : ReadAdvance<ReadAtomicLDW, 0>;
  def : ReadAdvance<ReadAtomicLDD, 0>;
  def : ReadAdvance<ReadAtomicSTW, 0>;
  def : ReadAdvance<ReadAtomicSTD, 0>;
  def : ReadAdvance<ReadFStoreData, 0>;
  def : ReadAdvance<ReadFMemBase, 0>;
  def : BulletAnyToFPRBypass<ReadFAdd16>;
  def : BulletAnyToFPRBypass<ReadFAdd32>;
  def : BulletAnyToFPRBypass<ReadFAdd64>;
  def : BulletAnyToFPRBypass<ReadFMul16>;
  def : BulletAnyToFPRBypass<ReadFMA16>;
  def : BulletAnyToFPRBypass<ReadFMul32>;
  def : BulletAnyToFPRBypass<ReadFMA32>;
  def : BulletAnyToFPRBypass<ReadFMul64>;
  def : BulletAnyToFPRBypass<ReadFMA64>;
  def : BulletAnyToFPRBypass<ReadFDiv16>;
  def : BulletAnyToFPRBypass<ReadFDiv32>;
  def : BulletAnyToFPRBypass<ReadFDiv64>;
  def : BulletAnyToFPRBypass<ReadFSqrt16>;
  def : BulletAnyToFPRBypass<ReadFSqrt32>;
  def : BulletAnyToFPRBypass<ReadFSqrt64>;
  def : BulletAnyToFPRBypass<ReadFCmp16>;
  def : BulletAnyToFPRBypass<ReadFCmp32>;
  def : BulletAnyToFPRBypass<ReadFCmp64>;
  def : BulletAnyToFPRBypass<ReadFSGNJ16>;
  def : BulletAnyToFPRBypass<ReadFSGNJ32>;
  def : BulletAnyToFPRBypass<ReadFSGNJ64>;
  def : BulletAnyToFPRBypass<ReadFMinMax16>;
  def : BulletAnyToFPRBypass<ReadFMinMax32>;
  def : BulletAnyToFPRBypass<ReadFMinMax64>;
  def : BulletAnyToFPRBypass<ReadFCvtF16ToI32>;
  def : BulletAnyToFPRBypass<ReadFCvtF16ToI64>;
  def : BulletAnyToFPRBypass<ReadFCvtF32ToI32>;
  def : BulletAnyToFPRBypass<ReadFCvtF32ToI64>;
  def : BulletAnyToFPRBypass<ReadFCvtF64ToI32>;
  def : BulletAnyToFPRBypass<ReadFCvtF64ToI64>;
  def : ReadAdvance<ReadFMA16Addend, 0>;
  def : ReadAdvance<ReadFMA32Addend, 0>;
  def : ReadAdvance<ReadFMA64Addend, 0>;
  def : ReadAdvance<ReadFCvtI32ToF16, 0>;
  def : ReadAdvance<ReadFCvtI32ToF32, 0>;
  def : ReadAdvance<ReadFCvtI32ToF64, 0>;
  def : ReadAdvance<ReadFCvtI64ToF16, 0>;
  def : ReadAdvance<ReadFCvtI64ToF32, 0>;
  def : ReadAdvance<ReadFCvtI64ToF64, 0>;
  def : BulletAnyToFPRBypass<ReadFCvtF32ToF64>;
  def : BulletAnyToFPRBypass<ReadFCvtF64ToF32>;
  def : BulletAnyToFPRBypass<ReadFCvtF16ToF32>;
  def : BulletAnyToFPRBypass<ReadFCvtF32ToF16>;
  def : BulletAnyToFPRBypass<ReadFCvtF16ToF64>;
  def : BulletAnyToFPRBypass<ReadFCvtF64ToF16>;
  def : BulletAnyToFPRBypass<ReadFMovF16ToI16>;
  def : ReadAdvance<ReadFMovI16ToF16, 0>;
  def : BulletAnyToFPRBypass<ReadFMovF32ToI32>;
  def : ReadAdvance<ReadFMovI32ToF32, 0>;
  def : BulletAnyToFPRBypass<ReadFMovF64ToI64>;
  def : ReadAdvance<ReadFMovI64ToF64, 0>;
  def : ReadAdvance<ReadFClass16, 0>;
  def : ReadAdvance<ReadFClass32, 0>;
  def : ReadAdvance<ReadFClass64, 0>;
  
  // 6. Configuration-Setting Instructions
  def : ReadAdvance<ReadVSETVLI, 2>;
  def : ReadAdvance<ReadVSETVL, 2>;
  
  // 7. Vector Loads and Stores
  def : ReadAdvance<ReadVLDX, 0>;
  def : ReadAdvance<ReadVSTX, 0>;
  defm "" : LMULReadAdvance<"ReadVSTEV", 0>;
  defm "" : LMULReadAdvance<"ReadVSTM", 0>;
  def : ReadAdvance<ReadVLDSX, 0>;
  def : ReadAdvance<ReadVSTSX, 0>;
  defm "" : LMULReadAdvance<"ReadVSTS8V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTS16V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTS32V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTS64V", 0>;
  defm "" : LMULReadAdvance<"ReadVLDUXV", 0>;
  defm "" : LMULReadAdvance<"ReadVLDOXV", 0>;
  defm "" : LMULReadAdvance<"ReadVSTUX8", 0>;
  defm "" : LMULReadAdvance<"ReadVSTUX16", 0>;
  defm "" : LMULReadAdvance<"ReadVSTUX32", 0>;
  defm "" : LMULReadAdvance<"ReadVSTUX64", 0>;
  defm "" : LMULReadAdvance<"ReadVSTUXV", 0>;
  defm "" : LMULReadAdvance<"ReadVSTUX8V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTUX16V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTUX32V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTUX64V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTOX8", 0>;
  defm "" : LMULReadAdvance<"ReadVSTOX16", 0>;
  defm "" : LMULReadAdvance<"ReadVSTOX32", 0>;
  defm "" : LMULReadAdvance<"ReadVSTOX64", 0>;
  defm "" : LMULReadAdvance<"ReadVSTOXV", 0>;
  defm "" : LMULReadAdvance<"ReadVSTOX8V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTOX16V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTOX32V", 0>;
  defm "" : LMULReadAdvance<"ReadVSTOX64V", 0>;
  // LMUL Aware
  def : ReadAdvance<ReadVST1R, 0>;
  def : ReadAdvance<ReadVST2R, 0>;
  def : ReadAdvance<ReadVST4R, 0>;
  def : ReadAdvance<ReadVST8R, 0>;
  
  // 12. Vector Integer Arithmetic Instructions
  defm : LMULReadAdvance<"ReadVIALUV", 0>;
  defm : LMULReadAdvance<"ReadVIALUX", 0>;
  defm : LMULReadAdvanceW<"ReadVIWALUV", 0>;
  defm : LMULReadAdvanceW<"ReadVIWALUX", 0>;
  defm : LMULReadAdvance<"ReadVExtV", 0>;
  defm : LMULReadAdvance<"ReadVICALUV", 0>;
  defm : LMULReadAdvance<"ReadVICALUX", 0>;
  defm : LMULReadAdvance<"ReadVShiftV", 0>;
  defm : LMULReadAdvance<"ReadVShiftX", 0>;
  defm : LMULReadAdvanceW<"ReadVNShiftV", 0>;
  defm : LMULReadAdvanceW<"ReadVNShiftX", 0>;
  defm : LMULReadAdvance<"ReadVICmpV", 0>;
  defm : LMULReadAdvance<"ReadVICmpX", 0>;
  defm : LMULReadAdvance<"ReadVIMinMaxV", 0>;
  defm : LMULReadAdvance<"ReadVIMinMaxX", 0>;
  defm : LMULReadAdvance<"ReadVIMulV", 0>;
  defm : LMULReadAdvance<"ReadVIMulX", 0>;
  defm : LMULSEWReadAdvance<"ReadVIDivV", 0>;
  defm : LMULSEWReadAdvance<"ReadVIDivX", 0>;
  defm : LMULReadAdvanceW<"ReadVIWMulV", 0>;
  defm : LMULReadAdvanceW<"ReadVIWMulX", 0>;
  defm : LMULReadAdvance<"ReadVIMulAddV", 0>;
  defm : LMULReadAdvance<"ReadVIMulAddX", 0>;
  defm : LMULReadAdvanceW<"ReadVIWMulAddV", 0>;
  defm : LMULReadAdvanceW<"ReadVIWMulAddX", 0>;
  defm : LMULReadAdvance<"ReadVIMergeV", 0>;
  defm : LMULReadAdvance<"ReadVIMergeX", 0>;
  defm : LMULReadAdvance<"ReadVIMovV", 0>;
  defm : LMULReadAdvance<"ReadVIMovX", 0>;
  
  // 13. Vector Fixed-Point Arithmetic Instructions
  defm "" : LMULReadAdvance<"ReadVSALUV", 0>;
  defm "" : LMULReadAdvance<"ReadVSALUX", 0>;
  defm "" : LMULReadAdvance<"ReadVAALUV", 0>;
  defm "" : LMULReadAdvance<"ReadVAALUX", 0>;
  defm "" : LMULReadAdvance<"ReadVSMulV", 0>;
  defm "" : LMULReadAdvance<"ReadVSMulX", 0>;
  defm "" : LMULReadAdvance<"ReadVSShiftV", 0>;
  defm "" : LMULReadAdvance<"ReadVSShiftX", 0>;
  defm "" : LMULReadAdvanceW<"ReadVNClipV", 0>;
  defm "" : LMULReadAdvanceW<"ReadVNClipX", 0>;
  
  // 14. Vector Floating-Point Instructions
  defm "" : LMULSEWReadAdvanceF<"ReadVFALUV", 0>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFALUF", -3>;
  defm "" : LMULSEWReadAdvanceFW<"ReadVFWALUV", 0>;
  defm "" : LMULSEWReadAdvanceFW<"ReadVFWALUF", -3>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFMulV", 0>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFMulF", -3>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFDivV", 0>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFDivF", -3>;
  defm "" : LMULSEWReadAdvanceFW<"ReadVFWMulV", 0>;
  defm "" : LMULSEWReadAdvanceFW<"ReadVFWMulF", -3>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFMulAddV", 0>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFMulAddF", -3>;
  defm "" : LMULSEWReadAdvanceFW<"ReadVFWMulAddV", 0>;
  defm "" : LMULSEWReadAdvanceFW<"ReadVFWMulAddF", -3>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFSqrtV", 0>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFRecpV", 0>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFMinMaxV", 0>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFMinMaxF", -3>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFSgnjV", 0>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFSgnjF", -3>;
  defm "" : LMULReadAdvance<"ReadVFCmpV", 0>;
  defm "" : LMULReadAdvance<"ReadVFCmpF", -3>;
  defm "" : LMULReadAdvance<"ReadVFClassV", 0>;
  defm "" : LMULReadAdvance<"ReadVFMergeV", 0>;
  defm "" : LMULReadAdvance<"ReadVFMergeF", -3>;
  defm "" : LMULReadAdvance<"ReadVFMovF", -3>;
  defm "" : LMULSEWReadAdvanceF<"ReadVFCvtIToFV", 0>;
  defm "" : LMULReadAdvance<"ReadVFCvtFToIV", 0>;
  defm "" : LMULSEWReadAdvanceW<"ReadVFWCvtIToFV", 0>;
  defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 0>;
  defm "" : LMULSEWReadAdvanceFW<"ReadVFWCvtFToFV", 0>;
  defm "" : LMULSEWReadAdvanceFW<"ReadVFNCvtIToFV", 0>;
  defm "" : LMULReadAdvanceW<"ReadVFNCvtFToIV", 0>;
  defm "" : LMULSEWReadAdvanceFW<"ReadVFNCvtFToFV", 0>;
  
  // 15. Vector Reduction Operations
  def : ReadAdvance<ReadVIRedV, 0>;
  def : ReadAdvance<ReadVIRedV0, 0>;
  def : ReadAdvance<ReadVIWRedV, 0>;
  def : ReadAdvance<ReadVIWRedV0, 0>;
  def : ReadAdvance<ReadVFRedV, 0>;
  def : ReadAdvance<ReadVFRedV0, 0>;
  def : ReadAdvance<ReadVFRedOV, 0>;
  def : ReadAdvance<ReadVFRedOV0, 0>;
  def : ReadAdvance<ReadVFWRedV, 0>;
  def : ReadAdvance<ReadVFWRedV0, 0>;
  def : ReadAdvance<ReadVFWRedOV, 0>;
  def : ReadAdvance<ReadVFWRedOV0, 0>;
  
  // 16. Vector Mask Instructions
  defm "" : LMULReadAdvance<"ReadVMALUV", 0>;
  defm "" : LMULReadAdvance<"ReadVMPopV", 0>;
  defm "" : LMULReadAdvance<"ReadVMFFSV", 0>;
  defm "" : LMULReadAdvance<"ReadVMSFSV", 0>;
  defm "" : LMULReadAdvance<"ReadVIotaV", 0>;
  
  // 17. Vector Permutation Instructions
  def : ReadAdvance<ReadVMovXS, 0>;
  def : ReadAdvance<ReadVMovSX_V, 0>;
  def : ReadAdvance<ReadVMovSX_X, 0>;
  def : ReadAdvance<ReadVMovFS, 0>;
  def : ReadAdvance<ReadVMovSF_V, 0>;
  def : ReadAdvance<ReadVMovSF_F, -3>;
  defm "" : LMULReadAdvance<"ReadVISlideV", 0>;
  defm "" : LMULReadAdvance<"ReadVISlideX", 0>;
  defm "" : LMULReadAdvance<"ReadVFSlideV", 0>;
  defm "" : LMULReadAdvance<"ReadVFSlideF", -3>;
  defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_data", 0>;
  defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_index", 0>;
  defm "" : LMULSEWReadAdvance<"ReadVRGatherEI16VV_data", 0>;
  defm "" : LMULSEWReadAdvance<"ReadVRGatherEI16VV_index", 0>;
  defm "" : LMULReadAdvance<"ReadVRGatherVX_data", 0>;
  defm "" : LMULReadAdvance<"ReadVRGatherVX_index", 0>;
  defm "" : LMULReadAdvance<"ReadVRGatherVI_data", 0>;
  defm "" : LMULSEWReadAdvance<"ReadVCompressV", 0>;
  // LMUL Aware
  def : ReadAdvance<ReadVMov1V, 0>;
  def : ReadAdvance<ReadVMov2V, 0>;
  def : ReadAdvance<ReadVMov4V, 0>;
  def : ReadAdvance<ReadVMov8V, 0>;
  
  // Others
  def : ReadAdvance<ReadVMask, 0>;
  def : ReadAdvance<ReadVMergeOp_WorstCase, 0>;
  foreach mx = SchedMxList in {
    def : ReadAdvance<!cast<SchedRead>("ReadVMergeOp_" # mx), 0>;
    foreach sew = SchedSEWSet<mx>.val in
      def : ReadAdvance<!cast<SchedRead>("ReadVMergeOp_" # mx  # "_E" # sew), 0>;
  }
  
  // Bitmanip
  def : BulletAnyToGPRBypass<ReadRotateImm>;
  def : BulletAnyToGPRBypass<ReadRotateImm32>;
  def : BulletAnyToGPRBypass<ReadRotateReg>;
  def : BulletAnyToGPRBypass<ReadRotateReg32>;
  def : BulletAnyToGPRBypass<ReadCLZ>;
  def : BulletAnyToGPRBypass<ReadCLZ32>;
  def : BulletAnyToGPRBypass<ReadCTZ>;
  def : BulletAnyToGPRBypass<ReadCTZ32>;
  def : ReadAdvance<ReadCPOP, 0>;
  def : ReadAdvance<ReadCPOP32, 0>;
  def : BulletAnyToGPRBypass<ReadORCB>;
  def : BulletAnyToGPRBypass<ReadIMinMax>;
  def : BulletAnyToGPRBypass<ReadREV8>;
  def : BulletAnyToGPRBypass<ReadSHXADD>;
  def : BulletAnyToGPRBypass<ReadSHXADD32>;
  // Single-bit instructions
  def : BulletAnyToGPRBypass<ReadSingleBit>;
  def : BulletAnyToGPRBypass<ReadSingleBitImm>;
  
  // Short forward branch
  // ALU, ALU_B, SFB to SFB
  // FIXME: Is this right?
  def : ReadAdvance<ReadSFBJmp, 2, [WriteIALU, WriteIALU32,
                                    WriteShiftImm, WriteShiftImm32,
                                    WriteShiftReg, WriteShiftReg32,
                                    WriteSHXADD, WriteSHXADD32,
                                    WriteRotateImm, WriteRotateImm32,
                                    WriteRotateReg, WriteRotateReg32,
                                    WriteCLZ, WriteCLZ32, WriteCTZ,
                                    WriteCTZ32, WriteCPOP, WriteCPOP32,
                                    WriteREV8, WriteORCB, WriteSFB]>;
  def : ReadAdvance<ReadSFBALU, 2, [WriteIALU, WriteIALU32,
                                    WriteShiftImm, WriteShiftImm32,
                                    WriteShiftReg, WriteShiftReg32,
                                    WriteSHXADD, WriteSHXADD32,
                                    WriteRotateImm, WriteRotateImm32,
                                    WriteRotateReg, WriteRotateReg32,
                                    WriteCLZ, WriteCLZ32, WriteCTZ,
                                    WriteCTZ32, WriteCPOP, WriteCPOP32,
                                    WriteREV8, WriteORCB, WriteSFB]>;
  
}
