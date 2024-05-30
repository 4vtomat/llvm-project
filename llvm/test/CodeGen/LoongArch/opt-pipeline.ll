;; When EXPENSIVE_CHECKS are enabled, the machine verifier appears between each
;; pass. Ignore it with 'grep -v'.
; RUN: llc --mtriple=loongarch32 -mattr=+d -O1 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefix=LAXX
; RUN: llc --mtriple=loongarch32 -mattr=+d -O2 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefix=LAXX
; RUN: llc --mtriple=loongarch32 -mattr=+d -O3 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefix=LAXX
; RUN: llc --mtriple=loongarch64 -mattr=+d -O1 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=LAXX,LA64
; RUN: llc --mtriple=loongarch64 -mattr=+d -O2 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=LAXX,LA64
; RUN: llc --mtriple=loongarch64 -mattr=+d -O3 --debug-pass=Structure %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | FileCheck %s --check-prefixes=LAXX,LA64

; REQUIRES: asserts

<<<<<<< HEAD
; CHECK-LABEL: Pass Arguments:
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Target Pass Configuration
; CHECK-NEXT: Machine Module Information
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Type-Based Alias Analysis
; CHECK-NEXT: Scoped NoAlias Alias Analysis
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT: Profile summary info
; CHECK-NEXT: Create Garbage Collector Module Metadata
; CHECK-NEXT: Machine Branch Probability Analysis
; CHECK-NEXT: Default Regalloc Eviction Advisor
; CHECK-NEXT: Default Regalloc Priority Advisor
; CHECK-NEXT:   ModulePass Manager
; CHECK-NEXT:     Pre-ISel Intrinsic Lowering
; CHECK-NEXT:     FunctionPass Manager
; CHECK-NEXT:       Expand large div/rem
; CHECK-NEXT:       Expand large fp convert
; CHECK-NEXT:       Expand powi functions
; CHECK-NEXT:       Expand vp.reduce functions
; CHECK-NEXT:       Expand Atomic instructions
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Canonicalize natural loops
; CHECK-NEXT:       Scalar Evolution Analysis
; CHECK-NEXT:       Loop Pass Manager
; CHECK-NEXT:         Canonicalize Freeze Instructions in Loops
; CHECK-NEXT:         Induction Variable Users
; CHECK-NEXT:         Loop Strength Reduction
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Merge contiguous icmps into a memcmp
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       Expand memcmp() to load/stores
; CHECK-NEXT:       Lower Garbage Collection Instructions
; CHECK-NEXT:       Shadow Stack GC Lowering
; CHECK-NEXT:       Lower constant intrinsics
; CHECK-NEXT:       Remove unreachable blocks from the CFG
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Post-Dominator Tree Construction
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Block Frequency Analysis
; CHECK-NEXT:       Constant Hoisting
; CHECK-NEXT:       Replace intrinsics with calls to vector library
; CHECK-NEXT:       Partially inline calls to library functions
; CHECK-NEXT:       Expand vector predication intrinsics
; CHECK-NEXT:       Scalarize Masked Memory Intrinsics
; CHECK-NEXT:       Expand reduction intrinsics
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       TLS Variable Hoist
; CHECK-NEXT:       CodeGen Prepare
; CHECK-NEXT:       Dominator Tree Construction
; CHECK-NEXT:       Exception handling preparation
; CHECK-NEXT:       Prepare callbr
; CHECK-NEXT:       Safe Stack instrumentation pass
; CHECK-NEXT:       Insert stack protectors
; CHECK-NEXT:       Module Verifier
; CHECK-NEXT:       Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:       Function Alias Analysis Results
; CHECK-NEXT:       Natural Loop Information
; CHECK-NEXT:       Post-Dominator Tree Construction
; CHECK-NEXT:       Branch Probability Analysis
; CHECK-NEXT:       Assignment Tracking Analysis
; CHECK-NEXT:       Lazy Branch Probability Analysis
; CHECK-NEXT:       Lazy Block Frequency Analysis
; CHECK-NEXT:       LoongArch DAG->DAG Pattern Instruction Selection
; CHECK-NEXT:       Finalize ISel and expand pseudo-instructions
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Early Tail Duplication
; CHECK-NEXT:       Optimize machine instruction PHIs
; CHECK-NEXT:       Slot index numbering
; CHECK-NEXT:       Merge disjoint stack slots
; CHECK-NEXT:       Local Stack Slot Allocation
; CHECK-NEXT:       Remove dead machine instructions
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Machine Natural Loop Construction
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       Early Machine Loop Invariant Code Motion
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Common Subexpression Elimination
; CHECK-NEXT:       MachinePostDominator Tree Construction
; CHECK-NEXT:       Machine Cycle Info Analysis
; CHECK-NEXT:       Machine code sinking
; CHECK-NEXT:       Peephole Optimizations
; CHECK-NEXT:       Remove dead machine instructions
; CHECK-NEXT:       LoongArch Pre-RA pseudo instruction expansion pass
; CHECK-NEXT:       Detect Dead Lanes
; CHECK-NEXT:       Init Undef Pass
; CHECK-NEXT:       Process Implicit Definitions
; CHECK-NEXT:       Remove unreachable machine basic blocks
; CHECK-NEXT:       Live Variable Analysis
; CHECK-NEXT:       Eliminate PHI nodes for register allocation
; CHECK-NEXT:       Two-Address instruction pass
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Slot index numbering
; CHECK-NEXT:       Live Interval Analysis
; CHECK-NEXT:       Register Coalescer
; CHECK-NEXT:       Rename Disconnected Subregister Components
; CHECK-NEXT:       Machine Instruction Scheduler
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       Debug Variable Analysis
; CHECK-NEXT:       Live Stack Slot Analysis
; CHECK-NEXT:       Virtual Register Map
; CHECK-NEXT:       Live Register Matrix
; CHECK-NEXT:       Bundle Machine CFG Edges
; CHECK-NEXT:       Spill Code Placement Analysis
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Greedy Register Allocator
; CHECK-NEXT:       Virtual Register Rewriter
; CHECK-NEXT:       Register Allocation Pass Scoring
; CHECK-NEXT:       Stack Slot Coloring
; CHECK-NEXT:       Machine Copy Propagation Pass
; CHECK-NEXT:       Machine Loop Invariant Code Motion
; CHECK-NEXT:       Remove Redundant DEBUG_VALUE analysis
; CHECK-NEXT:       Fixup Statepoint Caller Saved
; CHECK-NEXT:       PostRA Machine Sink
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       MachinePostDominator Tree Construction
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Shrink Wrapping analysis
; CHECK-NEXT:       Prologue/Epilogue Insertion & Frame Finalization
; CHECK-NEXT:       Machine Late Instructions Cleanup Pass
; CHECK-NEXT:       Control Flow Optimizer
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Tail Duplication
; CHECK-NEXT:       Machine Copy Propagation Pass
; CHECK-NEXT:       Post-RA pseudo instruction expansion pass
; CHECK-NEXT:       MachineDominator Tree Construction
; CHECK-NEXT:       Machine Natural Loop Construction
; CHECK-NEXT:       Post RA top-down list latency scheduler
; CHECK-NEXT:       Analyze Machine Code For Garbage Collection
; CHECK-NEXT:       Machine Block Frequency Analysis
; CHECK-NEXT:       MachinePostDominator Tree Construction
; CHECK-NEXT:       Branch Probability Basic Block Placement
; CHECK-NEXT:       Insert fentry calls
; CHECK-NEXT:       Insert XRay ops
; CHECK-NEXT:       Implement the 'patchable-function' attribute
; CHECK-NEXT:       Branch relaxation pass
; CHECK-NEXT:       Contiguously Lay Out Funclets
; CHECK-NEXT:       StackMap Liveness Analysis
; CHECK-NEXT:       Live DEBUG_VALUE analysis
; CHECK-NEXT:       Machine Sanitizer Binary Metadata
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       Stack Frame Layout Analysis
; CHECK-NEXT:       LoongArch pseudo instruction expansion pass
; CHECK-NEXT:       LoongArch atomic pseudo instruction expansion pass
; CHECK-NEXT:       Lazy Machine Block Frequency Analysis
; CHECK-NEXT:       Machine Optimization Remark Emitter
; CHECK-NEXT:       LoongArch Assembly Printer
; CHECK-NEXT:       Free MachineFunction
=======
; LAXX-LABEL: Pass Arguments:
; LAXX-NEXT: Target Library Information
; LAXX-NEXT: Target Pass Configuration
; LAXX-NEXT: Machine Module Information
; LAXX-NEXT: Target Transform Information
; LAXX-NEXT: Type-Based Alias Analysis
; LAXX-NEXT: Scoped NoAlias Alias Analysis
; LAXX-NEXT: Assumption Cache Tracker
; LAXX-NEXT: Profile summary info
; LAXX-NEXT: Create Garbage Collector Module Metadata
; LAXX-NEXT: Machine Branch Probability Analysis
; LAXX-NEXT: Default Regalloc Eviction Advisor
; LAXX-NEXT: Default Regalloc Priority Advisor
; LAXX-NEXT:   ModulePass Manager
; LAXX-NEXT:     Pre-ISel Intrinsic Lowering
; LAXX-NEXT:     FunctionPass Manager
; LAXX-NEXT:       Expand large div/rem
; LAXX-NEXT:       Expand large fp convert
; LAXX-NEXT:       Expand Atomic instructions
; LAXX-NEXT:       Module Verifier
; LAXX-NEXT:       Dominator Tree Construction
; LAXX-NEXT:       Basic Alias Analysis (stateless AA impl)
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       Canonicalize natural loops
; LAXX-NEXT:       Scalar Evolution Analysis
; LAXX-NEXT:       Loop Pass Manager
; LAXX-NEXT:         Canonicalize Freeze Instructions in Loops
; LAXX-NEXT:         Induction Variable Users
; LAXX-NEXT:         Loop Strength Reduction
; LAXX-NEXT:       Basic Alias Analysis (stateless AA impl)
; LAXX-NEXT:       Function Alias Analysis Results
; LAXX-NEXT:       Merge contiguous icmps into a memcmp
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       Lazy Branch Probability Analysis
; LAXX-NEXT:       Lazy Block Frequency Analysis
; LAXX-NEXT:       Expand memcmp() to load/stores
; LAXX-NEXT:       Lower Garbage Collection Instructions
; LAXX-NEXT:       Shadow Stack GC Lowering
; LAXX-NEXT:       Lower constant intrinsics
; LAXX-NEXT:       Remove unreachable blocks from the CFG
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       Post-Dominator Tree Construction
; LAXX-NEXT:       Branch Probability Analysis
; LAXX-NEXT:       Block Frequency Analysis
; LAXX-NEXT:       Constant Hoisting
; LAXX-NEXT:       Replace intrinsics with calls to vector library
; LAXX-NEXT:       Partially inline calls to library functions
; LAXX-NEXT:       Expand vector predication intrinsics
; LAXX-NEXT:       Scalarize Masked Memory Intrinsics
; LAXX-NEXT:       Expand reduction intrinsics
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       TLS Variable Hoist
; LAXX-NEXT:       CodeGen Prepare
; LAXX-NEXT:       Dominator Tree Construction
; LAXX-NEXT:       Exception handling preparation
; LAXX-NEXT:       Prepare callbr
; LAXX-NEXT:       Safe Stack instrumentation pass
; LAXX-NEXT:       Insert stack protectors
; LAXX-NEXT:       Module Verifier
; LAXX-NEXT:       Basic Alias Analysis (stateless AA impl)
; LAXX-NEXT:       Function Alias Analysis Results
; LAXX-NEXT:       Natural Loop Information
; LAXX-NEXT:       Post-Dominator Tree Construction
; LAXX-NEXT:       Branch Probability Analysis
; LAXX-NEXT:       Assignment Tracking Analysis
; LAXX-NEXT:       Lazy Branch Probability Analysis
; LAXX-NEXT:       Lazy Block Frequency Analysis
; LAXX-NEXT:       LoongArch DAG->DAG Pattern Instruction Selection
; LAXX-NEXT:       Finalize ISel and expand pseudo-instructions
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Early Tail Duplication
; LAXX-NEXT:       Optimize machine instruction PHIs
; LAXX-NEXT:       Slot index numbering
; LAXX-NEXT:       Merge disjoint stack slots
; LAXX-NEXT:       Local Stack Slot Allocation
; LAXX-NEXT:       Remove dead machine instructions
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       Machine Natural Loop Construction
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       Early Machine Loop Invariant Code Motion
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Common Subexpression Elimination
; LAXX-NEXT:       MachinePostDominator Tree Construction
; LAXX-NEXT:       Machine Cycle Info Analysis
; LAXX-NEXT:       Machine code sinking
; LAXX-NEXT:       Peephole Optimizations
; LAXX-NEXT:       Remove dead machine instructions
; LA64-NEXT:       LoongArch Optimize W Instructions
; LAXX-NEXT:       LoongArch Pre-RA pseudo instruction expansion pass
; LAXX-NEXT:       Detect Dead Lanes
; LAXX-NEXT:       Init Undef Pass
; LAXX-NEXT:       Process Implicit Definitions
; LAXX-NEXT:       Remove unreachable machine basic blocks
; LAXX-NEXT:       Live Variable Analysis
; LAXX-NEXT:       Eliminate PHI nodes for register allocation
; LAXX-NEXT:       Two-Address instruction pass
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       Slot index numbering
; LAXX-NEXT:       Live Interval Analysis
; LAXX-NEXT:       Register Coalescer
; LAXX-NEXT:       Rename Disconnected Subregister Components
; LAXX-NEXT:       Machine Instruction Scheduler
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       Debug Variable Analysis
; LAXX-NEXT:       Live Stack Slot Analysis
; LAXX-NEXT:       Virtual Register Map
; LAXX-NEXT:       Live Register Matrix
; LAXX-NEXT:       Bundle Machine CFG Edges
; LAXX-NEXT:       Spill Code Placement Analysis
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Optimization Remark Emitter
; LAXX-NEXT:       Greedy Register Allocator
; LAXX-NEXT:       Virtual Register Rewriter
; LAXX-NEXT:       Register Allocation Pass Scoring
; LAXX-NEXT:       Stack Slot Coloring
; LAXX-NEXT:       Machine Copy Propagation Pass
; LAXX-NEXT:       Machine Loop Invariant Code Motion
; LAXX-NEXT:       Remove Redundant DEBUG_VALUE analysis
; LAXX-NEXT:       Fixup Statepoint Caller Saved
; LAXX-NEXT:       PostRA Machine Sink
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       MachinePostDominator Tree Construction
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Optimization Remark Emitter
; LAXX-NEXT:       Shrink Wrapping analysis
; LAXX-NEXT:       Prologue/Epilogue Insertion & Frame Finalization
; LAXX-NEXT:       Machine Late Instructions Cleanup Pass
; LAXX-NEXT:       Control Flow Optimizer
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Tail Duplication
; LAXX-NEXT:       Machine Copy Propagation Pass
; LAXX-NEXT:       Post-RA pseudo instruction expansion pass
; LAXX-NEXT:       MachineDominator Tree Construction
; LAXX-NEXT:       Machine Natural Loop Construction
; LAXX-NEXT:       Post RA top-down list latency scheduler
; LAXX-NEXT:       Analyze Machine Code For Garbage Collection
; LAXX-NEXT:       Machine Block Frequency Analysis
; LAXX-NEXT:       MachinePostDominator Tree Construction
; LAXX-NEXT:       Branch Probability Basic Block Placement
; LAXX-NEXT:       Insert fentry calls
; LAXX-NEXT:       Insert XRay ops
; LAXX-NEXT:       Implement the 'patchable-function' attribute
; LAXX-NEXT:       Branch relaxation pass
; LAXX-NEXT:       Contiguously Lay Out Funclets
; LAXX-NEXT:       StackMap Liveness Analysis
; LAXX-NEXT:       Live DEBUG_VALUE analysis
; LAXX-NEXT:       Machine Sanitizer Binary Metadata
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Optimization Remark Emitter
; LAXX-NEXT:       Stack Frame Layout Analysis
; LAXX-NEXT:       LoongArch pseudo instruction expansion pass
; LAXX-NEXT:       LoongArch atomic pseudo instruction expansion pass
; LAXX-NEXT:       Lazy Machine Block Frequency Analysis
; LAXX-NEXT:       Machine Optimization Remark Emitter
; LAXX-NEXT:       LoongArch Assembly Printer
; LAXX-NEXT:       Free MachineFunction
>>>>>>> 855eef2
