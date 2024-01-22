// -flto causes a switch to llvm-bc object files.
// RUN: %clang -ccc-print-phases -c %s -flto 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILE-ACTIONS < %t %s
//
// CHECK-COMPILE-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILE-ACTIONS: 3: backend, {2}, lto-bc

// RUN: %clang -ccc-print-phases %s -flto 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILELINK-ACTIONS < %t %s
//
// CHECK-COMPILELINK-ACTIONS: 0: input, "{{.*}}lto.c", c
// CHECK-COMPILELINK-ACTIONS: 1: preprocessor, {0}, cpp-output
// CHECK-COMPILELINK-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILELINK-ACTIONS: 3: backend, {2}, lto-bc
// CHECK-COMPILELINK-ACTIONS: 4: linker, {3}, image

// llvm-bc and llvm-ll outputs need to match regular suffixes
// (unfortunately).
// RUN: %clang --target=x86_64-linux-gnu %s -flto -save-temps -### 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILELINK-SUFFIXES < %t %s
//
// CHECK-COMPILELINK-SUFFIXES: "-o" "{{.*}}lto.i" "-x" "c" "{{.*}}lto.c"
// CHECK-COMPILELINK-SUFFIXES: "-o" "{{.*}}lto.bc" {{.*}}"{{.*}}lto.i"
// CHECK-COMPILELINK-SUFFIXES: "-o" "{{.*}}lto.o" {{.*}}"{{.*}}lto.bc"
// CHECK-COMPILELINK-SUFFIXES: "{{.*}}a.{{(out|exe)}}" {{.*}}"{{.*}}lto.o"

// RUN: %clang --target=x86_64-linux-gnu %s -flto -S -### 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILE-SUFFIXES < %t %s
//
// CHECK-COMPILE-SUFFIXES: "-o" "{{.*}}lto.s" "-x" "c" "{{.*}}lto.c"

// RUN: not %clang %s -emit-llvm 2>&1 | FileCheck --check-prefix=LLVM-LINK %s
// LLVM-LINK: -emit-llvm cannot be used when linking

/// With ld.bfd or gold, link against LLVMgold.
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=bfd -flto=thin -### 2>&1 | FileCheck --check-prefix=LLVMGOLD %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=gold -flto=full -### 2>&1 | FileCheck --check-prefix=LLVMGOLD %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=gold -fno-lto -flto -### 2>&1 | FileCheck --check-prefix=LLVMGOLD %s
// LLVMGOLD: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"

/// lld does not need LLVMgold.
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -### 2>&1 | FileCheck --check-prefix=NO-LLVMGOLD %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=gold -flto -fno-lto -### 2>&1 | FileCheck --check-prefix=NO-LLVMGOLD %s
// NO-LLVMGOLD-NOT: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"

// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -O -### 2>&1 | FileCheck --check-prefix=O1 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -O1 -### 2>&1 | FileCheck --check-prefix=O1 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -Og -### 2>&1 | FileCheck --check-prefix=O1 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -O2 -### 2>&1 | FileCheck --check-prefix=O2 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -Os -### 2>&1 | FileCheck --check-prefix=O2 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -Oz -### 2>&1 | FileCheck --check-prefix=O2 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -O3 -### 2>&1 | FileCheck --check-prefix=O3 %s
// RUN: %clang --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_cross_linux_tree %s \
// RUN:   -fuse-ld=lld -flto -Ofast -### 2>&1 | FileCheck --check-prefix=O3 %s

// O1: -plugin-opt=O1
// O2: -plugin-opt=O2
// O3: -plugin-opt=O3

// -flto passes along an explicit debugger tuning argument.
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto -glldb 2> %t
// RUN: FileCheck -check-prefix=CHECK-TUNING-LLDB < %t %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto -g 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-TUNING < %t %s
//
// CHECK-TUNING-LLDB:   "-plugin-opt=-debugger-tune=lldb"
// CHECK-NO-TUNING-NOT: "-plugin-opt=-debugger-tune
//
// -flto=auto and -flto=jobserver pass along -flto=full
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=auto 2>&1 | FileCheck --check-prefix=FLTO-AUTO %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=jobserver 2>&1 | FileCheck --check-prefix=FLTO-JOBSERVER %s
//
// FLTO-AUTO: -flto=full
// FLTO-JOBSERVER: -flto=full
//

// Pass the last -flto argument.
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -flto 2>&1 | \
// RUN: FileCheck --check-prefix=FLTO-FULL %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -flto=full \
// RUN: 2>&1 | FileCheck --check-prefix=FLTO-FULL %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=full -flto=thin  \
// RUN: 2>&1 | FileCheck --check-prefix=FLTO-THIN %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -flto -flto=thin 2>&1 | \
// RUN: FileCheck --check-prefix=FLTO-THIN %s
//
// FLTO-FULL-NOT: -flto=thin
// FLTO-FULL: -flto=full
// FLTO-FULL-NOT: -flto=thin
//
// FLTO-THIN-NOT: -flto=full
// FLTO-THIN-NOT: "-flto"
// FLTO-THIN: -flto=thin
// FLTO-THIN-NOT: "-flto"
// FLTO-THIN-NOT: -flto=full

// if SIFIVE_CUSTOMIZATION
// Need to pass -target-abi option in RISC-V target.
// RUN: %clang --target=riscv32 %s -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV32-DEFAULT
// RUN: %clang --target=riscv64 %s -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV64-DEFAULT
// RV32-DEFAULT: "-plugin-opt=-target-abi=ilp32"
// RV64-DEFAULT: "-plugin-opt=-target-abi=lp64d"
//
// RUN: %clang --target=riscv32-unknown-elf %s -fuse-ld=gold -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV32-DEFAULT-ELF
// RUN: %clang --target=riscv32-unknown-elf %s -fuse-ld=lld -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV32-DEFAULT-ELF
// RUN: %clang --target=riscv32-unknown-linux-gnu %s -fuse-ld=gold -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV32-DEFAULT-LINUX
// RUN: %clang --target=riscv32-unknown-linux-gnu %s -fuse-ld=lld -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV32-DEFAULT-LINUX
//
// RV32-DEFAULT-ELF: "-plugin-opt=-target-abi=ilp32"
// RV32-DEFAULT-LINUX: "-plugin-opt=-target-abi=ilp32d"

// RUN: %clang --target=riscv64-unknown-elf %s -fuse-ld=gold -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV64-DEFAULT-ELF
// RUN: %clang --target=riscv64-unknown-elf %s -fuse-ld=lld -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV64-DEFAULT-ELF
// RUN: %clang --target=riscv64-unknown-linux-gnu %s -fuse-ld=gold -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV64-DEFAULT-LINUX
// RUN: %clang --target=riscv64-unknown-linux-gnu %s -fuse-ld=lld -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RV64-DEFAULT-LINUX
//
// RV64-DEFAULT-ELF: "-plugin-opt=-target-abi=lp64d"
// RV64-DEFAULT-LINUX: "-plugin-opt=-target-abi=lp64d"

// RUN: %clang --target=riscv32-unknown-linux-gnu %s -fuse-ld=gold -flto \
// RUN:   -mabi=ilp32f -### 2>&1 | FileCheck %s --check-prefix=RISCV-SPEC-ABI-1
// RUN: %clang --target=riscv32-unknown-linux-gnu %s -fuse-ld=gold -flto \
// RUN:   -mabi=ilp32d -### 2>&1 | FileCheck %s --check-prefix=RISCV-SPEC-ABI-2
// RUN: %clang --target=riscv64-unknown-linux-gnu %s -fuse-ld=lld -flto \
// RUN:   -mabi=lp64 -### 2>&1 | FileCheck %s --check-prefix=RISCV-SPEC-ABI-3
// RUN: %clang --target=riscv64-unknown-linux-gnu %s -fuse-ld=lld -flto \
// RUN:   -mabi=lp64f -### 2>&1 | FileCheck %s --check-prefix=RISCV-SPEC-ABI-4
//
// RISCV-SPEC-ABI-1: "-plugin-opt=-target-abi=ilp32f"
// RISCV-SPEC-ABI-2: "-plugin-opt=-target-abi=ilp32d"
// RISCV-SPEC-ABI-3: "-plugin-opt=-target-abi=lp64"
// RISCV-SPEC-ABI-4: "-plugin-opt=-target-abi=lp64f"

// Need to pass -mattr option in RISC-V target.
// RUN: %clang --target=riscv32-unknown-linux-gnu %s -fuse-ld=gold -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RISCV-SPEC-MARCH-1
// RUN: %clang --target=riscv32-unknown-linux-gnu %s -fuse-ld=gold -flto \
// RUN:   -march=rv32i -### 2>&1 | FileCheck %s --check-prefix=RISCV-SPEC-MARCH-2
// RUN: %clang --target=riscv32-unknown-linux-gnu %s -fuse-ld=gold -flto \
// RUN:   -march=rv32gc -### 2>&1 | FileCheck %s --check-prefix=RISCV-SPEC-MARCH-3
//
// RISCV-SPEC-MARCH-1: "-plugin-opt=-mattr=+m,+a,+f,+d,+c,+zicsr,-e,-h,-i,-sdext,-sdtrig,-shcounterenw,-shgatpa,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smepmp,-smstateen,-ss,-ssaia,-ssccptr,-sscofpmf,-sscounterenw,-ssstateen,-sstc,-sstvala,-sstvecd,-ssu64xl,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-v,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xsfpgflushdlone,-xsfvcp,-xsfvfhbfmin,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqdotq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-za128rs,-za64rs,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zca,-zcb,-zcd,-zce,-zcf,-zcmp,-zcmt,-zdinx,-zfa,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccif,-zicclsm,-ziccrse,-zicntr,-zifencei,-zihintntl,-zihintpause,-zihpm,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-zmmul,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfh,-zvfhmin,-zvkb,-zvkb0p1,-zvkg,-zvkg0p1,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknha0p1,-zvknhb,-zvknhb0p1,-zvkns,-zvks,-zvksc,-zvksed,-zvksed0p1,-zvksg,-zvksh,-zvksh0p1,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b,-experimental-smwg,-experimental-smwgd,-experimental-ssnpm,-experimental-sswg,-experimental-zacas,-experimental-zcmop,-experimental-zfbfmin,-experimental-zicfilp,-experimental-zicfiss,-experimental-zicond,-experimental-zimop,-experimental-ztso,-experimental-zvfbfmin,-experimental-zvfbfwma,+relax"
// RISCV-SPEC-MARCH-2: "-plugin-opt=-mattr=-a,-c,-d,-e,-f,-h,-i,-m,-sdext,-sdtrig,-shcounterenw,-shgatpa,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smepmp,-smstateen,-ss,-ssaia,-ssccptr,-sscofpmf,-sscounterenw,-ssstateen,-sstc,-sstvala,-sstvecd,-ssu64xl,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-v,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xsfpgflushdlone,-xsfvcp,-xsfvfhbfmin,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqdotq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-za128rs,-za64rs,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zca,-zcb,-zcd,-zce,-zcf,-zcmp,-zcmt,-zdinx,-zfa,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicsr,-zifencei,-zihintntl,-zihintpause,-zihpm,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-zmmul,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfh,-zvfhmin,-zvkb,-zvkb0p1,-zvkg,-zvkg0p1,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknha0p1,-zvknhb,-zvknhb0p1,-zvkns,-zvks,-zvksc,-zvksed,-zvksed0p1,-zvksg,-zvksh,-zvksh0p1,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b,-experimental-smwg,-experimental-smwgd,-experimental-ssnpm,-experimental-sswg,-experimental-zacas,-experimental-zcmop,-experimental-zfbfmin,-experimental-zicfilp,-experimental-zicfiss,-experimental-zicond,-experimental-zimop,-experimental-ztso,-experimental-zvfbfmin,-experimental-zvfbfwma,+relax"
// RISCV-SPEC-MARCH-3: "-plugin-opt=-mattr=+m,+a,+f,+d,+c,+zicsr,+zifencei,-e,-h,-i,-sdext,-sdtrig,-shcounterenw,-shgatpa,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smepmp,-smstateen,-ss,-ssaia,-ssccptr,-sscofpmf,-sscounterenw,-ssstateen,-sstc,-sstvala,-sstvecd,-ssu64xl,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-v,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xsfpgflushdlone,-xsfvcp,-xsfvfhbfmin,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqdotq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-za128rs,-za64rs,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zca,-zcb,-zcd,-zce,-zcf,-zcmp,-zcmt,-zdinx,-zfa,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccif,-zicclsm,-ziccrse,-zicntr,-zihintntl,-zihintpause,-zihpm,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-zmmul,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfh,-zvfhmin,-zvkb,-zvkb0p1,-zvkg,-zvkg0p1,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknha0p1,-zvknhb,-zvknhb0p1,-zvkns,-zvks,-zvksc,-zvksed,-zvksed0p1,-zvksg,-zvksh,-zvksh0p1,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b,-experimental-smwg,-experimental-smwgd,-experimental-ssnpm,-experimental-sswg,-experimental-zacas,-experimental-zcmop,-experimental-zfbfmin,-experimental-zicfilp,-experimental-zicfiss,-experimental-zicond,-experimental-zimop,-experimental-ztso,-experimental-zvfbfmin,-experimental-zvfbfwma,+relax"

// RUN: %clang --target=x86_64-unknown-linux-gnu %s -flto \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-TARGET-ABI
// CHECK-NO-TARGET-ABI-NOT: "-plugin-opt=-target-abi

// RUN: %clang --target=riscv32-unknown-linux-gnu %s -fuse-ld=gold -flto \
// RUN:   -mllvm -misched-bottomup=false \
// RUN:   -### 2>&1 | FileCheck %s --check-prefix=RISCV-TEST-MLLVM
//
// RISCV-TEST-MLLVM: "-plugin-opt=-misched-bottomup=false"

// RUN: %clang --target=riscv32-unknown-linux-gnu %S/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib/crt0.o \
// RUN:   -fuse-ld=gold -flto -mllvm -misched-bottomup=false -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RISCV-TEST-MLLVM-UNUSED-WARNING
//
// RISCV-TEST-MLLVM-UNUSED-WARNING-NOT: warning: argument unused during compilation: '-mllvm -misched-bottomup=false'
// endif SIFIVE_CUSTOMIZATION
// -flto passes along an explicit GlobalISel flag.
// RUN: %clang --target=riscv64-linux-gnu -### %s -flto -fglobal-isel 2> %t
// RUN: FileCheck --check-prefix=CHECK-GISEL < %t %s
// RUN: %clang --target=aarch64-linux-gnu -### %s -flto -fno-global-isel 2> %t
// RUN: FileCheck --check-prefix=CHECK-DISABLE-GISEL < %t %s
//
// CHECK-GISEL:         "-plugin-opt=-global-isel=1"
// CHECK-DISABLE-GISEL: "-plugin-opt=-global-isel=0"
