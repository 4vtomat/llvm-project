; RUN: opt -riscv-disable-all-loop-idiom=false -passes=riscv-loop-idiom -mtriple=riscv64-unknown-linux-gnu -mattr=+v -S < %s | FileCheck %s
; RUN: opt -riscv-disable-all-loop-idiom=false -passes=riscv-loop-idiom -mtriple=riscv64-unknown-linux-gnu -riscv-loop-idiom-customize-lmul=3 -mattr=+v -S < %s | FileCheck %s --check-prefix=LMUL8
; RUN: opt -riscv-disable-all-loop-idiom=false -passes='loop(riscv-loop-idiom),simplifycfg' -mtriple=riscv64-unknown-linux-gnu -mattr=+v -S < %s | FileCheck %s --check-prefix=LOOP-DEL

define i32 @compare_bytes_simple(ptr %a, ptr %b, i32 %len, i32 %n) {
; CHECK-LABEL: define i32 @compare_bytes_simple
; CHECK-SAME: (ptr [[A:%.*]], ptr [[B:%.*]], i32 [[LEN:%.*]], i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[LEN]], 1
; CHECK-NEXT:    br label [[MISMATCH_MIN_IT_CHECK:%.*]]
; CHECK:       mismatch_min_it_check:
; CHECK-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; CHECK-NEXT:    [[TC:%.*]] = zext i32 [[N]] to i64
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ule i32 [[TMP0]], [[N]]
; CHECK-NEXT:    br i1 [[TMP3]], label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0:![0-9]+]]
; CHECK:       mismatch_mem_check:
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP4]] to i64
; CHECK-NEXT:    [[TMP12:%.*]] = lshr i64 [[TMP7]], 12
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[A]], i64 [[TC]]
; CHECK-NEXT:    [[TMP10:%.*]] = ptrtoint ptr [[TMP8]] to i64
; CHECK-NEXT:    [[TMP13:%.*]] = lshr i64 [[TMP10]], 12
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
; CHECK-NEXT:    [[TMP14:%.*]] = lshr i64 [[TMP6]], 12
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[B]], i64 [[TC]]
; CHECK-NEXT:    [[TMP11:%.*]] = ptrtoint ptr [[TMP9]] to i64
; CHECK-NEXT:    [[TMP15:%.*]] = lshr i64 [[TMP11]], 12
; CHECK-NEXT:    [[TMP16:%.*]] = icmp ne i64 [[TMP12]], [[TMP13]]
; CHECK-NEXT:    [[TMP17:%.*]] = icmp ne i64 [[TMP14]], [[TMP15]]
; CHECK-NEXT:    [[TMP18:%.*]] = or i1 [[TMP16]], [[TMP17]]
; CHECK-NEXT:    br i1 [[TMP18]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_VECTOR_LOOP_PREHEADER:%.*]], !prof [[PROF1:![0-9]+]]
; CHECK:       mismatch_vector_loop_preheader:
; CHECK-NEXT:    br label [[MISMATCH_VECTOR_LOOP:%.*]]
; CHECK:       mismatch_vector_loop:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ [[TMP1]], [[MISMATCH_VECTOR_LOOP_PREHEADER]] ], [ [[INDEX_NEXT:%.*]], [[MISMATCH_VECTOR_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[AVL:%.*]] = sub nuw nsw i64 [[TC]], [[INDEX]]
; CHECK-NEXT:    [[RVL:%.*]] = call i32 @llvm.experimental.get.vector.length.i64(i64 %avl, i32 16, i1 true)
; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[INDEX]]
; CHECK-NEXT:    [[VP_LOAD1:%.*]] = call <vscale x 16 x i8> @llvm.vp.load.nxv16i8.p0(ptr [[TMP22]], <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i64 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), i32 [[RVL]])
; CHECK-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[INDEX]]
; CHECK-NEXT:    [[VP_LOAD2:%.*]] = call <vscale x 16 x i8> @llvm.vp.load.nxv16i8.p0(ptr [[TMP24]], <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i64 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), i32 [[RVL]])
; CHECK-NEXT:    [[VP_ICMP:%.*]] = call <vscale x 16 x i1> @llvm.vp.icmp.nxv16i8(<vscale x 16 x i8> [[VP_LOAD1]], <vscale x 16 x i8> [[VP_LOAD2]], metadata !"ne", <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i64 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), i32 [[RVL]])
; CHECK-NEXT:    [[FIRST:%.*]] = call i32 @llvm.vp.first.nxv16i1(<vscale x 16 x i1> [[VP_ICMP]], <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i64 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), i32 [[RVL]])
; CHECK-NEXT:    [[MISMATCH_FOUND:%.*]] = icmp ne i32 [[FIRST]], -1
; CHECK-NEXT:    br i1 [[MISMATCH_FOUND]], label [[MISMATCH_VECTOR_LOOP_FOUND:%.*]], label [[MISMATCH_VECTOR_LOOP_INC]]
; CHECK:       mismatch_vector_loop_inc:
; CHECK-NEXT:    [[RVL64:%.*]] = zext i32 [[RVL]] to i64
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw nsw i64 [[INDEX]], [[RVL64]]
; CHECK-NEXT:    [[EXIT_COND:%.*]] = icmp ne i64 [[INDEX_NEXT]], [[TC]]
; CHECK-NEXT:    br i1 [[EXIT_COND]], label [[MISMATCH_VECTOR_LOOP]], label [[MISMATCH_END:%.*]]
; CHECK:       mismatch_vector_loop_found:
; CHECK-NEXT:    [[FIRST_LCSSA:%.*]] = phi i32 [ [[FIRST]], [[MISMATCH_VECTOR_LOOP]] ]
; CHECK-NEXT:    [[INDEX_LCSSA:%.*]] = phi i64 [ [[INDEX]], [[MISMATCH_VECTOR_LOOP]] ]
; CHECK-NEXT:    [[TMP34:%.*]] = zext i32 [[FIRST_LCSSA]] to i64
; CHECK-NEXT:    [[TMP35:%.*]] = add nuw nsw i64 [[INDEX_LCSSA]], [[TMP34]]
; CHECK-NEXT:    [[TMP36:%.*]] = trunc i64 [[TMP35]] to i32
; CHECK-NEXT:    br label [[MISMATCH_END]]
; CHECK:       mismatch_loop_pre:
; CHECK-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ [[TMP0]], [[MISMATCH_MEM_CHECK]] ], [ [[TMP0]], [[MISMATCH_MIN_IT_CHECK]] ]
; CHECK-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; CHECK:       mismatch_loop:
; CHECK-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP43:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[TMP37:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; CHECK-NEXT:    [[TMP38:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[TMP37]]
; CHECK-NEXT:    [[TMP39:%.*]] = load i8, ptr [[TMP38]], align 1
; CHECK-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[TMP37]]
; CHECK-NEXT:    [[TMP41:%.*]] = load i8, ptr [[TMP40]], align 1
; CHECK-NEXT:    [[TMP42:%.*]] = icmp eq i8 [[TMP39]], [[TMP41]]
; CHECK-NEXT:    br i1 [[TMP42]], label [[MISMATCH_LOOP_INC]], label [[MISMATCH_END]]
; CHECK:       mismatch_loop_inc:
; CHECK-NEXT:    [[TMP43]] = add i32 [[MISMATCH_INDEX]], 1
; CHECK-NEXT:    [[TMP44:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], [[N]]
; CHECK-NEXT:    br i1 [[TMP44]], label [[MISMATCH_END]], label [[MISMATCH_LOOP]]
; CHECK:       mismatch_end:
; CHECK-NEXT:    [[MISMATCH_RESULT:%.*]] = phi i32 [ [[N]], [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ [[N]], [[MISMATCH_VECTOR_LOOP_INC]] ], [ [[TMP36]], [[MISMATCH_VECTOR_LOOP_FOUND]] ]
; CHECK-NEXT:    br i1 true, label [[BYTE_COMPARE:%.*]], label [[WHILE_COND:%.*]]
; CHECK:       while.cond:
; CHECK-NEXT:    [[LEN_ADDR:%.*]] = phi i32 [ [[LEN]], [[MISMATCH_END]] ], [ [[MISMATCH_RESULT]], [[WHILE_BODY:%.*]] ]
; CHECK-NEXT:    [[INC:%.*]] = add i32 [[MISMATCH_RESULT]], 1
; CHECK-NEXT:    [[CMP_NOT:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; CHECK-NEXT:    br i1 [[CMP_NOT]], label [[WHILE_END:%.*]], label [[WHILE_BODY]]
; CHECK:       while.body:
; CHECK-NEXT:    [[IDXPROM:%.*]] = zext i32 [[MISMATCH_RESULT]] to i64
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP45:%.*]] = load i8, ptr [[ARRAYIDX]], align 1
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP46:%.*]] = load i8, ptr [[ARRAYIDX2]], align 1
; CHECK-NEXT:    [[CMP_NOT2:%.*]] = icmp eq i8 [[TMP45]], [[TMP46]]
; CHECK-NEXT:    br i1 [[CMP_NOT2]], label [[WHILE_COND]], label [[WHILE_END]]
; CHECK:       byte.compare:
; CHECK-NEXT:    [[TMP47:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP47]], label [[WHILE_END]], label [[WHILE_END]]
; CHECK:       while.end:
; CHECK-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[MISMATCH_RESULT]], [[WHILE_BODY]] ], [ [[MISMATCH_RESULT]], [[WHILE_COND]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ]
; CHECK-NEXT:    ret i32 [[INC_LCSSA]]
;
; LMUL8-LABEL: define i32 @compare_bytes_simple
; LMUL8-SAME: (ptr [[A:%.*]], ptr [[B:%.*]], i32 [[LEN:%.*]], i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; LMUL8-NEXT:  entry:
; LMUL8-NEXT:    [[TMP0:%.*]] = add i32 [[LEN]], 1
; LMUL8-NEXT:    br label [[MISMATCH_MIN_IT_CHECK:%.*]]
; LMUL8:       mismatch_min_it_check:
; LMUL8-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; LMUL8-NEXT:    [[TC:%.*]] = zext i32 [[N]] to i64
; LMUL8-NEXT:    [[TMP3:%.*]] = icmp ule i32 [[TMP0]], [[N]]
; LMUL8-NEXT:    br i1 [[TMP3]], label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0:![0-9]+]]
; LMUL8:       mismatch_mem_check:
; LMUL8-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP1]]
; LMUL8-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP4]] to i64
; LMUL8-NEXT:    [[TMP12:%.*]] = lshr i64 [[TMP7]], 12
; LMUL8-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[A]], i64 [[TC]]
; LMUL8-NEXT:    [[TMP10:%.*]] = ptrtoint ptr [[TMP8]] to i64
; LMUL8-NEXT:    [[TMP13:%.*]] = lshr i64 [[TMP10]], 12
; LMUL8-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP1]]
; LMUL8-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
; LMUL8-NEXT:    [[TMP14:%.*]] = lshr i64 [[TMP6]], 12
; LMUL8-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[B]], i64 [[TC]]
; LMUL8-NEXT:    [[TMP11:%.*]] = ptrtoint ptr [[TMP9]] to i64
; LMUL8-NEXT:    [[TMP15:%.*]] = lshr i64 [[TMP11]], 12
; LMUL8-NEXT:    [[TMP16:%.*]] = icmp ne i64 [[TMP12]], [[TMP13]]
; LMUL8-NEXT:    [[TMP17:%.*]] = icmp ne i64 [[TMP14]], [[TMP15]]
; LMUL8-NEXT:    [[TMP18:%.*]] = or i1 [[TMP16]], [[TMP17]]
; LMUL8-NEXT:    br i1 [[TMP18]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_VECTOR_LOOP_PREHEADER:%.*]], !prof [[PROF1:![0-9]+]]
; LMUL8:       mismatch_vector_loop_preheader:
; LMUL8-NEXT:    br label [[MISMATCH_VECTOR_LOOP:%.*]]
; LMUL8:       mismatch_vector_loop:
; LMUL8-NEXT:    [[INDEX:%.*]] = phi i64 [ [[TMP1]], [[MISMATCH_VECTOR_LOOP_PREHEADER]] ], [ [[INDEX_NEXT:%.*]], [[MISMATCH_VECTOR_LOOP_INC:%.*]] ]
; LMUL8-NEXT:    [[AVL:%.*]] = sub nuw nsw i64 [[TC]], [[INDEX]]
; LMUL8-NEXT:    [[RVL:%.*]] = call i32 @llvm.experimental.get.vector.length.i64(i64 [[AVL]], i32 64, i1 true)
; LMUL8-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[INDEX]]
; LMUL8-NEXT:    [[VP_LOAD1:%.*]] = call <vscale x 64 x i8> @llvm.vp.load.nxv64i8.p0(ptr [[TMP22]], <vscale x 64 x i1> shufflevector (<vscale x 64 x i1> insertelement (<vscale x 64 x i1> poison, i1 true, i64 0), <vscale x 64 x i1> poison, <vscale x 64 x i32> zeroinitializer), i32 [[RVL]])
; LMUL8-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[INDEX]]
; LMUL8-NEXT:    [[VP_LOAD2:%.*]] = call <vscale x 64 x i8> @llvm.vp.load.nxv64i8.p0(ptr [[TMP24]], <vscale x 64 x i1> shufflevector (<vscale x 64 x i1> insertelement (<vscale x 64 x i1> poison, i1 true, i64 0), <vscale x 64 x i1> poison, <vscale x 64 x i32> zeroinitializer), i32 [[RVL]])
; LMUL8-NEXT:    [[VP_ICMP:%.*]] = call <vscale x 64 x i1> @llvm.vp.icmp.nxv64i8(<vscale x 64 x i8> [[VP_LOAD1]], <vscale x 64 x i8> [[VP_LOAD2]], metadata !"ne", <vscale x 64 x i1> shufflevector (<vscale x 64 x i1> insertelement (<vscale x 64 x i1> poison, i1 true, i64 0), <vscale x 64 x i1> poison, <vscale x 64 x i32> zeroinitializer), i32 [[RVL]])
; LMUL8-NEXT:    [[FIRST:%.*]] = call i32 @llvm.vp.first.nxv64i1(<vscale x 64 x i1> [[VP_ICMP]], <vscale x 64 x i1> shufflevector (<vscale x 64 x i1> insertelement (<vscale x 64 x i1> poison, i1 true, i64 0), <vscale x 64 x i1> poison, <vscale x 64 x i32> zeroinitializer), i32 [[RVL]])
; LMUL8-NEXT:    [[MISMATCH_FOUND:%.*]] = icmp ne i32 [[FIRST]], -1
; LMUL8-NEXT:    br i1 [[MISMATCH_FOUND]], label [[MISMATCH_VECTOR_LOOP_FOUND:%.*]], label [[MISMATCH_VECTOR_LOOP_INC]]
; LMUL8:       mismatch_vector_loop_inc:
; LMUL8-NEXT:    [[RVL64:%.*]] = zext i32 [[RVL]] to i64 
; LMUL8-NEXT:    [[INDEX_NEXT]] = add nuw nsw i64 [[INDEX]], [[RVL64]]
; LMUL8-NEXT:    [[EXIT_COND:%.*]] = icmp ne i64 [[INDEX_NEXT]], [[TC]]
; LMUL8-NEXT:    br i1 [[EXIT_COND]], label [[MISMATCH_VECTOR_LOOP]], label [[MISMATCH_END:%.*]]
; LMUL8:       mismatch_vector_loop_found:
; LMUL8-NEXT:    [[FIRST_LCSSA:%.*]] = phi i32 [ [[FIRST]], [[MISMATCH_VECTOR_LOOP]] ]
; LMUL8-NEXT:    [[INDEX_LCSSA:%.*]] = phi i64 [ [[INDEX]], [[MISMATCH_VECTOR_LOOP]] ]
; LMUL8-NEXT:    [[TMP34:%.*]] = zext i32 [[FIRST_LCSSA]] to i64
; LMUL8-NEXT:    [[TMP35:%.*]] = add nuw nsw i64 [[INDEX_LCSSA]], [[TMP34]]
; LMUL8-NEXT:    [[TMP36:%.*]] = trunc i64 [[TMP35]] to i32
; LMUL8-NEXT:    br label [[MISMATCH_END]]
; LMUL8:       mismatch_loop_pre:
; LMUL8-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ [[TMP0]], [[MISMATCH_MEM_CHECK]] ], [ [[TMP0]], [[MISMATCH_MIN_IT_CHECK]] ]
; LMUL8-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; LMUL8:       mismatch_loop:
; LMUL8-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP43:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; LMUL8-NEXT:    [[TMP37:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; LMUL8-NEXT:    [[TMP38:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[TMP37]]
; LMUL8-NEXT:    [[TMP39:%.*]] = load i8, ptr [[TMP38]], align 1
; LMUL8-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[TMP37]]
; LMUL8-NEXT:    [[TMP41:%.*]] = load i8, ptr [[TMP40]], align 1
; LMUL8-NEXT:    [[TMP42:%.*]] = icmp eq i8 [[TMP39]], [[TMP41]]
; LMUL8-NEXT:    br i1 [[TMP42]], label [[MISMATCH_LOOP_INC]], label [[MISMATCH_END]]
; LMUL8:       mismatch_loop_inc:
; LMUL8-NEXT:    [[TMP43]] = add i32 [[MISMATCH_INDEX]], 1
; LMUL8-NEXT:    [[TMP44:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], [[N]]
; LMUL8-NEXT:    br i1 [[TMP44]], label [[MISMATCH_END]], label [[MISMATCH_LOOP]]
; LMUL8:       mismatch_end:
; LMUL8-NEXT:    [[MISMATCH_RESULT:%.*]] = phi i32 [ [[N]], [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ [[N]], [[MISMATCH_VECTOR_LOOP_INC]] ], [ [[TMP36]], [[MISMATCH_VECTOR_LOOP_FOUND]] ]
; LMUL8-NEXT:    br i1 true, label [[BYTE_COMPARE:%.*]], label [[WHILE_COND:%.*]]
; LMUL8:       while.cond:
; LMUL8-NEXT:    [[LEN_ADDR:%.*]] = phi i32 [ [[LEN]], [[MISMATCH_END]] ], [ [[MISMATCH_RESULT]], [[WHILE_BODY:%.*]] ]
; LMUL8-NEXT:    [[INC:%.*]] = add i32 [[MISMATCH_RESULT]], 1
; LMUL8-NEXT:    [[CMP_NOT:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; LMUL8-NEXT:    br i1 [[CMP_NOT]], label [[WHILE_END:%.*]], label [[WHILE_BODY]]
; LMUL8:       while.body:
; LMUL8-NEXT:    [[IDXPROM:%.*]] = zext i32 [[MISMATCH_RESULT]] to i64
; LMUL8-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[IDXPROM]]
; LMUL8-NEXT:    [[TMP45:%.*]] = load i8, ptr [[ARRAYIDX]], align 1
; LMUL8-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[IDXPROM]]
; LMUL8-NEXT:    [[TMP46:%.*]] = load i8, ptr [[ARRAYIDX2]], align 1
; LMUL8-NEXT:    [[CMP_NOT2:%.*]] = icmp eq i8 [[TMP45]], [[TMP46]]
; LMUL8-NEXT:    br i1 [[CMP_NOT2]], label [[WHILE_COND]], label [[WHILE_END]]
; LMUL8:       byte.compare:
; LMUL8-NEXT:    [[TMP47:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; LMUL8-NEXT:    br i1 [[TMP47]], label [[WHILE_END]], label [[WHILE_END]]
; LMUL8:       while.end:
; LMUL8-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[MISMATCH_RESULT]], [[WHILE_BODY]] ], [ [[MISMATCH_RESULT]], [[WHILE_COND]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ]
; LMUL8-NEXT:    ret i32 [[INC_LCSSA]]
;
; LOOP-DEL-LABEL: define i32 @compare_bytes_simple
; LOOP-DEL-SAME: (ptr [[A:%.*]], ptr [[B:%.*]], i32 [[LEN:%.*]], i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; LOOP-DEL-NEXT:  entry:
; LOOP-DEL-NEXT:    [[TMP0:%.*]] = add i32 [[LEN]], 1
; LOOP-DEL-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; LOOP-DEL-NEXT:    [[TC:%.*]] = zext i32 [[N]] to i64
; LOOP-DEL-NEXT:    [[TMP3:%.*]] = icmp ule i32 [[TMP0]], [[N]]
; LOOP-DEL-NEXT:    br i1 [[TMP3]], label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0:![0-9]+]]
; LOOP-DEL:       mismatch_mem_check:
; LOOP-DEL-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP1]]
; LOOP-DEL-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP4]] to i64
; LOOP-DEL-NEXT:    [[TMP12:%.*]] = lshr i64 [[TMP7]], 12
; LOOP-DEL-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[A]], i64 [[TC]]
; LOOP-DEL-NEXT:    [[TMP10:%.*]] = ptrtoint ptr [[TMP8]] to i64
; LOOP-DEL-NEXT:    [[TMP13:%.*]] = lshr i64 [[TMP10]], 12
; LOOP-DEL-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP1]]
; LOOP-DEL-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
; LOOP-DEL-NEXT:    [[TMP14:%.*]] = lshr i64 [[TMP6]], 12
; LOOP-DEL-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[B]], i64 [[TC]]
; LOOP-DEL-NEXT:    [[TMP11:%.*]] = ptrtoint ptr [[TMP9]] to i64
; LOOP-DEL-NEXT:    [[TMP15:%.*]] = lshr i64 [[TMP11]], 12
; LOOP-DEL-NEXT:    [[TMP16:%.*]] = icmp ne i64 [[TMP12]], [[TMP13]]
; LOOP-DEL-NEXT:    [[TMP17:%.*]] = icmp ne i64 [[TMP14]], [[TMP15]]
; LOOP-DEL-NEXT:    [[TMP18:%.*]] = or i1 [[TMP16]], [[TMP17]]
; LOOP-DEL-NEXT:    br i1 [[TMP18]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_VECTOR_LOOP:%.*]], !prof [[PROF1:![0-9]+]]
; LOOP-DEL:       mismatch_vector_loop:
; LOOP-DEL-NEXT:    [[INDEX:%.*]] = phi i64 [ [[INDEX_NEXT:%.*]], [[MISMATCH_VECTOR_LOOP_INC:%.*]] ], [ [[TMP1]], [[MISMATCH_MEM_CHECK]] ]
; LOOP-DEL-NEXT:    [[AVL:%.*]] = sub nuw nsw i64 [[TC]], [[INDEX]]
; LOOP-DEL-NEXT:    [[RVL:%.*]] = call i32 @llvm.experimental.get.vector.length.i64(i64 [[AVL]], i32 16, i1 true)
; LOOP-DEL-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[INDEX]]
; LOOP-DEL-NEXT:    [[VP_LOAD1:%.*]] = call <vscale x 16 x i8> @llvm.vp.load.nxv16i8.p0(ptr [[TMP22]], <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i64 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), i32 [[RVL]])
; LOOP-DEL-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[INDEX]]
; LOOP-DEL-NEXT:    [[VP_LOAD2:%.*]] = call <vscale x 16 x i8> @llvm.vp.load.nxv16i8.p0(ptr [[TMP24]], <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i64 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), i32 [[RVL]])
; LOOP-DEL-NEXT:    [[VP_ICMP:%.*]] = call <vscale x 16 x i1> @llvm.vp.icmp.nxv16i8(<vscale x 16 x i8> [[VP_LOAD1]], <vscale x 16 x i8> [[VP_LOAD2]], metadata !"ne", <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i64 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), i32 [[RVL]])
; LOOP-DEL-NEXT:    [[FIRST:%.*]] = call i32 @llvm.vp.first.nxv16i1(<vscale x 16 x i1> [[VP_ICMP]], <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i64 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), i32 [[RVL]])
; LOOP-DEL-NEXT:    [[MISMATCH_FOUND:%.*]] = icmp ne i32 [[FIRST]], -1
; LOOP-DEL-NEXT:    br i1 [[MISMATCH_FOUND]], label [[MISMATCH_VECTOR_LOOP_FOUND:%.*]], label [[MISMATCH_VECTOR_LOOP_INC]]
; LOOP-DEL:       mismatch_vector_loop_inc:
; LOOP-DEL-NEXT:    [[RVL64:%.*]] = zext i32 [[RVL]] to i64
; LOOP-DEL-NEXT:    [[INDEX_NEXT]] = add nuw nsw i64 [[INDEX]], [[RVL64]]
; LOOP-DEL-NEXT:    [[EXIT_COND:%.*]] = icmp ne i64 [[INDEX_NEXT]], [[TC]]
; LOOP-DEL-NEXT:    br i1 [[EXIT_COND]], label [[MISMATCH_VECTOR_LOOP]], label [[MISMATCH_END:%.*]]
; LOOP-DEL:       mismatch_vector_loop_found:
; LOOP-DEL-NEXT:    [[FIRST_LCSSA:%.*]] = phi i32 [ [[FIRST]], [[MISMATCH_VECTOR_LOOP]] ]
; LOOP-DEL-NEXT:    [[INDEX_LCSSA:%.*]] = phi i64 [ [[INDEX]], [[MISMATCH_VECTOR_LOOP]] ]
; LOOP-DEL-NEXT:    [[TMP34:%.*]] = zext i32 [[FIRST_LCSSA]] to i64
; LOOP-DEL-NEXT:    [[TMP35:%.*]] = add nuw nsw i64 [[INDEX_LCSSA]], [[TMP34]]
; LOOP-DEL-NEXT:    [[TMP36:%.*]] = trunc i64 [[TMP35]] to i32
; LOOP-DEL-NEXT:    br label [[MISMATCH_END]]
; LOOP-DEL:       mismatch_loop_pre:
; LOOP-DEL-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ [[TMP0]], [[MISMATCH_MEM_CHECK]] ], [ [[TMP0]], %entry ]
; LOOP-DEL-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; LOOP-DEL:       mismatch_loop:
; LOOP-DEL-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP43:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; LOOP-DEL-NEXT:    [[TMP37:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; LOOP-DEL-NEXT:    [[TMP38:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[TMP37]]
; LOOP-DEL-NEXT:    [[TMP39:%.*]] = load i8, ptr [[TMP38]], align 1
; LOOP-DEL-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[TMP37]]
; LOOP-DEL-NEXT:    [[TMP41:%.*]] = load i8, ptr [[TMP40]], align 1
; LOOP-DEL-NEXT:    [[TMP42:%.*]] = icmp eq i8 [[TMP39]], [[TMP41]]
; LOOP-DEL-NEXT:    br i1 [[TMP42]], label [[MISMATCH_LOOP_INC]], label [[MISMATCH_END]]
; LOOP-DEL:       mismatch_loop_inc:
; LOOP-DEL-NEXT:    [[TMP43]] = add i32 [[MISMATCH_INDEX]], 1
; LOOP-DEL-NEXT:    [[TMP44:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], [[N]]
; LOOP-DEL-NEXT:    br i1 [[TMP44]], label [[MISMATCH_END]], label [[MISMATCH_LOOP]]
; LOOP-DEL:       while.end:
; LOOP-DEL-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[N]], [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ [[N]], [[MISMATCH_VECTOR_LOOP_INC]] ], [ [[TMP36]], [[MISMATCH_VECTOR_LOOP_FOUND]] ]
; LOOP-DEL-NEXT:    ret i32 [[INC_LCSSA]]
;
entry:
  br label %while.cond

while.cond:
  %len.addr = phi i32 [ %len, %entry ], [ %inc, %while.body ]
  %inc = add i32 %len.addr, 1
  %cmp.not = icmp eq i32 %inc, %n
  br i1 %cmp.not, label %while.end, label %while.body

while.body:
  %idxprom = zext i32 %inc to i64
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %idxprom
  %0 = load i8, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %idxprom
  %1 = load i8, ptr %arrayidx2
  %cmp.not2 = icmp eq i8 %0, %1
  br i1 %cmp.not2, label %while.cond, label %while.end

while.end:
  %inc.lcssa = phi i32 [ %inc, %while.body ], [ %inc, %while.cond ]
  ret i32 %inc.lcssa
}