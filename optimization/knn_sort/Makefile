IDIR=../../include

_UTILS=utils.h mat.h tsc_x86.h ksort.h quadsort.h
SORT=bench_sort.o
KNN=bench_knn.o

# compiler flags
CC=gcc
CFLAGS=-I${IDIR} -w -mfma -lm# -fopt-info-vec-all
ENABLE_VEC=-mavx2
AVX512=-mavx512f -mavx512cd -mavx512bw -mavx512vl -mavx512dq
SANITIZE=-fsanitize=address -g

OPT_FAST=-O3 -fno-ipa-cp-clone
OPT_FAST_NATIVE=${OPT_FAST} -march=native
OPT_FAST_NATIVE_NOVEC=${OPT_FAST_NATIVE} -fno-tree-vectorize

OPT_MED=-O2
OPT_MED_NATIVE=${OPT_MED} -march=native
OPT_MED_NATIVE_NOVEC=${OPT_FAST_NATIVE} -fno-tree-vectorize

# some templates
DEBUG=${CFLAGS} ${SANITIZE}
VEC=${CFLAGS} ${ENABLE_VEC} ${OPT_FAST_NATIVE} 

# fix paths
UTILS = $(patsubst %,$(IDIR)/%,$(_UTILS))

# make rules
all: knn_performance.out sort_performance.out

sort_performance.out: bench_sort.c ${UTILS} sort_opt.h alg1_sort.h
	${CC} -std=c99 -o no-flags-sort $< ${CFLAGS} ${ENABLE_VEC}
	${CC} -std=c99 -o O3-sort $< ${CFLAGS} ${OPT_FAST} ${ENABLE_VEC}
	${CC} -std=c99 -o O3-sort-native $< ${CFLAGS} ${OPT_FAST_NATIVE} ${ENABLE_VEC}
	${CC} -std=c99 -o O3-sort-native-novec $< ${CFLAGS}  ${OPT_FAST_NATIVE_NOVEC} ${ENABLE_VEC}
	${CC} -std=c99 -o O2-sort $< ${CFLAGS} ${OPT_MED} ${ENABLE_VEC}
	${CC} -std=c99 -o O2-sort-native $< ${CFLAGS} ${OPT_MED_NATIVE} ${ENABLE_VEC}
	${CC} -std=c99 -o O2-sort-native-novec $< ${CFLAGS} ${OPT_MED_NATIVE_NOVEC} ${ENABLE_VEC}
#	#${CC} -std=c99 -o O3-SIMD $< ${CFLAGS} ${OPT_SIMD}
#
knn_performance.out: bench_knn.c ${UTILS} knn_opt.h alg1_knn.h
	${CC} -std=c99 -o no-flags-knn $< ${CFLAGS}  ${ENABLE_VEC}
	${CC} -std=c99 -o O3-knn $< ${CFLAGS} ${OPT_FAST} ${ENABLE_VEC}
	${CC} -std=c99 -o O3-knn-ffast $< ${CFLAGS} ${OPT_FAST} ${AVX512} -ffast-math
	${CC} -std=c99 -o O3-knn-unrolld $< ${CFLAGS} ${OPT_FAST} ${AVX512} -funroll-loops ${SANITIZE}
	${CC} -std=c99 -o O3-knn-ffast-unrolld $< ${CFLAGS} ${OPT_FAST} ${AVX512} -ffast-math -funroll-loops
	${CC} -std=c99 -o O3-knn-native $< ${CFLAGS} ${OPT_FAST_NATIVE}  ${ENABLE_VEC}
	${CC} -std=c99 -o O3-knn-ffast-native $< ${CFLAGS} ${OPT_FAST_NATIVE} ${AVX512} -ffast-math
	${CC} -std=c99 -o O3-knn-unrolld-native $< ${CFLAGS} ${OPT_FAST_NATIVE} ${AVX512} -funroll-loops
	${CC} -std=c99 -o O3-knn-ffast-unrolld-native $< ${CFLAGS} ${OPT_FAST_NATIVE} ${AVX512} -ffast-math -funroll-loops

.PHONY: all clean

clean:
	-rm no-flags-* O3-* O2-*