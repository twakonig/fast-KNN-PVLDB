IDIR=../include
DIR1=../src/alg1
DIR2=../src/alg2

CC=gcc
CFLAGS=-I${IDIR} -I${DIR1} -I${DIR2} -W -Wall -Werror -Wno-unused-function -Wfatal-errors -mavx -mavx2 -msse -mfma -DNDEBUG
#CFLAGS=-I${IDIR} -I${DIR1} -I${DIR2} -mavx -msse -mfma
OPT_NOVEC=-O3 -fno-tree-vectorize
OPT_VEC=-O3 -march=native
OPT_SIMD=-O3 -fno-tree-vectorize -march=native
LIBS=-lm

UTILS=utils.h mat.h cnpy.h read_input.h tsc_x86.h

all: l2norm_performance

l2norm_performance: main.c
	${CC} -std=c99 -o no-flags-l2 $< ${LIBS} ${CFLAGS}
	${CC} -std=c99 -o O3-novec-l2  $< ${LIBS} ${CFLAGS} ${OPT_NOVEC}
	${CC} -std=c99 -o O3-vec-l2  $< ${LIBS} ${CFLAGS} ${OPT_VEC}
	${CC} -std=c99 -o O3-SIMD  $< ${LIBS} ${CFLAGS} ${OPT_SIMD}

.PHONY: all clean


