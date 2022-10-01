IDIR=../../include
DIR1=../../src/alg1
DIR2=../../src/alg2

CC=gcc-11
CFLAGS=-I${IDIR} -I${DIR1} -I${DIR2} -W -Wall -Werror -Wno-unused-function -Wfatal-errors -mavx -msse -mfma
#CFLAGS=-I${IDIR} -I${DIR1} -I${DIR2} -mavx -msse -mfma
OPT_NOVEC=-O3 -fno-tree-vectorize
OPT_VEC=-O3 -ffast-math -march=native
OPT_SIMD=-O3 -fno-tree-vectorize -march=native
LIBS=-lm

UTILS=utils.h mat.h cnpy.h read_input.h tsc_x86.h

all: shuffle-opt

shuffle-opt: main.c
	${CC} -std=c99 -o no-flags-shuffle $< ${LIBS} ${CFLAGS}
	${CC} -std=c99 -o O3-novec-shuffle  $< ${LIBS} ${CFLAGS} ${OPT_NOVEC}
	${CC} -std=c99 -o O3-vec-shuffle  $< ${LIBS} ${CFLAGS} ${OPT_VEC}
	${CC} -std=c99 -o O3-SIMD  $< ${LIBS} ${CFLAGS} ${OPT_SIMD}

.PHONY: all clean


