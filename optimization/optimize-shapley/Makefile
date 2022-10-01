IDIR=../include
#DIR1=../src/alg1
#DIR2=../src/alg2
# -I${DIR1} -I${DIR2}

CC=gcc
CFLAGS=-I${IDIR} -W -Wall -Werror -Wno-unused-function -Wfatal-errors -mavx2 -mfma
#CFLAGS=-I${IDIR} -I${DIR1} -I${DIR2} -mavx -msse -mfma -fopt-info-all
OPT_O2=-O2 -march=native
OPT_VEC=-O3 -march=native
OPT_SIMD=-O3 -fno-tree-vectorize -march=native
LIBS=-lm

UTILS=utils.h mat.h cnpy.h read_input.h tsc_x86.h

all: shapley_performance

shapley_performance: main.c
	${CC} -std=c99 -o no-flags $< ${LIBS} ${CFLAGS}
#${CC} -std=c99 -o O2-vec $< ${LIBS} ${CFLAGS} ${OPT_O2}
	${CC} -std=c99 -o O3-vec  $< ${LIBS} ${CFLAGS} ${OPT_VEC}
#${CC} -std=c99 -o O3-SIMD  $< ${LIBS} ${CFLAGS} ${OPT_SIMD}

.PHONY: all clean
