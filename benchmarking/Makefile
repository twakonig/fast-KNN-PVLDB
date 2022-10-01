IDIR=../include
DIR1=../src/alg1
DIR2=../src/alg2

CC=gcc
CFLAGS=-I${IDIR} -I${DIR1} -I${DIR2} -W -Wall -Wno-unused-function
OPT_NOVEC=-O3 -fno-tree-vectorize
OPT_VEC=-O3 -ffast-math -march=native -mavx2
LIBS=-lm

UTILS=utils.h mat.h cnpy.h read_input.h tsc_x86.h papi_wrapper.h

all: benchmark_cycles benchmark_cycles_alg2_diff_K

benchmark_cycles: benchmark_cycles.c
	${CC} -std=c99 -o no-flags-cycles $< ${LIBS} ${CFLAGS}
	${CC} -std=c99 -o O3-novec-flags-cycles  $< ${LIBS} ${CFLAGS} ${OPT_NOVEC}
	${CC} -std=c99 -o O3-vec-flags-cycles  $< ${LIBS} ${CFLAGS} ${OPT_VEC}

benchmark_cycles_alg2_diff_K: benchmark_cycles_alg2_diff_K.c
	${CC} -std=c99 -o O3-vec-flags-cycles-alg2-diff-K $< ${LIBS} ${CFLAGS} ${OPT_VEC}

benchmark_flops: benchmark_flops.c
	${CC} -std=c99 -o no-flags-flops $< ${LIBS} ${CFLAGS}

.PHONY: all clean

clean:
	-rm no-flags* O3-*
