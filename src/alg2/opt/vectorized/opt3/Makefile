IDIR=../../../../../include

CC=gcc
CFLAGS=-I${IDIR} -W -Wall -Wno-unused-function -Wfatal-errors -mavx -msse -mfma
OPT_NOVEC=-O3 -fno-tree-vectorize
OPT_VEC=-O3 -ffast-math -march=native -mavx2
LIBS=-lm

all: benchmark_cycles testing_infra bottlenecks

benchmark_cycles: benchmark_cycles.c
	${CC} -std=c99 -o no-flags-cycles $< ${LIBS} ${CFLAGS}
	${CC} -std=c99 -o O3-novec-flags-cycles  $< ${LIBS} ${CFLAGS} ${OPT_NOVEC}
	${CC} -std=c99 -o O3-vec-flags-cycles  $< ${LIBS} ${CFLAGS} ${OPT_VEC}


testing_infra: testing_infra.c
	${CC} -std=c99 -o alg2opt3.out $< ${LIBS} ${CFLAGS} ${OPT_VEC}

bottlenecks: bottlenecks.c
	${CC} -std=c99 -o no-flags-bottlenecks $< ${LIBS} ${CFLAGS}
	${CC} -std=c99 -o O3-novec-flags-bottlenecks  $< ${LIBS} ${CFLAGS} ${OPT_NOVEC}
	${CC} -std=c99 -o O3-vec-flags-bottlenecks  $< ${LIBS} ${CFLAGS} ${OPT_VEC}

.PHONY: all clean

clean:
	-rm no-flags* O3-* *.out
