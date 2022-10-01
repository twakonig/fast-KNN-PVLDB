IDIR=../../../../../include

CC=gcc
CFLAGS=-I${IDIR} -W -Wall -Wno-unused-function -Wfatal-errors -mfma -fno-tree-vectorize
OPT=-O3
OPT_FFM=-O3 -ffast-math -march=native
LIBS=-lm

all: benchmark_cycles testing_infra bottlenecks

benchmark_cycles: benchmark_cycles.c
	${CC} -std=c99 -o no-flags-cycles $< ${LIBS} ${CFLAGS}
	${CC} -std=c99 -o O3-flags-cycles  $< ${LIBS} ${CFLAGS} ${OPT}
	${CC} -std=c99 -o O3-ffm-flags-cycles  $< ${LIBS} ${CFLAGS} ${OPT_FFM}

testing_infra: testing_infra.c
	${CC} -std=c99 -o alg2opt1.out $< ${LIBS} ${CFLAGS} ${OPT_FFM}

bottlenecks: bottlenecks.c
	${CC} -std=c99 -o no-flags-bottlenecks $< ${LIBS} ${CFLAGS}
	${CC} -std=c99 -o O3-flags-bottlenecks  $< ${LIBS} ${CFLAGS} ${OPT}
	${CC} -std=c99 -o O3-ffm-flags-bottlenecks  $< ${LIBS} ${CFLAGS} ${OPT_FFM}

.PHONY: all clean

clean:
	-rm no-flags* O3-* *.out
