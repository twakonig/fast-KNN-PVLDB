IDIR=../../include
DIR1=../../src/alg1
DIR2=../../src/alg2

CC=gcc
CFLAGS=-I${IDIR} -W -Wall -Wno-unused-function -Wfatal-errors
CFLAGS_NOERRORS= -I${IDIR}
SPEED=-ffast-math -mavx -mfma
LIBS=-lm
SANITIZE=-fsanitize=address -g
OPT_MED=-O2 -DUSE_RESTRICT
OPT_FAST=-O3 -march=native -DUSE_RESTRICT

UTILS=utils.h mat.h cnpy.h read_input.h tsc_x86.h papi_wrapper.h

all: alg2_heap_opt performance_benchmark

alg2_heap_opt: testing_infra.c
	${CC} -o alg2_heap_opt.out $< ${LIBS} ${CFLAGS}

performance_benchmark: benchmark_heap_opt.c
	${CC} -o benchmark-no-flags-heap-opt $< ${LIBS} ${CFLAGS}
	${CC} -o benchmark-O2-heap-opt  $< ${LIBS} ${CFLAGS} ${OPT_MED} ${SPEED}
	${CC} -o benchmark-O3-heap-opt $< ${LIBS} ${CFLAGS} ${OPT_FAST} ${SPEED}


.PHONY: all clean

clean:
	-rm *.out benchmark-no-flags-* benchmark-O2-* benchmark-O3-*
