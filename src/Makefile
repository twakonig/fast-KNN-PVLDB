# source directories for headers and source files
DIR1=./alg1
DIR2=./alg2
IDIR=../include

# binaries and everything else that needs to be included
_UTILS=utils.h mat.h cnpy.h read_input.h tsc_x86.h ksort.h quadsort.h
_OBJ=testing_infra.o
BOTTLENECKS=bottlenecks.o
BOTTLENECKS_OPT_1=bottlenecks_l2_sort.o
BOTTLENECKS_OPT_2=bottlenecks_l2_sort_knn.o
MAIN=main.o

# compiler flags
CC=gcc
CFLAGS=-I${IDIR} -W -Wall -Wno-unused-function -Wfatal-errors -lm -Wno-implicit-fallthrough -Wno-sign-compare -fno-ipa-cp-clone
ENABLE_VEC=-mfma -mavx512f -mavx512cd -mavx512bw -mavx512vl -mavx512dq
SANITIZE=-fsanitize=address -g

OPT_FAST=-O3
OPT_FAST_NATIVE=${OPT_FAST} -march=native
OPT_FAST_NATIVE_NOVEC=${OPT_FAST_NATIVE} -fno-tree-vectorize

OPT_MED=-O2
OPT_MED_NATIVE=${OPT_MED} -march=native
OPT_MED_NATIVE_NOVEC=${OPT_FAST_NATIVE} -fno-tree-vectorize

# some templates
DEBUG=${CFLAGS} ${SANITIZE}
VEC=${CFLAGS} ${ENABLE_VEC} ${OPT_FAST_NATIVE}

# fix paths
OBJ1 = $(patsubst %,$(DIR1)/%,$(_OBJ))
OBJ2 = $(patsubst %,$(DIR2)/%,$(_OBJ))
MAIN1 = $(patsubst %,$(DIR1)/%,$(MAIN))
MAIN2 = $(patsubst %,$(DIR2)/%,$(MAIN))
BOTTLENECKS1 = $(patsubst %,$(DIR1)/%,$(BOTTLENECKS))
BOTTLENECKS1_OPT_1 = $(patsubst %,$(DIR1)/%,$(BOTTLENECKS_OPT_1))
BOTTLENECKS1_OPT_2 = $(patsubst %,$(DIR1)/%,$(BOTTLENECKS_OPT_2))
BOTTLENECKS2 = $(patsubst %,$(DIR2)/%,$(BOTTLENECKS))
UTILS = $(patsubst %,$(IDIR)/%,$(_UTILS))

# make rules
all: alg1.out alg2.out main1.out main2.out bottlenecks1_sc.out bottlenecks1_vec.out bottlenecks2_sc.out bottlenecks2_vec.out

alg1.out: ${DIR1}/testing_infra.c ${UTILS}
	${CC} -o $@ $< ${CFLAGS} ${ENABLE_VEC} ${OPT_FAST_NATIVE};

alg2.out: ${DIR2}/testing_infra.c ${UTILS}
	${CC} -o $@ $< ${CFLAGS} ${ENABLE_VEC} ${OPT_FAST_NATIVE};

main1.out: ${DIR1}/main.c ${UTILS}
	${CC} -o $@ $< ${CFLAGS} ${ENABLE_VEC} ${OPT_FAST_NATIVE};

main2.out: ${DIR2}/main.c ${UTILS}
	${CC} -o $@ $< ${CFLAGS} ${ENABLE_VEC} ${OPT_FAST_NATIVE};

bottlenecks1_vec.out: ${DIR1}/bottlenecks.c ${UTILS}
	${CC} -o $@ $< ${CFLAGS} ${ENABLE_VEC} ${OPT_FAST_NATIVE};

bottlenecks1_sc.out: ${DIR1}/bottlenecks.c ${UTILS}
	${CC} -o $@ $< ${CFLAGS} ${OPT_FAST_NATIVE_NOVEC};

bottlenecks2_vec.out: ${DIR2}/bottlenecks.c ${UTILS}
	${CC} -o $@ $< ${CFLAGS} ${ENABLE_VEC} ${OPT_FAST_NATIVE} -ffast-math;

bottlenecks2_sc.out: ${DIR2}/bottlenecks.c ${UTILS}
	${CC} -o $@ $< ${CFLAGS} ${OPT_FAST_NATIVE_NOVEC} -ffast-math;



.PHONY: all clean

clean:
	-rm *.out
