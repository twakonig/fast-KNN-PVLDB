//
// Created by jokics on 22/05/22.
//

#ifndef TEAM36_ALG2_HEAP_H
#define TEAM36_ALG2_HEAP_H

#endif

#include <stdio.h>
#include <stdlib.h>
#include "tsc_x86.h"
#include "benchmark_heap_opt.h"
#include "alg2_heap_opt.h"
#include "../../src/alg2/alg2.h"

void register_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &knn_mc_approximation;
    userFuncs[1] = &knn_mc_approximation_heap_opt;
}

int main(int argc, char **argv) {
    // read in command line arguments
    if (argc!=5) {printf("Incorrect number of arguments, please use (N) (K) (seed) (nfuncs). N = input size/dimension. K = number of nearest neighbours."
                         "nfuncs = number of function implementations you want to register for testing. \n"); return -1;}
    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    int seed = atoi(argv[3]);
    int nfuncs = atoi(argv[4]);

    srand(seed);

    // array of registered functions, nfuncs function pointers
    functionptr* userFuncs = malloc(nfuncs*sizeof(functionptr));
    register_functions(userFuncs);
    validate(userFuncs, nfuncs, N, K, seed);
    time_and_print(userFuncs, nfuncs, N, K);
    free(userFuncs);
    return 0;
}
