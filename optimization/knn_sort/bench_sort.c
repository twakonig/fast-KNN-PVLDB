//
// Created by lucasck on 18/05/22.
// Modified by jokics on 19/05/22.
//

#include <stdio.h>
#include <stdlib.h>
#include "sort_opt.h"
#include "alg1_sort.h"

void register_scalar_functions(functionptr_argsort* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0].as_baseline_func = &argsort; // baseline
    userFuncs[1].as_opt_func = &sort_qsort;
    userFuncs[2].as_opt_func = &sort_klib_mergesort;
    userFuncs[3].as_opt_func = &sort_quadsort;
}


int main(int argc, char **argv) {
    // read in command line arguments
    if (argc!=4) {printf("Incorrect number of arguments, please use (N) (seed) (nfuncs). N = input size/dimension. "
                         "nfuncs = number of function implementations you want to register for testing. \n"); return -1;}
    int N = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int nfuncs= atoi(argv[3]);

    srand(seed);

    // array of registered functions (register in alg1_sort.h), nfunc function pointers
    functionptr_argsort* userFuncs = malloc(nfuncs*sizeof(functionptr_argsort));
    register_scalar_functions(userFuncs);
    validate(userFuncs, nfuncs, N);
    time_and_print_sort(userFuncs, nfuncs, N);
    free(userFuncs);
    return 0;
}
