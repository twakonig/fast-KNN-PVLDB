//
// Created by lucasck on 18/05/22.
//

#include <stdio.h>
#include <stdlib.h>
#include "knn_opt.h"
#include "alg1_knn.h"


void register_scalar_functions(functionptr_knn* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
// <----------------- naive optimizations -------------->
//    userFuncs[0].as_baseline_func = &get_true_knn_base;
//    userFuncs[1].as_baseline_func = &get_true_knn_opt1;
//    userFuncs[2].as_baseline_func = &get_true_knn_opt2;
//    userFuncs[3].as_baseline_func = &get_true_knn_opt3;
//    userFuncs[4].as_baseline_func = &get_true_knn_opt4;
//    userFuncs[5].as_baseline_func = &get_true_knn_opt5;
//    userFuncs[6].as_baseline_func = &get_true_knn_opt6;
//    userFuncs[7].as_opt_func = &get_true_knn_opt7;

// <----------------- scalar blocking ------------------>
//    userFuncs[0].as_baseline_func = &get_true_knn_opt8;
//    userFuncs[1].as_baseline_func = &get_true_knn_opt9;
//    userFuncs[2].as_baseline_func = &get_true_knn_opt10;
//    userFuncs[3].as_baseline_func = &get_true_knn_opt11;
//    userFuncs[4].as_baseline_func = &get_true_knn_opt12;
//    userFuncs[5].as_baseline_func = &get_true_knn_opt13;

// <----------------- vectorized blocking ------------------>
// current vectorized best is opt4 (on master), here also opt4, so we aim to beat that as baseline.
    userFuncs[0].as_baseline_func = &get_true_knn_opt4;
    userFuncs[1].as_baseline_func = &get_true_knn_opt14;
    userFuncs[2].as_baseline_func = &get_true_knn_opt15;
    userFuncs[3].as_baseline_func = &get_true_knn_opt16;
    userFuncs[4].as_baseline_func = &get_true_knn_opt17;
    userFuncs[5].as_baseline_func = &get_true_knn_opt18;
    userFuncs[6].as_baseline_func = &get_true_knn_opt19;
    userFuncs[7].as_baseline_func = &get_true_knn_opt20;
    userFuncs[8].as_baseline_func = &get_true_knn_opt21;
    userFuncs[9].as_baseline_func = &get_true_knn_opt22;
}

int main(int argc, char **argv) {
    // read in command line arguments
    if (argc!=5) {printf("Incorrect number of arguments, please use (N) (seed) (nfuncs) (mode). N = input size/dimension. "
                         "nfuncs = number of function implementations you want to register for testing. seed = seed for generating random numbers. mode = 0 if not testing any implementation that takes a knn_mat as one of its arguments, 1 if the last function in userFuncs does \n"); return -1;}
    int N = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int nfuncs= atoi(argv[3]);
    int mode = atoi(argv[4]);
    srand(seed);

    // array of registered functions (register in alg1_sort.h), nfunc function pointers
    functionptr_knn* userFuncs = malloc(nfuncs*sizeof(functionptr_knn));
    register_scalar_functions(userFuncs);
    validate(userFuncs, nfuncs, N, mode);
    time_and_print_knn(userFuncs, nfuncs, N, mode);
    free(userFuncs);
    return 0;
}
