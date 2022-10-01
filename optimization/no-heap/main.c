#include <stdio.h>
#include <stdlib.h>
#include "alg2_no_heap.h"
#include "alg2_no_heap_vectorized.h"
//#include "l2experiments.h"


int main(int argc, char **argv) {
    // read in command line arguments
    if (argc!=6) {printf("Incorrect number of arguments, please use (N) (seed) (nfuncs) (mode) (reps). N = input size/dimension. "
                         "nfuncs = number of function implementations you want to register for testing. mode = SCALAR (0) or SIMD (1) implementation.\n"); return -1;}
    int N = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int nfuncs= atoi(argv[3]);
    // TODO: change to string, cases: SCALAR(0), SIMD(1)
    int mode = atoi(argv[4]);
    int reps = atoi(argv[5]);
    srand(seed);

    // build and initialize random arrays of floats //TODO Change: Update to new data types needed.
    mat kernel_approx;
    mat sp_approx;
    mat x_trn;
    mat x_tst;
    int* y_trn = malloc(N*sizeof(int));
    int* y_tst = malloc(N*sizeof(int));

    build(&sp_approx, N, N);
    build(&kernel_approx, N, N);
    build(&x_trn, N, N);
    build(&x_tst, N, N);

    initialize_mat(&kernel_approx, 0.0);
    initialize_mat(&sp_approx, 0.0);
    initialize_rand_array(y_trn, N);
    initialize_rand_array(y_tst, N);
    initialize_rand_mat(&x_trn);
    initialize_rand_mat(&x_tst);
    // TODO: figure out aligned_alloc
    //float* c = (float*) aligned_alloc(16, N*sizeof(float));

    // array of registered functions (register in l2norm.h), nfunc function pointers
    functionptr* userFuncs = malloc(nfuncs*sizeof(functionptr));

    // decide to run either SCALAR or SIMD implementations of code
    switch (mode) {
        // SCALAR implementation
        case 0:
            register_scalar_functions(userFuncs);
            validate(userFuncs, nfuncs, &kernel_approx,&sp_approx, &x_trn, y_trn, &x_tst, y_tst, 1, 130); //TODO Change: new function signature
            destroy(&sp_approx);
            destroy(&kernel_approx);
            destroy(&x_tst);
            destroy(&x_trn);
            free(y_trn);
            free(y_tst);
            // average performance of registered functions [cycles]
            time_and_print(userFuncs, nfuncs, N, reps);
            free(userFuncs);
            break;
        // SIMD implementation
        case 1:
            register_simd_functions(userFuncs);
            validate(userFuncs, nfuncs, &kernel_approx,&sp_approx, &x_trn, y_trn, &x_tst, y_tst, 1, 130); //TODO Change: new function signature
            destroy(&sp_approx);
            destroy(&kernel_approx);
            destroy(&x_tst);
            destroy(&x_trn);
            free(y_trn);
            free(y_tst);
            // average performance of registered functions [cycles]
            time_and_print(userFuncs, nfuncs, N, reps);
            free(userFuncs);
            break;
        default:
            printf("Mode does not exist. Enter 0 (SCALAR), 1 (SIMD) or 2 (EXPERIMENT) for the corresponding implementation to run.");
            destroy(&sp_approx);
            destroy(&kernel_approx);
            destroy(&x_tst);
            destroy(&x_trn);
            free(y_trn);
            free(y_tst);
            free(userFuncs);
            return -1;
    }
    return 0;
}