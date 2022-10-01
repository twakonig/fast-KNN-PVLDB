#include <stdio.h>
#include <stdlib.h>
#include "l2norm.h"
#include "l2norm_vectorized.h"
#include "l2experiments.h"


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

    // build and initialize random arrays of floats
    float* a = malloc(N*sizeof(float));
    float* b = malloc(N*sizeof(float));
    initialize_rand_float_array(a, N);
    initialize_rand_float_array(b, N);
    // TODO: figure out aligned_alloc
    //float* c = (float*) aligned_alloc(16, N*sizeof(float));

    // array of registered functions (register in l2norm.h), nfunc function pointers
    functionptr* userFuncs = malloc(nfuncs*sizeof(functionptr));

    // decide to run either SCALAR or SIMD implementations of code
    switch (mode) {
        // SCALAR implementation
        case 0:
            register_scalar_functions(userFuncs);
            validate(userFuncs, nfuncs, a, b, N);
            free(a);
            free(b);
            // average performance of registered functions [cycles]
            time_and_print(userFuncs, nfuncs, N, reps);
            free(userFuncs);
            break;
        // SIMD implementation
        case 1:
            register_simd_functions(userFuncs);
            validate(userFuncs, nfuncs, a, b, N);
            free(a);
            free(b);
            // average performance of registered functions [cycles]
            time_and_print(userFuncs, nfuncs, N, reps);
            free(userFuncs);
            break;
        /*// l2experiments implementation
        case 2:
            register_test_functions(userFuncs);
            validate(userFuncs, nfuncs, a, b, N);
            free(a);
            free(b);
            // average performance of registered functions [cycles]
            time_and_print(userFuncs, nfuncs, N, reps);
            free(userFuncs);
            break;*/
        default:
            printf("Mode does not exist. Enter 0 (SCALAR), 1 (SIMD) or 2 (EXPERIMENT) for the corresponding implementation to run.");
            free(a);
            free(b);
            free(userFuncs);
            return -1;
    }
    return 0;
}