#include <stdio.h>
#include <stdlib.h>
#include "compute_shapley.h"

int main(int argc, char **argv) {
    // read in command line arguments
    if (argc!=6) {printf("Incorrect number of arguments, please use (N) (seed) (nfuncs) (mode) (reps). N = input size/dimension. "
                         "nfuncs = number of function implementations you want to register for testing. mode = SCALAR (0) or SIMD (1) implementation.\n"); return -1;}
    int N = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int nfuncs= atoi(argv[3]);
    int mode = atoi(argv[4]);
    int reps = atoi(argv[5]);
    srand(seed);
    int K = 1;

    int* y_trn = malloc(N*sizeof(int));
    int* y_tst = malloc(N*sizeof(int));
    int_mat x_tst_knn_gt;
    build_int_mat(&x_tst_knn_gt, N, N);

    initialize_rand_array(y_trn, N);
    initialize_rand_array(y_tst, N);
    initialize_rand_int_mat(&x_tst_knn_gt, N);


    // array of registered functions (register in l2norm.h), nfunc function pointers
    svfctptr* userFuncs = malloc(nfuncs*sizeof(svfctptr));

    // decide to run either SCALAR or SIMD implementations of code
    switch (mode) {
        // SCALAR implementation
        case 0:
            register_comp_shapley(userFuncs);
            validate_shapley(userFuncs, nfuncs, y_trn, y_tst, &x_tst_knn_gt, N, K);
            free(y_trn);
            free(y_tst);
            destroy_int_mat(&x_tst_knn_gt);
            // average performance of registered functions [cycles]
            time_and_print(userFuncs, nfuncs, N, reps, K);
            free(userFuncs);
            break;
        /*// SIMD implementation
        case 1:
            register_simd_functions(userFuncs);
            validate(userFuncs, nfuncs, a, b, N);
            free(a);
            free(b);
            // average performance of registered functions [cycles]
            time_and_print(userFuncs, nfuncs, N, reps);
            free(userFuncs);
            break;*/
        default:
            printf("Mode does not exist. Enter 0 (SCALAR), 1 (SIMD) or 2 (EXPERIMENT) for the corresponding implementation to run.");
            free(y_trn);
            free(y_tst);
            destroy_int_mat(&x_tst_knn_gt);
            free(userFuncs);
            return -1;
    }
    return 0;
}
