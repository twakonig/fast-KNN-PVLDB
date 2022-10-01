#include <stdio.h>
#include <stdlib.h>

#include "../include/mat.h"
#include "../src/alg1/alg1.h"
#include "../src/alg2/alg2.h"
#include "../include/tsc_x86.h"


// max matrix size (n x n)
// warmup iterations
#define NUM_WARMUP 50
// num. of iterations (measurements) per n
int NUM_RUNS;
int K;

void warmup(int N) {
    int_mat x_tst_knn_gt;
    mat sp_gt;

    // warm up CPU first
    // build matrices
    int* y_trn = malloc(N*sizeof(int));
    int* y_tst = malloc(N*sizeof(int));
    build(&sp_gt, N, N);
    build_int_mat(&x_tst_knn_gt, N, N);

    for(int it = 0; it < NUM_WARMUP; ++it ) {
        // randomly initialize all data containers
        initialize_rand_int_mat(&x_tst_knn_gt, N);
        initialize_rand_mat(&sp_gt);
        initialize_rand_array(y_trn, N);
        initialize_rand_array(y_tst, N);

        // run function shapley on input
        compute_single_unweighted_knn_class_shapley(&sp_gt, y_trn, y_tst, &x_tst_knn_gt, K);
    }

    // free all
    destroy_int_mat(&x_tst_knn_gt);
    destroy(&sp_gt);
    free(y_trn);
    free(y_tst);
}

void benchmarkAlg1(int n) {
    int_mat x_tst_knn_gt;
    mat sp_gt;
    mat x_trn;
    mat x_tst;
    myInt64 start, cycles;

    //warmup
    warmup(256);

    //malloc structs
    int* y_trn = malloc(n*sizeof(int));
    int* y_tst = malloc(n*sizeof(int));
    build(&x_trn, n, n);
    build(&x_tst, n, n);
    build(&sp_gt, n, n);
    build_int_mat(&x_tst_knn_gt, n, n);

    // Actual benchmarking, run NUM_RUNS times.
    for(int it = 0; it < NUM_RUNS; ++it ) {
        // randomly initialize all data containers
        initialize_rand_mat(&x_trn);
        initialize_rand_mat(&x_tst);
        initialize_mat(&sp_gt, 0.0);
        initialize_rand_array(y_trn, n);
        initialize_rand_array(y_tst, n);

        // START, read TSC
        start = start_tsc();
        get_true_knn(&x_tst_knn_gt, &x_trn, &x_tst);
        compute_single_unweighted_knn_class_shapley(&sp_gt, y_trn, y_tst, &x_tst_knn_gt, K);
        // END, read TSC
        cycles = stop_tsc(start);

        // // print statement for "manual inspection"
        // printf("Alg1 with KNN, n=%d, it=%d, cycles=%lld\n", n, it, cycles);

        // print statement for .csv file, used for plotting with matplotlib in next step
        printf("%d, %llu\n", n, cycles);
    }
    // free all
    destroy(&x_tst);
    destroy(&x_trn);
    destroy_int_mat(&x_tst_knn_gt);
    destroy(&sp_gt);
    free(y_trn);
    free(y_tst);
}

// adjust T as needed
void benchmarkAlg2(int n) {
    mat sp_approx;
    mat x_trn;
    mat x_tst;
    myInt64 start, cycles;

    //warmup
    warmup(256);

    // malloc structs
    int* y_trn = malloc(n*sizeof(int));
    int* y_tst = malloc(n*sizeof(int));
    build(&sp_approx, n, n);
    build(&x_trn, n, n);
    build(&x_tst, n, n);

    //Actual benchmarking
    for(int it = 0; it < NUM_RUNS; ++it ) {
        // randomly initialize all data containers
        initialize_mat(&sp_approx, 0.0);
        initialize_rand_array(y_trn, n);
        initialize_rand_array(y_tst, n);
        initialize_rand_mat(&x_trn);
        initialize_rand_mat(&x_tst);

        // START, read TSC
        start = start_tsc();
        knn_mc_approximation(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, K, 32);
        // END, read TSC
        cycles = stop_tsc(start);

        // // print statement for "manual inspection"
        //printf("Alg2, n=%d, it=%d, cycles=%lld\n", n, it, cycles);

        // print statement for .csv file, used for plotting with matplotlib in next step
        printf("%d, %llu \n", n, cycles);

    }
    // free all
    destroy(&sp_approx);
    destroy(&x_tst);
    destroy(&x_trn);
    free(y_trn);
    free(y_tst);
}

int main(int argc, char **argv) {

    if (argc!=6) {printf("Incorrect number of arguments, please use (mode) (n) (K) (seed) (num_runs).\n Modes: 1 (alg1) or 2 (alg2).\n"); return -1;}

    // mode for benchmarking. key:
    int mode = atoi(argv[1]);
    int n = atoi(argv[2]);
    K = atoi(argv[3]);
    int seed = atoi(argv[4]);
    NUM_RUNS = atoi(argv[5]);

    srand(seed);

    switch (mode) {
        case 0:
            printf("Mode does not exist. Enter 1 (alg1) or 2 (alg2).");
            return -1;
        case 1:
            benchmarkAlg1(n);
            break;
        case 2:
            benchmarkAlg2(n);
            break;
        default:
            printf("Mode does not exist. Enter 1 (alg1) or 2 (alg2).");
            return -1;
    }

    return 0;
}
