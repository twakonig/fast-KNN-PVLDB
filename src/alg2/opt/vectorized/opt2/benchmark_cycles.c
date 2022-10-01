#include <stdio.h>
#include <stdlib.h>
#include "opt2.h"
#include "mat.h"
#include "tsc_x86.h"

// warmup iterations
#define NUM_WARMUP 20
// num. of iterations (measurements) per n
int NUM_RUNS;
int n;
int K;
int seed;

void benchmarkAlg2Opt2() {
    mat x_tst_knn_gt;
    mat sp_approx;
    mat x_trn;
    mat x_tst;
    myInt64 start, cycles;

    // warmup
    // malloc structs
    int* y_trn = malloc(256*sizeof(int));
    int* y_tst = malloc(256*sizeof(int));
    build(&sp_approx, 256, 256);
    build(&x_tst_knn_gt, 256, 256);
    build(&x_trn, 256, 256);
    build(&x_tst, 256, 256);

    for(int it = 0; it < NUM_WARMUP; ++it ) {
        // randomly initialize all data containers
        initialize_rand_mat(&x_tst_knn_gt);
        initialize_mat(&sp_approx, 0.0);
        initialize_rand_array(y_trn, 256);
        initialize_rand_array(y_tst, 256);
        initialize_rand_mat(&x_trn);
        initialize_rand_mat(&x_tst);

        knn_mc_approximation(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, K, 32);
    }

    destroy(&x_tst_knn_gt);
    destroy(&sp_approx);
    destroy(&x_tst);
    destroy(&x_trn);
    free(y_trn);
    free(y_tst);

    y_trn = malloc(n*sizeof(int));
    y_tst = malloc(n*sizeof(int));
    build(&sp_approx, n, n);
    build(&x_tst_knn_gt, n, n);
    build(&x_trn, n, n);
    build(&x_tst, n, n);

    //Actual benchmarking
    for(int it = 0; it < NUM_RUNS; ++it ) {
        // randomly initialize all data containers
        initialize_rand_mat(&x_tst_knn_gt);
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
    destroy(&x_tst_knn_gt);
    destroy(&sp_approx);
    destroy(&x_tst);
    destroy(&x_trn);
    free(y_trn);
    free(y_tst);
}

int main(int argc, char **argv) {

    if (argc!=5) {printf("Incorrect number of arguments, please use (n) (K) (seed) (num_runs).\n"); return -1;}

    // mode for benchmarking. key:
    n = atoi(argv[1]);
    K = atoi(argv[2]);
    seed = atoi(argv[3]);
    NUM_RUNS = atoi(argv[4]);

    srand(seed);

    benchmarkAlg2Opt2();

    return 0;
}
