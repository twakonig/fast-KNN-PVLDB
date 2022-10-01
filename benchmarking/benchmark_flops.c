#include <stdio.h>
#include <stdlib.h>
#include "../include/mat.h"
#include "alg1_flops.h"
#include "alg2_flops.h"

// num. of iterations (measurements) per n
#define NUM_RUNS 30


// max matrix size (n x n)
// idea: let N vary from 2^4 to 2^14 (would take hours though)
// for large N: alg2 killed
//#define N_MAX pow(2, 8)


void benchmarkAlg1(int nmax) {
    int_mat x_tst_knn_gt;
    mat sp_gt;
    mat x_trn;
    mat x_tst;

    long long unsigned int flops;
    long long unsigned int true_knn_flops;

    // first line of .csv file
    //printf("n, flops\n");

    // Actual benchmarking, run NUM_RUNS times.
    for(int n = 16; n <= nmax; n*=2 ) {

        //malloc structs
        int* y_trn = malloc(n*sizeof(int));
        int* y_tst = malloc(n*sizeof(int));
        build(&x_trn, n, n);
        build(&x_tst, n, n);
        build(&sp_gt, n, n);
        build_int_mat(&x_tst_knn_gt, n, n);


        // randomly initialize all data containers
        initialize_rand_mat(&x_trn);
        initialize_rand_mat(&x_tst);
        initialize_mat(&sp_gt, 0.0);
        initialize_rand_array(y_trn, n);
        initialize_rand_array(y_tst, n);

        flops = 0;
        true_knn_flops = 0;

        get_true_knn(&x_tst_knn_gt, &x_trn, &x_tst, &true_knn_flops);

        compute_single_unweighted_knn_class_shapley(&sp_gt, y_trn, y_tst, &x_tst_knn_gt, 1, &flops);

        // // print statement for "manual inspection"
        //printf("Alg1 without knn, n=%d, it=%d, cycles=%lld\n", n, it, cycles);

        // print statement for .csv file, used for plotting with matplotlib in next step
        printf("%d, %llu\n", n, flops);

        // free all
        destroy(&x_tst);
        destroy(&x_trn);
        destroy_int_mat(&x_tst_knn_gt);
        destroy(&sp_gt);
        free(y_trn);
        free(y_tst);
    }
}

void benchmarkAlg1WithKNN(int nmax) {
    int_mat x_tst_knn_gt;
    mat sp_gt;
    mat x_trn;
    mat x_tst;

    long long unsigned int flops = 0;

    // first line of .csv file
    //printf("n, flops\n");

    // Actual benchmarking, run NUM_RUNS times.
    for(int n = 16; n <= nmax; n*=2 ) {

        //malloc structs
        int* y_trn = malloc(n*sizeof(int));
        int* y_tst = malloc(n*sizeof(int));
        build(&x_trn, n, n);
        build(&x_tst, n, n);
        build(&sp_gt, n, n);
        build_int_mat(&x_tst_knn_gt, n, n);

        // randomly initialize all data containers
        initialize_rand_mat(&x_trn);
        initialize_rand_mat(&x_tst);
        initialize_mat(&sp_gt, 0.0);
        initialize_rand_array(y_trn, n);
        initialize_rand_array(y_tst, n);

        flops = 0;

        get_true_knn(&x_tst_knn_gt, &x_trn, &x_tst, &flops);

        compute_single_unweighted_knn_class_shapley(&sp_gt, y_trn, y_tst, &x_tst_knn_gt, 1, &flops);

        // // print statement for "manual inspection"
        //printf("Alg1 without knn, n=%d, it=%d, cycles=%lld\n", n, it, cycles);

        // print statement for .csv file, used for plotting with matplotlib in next step
        printf("%d, %llu\n", n, flops);

        // free all
        destroy(&x_tst);
        destroy(&x_trn);
        destroy_int_mat(&x_tst_knn_gt);
        destroy(&sp_gt);
        free(y_trn);
        free(y_tst);
    }
}

void benchmarkAlg2(int n) {
    mat sp_approx;
    mat x_trn;
    mat x_tst;

    long long unsigned int flops;

    // first line of .csv file
    //printf("n, flops\n");


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

        flops = 0;

        knn_mc_approximation(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, 1, 130, &flops);

        // // print statement for "manual inspection"
        //printf("Alg2, n=%d, it=%d, cycles=%lld\n", n, it, cycles);

        // print statement for .csv file, used for plotting with matplotlib in next step
        printf("%d, %llu \n", n, flops);

    }
    // free all
    destroy(&sp_approx);
    destroy(&x_tst);
    destroy(&x_trn);
    free(y_trn);
    free(y_tst);
}



int main(int argc, char **argv) {

    if (argc!=4) {printf("Incorrect number of arguments, please use (mode) (n) (seed).\n Modes: 0 (alg1), 1 (alg1 with KNN) or 2 (alg2).\n"); return -1;}

    // mode for benchmarking. key:
    int mode = atoi(argv[1]);
    int n = atoi(argv[2]);
    int seed = atoi(argv[3]);

    srand(seed);

    switch (mode) {
        case 0:
            benchmarkAlg1(n);
            break;
        case 1:
            benchmarkAlg1WithKNN(n);
            break;
        case 2:
            benchmarkAlg2(n);
            break;
        default:
            printf("Mode does not exist. Enter 0 (alg1), 1 (alg1 with KNN) or 2 (alg2).");
            return -1;
    }

    return 0;
}
