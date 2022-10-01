//
// Created by jokics on 22/05/22.
//

#pragma once

#include "utils.h"
#include "tsc_x86.h"
#include "mat.h"

#define EPS (1e-3)
#define CYCLES_REQUIRED 1e5
#define REP 10

typedef void (*functionptr)(mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T);

// check for correct result
void validate(functionptr* userFuncs, int nfuncs, int N, int K, int seed) {
    mat x_trn, x_tst, sp_approx_baseline, sp_approx;
    // malloc structs
    int* y_trn = malloc(N*sizeof(int));
    int* y_tst = malloc(N*sizeof(int));
    build(&x_trn, N, N);
    build(&x_tst, N, N);
    build(&sp_approx_baseline, N, N);
    build(&sp_approx, N, N);


    // randomly initialize all data containers
    initialize_rand_mat(&x_trn);
    initialize_rand_mat(&x_tst);
    initialize_mat(&sp_approx_baseline, 0.0);
    initialize_mat(&sp_approx, 0.0);
    initialize_rand_array(y_trn, N);
    initialize_rand_array(y_tst, N);

    functionptr f;
    f = userFuncs[0];
    srand(seed);
    f(&sp_approx_baseline, &x_trn, y_trn, &x_tst, y_tst, K, 130);

    for(int i = 1; i < nfuncs; i++){
        f = userFuncs[i];
        srand(seed);
        f(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, K, 130);

        for(int j = 0; j < sp_approx.n1; j++){
            for(int k = 0; k < sp_approx.n2; k++){
                if(mat_get(&sp_approx_baseline, j, k) != mat_get(&sp_approx, j, k)){
                    printf("ERROR: function %d is incorrect.\n", i);
                }
            }
        }
    }
    return;
}

/*
 * perf_test adapted from homework 2, ASL 2022, How to Write Fast Numerical Code 263-2300 - ETH Zurich
 * returns #cycles required on average (mean) per iteration of the function
 */
double perf_test(functionptr f, int N, int K) {
    double cycles;
    long unsigned num_runs = 10;
    double multiplier = 1;
    myInt64 start, end;

    mat x_trn, x_tst, sp_approx;

    //malloc structs
    int* y_trn = malloc(N*sizeof(int));
    int* y_tst = malloc(N*sizeof(int));
    build(&x_trn, N, N);
    build(&x_tst, N, N);
    build(&sp_approx, N, N);


    // randomly initialize all data containers
    initialize_rand_mat(&x_trn);
    initialize_rand_mat(&x_tst);
    initialize_mat(&sp_approx, 0.0);
    initialize_rand_array(y_trn, N);
    initialize_rand_array(y_tst, N);

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, K, 130);
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);

    } while (multiplier > 2);

    // Actual performance measurements repeated REP times.
    // We simply store all results and compute medians during post-processing.
    double total_cycles = 0;
    for (size_t j = 0; j < REP; j++) {
        start = start_tsc();
        for (size_t i = 0; i < num_runs; ++i) {
            f(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, K, 130);
        }
        end = stop_tsc(start);
        cycles = ((double)end) / num_runs;
        total_cycles += cycles;
    }
    // compute mean
    total_cycles /= REP;

    // free the malloced memory
    destroy(&x_trn);
    destroy(&x_tst);
    destroy(&sp_approx);
    free(y_trn);
    free(y_tst);
    cycles = total_cycles;
    return cycles;
}

void time_and_print(functionptr* userFuncs, int nfuncs, int N, int K) {
    // average performance in cycles
    double performance, performance_base;
    double speedup;
    // measure RUNTIME [cycles] of registered functions
    for (int i = 0; i < nfuncs; i++) {
        performance = perf_test(userFuncs[i], N, K);
        // function registered as 0 serves as base
        if (i == 0) {
            performance_base = performance;
            printf("%d \t\t %f", N, performance_base);
            continue;
        }
        speedup = performance_base / performance;
        printf(" \t\t %.4fx", speedup);
        // manual inspection
        //printf("N = %d. Running implementation %d. Cycles: %lf. Speed-up: %lfx\n", N, i, performance, speedup);
    }
}
