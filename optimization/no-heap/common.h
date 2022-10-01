#pragma once

#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "../../include/mat.h"
#include <stdio.h>
#include <stdlib.h>

#define EPS (1e-3)
#define CYCLES_REQUIRED 1e5
//#define REP 100
typedef void (*functionptr)(mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T); //TODO Change: new function signature

float compute_mat_error(mat* a, mat* b){
    float error = 0.0;
    for (int i = 0; i < a->n1; i++){
        for (int j = 0; j < a->n2; j++){ // N
            error += fabs(mat_get(a, i, j) - mat_get(b, i, j));
        }
    }
    return error;
}

// check for correct result //TODO Change: update for correct validation
void validate(functionptr* userFuncs, int nfuncs, mat* kernel_approx, mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T) {
    float error;
    // check result of registered functions
    for (int i = 0; i < nfuncs; i++) {
        functionptr f = userFuncs[i];
        // make function 0 (first one to be registered) the base/kernel version
        if (i == 0) {
            f(kernel_approx, x_trn, y_trn, x_tst, y_tst, K, T);
            continue;
        }
        f(sp_approx, x_trn, y_trn, x_tst, y_tst, K, T);
        error = compute_mat_error(kernel_approx, sp_approx);
        if (error > EPS) {
            printf("ERROR, the result of function %d is wrong. |error| = %lf\n", i, error);
        }
    }
}

/*
 * perf_test adapted from homework 2, ASL 2022, How to Write Fast Numerical Code 263-2300 - ETH Zurich
 * returns #cycles required on average (mean) per iteration of the function
 */
double perf_test(functionptr f, int N, int reps) {
    double cycles = 0.;
    long unsigned num_runs = 10;
    double multiplier = 1;
    myInt64 start, end;

    // build and initialize matrices //TODO Change: update for needed data types
    mat sp_approx;
    mat x_trn;
    mat x_tst;
    int* y_trn = malloc(N*sizeof(int));
    int* y_tst = malloc(N*sizeof(int));

    build(&sp_approx, N, N);
    build(&x_trn, N, N);
    build(&x_tst, N, N);

    initialize_mat(&sp_approx, 0.0);
    initialize_rand_array(y_trn, N);
    initialize_rand_array(y_tst, N);
    initialize_rand_mat(&x_trn);
    initialize_rand_mat(&x_tst);



    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, 1, 130); //TODO Change: new function signature
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);

    } while (multiplier > 2);


    // Actual performance measurements repeated REP times.
    // We simply store all results and compute medians during post-processing.
    double total_cycles = 0;
    for (int j = 0; j < reps; j++) {

        start = start_tsc();
        for (size_t i = 0; i < num_runs; ++i) {
            f(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, 1, 130); //TODO Change: new function signature
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;
    }
    // compute mean
    total_cycles /= reps;

    // free the malloced memory
    destroy(&sp_approx);
    destroy(&x_tst);
    destroy(&x_trn);
    free(y_trn);
    free(y_tst);
    cycles = total_cycles;
    return  cycles;
}

void time_and_print(functionptr* userFuncs, int nfuncs, int N, int reps) {
    // average performance in cycles
    double performance, performance_base;
    double speedup;
    // measure RUNTIME [cycles] of registered functions
    for (int i = 0; i < nfuncs; i++) {
        performance = perf_test(userFuncs[i], N, reps);
        // function registered as 0 serves as base
        if (i == 0) {
            performance_base = performance;
            printf("%d \t %f", N, performance_base);
            continue;
        }
        speedup = performance_base / performance;
        printf(" \t %.4fx", speedup);
        // manual inspection
        //printf("N = %d. Running implementation %d. Cycles: %lf. Speed-up: %lfx\n", N, i, performance, speedup);
    }
    printf("\n");
}