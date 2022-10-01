#pragma once

#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "../../include/mat.h"
#include <stdio.h>
#include <stdlib.h>

#define EPS (1e-3)
#define CYCLES_REQUIRED 1e8
//#define REP 100
typedef float (*functionptr)(int* y_trn, int y_tst, int size, int K); //TODO Change: new function signature


// random array initialization; floats between 0 and 1
void initialize_rand_int_array_range(int* a, int n, int range) {
    for(int i = 0; i < n; i++){
        a[i] = rand() % range;
    }
}


float compute_row_error(mat* a, mat* b, int i_tst){
    float error = 0.0;
    for (int j = 0; j < a->n2; j++){ // N
        error += fabs(mat_get(a, i_tst, j) - mat_get(b, i_tst, j));
    }
    return error;
}

// check for correct result //TODO Change: update for correct validation
void validate(functionptr* userFuncs, int nfuncs, int* y_trn, int y_tst, int size, int K) {
    float error;
    float res_kernel, res_f;
    // check result of registered functions
    for (int i = 0; i < nfuncs; i++) {
        functionptr f = userFuncs[i];
        // make function 0 (first one to be registered) the base/kernel version
        if (i == 0) {
            res_kernel = f(y_trn, y_tst, size, K);
            continue;
        }
        res_f = f(y_trn, y_tst, size, K);
        error = fabs(res_f - res_kernel);
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
    long unsigned num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;

    // build and initialize matrices //TODO Change: update for needed data types
    int* a = malloc(N*sizeof(int));
    initialize_rand_int_array_range(a, N, 10);
    int k = 3;
    int y_tst = 0;


    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(a, y_tst, N, k); //TODO Change: new function signature
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
            f(a, y_tst, N, k); //TODO Change: new function signature
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;
    }
    // compute mean
    total_cycles /= reps;

    // free the malloced memory
    free(a);
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