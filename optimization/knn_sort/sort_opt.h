//
// Created by lucasck on 18/05/22.
// Modified by jokics on 19/05/22.
//

#pragma once

#include "../../include/tsc_x86.h"
#include "../../include/mat.h"

#define EPS (1e-3)
#define CYCLES_REQUIRED 1e8
#define REP 100

typedef float (*functionptr_argsort_baseline)(int idx_arr[], float dist_gt[], size_t len);
typedef float (*functionptr_argsort_opt)(pair_t dist_gt[], size_t len);
typedef union {
    functionptr_argsort_baseline as_baseline_func;
    functionptr_argsort_opt as_opt_func;
} functionptr_argsort;


// check for correct result
void validate(functionptr_argsort* userFuncs, int nfuncs, int N) {

    // build and initialize random arrays of floats and
    // corresponding indices for baseline argsort implementation
    int* idx_arr = malloc(N*sizeof(int));
    float* dist_gt = malloc(N*sizeof(float));

    initialize_incr_int_array(idx_arr, N);
    initialize_rand_float_array(dist_gt, N);


//    printf("\n");
    int error;
    // check result of registered functions
    for (int i = 0; i < nfuncs; i++) {
        error = 0;
        functionptr_argsort f = userFuncs[i];
        if(i == 0){
            f.as_baseline_func(idx_arr, dist_gt, N);
//            for(int i=0; i < N; i++){
//                printf("%d ",idx_arr[i]);
//            }
//            printf("\n");
        }
        else{
            // build and initalize random array of structs (value, index)
            // for optimized implementations of argsort
            pair_t* distances;
            distances = malloc(N * sizeof(pair_t));

            for(int j = 0; j < N; j++){
                distances[j].value = dist_gt[j];
                distances[j].index = j;
            }

            f.as_opt_func(distances, N);
//            for(int i=0; i < N; i++){
//                printf("%d ",distances[i].index);
//            }
//            printf("\n");

            for(int j = 0; j < N; j++){
                if(distances[j].index != idx_arr[j]){
                    error = 1;
                }
            }
            if (error == 1) {
                printf("ERROR: function %d is incorrect.", i+1);
            }
            free(distances);
        }
    }

    free(idx_arr);
    free(dist_gt);
}

/*
 * perf_test adapted from homework 2, ASL 2022, How to Write Fast Numerical Code 263-2300 - ETH Zurich
 * returns #cycles required on average (mean) per iteration of the function
 */
double perf_test_sort(int num_func, functionptr_argsort f, int N) {

    double cycles;
    long unsigned num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    myInt64 cycles_numruns = 0;

    // build and initialize random arrays of floats and
    // corresponding indices for baseline argsort implementation
    int* idx_arr = malloc(N*sizeof(int));
    float* dist_gt = malloc(N*sizeof(float));

    initialize_incr_int_array(idx_arr, N);
    initialize_rand_float_array(dist_gt, N);


    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;

        for (size_t i = 0; i < num_runs; i++) {

            // build and initalize random array of structs (value, index)
            // for optimized implementations of argsort
            pair_t* distances;
            distances = malloc(N * sizeof(pair_t));
            initialize_rand_struct(distances, N);

            if(num_func == 0){
                start = start_tsc();
                f.as_baseline_func(idx_arr, dist_gt, N);
            }
            else{
                start = start_tsc();
                f.as_opt_func(distances, N);
            }
            end = stop_tsc(start);
            cycles_numruns += end;
            free(distances);
        }


        cycles = (double)cycles_numruns;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        cycles_numruns = 0;

    } while (multiplier > 2);


    // Actual performance measurements repeated REP times.
    // We simply store all results and compute medians during post-processing.
    double total_cycles = 0;
    cycles_numruns = 0;

    for (size_t j = 0; j < REP; j++) {
        for (size_t i = 0; i < num_runs; ++i) {

            // build and initalize random array of structs (value, index)
            // for optimized implementations of argsort
            pair_t* distances;
            distances = malloc(N * sizeof(pair_t));
            initialize_rand_struct(distances, N);

            if(num_func == 0){
                start = start_tsc();
                f.as_baseline_func(idx_arr, dist_gt, N);
            }
            else{
                start = start_tsc();
                f.as_opt_func(distances, N);
            }
            end = stop_tsc(start);
            cycles_numruns += end;
            free(distances);
        }

        cycles = ((double)cycles_numruns) / num_runs;
        total_cycles += cycles;
        cycles_numruns = 0;
    }
    // compute mean
    total_cycles /= REP;

    // free the malloced memory
    free(idx_arr);
    free(dist_gt);
    cycles = total_cycles;
    return cycles;
}

void time_and_print_sort(functionptr_argsort* userFuncs, int nfuncs, int N) {
    // average performance in cycles
    double performance, performance_base;
    double speedup;
    // measure RUNTIME [cycles] of registered functions
    for (int i = 0; i < nfuncs; i++) {
        performance = perf_test_sort(i, userFuncs[i], N);
        // function registered as 0 serves as base
        if (i == 0) {
            performance_base = performance;
            printf("%d \t\t %f", N, performance_base);
            continue;
        }
        speedup = performance_base / performance;
        printf(" \t\t %.4fx", speedup);
        //printf("N = %d. Running implementation %d. Cycles: %lf. Speed-up: %lfx\n", N, i, performance, speedup);
    }
}

