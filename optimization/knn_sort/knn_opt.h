//
// Created by lucasck on 24/05/22.
//

#ifndef TEAM36_KNN_OPT_H
#define TEAM36_KNN_OPT_H

#define CYCLES_REQUIRED 1e8
#define REP 4
#define FLT_EPSILON 1e-6
#include "tsc_x86.h"
#include "mat.h"
#include <math.h>
#include <stdbool.h>

typedef float (*functionptr_knn_baseline)(mat *knn, mat *x_trn, mat *x_tst);
typedef float (*functionptr_knn_opt)(knn *knn, mat *x_trn, mat *x_tst);
typedef union {
    functionptr_knn_baseline as_baseline_func;
    functionptr_knn_opt as_opt_func;
} functionptr_knn;

void compute_col_mean(mat* m, float* res, size_t M){

    for(int i = 0; i < m->n2; i++){
        res[i] = 0;
        for(int j = 0; j < m->n1; j++){
            res[i] += mat_get(m, j, i);
        }
        res[i] /= M;
    }
}


// check for correct result
void validate(functionptr_knn* userFuncs, int nfuncs, int N, int mode) {

    // build and initialize random arrays of floats and
    // corresponding indices for baseline argsort implementation
    mat x_tst_knn_base, x_tst_knn_opt;
    knn x_tst_knn_aos;

    mat x_trn;
    mat x_tst;

    build(&x_trn, N, N);
    build(&x_tst, N, N);
    build(&x_tst_knn_base, N, N);


    // randomly initialize all data containers
    initialize_rand_mat(&x_trn);
    initialize_rand_mat(&x_tst);

    int error;
    // check result of registered functions
    for (int i = 0; i < nfuncs; i++) {
        error = 0;
        functionptr_knn f = userFuncs[i];
        if (i == 0) f.as_baseline_func(&x_tst_knn_base, &x_trn, &x_tst); // base will always be first
        else if (i == nfuncs - 1 && mode == 1) { // mode 1 means last optimization has aos knn
            build_knn(&x_tst_knn_aos, N, N);
            f.as_opt_func(&x_tst_knn_aos, &x_trn, &x_tst);

            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                        error += abs(mat_get(&x_tst_knn_base, j, k) - knn_get(&x_tst_knn_aos, j, k));
                }
            }
            destroy_knn(&x_tst_knn_aos);
        }
        else { // nothing weird is happening
            build(&x_tst_knn_opt, N, N);
            f.as_baseline_func(&x_tst_knn_opt, &x_trn, &x_tst);

            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                        error += abs(mat_get(&x_tst_knn_base, j, k) - mat_get(&x_tst_knn_opt, j, k));
                }
            }
            destroy(&x_tst_knn_opt);
        }
        if (error != 0) {
            printf("ERROR: function %d is incorrect. |Error| = %d \n", i + 1, error);
        }
    }

    destroy(&x_tst_knn_base);
    destroy(&x_tst);
    destroy(&x_trn);
}

double perf_test_knn(functionptr_knn f, int N, int nfuncs, int nfunc, int mode) {
    double cycles;
    long unsigned num_runs = 3;
    double total_cycles = 0;
    myInt64 start, end;

    mat x_trn;
    mat x_tst;

    build(&x_trn, N, N);
    build(&x_tst, N, N);

    // randomly initialize all data containers
    initialize_rand_mat(&x_trn);
    initialize_rand_mat(&x_tst);

    if (nfunc == nfuncs - 1 && mode == 1){
        knn x_tst_knn_aos;

        for (size_t j = 0; j < REP; j++) {
            build_knn(&x_tst_knn_aos, N, N);

            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                f.as_opt_func(&x_tst_knn_aos, &x_trn, &x_tst);
            }
            end = stop_tsc(start);
            destroy_knn(&x_tst_knn_aos);
            cycles = ((double)end) / num_runs;
            total_cycles += cycles;
        }
    }else{
        mat x_tst_knn;

        // Actual performance measurements repeated REP times.
        // We simply store all results and compute medians during post-processing.

        for (size_t j = 0; j < REP; j++) {
            build(&x_tst_knn, N, N);

            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                f.as_baseline_func(&x_tst_knn, &x_trn, &x_tst);
            }

            end = stop_tsc(start);
            destroy(&x_tst_knn);

            cycles = ((double)end) / num_runs;
            total_cycles += cycles;
        }
    }

    // compute mean
    total_cycles /= REP;

    // free the malloced memory
    destroy(&x_trn);
    destroy(&x_tst);

    cycles = total_cycles;
    return  cycles;
}

void time_and_print_knn(functionptr_knn* userFuncs, int nfuncs, int N, int mode) {
    // average performance in cycles
    double cycles;
    double multiplier = 1;
    long unsigned num_runs = 3;
    double performance, performance_base;
    double speedup;
    myInt64 start, end;

    mat x_tst_knn;
    mat x_trn;
    mat x_tst;

    // randomly initialize all data containers
    build(&x_trn, N, N);
    build(&x_tst, N, N);
    build(&x_tst_knn, N, N);
    initialize_rand_mat(&x_trn);
    initialize_rand_mat(&x_tst);

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            userFuncs[0].as_baseline_func(&x_tst_knn, &x_trn, &x_tst);
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);

    } while (multiplier > 2);
    // destroy the sorted knn matrix since it has sorted indices
    destroy(&x_tst_knn);

    // measure RUNTIME [cycles] of registered functions
    for (int i = 0; i < nfuncs; i++) {
        performance = perf_test_knn(userFuncs[i], N, nfuncs, i, mode);
        // function registered as 0 serves as base
        if (i == 0) {
            performance_base = performance;
            printf("%d \t\t %f", N, performance_base);
            continue;
        }
        speedup = performance_base / performance;
        printf(" \t\t %.4fx", speedup);
    }
    printf("\n");
    // free the malloced memory
    destroy(&x_trn);
    destroy(&x_tst);
}

#endif //TEAM36_KNN_OPT_H
