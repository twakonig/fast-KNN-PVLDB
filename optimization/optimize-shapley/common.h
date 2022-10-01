#pragma once

#define EPS (1e-5)
#define CYCLES_REQUIRED 1e8
//#define REP 100
typedef void (*svfctptr)(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K);


// compare computed matrix entries (base and optimizations)
void compare_mat(mat* sp_kernel, mat* sp_opt, int N){
    float error, rel_error;
    int ctr = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            error = fabs(sp_kernel->data[i*N + j] - sp_opt->data[i*N + j]);
            rel_error = error / fabs(sp_kernel->data[i*N + j]);
            if (error > EPS) {
                printf("ERROR, shapley entry at (%d, %d) is wrong. |error| = %lf, |rel_error| = %lf\n", i, j, error,
                       rel_error);
                ctr += 1;
            }
        }
    }
//    if(ctr == 0)
//        printf("Shapley values CORRECT.\n");
}

// check for correct result of compute_shapley
void validate_shapley(svfctptr* userFuncs, int nfuncs, int* y_trn, int* y_tst, int_mat* x_tst_knn_gt, int N, int K) {
    // matrices for intermediate shapley values
    mat sp_kernel, sp_opt;
    // build and set to 0.0
    build(&sp_kernel, N, N);
    build(&sp_opt, N, N);
    for(int i = 0; i < sp_kernel.n1; i++){
        for(int j = 0; j < sp_kernel.n2; j++){
            mat_set(&sp_kernel, i, j, 0.0);
            mat_set(&sp_opt, i, j, 0.0);
        }
    }

    // check result of registered functions
    for (int i = 0; i < nfuncs; i++) {
        svfctptr comp_sp = userFuncs[i];
        if (i == 0) {
            comp_sp(&sp_kernel, y_trn, y_tst, x_tst_knn_gt, K);
            continue;
        }
        comp_sp(&sp_opt, y_trn, y_tst, x_tst_knn_gt, K);
        compare_mat(&sp_kernel, &sp_opt, N);
    }
    // from build (malloc)
    destroy(&sp_kernel);
    destroy(&sp_opt);
}

/*
 * perf_test adapted from homework 2, ASL 2022, How to Write Fast Numerical Code 263-2300 - ETH Zurich
 * returns #cycles required on average (mean) per iteration of the function
 */
double perf_test(svfctptr f, int N, int reps, int K) {
    double cycles = 0.;
    long unsigned num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;

    // build and initialize random arrays of floats
    int* y_trn = malloc(N*sizeof(int));
    int* y_tst = malloc(N*sizeof(int));
    int_mat x_tst_knn_gt;
    mat sp_gt;
    build_int_mat(&x_tst_knn_gt, N, N);
    build(&sp_gt, N, N);
    initialize_rand_array(y_trn, N);
    initialize_rand_array(y_tst, N);
    initialize_rand_int_mat(&x_tst_knn_gt, N);
    for(int i = 0; i < sp_gt.n1; i++){
        for(int j = 0; j < sp_gt.n2; j++){
            mat_set(&sp_gt, i, j, 0.0);
        }
    }

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(&sp_gt, y_trn, y_tst, &x_tst_knn_gt, K);
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
            f(&sp_gt, y_trn, y_tst, &x_tst_knn_gt, K);
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;
    }
    // compute mean
    total_cycles /= reps;

    // free the malloced memory
    free(y_trn);
    free(y_tst);
    destroy_int_mat(&x_tst_knn_gt);
    destroy(&sp_gt);
    cycles = total_cycles;
    return  cycles;
}

void time_and_print(svfctptr* userFuncs, int nfuncs, int N, int reps, int K) {
    // average performance in cycles
    double performance, performance_base;
    double speedup;
    // measure RUNTIME [cycles] of registered functions
    for (int i = 0; i < nfuncs; i++) {
        performance = perf_test(userFuncs[i], N, reps, K);
        // function registered as 0 serves as base
        if (i == 0) {
            performance_base = performance;
            printf("%d \t %.4fx", N, performance_base);
            continue;
        }
        speedup = performance_base / performance;
        printf(" \t %.4fx", speedup);
        // manual inspection
        //printf("N = %d. Running implementation %d. Cycles: %lf. Speed-up: %lfx\n", N, i, performance, speedup);
    }
    printf("\n");
}

