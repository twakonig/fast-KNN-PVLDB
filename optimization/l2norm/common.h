#pragma once

#define EPS (1e-3)
#define CYCLES_REQUIRED 1e8
//#define REP 100
typedef float (*functionptr)(float arr1[], float arr2[], size_t len);

// random array initialization; floats between 0 and 1
void initialize_rand_float_array(float* a, int n) {
    for(int i = 0; i < n; i++){
        a[i] = (float)rand() / RAND_MAX;
    }
}

void print_float_array(float* a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.6f\n", a[i]);
    }
}

// check for correct result of l2norm
void validate(functionptr* userFuncs, int nfuncs, float* a, float* b, int N) {
    float res_kernel, res_f, error;
    double rel_error;
    res_kernel = l2norm(a, b, N);
    // check result of registered functions
    for (int i = 1; i < nfuncs; i++) {
        functionptr f = userFuncs[i];
        // make function 0 (first one to be registered) the base/kernel version
        if (i == 1) {
            res_kernel = f(a, b, N);
            continue;
        }
        res_f = f(a, b, N);
        error = fabs(res_kernel - res_f);
        rel_error = error / fabs(res_kernel);
        if (error > EPS) {
            printf("ERROR, the result of function %d is wrong. |error| = %lf, |rel_error| = %lf\n", i, error, rel_error);
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

    // build and initialize random arrays of floats
    float* a = malloc(N*sizeof(float));
    float* b = malloc(N*sizeof(float));
    initialize_rand_float_array(a, N);
    initialize_rand_float_array(b, N);

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(a, b, N);
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
            f(a, b, N);
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;
    }
    // compute mean
    total_cycles /= reps;

    // free the malloced memory
    free(a);
    free(b);
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
        // care for basic case where sqrt is ignored
        if (i == 1) {
            continue;
        }

        speedup = performance_base / performance;
        printf(" \t %.4fx", speedup);
        // manual inspection
        //printf("N = %d. Running implementation %d. Cycles: %lf. Speed-up: %lfx\n", N, i, performance, speedup);
    }
    printf("\n");
}