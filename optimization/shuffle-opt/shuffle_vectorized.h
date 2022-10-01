#pragma once
#include <immintrin.h>
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "common.h"

void shuffle_scalar_baseline(int *arr, int n) {
    int i, j, tmp;

    for (i = n - 1; i >= 0; i--) {
        j = rand() % (i + 1);
        tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

void register_simd_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &shuffle_scalar_baseline;

}
