#pragma once
#include <immintrin.h>
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "common.h"

float unweighted_knn_utility_scalar_basline(int* y_trn, int y_tst, int size, int K){
    float sum = 0.0;
    for (int i = 0; i < size; i++){
        sum += (y_trn[i] == y_tst) ? 1.0 : 0.0;
    }
    return sum / K;
}

void register_simd_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &unweighted_knn_utility_scalar_basline;

}
