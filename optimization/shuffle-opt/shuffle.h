#pragma once
#include <immintrin.h>
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "common.h"


void shuffle_baseline(int *arr, int n) {
    int i, j, tmp;

    for (i = n - 1; i >= 0; i--) {
        j = rand() % (i + 1);
        tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

// Credits: https://lemire.me/blog/2016/06/30/fast-random-shuffling/
void shuffle_opt1(int *arr, int n) {
    int i, j, tmp;

    for (i = n - 1; i >= 0; i--) {
        j = (uint64_t) rand() * (uint64_t) (i + 1) >> 32;
        tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

// Credits: https://lemire.me/blog/2016/06/30/fast-random-shuffling/
void shuffle_opt1_unrolled(int *arr, int n) {
    int i, j, tmp, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    int j1, j2, j3, j4 , j5, j6, j7;


    for (i = n - 1; i >= 0; i-=8) {
        j = (uint64_t) rand() * (uint64_t) (i + 1) >> 32;
        j1 = (uint64_t) rand() * (uint64_t) (i) >> 32;
        j2 = (uint64_t) rand() * (uint64_t) (i - 1) >> 32;
        j3 = (uint64_t) rand() * (uint64_t) (i - 2) >> 32;
        j4 = (uint64_t) rand() * (uint64_t) (i - 3) >> 32;
        j5 = (uint64_t) rand() * (uint64_t) (i - 4) >> 32;
        j6 = (uint64_t) rand() * (uint64_t) (i - 5) >> 32;
        j7 = (uint64_t) rand() * (uint64_t) (i - 6) >> 32;

        tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;

        tmp1 = arr[j1];
        arr[j1] = arr[i-1];
        arr[i-1] = tmp1;

        tmp2 = arr[j2];
        arr[j2] = arr[i-2];
        arr[i-2] = tmp2;

        tmp3 = arr[j3];
        arr[j3] = arr[i-3];
        arr[i-3] = tmp3;

        tmp4 = arr[j4];
        arr[j4] = arr[i-4];
        arr[i-4] = tmp4;

        tmp5 = arr[j5];
        arr[j5] = arr[i-5];
        arr[i-5] = tmp5;

        tmp6 = arr[j6];
        arr[j6] = arr[i-6];
        arr[i-6] = tmp6;

        tmp7 = arr[j7];
        arr[j7] = arr[i-7];
        arr[i-7] = tmp7;
    }
}


void register_scalar_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &shuffle_baseline;
    userFuncs[1] = &shuffle_opt1;
    userFuncs[2] = &shuffle_opt1_unrolled;
}