#pragma once
#include <immintrin.h>
#include "../include/utils.h"
#include "../include/tsc_x86.h"
#include "common.h"

/*---------FILE CONTENT------------
 * all SIMD implementations of l2norm function
 * best performing version, to be used for timing: l2normsq_vectorized32
 *
 * Optimization techniques used: replacement of library calls, removal of sqrt (not needed), loop unrolling,
 * separate accumulators, scalar replacement
 *
 * !!MAKE SURE NOT TO USE -ffast-math FLAG -> PROBLEMS WITH FLOATING POINT ARITHMETIC!!
 * */

// 1st function registered must be base in terms of cycles
// 2nd function registered must be base in terms of correctness

// base implementation, not optimized
float l2norm_scalar_base(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    for (size_t i = 0; i < len; i++) {
        res += pow(arr1[i] - arr2[i], 2);
    }
    return sqrt(res);
}

// base implementation, not optimized
float l2normsq_scalar_gt(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    for (size_t i = 0; i < len; i++) {
        res += pow(arr1[i] - arr2[i], 2);
    }
    return res;
}

/*// straightforward try using SIMD instructions
float l2normsq_vectorized1(float a[], float b[], size_t len){
    // ps vector can hold 8 elements
    float* vresult = malloc(8*sizeof(float));
    float res_scalar = 0.0;
    __m256 a_vec, b_vec;
    __m256 sub_vec, mul_vec, res_vec;

    res_vec = _mm256_setzero_ps();

    for (size_t i = 0; i < len; i+=8) {
        // load data
        a_vec = _mm256_loadu_ps(a+i);
        b_vec = _mm256_loadu_ps(b+i);
        // computations
        sub_vec = _mm256_sub_ps(a_vec, b_vec);
        mul_vec = _mm256_mul_ps(sub_vec, sub_vec);
        res_vec = _mm256_add_ps(res_vec, mul_vec);
    }
    // store to vresult
    _mm256_storeu_ps(vresult, res_vec);

    // TODO: improve this
    // sum up vector elements and take sqrt
    for (int k = 0; k < 8; k++) {
        res_scalar += vresult[k];
    }
    free(vresult);
    return res_scalar;
}*/

// replace mul and add by fma
float l2normsq_vectorized8(float a[], float b[], size_t len){
    float* vresult = malloc(8*sizeof(float));
    float res_scalar = 0.0;
    __m256 a_vec, b_vec;
    __m256 sub_vec, res_vec;

    res_vec = _mm256_setzero_ps();

    for (size_t i = 0; i < len; i+=8) {
        // load data
        a_vec = _mm256_loadu_ps(a+i);
        b_vec = _mm256_loadu_ps(b+i);
        // computations
        sub_vec = _mm256_sub_ps(a_vec, b_vec);
        res_vec = _mm256_fmadd_ps(sub_vec, sub_vec, res_vec);
    }
    // store to vresult
    _mm256_storeu_ps(vresult, res_vec);

    // TODO: improve this
    // sum up vector elements and take sqrt
    for (int k = 0; k < 8; k++) {
        res_scalar += vresult[k];
    }
    free(vresult);
    return res_scalar;
}

// unroll loop by factor 2
float l2normsq_vectorized16(float a[], float b[], size_t len){
    float* vresult = malloc(8*sizeof(float));
    float res_scalar = 0.0;
    __m256 a_vec0, b_vec0, a_vec1, b_vec1;
    __m256 sub_vec0, res_vec0, sub_vec1, res_vec1;
    __m256 res_vec;

    res_vec0 = _mm256_setzero_ps();
    res_vec1 = _mm256_setzero_ps();
    res_vec = _mm256_setzero_ps();

    for (size_t i = 0; i < len; i+=16) {
        // load data
        a_vec0 = _mm256_loadu_ps(a+i);
        b_vec0 = _mm256_loadu_ps(b+i);
        a_vec1 = _mm256_loadu_ps(a+i+8);
        b_vec1 = _mm256_loadu_ps(b+i+8);
        // computations
        sub_vec0 = _mm256_sub_ps(a_vec0, b_vec0);
        sub_vec1 = _mm256_sub_ps(a_vec1, b_vec1);
        res_vec0 = _mm256_fmadd_ps(sub_vec0, sub_vec0, res_vec0);
        res_vec1 = _mm256_fmadd_ps(sub_vec1, sub_vec1, res_vec1);
    }
    // add up the separate accumulators
    res_vec = _mm256_add_ps(res_vec0, res_vec1);
    // store to vresult
    _mm256_storeu_ps(vresult, res_vec);

    // TODO: improve this
    // sum up vector elements and take sqrt
    for (int k = 0; k < 8; k++) {
        res_scalar += vresult[k];
    }
    free(vresult);
    return res_scalar;
}

// unroll loop by factor 4
float l2normsq_vectorized32(float a[], float b[], size_t len){
    float* vresult = malloc(8*sizeof(float));
    float res_scalar = 0.0;
    __m256 a_vec0, b_vec0, a_vec1, b_vec1, a_vec2, b_vec2, a_vec3, b_vec3;
    __m256 sub_vec0, res_vec0, sub_vec1, res_vec1, sub_vec2, res_vec2, sub_vec3, res_vec3;
    __m256 tmp_vec0, tmp_vec1, res_vec;

    res_vec0 = _mm256_setzero_ps();
    res_vec1 = _mm256_setzero_ps();
    res_vec2 = _mm256_setzero_ps();
    res_vec3 = _mm256_setzero_ps();
    res_vec = _mm256_setzero_ps();

    for (size_t i = 0; i < len; i+=32) {
        // load data
        a_vec0 = _mm256_loadu_ps(a+i);
        b_vec0 = _mm256_loadu_ps(b+i);
        a_vec1 = _mm256_loadu_ps(a+i+8);
        b_vec1 = _mm256_loadu_ps(b+i+8);
        a_vec2 = _mm256_loadu_ps(a+i+16);
        b_vec2 = _mm256_loadu_ps(b+i+16);
        a_vec3 = _mm256_loadu_ps(a+i+24);
        b_vec3 = _mm256_loadu_ps(b+i+24);
        // computations
        sub_vec0 = _mm256_sub_ps(a_vec0, b_vec0);
        sub_vec1 = _mm256_sub_ps(a_vec1, b_vec1);
        sub_vec2 = _mm256_sub_ps(a_vec2, b_vec2);
        sub_vec3 = _mm256_sub_ps(a_vec3, b_vec3);
        res_vec0 = _mm256_fmadd_ps(sub_vec0, sub_vec0, res_vec0);
        res_vec1 = _mm256_fmadd_ps(sub_vec1, sub_vec1, res_vec1);
        res_vec2 = _mm256_fmadd_ps(sub_vec2, sub_vec2, res_vec2);
        res_vec3 = _mm256_fmadd_ps(sub_vec3, sub_vec3, res_vec3);
    }
    // add up the separate accumulators
    tmp_vec0 = _mm256_add_ps(res_vec0, res_vec1);
    tmp_vec1 = _mm256_add_ps(res_vec2, res_vec3);
    res_vec = _mm256_add_ps(tmp_vec0, tmp_vec1);
    // store to vresult
    _mm256_storeu_ps(vresult, res_vec);

    // sum up vector elements and take sqrt
    for (int k = 0; k < 8; k++) {
        res_scalar += vresult[k];
    }
    free(vresult);
    return res_scalar;
}



void register_simd_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &l2norm_scalar_base;
    userFuncs[1] = &l2normsq_scalar_gt;
    userFuncs[2] = &l2normsq_vectorized8;
    userFuncs[3] = &l2normsq_vectorized16;
    userFuncs[4] = &l2normsq_vectorized32;
}