//
// Created by lucasck on 19/05/22.
//

// <----------------------------------------------------->
// This file is divided in 2 parts: 1) function definitions for (vectorized) horizontal sums and l2norm and 2) optimizations
// Optimizations -> get_true_knn_base represents the base we started with

// opts 1 -- 7 explore various optimizations wrt locality and sorting 
//      -> referencing, unrolling, copying & referencing, klib mergesort vs quadsort, AoS.

// Opts 8 -- 13 are scalar blocking
// Opts 8, 10, 12 explore blocking with various block sizes
// Opts 9, 11, 13 are identical to their counterparts, save for optimizations done through pragmas

// Opts 14 -- 22 are vectorized blocking, and divided as follows
// Opts 14 -- 17 do 4x8 blocking with various improvements (cumulative)

// Opts 18 -- 21 do 8x8 blocking with various improvements (cumulative)
// Opt 22 does 8x16 blocking
// <----------------------------------------------------->


#include "../../include/tsc_x86.h"
#include "../../src/alg1/alg1.h"
#include "../../include/utils.h"
#include "../../include/quadsort.h"
#include "../../include/ksort.h"
#include <immintrin.h>

#define pair_lt(a, b) ((a).value < (b).value)

KSORT_INIT(pair, pair_t, pair_lt)

KSORT_INIT_GENERIC(float)

// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

float hsum256_ps_avx(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}


// L2-norms used throughout optimizations -> vectorized && using arrays copied into memory or referenced by pointer

float l2norm_opt(float a[], float b[], size_t len) {
    float *vresult = malloc(8 * sizeof(float));
    float res_scalar = 0.0;
    __m256 a_vec0, b_vec0, a_vec1, b_vec1, a_vec2, b_vec2, a_vec3, b_vec3;
    __m256 sub_vec0, res_vec0, sub_vec1, res_vec1, sub_vec2, res_vec2, sub_vec3, res_vec3;
    __m256 tmp_vec0, tmp_vec1, res_vec;

    res_vec0 = _mm256_setzero_ps();
    res_vec1 = _mm256_setzero_ps();
    res_vec2 = _mm256_setzero_ps();
    res_vec3 = _mm256_setzero_ps();
    res_vec = _mm256_setzero_ps();

    for (size_t i = 0; i < len; i += 32) {
        // load data
        a_vec0 = _mm256_loadu_ps(a + i);
        b_vec0 = _mm256_loadu_ps(b + i);
        a_vec1 = _mm256_loadu_ps(a + i + 8);
        b_vec1 = _mm256_loadu_ps(b + i + 8);
        a_vec2 = _mm256_loadu_ps(a + i + 16);
        b_vec2 = _mm256_loadu_ps(b + i + 16);
        a_vec3 = _mm256_loadu_ps(a + i + 24);
        b_vec3 = _mm256_loadu_ps(b + i + 24);
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
    // sum up vector elements
    for (int k = 0; k < 8; k++) {
        res_scalar += vresult[k];
    }
    free(vresult);
    return res_scalar;
}

float l2norm_opt_ptr(float a[], float b[], size_t strt1, size_t strt2, size_t len) {
    float *vresult = _mm_malloc(8 * sizeof(float), 32);
    float res_scalar = 0.0;
    __m256 a_vec0, b_vec0, a_vec1, b_vec1, a_vec2, b_vec2, a_vec3, b_vec3;
    __m256 sub_vec0, res_vec0, sub_vec1, res_vec1, sub_vec2, res_vec2, sub_vec3, res_vec3;
    __m256 tmp_vec0, tmp_vec1, res_vec;

    res_vec0 = _mm256_setzero_ps();
    res_vec1 = _mm256_setzero_ps();
    res_vec2 = _mm256_setzero_ps();
    res_vec3 = _mm256_setzero_ps();
    res_vec = _mm256_setzero_ps();

    int offset1 = strt1 * len;
    int offset2 = strt2 * len;

    for (size_t i = 0; i < len; i += 32) {
        // load data
        a_vec0 = _mm256_load_ps(a + i + offset1);
        b_vec0 = _mm256_load_ps(b + i + offset2);
        a_vec1 = _mm256_load_ps(a + i + 8 + offset1);
        b_vec1 = _mm256_load_ps(b + i + 8 + offset2);
        a_vec2 = _mm256_load_ps(a + i + 16 + offset1);
        b_vec2 = _mm256_load_ps(b + i + 16 + offset2);
        a_vec3 = _mm256_load_ps(a + i + 24 + offset1);
        b_vec3 = _mm256_load_ps(b + i + 24 + offset2);
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
    _mm256_store_ps(vresult, res_vec);
    // sum up vector elements
    for (int k = 0; k < 8; k++) {
        res_scalar += vresult[k];
    }
    _mm_free(vresult);
    return res_scalar;
}

float l2norm_opt_ptr2(float a[], float *b, size_t strt_b, size_t len) {
    float *vresult = _mm_malloc(8 * sizeof(float), 32);
    float res_scalar = 0.0;
    __m256 a_vec0, b_vec0, a_vec1, b_vec1, a_vec2, b_vec2, a_vec3, b_vec3;
    __m256 sub_vec0, res_vec0, sub_vec1, res_vec1, sub_vec2, res_vec2, sub_vec3, res_vec3;
    __m256 tmp_vec0, tmp_vec1, res_vec;

    res_vec0 = _mm256_setzero_ps();
    res_vec1 = _mm256_setzero_ps();
    res_vec2 = _mm256_setzero_ps();
    res_vec3 = _mm256_setzero_ps();
    res_vec = _mm256_setzero_ps();
    
    int offset = strt_b * len;

    for (size_t i = 0; i < len; i += 32) {
        // load data
        a_vec0 = _mm256_load_ps(a + i);
        b_vec0 = _mm256_load_ps(b + i + offset);
        a_vec1 = _mm256_load_ps(a + i + 8);
        b_vec1 = _mm256_load_ps(b + i + 8 + offset);
        a_vec2 = _mm256_load_ps(a + i + 16);
        b_vec2 = _mm256_load_ps(b + i + 16 + offset);
        a_vec3 = _mm256_load_ps(a + i + 24);
        b_vec3 = _mm256_load_ps(b + i + 24 + offset);
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
    _mm256_store_ps(vresult, res_vec);
    // sum up vector elements
    for (int k = 0; k < 8; k++) {
        res_scalar += vresult[k];
    }
    _mm_free(vresult);
    return res_scalar;
}

void get_true_knn_base(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;

    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
        float dist_gt[N];
        int idx_arr[N];

        for (int i_trn = 0; i_trn < N; i_trn++) {
            float trn_row[d];
            float tst_row[d];
            for (int j = 0; j < d; j++) {
                trn_row[j] = data_trn[i_trn * d + j];
                tst_row[j] = data_tst[i_tst * d + j];
            }
            idx_arr[i_trn] = i_trn;
            dist_gt[i_trn] = l2norm(trn_row, tst_row, d);
        }
        argsort(idx_arr, dist_gt, N);
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = idx_arr[k];
        }
    }
}

// unroll factor 1, copy into memory
void get_true_knn_opt1(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    pair_t *distances;
    distances = malloc(N * sizeof(pair_t));
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    float trn_row[d];
    float tst_row[d];


    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
        for (int j = 0; j < d; j++) {
            tst_row[j] = data_tst[i_tst * d + j];
        }
        for (int i_trn = 0; i_trn < N; i_trn++) {

            for (int j = 0; j < d; j++) {
                trn_row[j] = data_trn[i_trn * d + j];
            }
            distances[i_trn].value = l2norm_opt(trn_row, tst_row, d);
            distances[i_trn].index = i_trn;
        }
        quadsort(distances, N, sizeof(pair_t), cmp);
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = distances[k].index;
        }
    }
    free(distances);
}

// unroll factor 1, pass by reference
void get_true_knn_opt2(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    pair_t *distances;
    distances = malloc(N * sizeof(pair_t));
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
        for (int i_trn = 0; i_trn < N; i_trn++) {
            distances[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, i_tst, d);
            distances[i_trn].index = i_trn;
        }

        quadsort(distances, N, sizeof(pair_t), cmp);
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = distances[k].index;
        }
    }
    free(distances);
}

// unroll factor 1, pass by reference only the test row
void get_true_knn_opt3(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    pair_t *distances;
    distances = malloc(N * sizeof(pair_t));
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    float tst_row[d];
    int i_tstd, i_tstN;

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
        i_tstd = i_tst * d;
        i_tstN = i_tst * N;
        for (int j = 0; j < d; j++) {
            tst_row[j] = data_tst[i_tstd + j];
        }
        for (int i_trn = 0; i_trn < N; i_trn++) {
            distances[i_trn].value = l2norm_opt_ptr2(tst_row, data_trn, i_trn, d);
            distances[i_trn].index = i_trn;
        }
        quadsort(distances, N, sizeof(pair_t), cmp);
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tstN + k] = distances[k].index;
        }
    }
    free(distances);
}

// unroll factor 4, pass by reference
void get_true_knn_opt4(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    pair_t *distances, *distances_i, *distances_ii, *distances_iii;
    distances = malloc(N * sizeof(pair_t));
    distances_i = malloc(N * sizeof(pair_t));
    distances_ii = malloc(N * sizeof(pair_t));
    distances_iii = malloc(N * sizeof(pair_t));
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    int i_tstd, i_tstdd, i_tstddd, i_tstdddd, i_tstN, i_tstNN, i_tstNNN, i_tstNNNN;

    for (int i_tst = 0; i_tst < N_tst; i_tst += 4) {
        i_tstN = i_tst * N;
        i_tstNN = (i_tst + 1) * N;
        i_tstNNN = (i_tst + 2) * N;
        i_tstNNNN = (i_tst + 3) * N;

#pragma GCC ivdep
        for (int i_trn = 0; i_trn < N; i_trn++) {
            distances[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, i_tst, d);
            distances_i[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 1), d);
            distances_ii[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 2), d);
            distances_iii[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 3), d);
            distances[i_trn].index = i_trn;
            distances_i[i_trn].index = i_trn;
            distances_ii[i_trn].index = i_trn;
            distances_iii[i_trn].index = i_trn;
        }
        quadsort(distances, N, sizeof(pair_t), cmp);
        quadsort(distances_i, N, sizeof(pair_t), cmp);
        quadsort(distances_ii, N, sizeof(pair_t), cmp);
        quadsort(distances_iii, N, sizeof(pair_t), cmp);

#pragma GCC ivdep
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tstNNNN + k] = distances_iii[k].index;
            x_tst_knn_gt->data[i_tstNNN + k] = distances_ii[k].index;
            x_tst_knn_gt->data[i_tstNN + k] = distances_i[k].index;
            x_tst_knn_gt->data[i_tstN + k] = distances[k].index;
        }
    }
    free(distances);
    free(distances_i);
    free(distances_ii);
    free(distances_iii);
}

// unroll factor 4, copy in memory
void get_true_knn_opt5(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;

    for (int i_tst = 0; i_tst < N_tst; i_tst += 4) {
        pair_t *dist_gt = malloc(N * sizeof(pair_t));
        pair_t *dist_gt_i = malloc(N * sizeof(pair_t));
        pair_t *dist_gt_ii = malloc(N * sizeof(pair_t));
        pair_t *dist_gt_iii = malloc(N * sizeof(pair_t));

        for (int i_trn = 0; i_trn < N; i_trn++) {
            float trn_row[d];
            float tst_row[d];
            float tst_row_i[d];
            float tst_row_ii[d];
            float tst_row_iii[d];
            for (int j = 0; j < d; j++) {
                trn_row[j] = data_trn[i_trn * d + j];
                tst_row[j] = data_tst[i_tst * d + j];
                tst_row_i[j] = data_tst[(i_tst + 1) * d + j];
                tst_row_ii[j] = data_tst[(i_tst + 2) * d + j];
                tst_row_iii[j] = data_tst[(i_tst + 3) * d + j];
            }
            dist_gt[i_trn].index = i_trn;
            dist_gt[i_trn].value = l2norm_opt(trn_row, tst_row, d);
            dist_gt_i[i_trn].index = i_trn;
            dist_gt_i[i_trn].value = l2norm_opt(trn_row, tst_row_i, d);
            dist_gt_ii[i_trn].index = i_trn;
            dist_gt_ii[i_trn].value = l2norm_opt(trn_row, tst_row_ii, d);
            dist_gt_iii[i_trn].index = i_trn;
            dist_gt_iii[i_trn].value = l2norm_opt(trn_row, tst_row_iii, d);

        }
        quadsort(dist_gt, N, sizeof(pair_t), cmp);
        quadsort(dist_gt_i, N, sizeof(pair_t), cmp);
        quadsort(dist_gt_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_gt_iii, N, sizeof(pair_t), cmp);
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_gt[k].index;
            x_tst_knn_gt->data[(i_tst + 1) * N + k] = dist_gt_i[k].index;
            x_tst_knn_gt->data[(i_tst + 2) * N + k] = dist_gt_ii[k].index;
            x_tst_knn_gt->data[(i_tst + 3) * N + k] = dist_gt_iii[k].index;
        }
        free(dist_gt);
        free(dist_gt_i);
        free(dist_gt_ii);
        free(dist_gt_iii);
    }
}

// unroll factor 8, pass by reference
void get_true_knn_opt6(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    pair_t *distances, *distances_i, *distances_ii, *distances_iii, *distances_iiii, *distances_iiiii, *distances_iiiiii, *distances_iiiiiii;
    distances = malloc(N * sizeof(pair_t));
    distances_i = malloc(N * sizeof(pair_t));
    distances_ii = malloc(N * sizeof(pair_t));
    distances_iii = malloc(N * sizeof(pair_t));
    distances_iiii = malloc(N * sizeof(pair_t));
    distances_iiiii = malloc(N * sizeof(pair_t));
    distances_iiiiii = malloc(N * sizeof(pair_t));
    distances_iiiiiii = malloc(N * sizeof(pair_t));
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    int i_tstN, i_tstNN, i_tstNNN, i_tstNNNN, i_tstNNNNN, i_tstNNNNNN, i_tstNNNNNNN, i_tstNNNNNNNN;

    for (int i_tst = 0; i_tst < N_tst; i_tst += 8) {
        i_tstN = i_tst * N;
        i_tstNN = (i_tst + 1) * N;
        i_tstNNN = (i_tst + 2) * N;
        i_tstNNNN = (i_tst + 3) * N;
        i_tstNNNNN = (i_tst + 4) * N;
        i_tstNNNNNN = (i_tst + 5) * N;
        i_tstNNNNNNN = (i_tst + 6) * N;
        i_tstNNNNNNNN = (i_tst + 7) * N;
#pragma GCC ivdep
        for (int i_trn = 0; i_trn < N; i_trn++) {
            distances[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, i_tst, d);
            distances_i[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 1), d);
            distances_ii[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 2), d);
            distances_iii[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 3), d);
            distances_iiii[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 4), d);
            distances_iiiii[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 5), d);
            distances_iiiiii[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 6), d);
            distances_iiiiiii[i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, (i_tst + 7), d);
            distances[i_trn].index = i_trn;
            distances_i[i_trn].index = i_trn;
            distances_ii[i_trn].index = i_trn;
            distances_iii[i_trn].index = i_trn;
            distances_iiii[i_trn].index = i_trn;
            distances_iiiii[i_trn].index = i_trn;
            distances_iiiiii[i_trn].index = i_trn;
            distances_iiiiiii[i_trn].index = i_trn;
        }
        quadsort(distances, N, sizeof(pair_t), cmp);
        quadsort(distances_i, N, sizeof(pair_t), cmp);
        quadsort(distances_ii, N, sizeof(pair_t), cmp);
        quadsort(distances_iii, N, sizeof(pair_t), cmp);
        quadsort(distances_iiii, N, sizeof(pair_t), cmp);
        quadsort(distances_iiiii, N, sizeof(pair_t), cmp);
        quadsort(distances_iiiiii, N, sizeof(pair_t), cmp);
        quadsort(distances_iiiiiii, N, sizeof(pair_t), cmp);
#pragma GCC ivdep
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tstNNNNNNNN + k] = distances_iiiiiii[k].index;
            x_tst_knn_gt->data[i_tstNNNNNNN + k] = distances_iiiiii[k].index;
            x_tst_knn_gt->data[i_tstNNNNNN + k] = distances_iiiii[k].index;
            x_tst_knn_gt->data[i_tstNNNNN + k] = distances_iiii[k].index;
            x_tst_knn_gt->data[i_tstNNNN + k] = distances_iii[k].index;
            x_tst_knn_gt->data[i_tstNNN + k] = distances_ii[k].index;
            x_tst_knn_gt->data[i_tstNN + k] = distances_i[k].index;
            x_tst_knn_gt->data[i_tstN + k] = distances[k].index;
        }
    }
    free(distances);
    free(distances_i);
    free(distances_ii);
    free(distances_iii);
    free(distances_iiii);
    free(distances_iiiii);
    free(distances_iiiiii);
    free(distances_iiiiiii);

}

// AoS for the knn matrix, now holding both indices and l2-norm values
void get_true_knn_opt7(knn *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    pair_t *knn_mat = x_tst_knn_gt->data; // exploring whether making the knn matrix be an AoS would help

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
        for (int i_trn = 0; i_trn < N; i_trn++) {
            x_tst_knn_gt ->data[i_tst * N + i_trn].value = l2norm_opt_ptr(data_trn, data_tst, i_trn, i_tst, d);
        }
       quadsort(x_tst_knn_gt ->data + i_tst * N, N, sizeof(pair_t), cmp);
    }
}

// scalar blocking Nb = 4
void get_true_knn_opt8(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 4;
    pair_t *dist_arr;
    dist_arr = malloc(N * sizeof(pair_t));
    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;

    for (bi = 0; bi < N_tst; bi += Nb)
        for (bj = 0; bj < N; bj += Nb)
            for (bk = 0; bk < d; bk += Nb)
                for (i = bi; i < Nb + bi; i++)
                    for (j = bj; j < Nb + bj; j++) {
                        float tmp = distances.data[(i) * N + j];
                        for (k = bk; k < Nb + bk; k++) {
                            float val = (data_tst[i * N + k] - data_trn[j * N + k]);
                            tmp += val * val;
                        }
                        distances.data[(i) * N + j] = tmp;
                    }

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
        for (int i_trn = 0; i_trn < N; i_trn++) {
            dist_arr[i_trn].value = distances.data[i_tst * N + i_trn];
            dist_arr[i_trn].index = i_trn;
        }

        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
        }
    }
    free(dist_arr);
    destroy(&distances);
}

// scalar blocking Nb = 4
void get_true_knn_opt9(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 4;
    pair_t *__restrict__ dist_arr;
    dist_arr = malloc(N * sizeof(pair_t));
    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float *__restrict__ data_trn = x_trn->data;
    float *__restrict__ data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;

    for (bi = 0; bi < N_tst; bi += Nb)
        for (bj = 0; bj < N; bj += Nb)
            for (bk = 0; bk < d; bk += Nb)
                for (i = bi; i < Nb + bi; i++)
#pragma ivdep
                        for (j = bj; j < Nb + bj; j++) {
                            float tmp = distances.data[(i) * N + j];
#pragma ivdep
#pragma unroll(4)
                            for (k = bk; k < Nb + bk; k++) {
                                float val = (data_tst[i * N + k] - data_trn[j * N + k]);
                                tmp += val * val;
                            }
                            distances.data[(i) * N + j] = tmp;
                        }

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
#pragma ivdep
        for (int i_trn = 0; i_trn < N; i_trn++) {
            dist_arr[i_trn].value = distances.data[i_tst * N + i_trn];
            dist_arr[i_trn].index = i_trn;
        }

        quadsort(dist_arr, N, sizeof(pair_t), cmp);
#pragma ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
        }
    }
    free(dist_arr);
    destroy(&distances);
}

// scalar blocking Nb = 8
void get_true_knn_opt10(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 8;
    pair_t *dist_arr;
    dist_arr = malloc(N * sizeof(pair_t));
    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;

    for (bi = 0; bi < N_tst; bi += Nb)
        for (bj = 0; bj < N; bj += Nb)
            for (bk = 0; bk < d; bk += Nb)
                for (i = bi; i < Nb + bi; i++)
                    for (j = bj; j < Nb + bj; j++) {
                        float tmp = distances.data[(i) * N + j];
                        for (k = bk; k < Nb + bk; k++) {
                            float val = (data_tst[(i) * N + k] - data_trn[(j) * N + k]);
                            tmp += val * val;
                        }
                        distances.data[(i) * N + j] = tmp;
                    }

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
        for (int i_trn = 0; i_trn < N; i_trn++) {
            dist_arr[i_trn].value = distances.data[i_tst * N + i_trn];
            dist_arr[i_trn].index = i_trn;
        }

        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
        }
    }
    free(dist_arr);
    destroy(&distances);
}

// scalar blocking Nb = 8
void get_true_knn_opt11(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 8;
    pair_t *__restrict__ dist_arr;
    dist_arr = malloc(N * sizeof(pair_t));
    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float *__restrict__ data_trn = x_trn->data;
    float *__restrict__ data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;

    for (bi = 0; bi < N_tst; bi += Nb)
        for (bj = 0; bj < N; bj += Nb)
            for (bk = 0; bk < d; bk += Nb)
                for (i = bi; i < Nb + bi; i++)
#pragma ivdep
                    for (j = bj; j < Nb + bj; j++) {
                        float tmp = distances.data[(i) * N + j];
#pragma ivdep
#pragma unroll (8)
                        for (k = bk; k < Nb + bk; k++) {
                            float val = (data_tst[(i) * N + k] - data_trn[(j) * N + k]);
                            tmp += val * val;
                        }
                        distances.data[(i) * N + j] = tmp;
                    }

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
#pragma ivdep
        for (int i_trn = 0; i_trn < N; i_trn++) {
            dist_arr[i_trn].value = distances.data[i_tst * N + i_trn];
            dist_arr[i_trn].index = i_trn;
        }

        quadsort(dist_arr, N, sizeof(pair_t), cmp);
#pragma ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
        }
    }
    free(dist_arr);
    destroy(&distances);
}

// scalar blocking Nb = 16
void get_true_knn_opt12(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 16;
    pair_t *dist_arr;
    dist_arr = malloc(N * sizeof(pair_t));
    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;

    for (bi = 0; bi < N_tst; bi += Nb)
        for (bj = 0; bj < N; bj += Nb)
            for (bk = 0; bk < d; bk += Nb)
                for (i = bi; i < Nb + bi; i++)
                    for (j = bj; j < Nb + bj; j++) {
                        float tmp = distances.data[(i) * N + j];
                        for (k = bk; k < Nb + bk; k++) {
                            float val = (data_tst[i * N + k] - data_trn[j * N + k]);
                            tmp += val * val;
                        }
                        distances.data[(i) * N + j] = tmp;
                    }

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
        for (int i_trn = 0; i_trn < N; i_trn++) {
            dist_arr[i_trn].value = distances.data[i_tst * N + i_trn];
            dist_arr[i_trn].index = i_trn;
        }

        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
        }
    }
    free(dist_arr);
    destroy(&distances);
}

// scalar blocking Nb = 16
void get_true_knn_opt13(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 16;
    pair_t *__restrict__ dist_arr;
    dist_arr = malloc(N * sizeof(pair_t));
    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float *__restrict__ data_trn = x_trn->data;
    float *__restrict__ data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;

    for (bi = 0; bi < N_tst; bi += Nb)
        for (bj = 0; bj < N; bj += Nb)
            for (bk = 0; bk < d; bk += Nb)
                for (i = bi; i < Nb + bi; i++)
#pragma ivdep
                    for (j = bj; j < Nb + bj; j++) {
                        float tmp = distances.data[(i) * N + j];
#pragma ivdep
#pragma unroll (16)
                        for (k = bk; k < Nb + bk; k++) {
                            float val = (data_tst[i * N + k] - data_trn[j * N + k]);
                            tmp += val * val;
                        }
                        distances.data[(i) * N + j] = tmp;
                    }

    for (int i_tst = 0; i_tst < N_tst; i_tst++) {
#pragma ivdep
        for (int i_trn = 0; i_trn < N; i_trn++) {
            dist_arr[i_trn].value = distances.data[i_tst * N + i_trn];
            dist_arr[i_trn].index = i_trn;
        }

        quadsort(dist_arr, N, sizeof(pair_t), cmp);
#pragma ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
        }
    }
    free(dist_arr);
    destroy(&distances);
}

// 4x8 vectorized - horizontal sum inside + 4 AoS arrays + float accumulators
void get_true_knn_opt14(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 4;

    pair_t *__restrict__ dist_arr = malloc(d * sizeof(pair_t));
    pair_t *__restrict__ dist_arr_i = malloc(d * sizeof(pair_t));
    pair_t *__restrict__ dist_arr_ii = malloc(d * sizeof(pair_t));
    pair_t *__restrict__ dist_arr_iii = malloc(d * sizeof(pair_t));

    float *__restrict__ data_trn = x_trn->data;
    float *__restrict__ data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;
    __m256 ts_vec0, tr_vec0, ts_vec1, tr_vec1, ts_vec2, tr_vec2, ts_vec3, tr_vec3;
    __m256 tmp0, tmp1, tmp2, tmp3;

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3, vec_sum4, vec_sum5, vec_sum6, vec_sum7;
    __m256 sub_vec0, sub_vec1, sub_vec2, sub_vec3, sub_vec4, sub_vec5, sub_vec6, sub_vec7;
    __m256 sub_vec8, sub_vec9, sub_vec10, sub_vec11, sub_vec12, sub_vec13, sub_vec14;
    __m256 sub_vec15, sub_vec16, sub_vec17, sub_vec18, sub_vec19, sub_vec20, sub_vec21;
    __m256 sub_vec22, sub_vec23, sub_vec24, sub_vec25, sub_vec26, sub_vec27, sub_vec28;
    __m256 sub_vec29, sub_vec30, sub_vec31, sub_vec32;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;

    for (bi = 0; bi < N_tst; bi += Nb) {
        for (bj = 0; bj < N; bj += Nb) {

            f0 = f1 = f2 = f3 = f4 = f5 = f6 = f7 = 0;
            f8 = f9 = f10 = f11 = f12 = f13 = f14 = f15 = 0;

            for (bk = 0; bk < d; bk += 8) {

                ts_vec0 = _mm256_load_ps(data_tst + bi * N + bk); // 4x8 blocks from test matrix
                ts_vec1 = _mm256_load_ps(data_tst + (bi + 1) * N + bk);
                ts_vec2 = _mm256_load_ps(data_tst + (bi + 2) * N + bk);
                ts_vec3 = _mm256_load_ps(data_tst + (bi + 3) * N + bk);

                tr_vec0 = _mm256_load_ps(data_trn + bj * N + bk); // 4x8 blocks from train matrix
                tr_vec1 = _mm256_load_ps(data_trn + (bj + 1) * N + bk);
                tr_vec2 = _mm256_load_ps(data_trn + (bj + 2) * N + bk);
                tr_vec3 = _mm256_load_ps(data_trn + (bj + 3) * N + bk);

                sub_vec0 = _mm256_sub_ps(ts_vec0, tr_vec0); // pairwise distances between test and train blocks
                sub_vec1 = _mm256_sub_ps(ts_vec0, tr_vec1); // 1x4 per code block, total 4x8 values computed
                sub_vec2 = _mm256_sub_ps(ts_vec0, tr_vec2);
                sub_vec3 = _mm256_sub_ps(ts_vec0, tr_vec3);

                sub_vec4 = _mm256_sub_ps(ts_vec1, tr_vec0);
                sub_vec5 = _mm256_sub_ps(ts_vec1, tr_vec1);
                sub_vec6 = _mm256_sub_ps(ts_vec1, tr_vec2);
                sub_vec7 = _mm256_sub_ps(ts_vec1, tr_vec3);

                sub_vec8 = _mm256_sub_ps(ts_vec2, tr_vec0);
                sub_vec9 = _mm256_sub_ps(ts_vec2, tr_vec1);
                sub_vec10 = _mm256_sub_ps(ts_vec2, tr_vec2);
                sub_vec11 = _mm256_sub_ps(ts_vec2, tr_vec3);

                sub_vec12 = _mm256_sub_ps(ts_vec3, tr_vec0);
                sub_vec13 = _mm256_sub_ps(ts_vec3, tr_vec1);
                sub_vec14 = _mm256_sub_ps(ts_vec3, tr_vec2);
                sub_vec15 = _mm256_sub_ps(ts_vec3, tr_vec3);

                sub_vec0 = _mm256_mul_ps(sub_vec0, sub_vec0); // squared distance, 4x8
                sub_vec1 = _mm256_mul_ps(sub_vec1, sub_vec1);
                sub_vec2 = _mm256_mul_ps(sub_vec2, sub_vec2);
                sub_vec3 = _mm256_mul_ps(sub_vec3, sub_vec3);
                sub_vec4 = _mm256_mul_ps(sub_vec4, sub_vec4);
                sub_vec5 = _mm256_mul_ps(sub_vec5, sub_vec5);
                sub_vec6 = _mm256_mul_ps(sub_vec6, sub_vec6);
                sub_vec7 = _mm256_mul_ps(sub_vec7, sub_vec7);

                sub_vec8 = _mm256_mul_ps(sub_vec8, sub_vec8);
                sub_vec9 = _mm256_mul_ps(sub_vec9, sub_vec9);
                sub_vec10 = _mm256_mul_ps(sub_vec10, sub_vec10);
                sub_vec11 = _mm256_mul_ps(sub_vec11, sub_vec11);
                sub_vec12 = _mm256_mul_ps(sub_vec12, sub_vec12);
                sub_vec13 = _mm256_mul_ps(sub_vec13, sub_vec13);
                sub_vec14 = _mm256_mul_ps(sub_vec14, sub_vec14);
                sub_vec15 = _mm256_mul_ps(sub_vec15, sub_vec15);

                f0 += sum8(sub_vec0); // horizontal add on 4x8 -> 8 values (sums) to be stored.
                f1 += sum8(sub_vec1); // both sum8 and this one seem to work
                f2 += sum8(sub_vec2);
                f3 += sum8(sub_vec3);

                f4 += sum8(sub_vec4);
                f5 += sum8(sub_vec5);
                f6 += sum8(sub_vec6);
                f7 += sum8(sub_vec7);
                f8 += sum8(sub_vec8);
                f9 += sum8(sub_vec9);
                f10 += sum8(sub_vec10);
                f11 += sum8(sub_vec11);
                f12 += sum8(sub_vec12);
                f13 += sum8(sub_vec13);
                f14 += sum8(sub_vec14);
                f15 += sum8(sub_vec15);
            }

            dist_arr[bj].value = f0;
            dist_arr[bj].index = bj;
            dist_arr[bj + 1].value = f1;
            dist_arr[bj + 1].index = bj + 1;
            dist_arr[bj + 2].value = f2;
            dist_arr[bj + 2].index = bj + 2;
            dist_arr[bj + 3].value = f3;
            dist_arr[bj + 3].index = bj + 3;

            dist_arr_i[bj].value = f4;
            dist_arr_i[bj].index = bj;
            dist_arr_i[bj + 1].value = f5;
            dist_arr_i[bj + 1].index = bj + 1;
            dist_arr_i[bj + 2].value = f6;
            dist_arr_i[bj + 2].index = bj + 2;
            dist_arr_i[bj + 3].value = f7;
            dist_arr_i[bj + 3].index = bj + 3;

            dist_arr_ii[bj].value = f8;
            dist_arr_ii[bj].index = bj;
            dist_arr_ii[bj + 1].value = f9;
            dist_arr_ii[bj + 1].index = bj + 1;
            dist_arr_ii[bj + 2].value = f10;
            dist_arr_ii[bj + 2].index = bj + 2;
            dist_arr_ii[bj + 3].value = f11;
            dist_arr_ii[bj + 3].index = bj + 3;

            dist_arr_iii[bj].value = f12;
            dist_arr_iii[bj].index = bj;
            dist_arr_iii[bj + 1].value = f13;
            dist_arr_iii[bj + 1].index = bj + 1;
            dist_arr_iii[bj + 2].value = f14;
            dist_arr_iii[bj + 2].index = bj + 2;
            dist_arr_iii[bj + 3].value = f15;
            dist_arr_iii[bj + 3].index = bj + 3;
        }
        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_i, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iii, N, sizeof(pair_t), cmp);

#pragma GCC ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[bi * N + k] = dist_arr[k].index;
            x_tst_knn_gt->data[(bi + 1) * N + k] = dist_arr_i[k].index;
            x_tst_knn_gt->data[(bi + 2) * N + k] = dist_arr_ii[k].index;
            x_tst_knn_gt->data[(bi + 3) * N + k] = dist_arr_iii[k].index;
        }
    }

    free(dist_arr);
    free(dist_arr_i);
    free(dist_arr_ii);
    free(dist_arr_iii);

}

// 4x8 vectorized - horizontal sum inside + 4 AoS arrays + vector accumulators
void get_true_knn_opt15(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 4;

    pair_t *__restrict__  dist_arr = malloc(d * sizeof(pair_t));
    pair_t *__restrict__  dist_arr_i = malloc(d * sizeof(pair_t));
    pair_t *__restrict__  dist_arr_ii = malloc(d * sizeof(pair_t));
    pair_t *__restrict__  dist_arr_iii = malloc(d * sizeof(pair_t));

    float *__restrict__ data_trn = x_trn->data;
    float *__restrict__ data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;
    __m256 ts_vec0, tr_vec0, ts_vec1, tr_vec1, ts_vec2, tr_vec2, ts_vec3, tr_vec3;
    __m256 tmp0, tmp1, tmp2, tmp3;

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3, vec_sum4, vec_sum5, vec_sum6, vec_sum7;
    __m256 sub_vec0, sub_vec1, sub_vec2, sub_vec3, sub_vec4, sub_vec5, sub_vec6, sub_vec7;
    __m256 sub_vec8, sub_vec9, sub_vec10, sub_vec11, sub_vec12, sub_vec13, sub_vec14;
    __m256 sub_vec15, sub_vec16, sub_vec17, sub_vec18, sub_vec19, sub_vec20, sub_vec21;
    __m256 sub_vec22, sub_vec23, sub_vec24, sub_vec25, sub_vec26, sub_vec27, sub_vec28;
    __m256 sub_vec29, sub_vec30, sub_vec31, sub_vec32;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;

    for (bi = 0; bi < N_tst; bi += Nb) {
        for (bj = 0; bj < N; bj += Nb) {

            tmp0 = _mm256_setzero_ps(); // accumulators
            tmp1 = _mm256_setzero_ps();
            tmp2 = _mm256_setzero_ps();
            tmp3 = _mm256_setzero_ps();

            for (bk = 0; bk < d; bk += 8) {

                ts_vec0 = _mm256_load_ps(data_tst + bi * N + bk); // 4x8 blocks from test matrix
                ts_vec1 = _mm256_load_ps(data_tst + (bi + 1) * N + bk);
                ts_vec2 = _mm256_load_ps(data_tst + (bi + 2) * N + bk);
                ts_vec3 = _mm256_load_ps(data_tst + (bi + 3) * N + bk);

                tr_vec0 = _mm256_load_ps(data_trn + bj * N + bk); // 4x8 blocks from train matrix
                tr_vec1 = _mm256_load_ps(data_trn + (bj + 1) * N + bk);
                tr_vec2 = _mm256_load_ps(data_trn + (bj + 2) * N + bk);
                tr_vec3 = _mm256_load_ps(data_trn + (bj + 3) * N + bk);

                sub_vec0 = _mm256_sub_ps(ts_vec0, tr_vec0); // pairwise distances between test and train blocks
                sub_vec1 = _mm256_sub_ps(ts_vec0, tr_vec1); // 1x4 per code block, total 4x8 values computed
                sub_vec2 = _mm256_sub_ps(ts_vec0, tr_vec2);
                sub_vec3 = _mm256_sub_ps(ts_vec0, tr_vec3);

                sub_vec4 = _mm256_sub_ps(ts_vec1, tr_vec0);
                sub_vec5 = _mm256_sub_ps(ts_vec1, tr_vec1);
                sub_vec6 = _mm256_sub_ps(ts_vec1, tr_vec2);
                sub_vec7 = _mm256_sub_ps(ts_vec1, tr_vec3);

                sub_vec8 = _mm256_sub_ps(ts_vec2, tr_vec0);
                sub_vec9 = _mm256_sub_ps(ts_vec2, tr_vec1);
                sub_vec10 = _mm256_sub_ps(ts_vec2, tr_vec2);
                sub_vec11 = _mm256_sub_ps(ts_vec2, tr_vec3);

                sub_vec12 = _mm256_sub_ps(ts_vec3, tr_vec0);
                sub_vec13 = _mm256_sub_ps(ts_vec3, tr_vec1);
                sub_vec14 = _mm256_sub_ps(ts_vec3, tr_vec2);
                sub_vec15 = _mm256_sub_ps(ts_vec3, tr_vec3);

                sub_vec0 = _mm256_mul_ps(sub_vec0, sub_vec0); // squared distance, 4x8
                sub_vec1 = _mm256_mul_ps(sub_vec1, sub_vec1);
                sub_vec2 = _mm256_mul_ps(sub_vec2, sub_vec2);
                sub_vec3 = _mm256_mul_ps(sub_vec3, sub_vec3);
                sub_vec4 = _mm256_mul_ps(sub_vec4, sub_vec4);
                sub_vec5 = _mm256_mul_ps(sub_vec5, sub_vec5);
                sub_vec6 = _mm256_mul_ps(sub_vec6, sub_vec6);
                sub_vec7 = _mm256_mul_ps(sub_vec7, sub_vec7);

                sub_vec8 = _mm256_mul_ps(sub_vec8, sub_vec8);
                sub_vec9 = _mm256_mul_ps(sub_vec9, sub_vec9);
                sub_vec10 = _mm256_mul_ps(sub_vec10, sub_vec10);
                sub_vec11 = _mm256_mul_ps(sub_vec11, sub_vec11);
                sub_vec12 = _mm256_mul_ps(sub_vec12, sub_vec12);
                sub_vec13 = _mm256_mul_ps(sub_vec13, sub_vec13);
                sub_vec14 = _mm256_mul_ps(sub_vec14, sub_vec14);
                sub_vec15 = _mm256_mul_ps(sub_vec15, sub_vec15);

                f0 = sum8(sub_vec0); // horizontal add on 4x8 -> 8 values (sums) to be stored.
                f1 = sum8(sub_vec1); // both sum8 and this one seem to work
                f2 = sum8(sub_vec2);
                f3 = sum8(sub_vec3);

                f4 = sum8(sub_vec4);
                f5 = sum8(sub_vec5);
                f6 = sum8(sub_vec6);
                f7 = sum8(sub_vec7);
                f8 = sum8(sub_vec8);
                f9 = sum8(sub_vec9);
                f10 = sum8(sub_vec10);
                f11 = sum8(sub_vec11);
                f12 = sum8(sub_vec12);
                f13 = sum8(sub_vec13);
                f14 = sum8(sub_vec14);
                f15 = sum8(sub_vec15);

                vec_sum0 = _mm256_set_ps(0, 0, 0, 0, f3, f2, f1, f0);
                vec_sum1 = _mm256_set_ps(0, 0, 0, 0, f7, f6, f5, f4);
                vec_sum2 = _mm256_set_ps(0, 0, 0, 0, f11, f10, f9, f8);
                vec_sum3 = _mm256_set_ps(0, 0, 0, 0, f15, f14, f13, f12);

                tmp0 = _mm256_add_ps(tmp0, vec_sum0); // accumulate
                tmp1 = _mm256_add_ps(tmp1, vec_sum1); // accumulate
                tmp2 = _mm256_add_ps(tmp2, vec_sum2); // accumulate
                tmp3 = _mm256_add_ps(tmp3, vec_sum3); // accumulate

            }

            for (i = 0; i < 4; i++) { // this should get vectorized
                dist_arr[bj + i].value = tmp0[i];
                dist_arr[bj + i].index = bj + i;

                dist_arr_i[bj + i].value = tmp1[i];
                dist_arr_i[bj + i].index = bj + i;

                dist_arr_ii[bj + i].value = tmp2[i];
                dist_arr_ii[bj + i].index = bj + i;

                dist_arr_iii[bj + i].value = tmp3[i];
                dist_arr_iii[bj + i].index = bj + i;
            }
        }
        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_i, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iii, N, sizeof(pair_t), cmp);

#pragma GCC ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[bi * N + k] = dist_arr[k].index;
            x_tst_knn_gt->data[(bi + 1) * N + k] = dist_arr_i[k].index;
            x_tst_knn_gt->data[(bi + 2) * N + k] = dist_arr_ii[k].index;
            x_tst_knn_gt->data[(bi + 3) * N + k] = dist_arr_iii[k].index;
        }
    }

    free(dist_arr);
    free(dist_arr_i);
    free(dist_arr_ii);
    free(dist_arr_iii);

}

// 4x8 vectorized - horizontal sum outside + 4 AoS arrays
void get_true_knn_opt16(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 4;

    pair_t *__restrict__  dist_arr = malloc(d * sizeof(pair_t));
    pair_t *__restrict__  dist_arr_i = malloc(d * sizeof(pair_t));
    pair_t *__restrict__  dist_arr_ii = malloc(d * sizeof(pair_t));
    pair_t *__restrict__  dist_arr_iii = malloc(d * sizeof(pair_t));

    float *__restrict__ data_trn = x_trn->data;
    float *__restrict__ data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;
    __m256 ts_vec0, tr_vec0, ts_vec1, tr_vec1, ts_vec2, tr_vec2, ts_vec3, tr_vec3;
    __m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11, acc12, acc13, acc14, acc15;

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3, vec_sum4, vec_sum5, vec_sum6, vec_sum7;
    __m256 sub_vec0, sub_vec1, sub_vec2, sub_vec3, sub_vec4, sub_vec5, sub_vec6, sub_vec7;
    __m256 sub_vec8, sub_vec9, sub_vec10, sub_vec11, sub_vec12, sub_vec13, sub_vec14;
    __m256 sub_vec15, sub_vec16, sub_vec17, sub_vec18, sub_vec19, sub_vec20, sub_vec21;
    __m256 sub_vec22, sub_vec23, sub_vec24, sub_vec25, sub_vec26, sub_vec27, sub_vec28;
    __m256 sub_vec29, sub_vec30, sub_vec31, sub_vec32;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;

    for (bi = 0; bi < N_tst; bi += Nb) {
        for (bj = 0; bj < N; bj += Nb) {

            for (i = 0; i < 4; i++) { // this should get vectorized
                dist_arr[bj + i].index = bj + i;
                dist_arr_i[bj + i].index = bj + i;
                dist_arr_ii[bj + i].index = bj + i;
                dist_arr_iii[bj + i].index = bj + i;
            }

            acc0 = _mm256_setzero_ps(); // accumulators
            acc1 = _mm256_setzero_ps();
            acc2 = _mm256_setzero_ps();
            acc3 = _mm256_setzero_ps();

            acc4 = _mm256_setzero_ps(); // accumulators
            acc5 = _mm256_setzero_ps();
            acc6 = _mm256_setzero_ps();
            acc7 = _mm256_setzero_ps();

            acc8 = _mm256_setzero_ps(); // accumulators
            acc9 = _mm256_setzero_ps();
            acc10 = _mm256_setzero_ps();
            acc11 = _mm256_setzero_ps();

            acc12 = _mm256_setzero_ps(); // accumulators
            acc13 = _mm256_setzero_ps();
            acc14 = _mm256_setzero_ps();
            acc15 = _mm256_setzero_ps();
            
            for (bk = 0; bk < d; bk += 8) {

                ts_vec0 = _mm256_load_ps(data_tst + bi * N + bk); // 4x8 blocks from test matrix
                ts_vec1 = _mm256_load_ps(data_tst + (bi + 1) * N + bk);
                ts_vec2 = _mm256_load_ps(data_tst + (bi + 2) * N + bk);
                ts_vec3 = _mm256_load_ps(data_tst + (bi + 3) * N + bk);

                tr_vec0 = _mm256_load_ps(data_trn + bj * N + bk); // 4x8 blocks from train matrix
                tr_vec1 = _mm256_load_ps(data_trn + (bj + 1) * N + bk);
                tr_vec2 = _mm256_load_ps(data_trn + (bj + 2) * N + bk);
                tr_vec3 = _mm256_load_ps(data_trn + (bj + 3) * N + bk);

                sub_vec0 = _mm256_sub_ps(ts_vec0, tr_vec0); // pairwise distances between test and train blocks
                sub_vec1 = _mm256_sub_ps(ts_vec0, tr_vec1); // 1x4 per code block, total 4x8 values computed
                sub_vec2 = _mm256_sub_ps(ts_vec0, tr_vec2);
                sub_vec3 = _mm256_sub_ps(ts_vec0, tr_vec3);

                sub_vec4 = _mm256_sub_ps(ts_vec1, tr_vec0);
                sub_vec5 = _mm256_sub_ps(ts_vec1, tr_vec1);
                sub_vec6 = _mm256_sub_ps(ts_vec1, tr_vec2);
                sub_vec7 = _mm256_sub_ps(ts_vec1, tr_vec3);

                sub_vec8 = _mm256_sub_ps(ts_vec2, tr_vec0);
                sub_vec9 = _mm256_sub_ps(ts_vec2, tr_vec1);
                sub_vec10 = _mm256_sub_ps(ts_vec2, tr_vec2);
                sub_vec11 = _mm256_sub_ps(ts_vec2, tr_vec3);

                sub_vec12 = _mm256_sub_ps(ts_vec3, tr_vec0);
                sub_vec13 = _mm256_sub_ps(ts_vec3, tr_vec1);
                sub_vec14 = _mm256_sub_ps(ts_vec3, tr_vec2);
                sub_vec15 = _mm256_sub_ps(ts_vec3, tr_vec3);

                sub_vec0 = _mm256_mul_ps(sub_vec0, sub_vec0); // squared distance, 4x8
                sub_vec1 = _mm256_mul_ps(sub_vec1, sub_vec1);
                sub_vec2 = _mm256_mul_ps(sub_vec2, sub_vec2);
                sub_vec3 = _mm256_mul_ps(sub_vec3, sub_vec3);
                sub_vec4 = _mm256_mul_ps(sub_vec4, sub_vec4);
                sub_vec5 = _mm256_mul_ps(sub_vec5, sub_vec5);
                sub_vec6 = _mm256_mul_ps(sub_vec6, sub_vec6);
                sub_vec7 = _mm256_mul_ps(sub_vec7, sub_vec7);

                sub_vec8 = _mm256_mul_ps(sub_vec8, sub_vec8);
                sub_vec9 = _mm256_mul_ps(sub_vec9, sub_vec9);
                sub_vec10 = _mm256_mul_ps(sub_vec10, sub_vec10);
                sub_vec11 = _mm256_mul_ps(sub_vec11, sub_vec11);
                sub_vec12 = _mm256_mul_ps(sub_vec12, sub_vec12);
                sub_vec13 = _mm256_mul_ps(sub_vec13, sub_vec13);
                sub_vec14 = _mm256_mul_ps(sub_vec14, sub_vec14);
                sub_vec15 = _mm256_mul_ps(sub_vec15, sub_vec15);

                acc0 = _mm256_add_ps(acc0, sub_vec0); // accumulate
                acc1 = _mm256_add_ps(acc1, sub_vec1);
                acc2 = _mm256_add_ps(acc2, sub_vec2);
                acc3 = _mm256_add_ps(acc3, sub_vec3);

                acc4 = _mm256_add_ps(acc4, sub_vec4);
                acc5 = _mm256_add_ps(acc5, sub_vec5);
                acc6 = _mm256_add_ps(acc6, sub_vec6);
                acc7 = _mm256_add_ps(acc7, sub_vec7);

                acc8 = _mm256_add_ps(acc8, sub_vec8);
                acc9 = _mm256_add_ps(acc9, sub_vec9);
                acc10 = _mm256_add_ps(acc10, sub_vec10);
                acc11 = _mm256_add_ps(acc11, sub_vec11);

                acc12 = _mm256_add_ps(acc12, sub_vec12);
                acc13 = _mm256_add_ps(acc13, sub_vec13);
                acc14 = _mm256_add_ps(acc14, sub_vec14);
                acc15 = _mm256_add_ps(acc15, sub_vec15);
            }

            dist_arr[bj].value = sum8(acc0); // horizontal add on 4x8 -> 8 values (sums) to be stored.
            dist_arr[bj + 1].value = sum8(acc1); // both sum8 and this one seem to work
            dist_arr[bj + 2].value = sum8(acc2);
            dist_arr[bj + 3].value = sum8(acc3);

            dist_arr_i[bj].value = sum8(acc4);
            dist_arr_i[bj + 1].value = sum8(acc5);
            dist_arr_i[bj + 2].value = sum8(acc6);
            dist_arr_i[bj + 3].value = sum8(acc7);

            dist_arr_ii[bj].value = sum8(acc8);
            dist_arr_ii[bj + 1].value = sum8(acc9);
            dist_arr_ii[bj + 2].value = sum8(acc10);
            dist_arr_ii[bj + 3].value = sum8(acc11);

            dist_arr_iii[bj].value = sum8(acc12);
            dist_arr_iii[bj + 1].value = sum8(acc13);
            dist_arr_iii[bj + 2].value = sum8(acc14);
            dist_arr_iii[bj + 3].value = sum8(acc15);

        }
        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_i, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iii, N, sizeof(pair_t), cmp);

#pragma GCC ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[bi * N + k] = dist_arr[k].index;
            x_tst_knn_gt->data[(bi + 1) * N + k] = dist_arr_i[k].index;
            x_tst_knn_gt->data[(bi + 2) * N + k] = dist_arr_ii[k].index;
            x_tst_knn_gt->data[(bi + 3) * N + k] = dist_arr_iii[k].index;
        }
    }

    free(dist_arr);
    free(dist_arr_i);
    free(dist_arr_ii);
    free(dist_arr_iii);

}

// 4x8 vectorized - horizontal sum outside + fmas + 4 AoS arrays
void get_true_knn_opt17(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 4;

    pair_t *__restrict__  dist_arr = malloc(d * sizeof(pair_t));
    pair_t *__restrict__  dist_arr_i = malloc(d * sizeof(pair_t));
    pair_t *__restrict__  dist_arr_ii = malloc(d * sizeof(pair_t));
    pair_t *__restrict__  dist_arr_iii = malloc(d * sizeof(pair_t));

    float *__restrict__ data_trn = x_trn->data;
    float *__restrict__ data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;
    __m256 ts_vec0, tr_vec0, ts_vec1, tr_vec1, ts_vec2, tr_vec2, ts_vec3, tr_vec3;
    __m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11, acc12, acc13, acc14, acc15;

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3, vec_sum4, vec_sum5, vec_sum6, vec_sum7;
    __m256 sub_vec0, sub_vec1, sub_vec2, sub_vec3, sub_vec4, sub_vec5, sub_vec6, sub_vec7;
    __m256 sub_vec8, sub_vec9, sub_vec10, sub_vec11, sub_vec12, sub_vec13, sub_vec14;
    __m256 sub_vec15, sub_vec16, sub_vec17, sub_vec18, sub_vec19, sub_vec20, sub_vec21;
    __m256 sub_vec22, sub_vec23, sub_vec24, sub_vec25, sub_vec26, sub_vec27, sub_vec28;
    __m256 sub_vec29, sub_vec30, sub_vec31, sub_vec32;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;

    for (bi = 0; bi < N_tst; bi += Nb) {
        for (bj = 0; bj < N; bj += Nb) {

            for (i = 0; i < 4; i++) { // this should get vectorized
                dist_arr[bj + i].index = bj + i;
                dist_arr_i[bj + i].index = bj + i;
                dist_arr_ii[bj + i].index = bj + i;
                dist_arr_iii[bj + i].index = bj + i;
            }

            acc0 = _mm256_setzero_ps(); // accumulators
            acc1 = _mm256_setzero_ps();
            acc2 = _mm256_setzero_ps();
            acc3 = _mm256_setzero_ps();

            acc4 = _mm256_setzero_ps(); // accumulators
            acc5 = _mm256_setzero_ps();
            acc6 = _mm256_setzero_ps();
            acc7 = _mm256_setzero_ps();

            acc8 = _mm256_setzero_ps(); // accumulators
            acc9 = _mm256_setzero_ps();
            acc10 = _mm256_setzero_ps();
            acc11 = _mm256_setzero_ps();

            acc12 = _mm256_setzero_ps(); // accumulators
            acc13 = _mm256_setzero_ps();
            acc14 = _mm256_setzero_ps();
            acc15 = _mm256_setzero_ps();

            for (bk = 0; bk < d; bk += 8) {

                ts_vec0 = _mm256_load_ps(data_tst + bi * N + bk); // 4x8 blocks from test matrix
                ts_vec1 = _mm256_load_ps(data_tst + (bi + 1) * N + bk);
                ts_vec2 = _mm256_load_ps(data_tst + (bi + 2) * N + bk);
                ts_vec3 = _mm256_load_ps(data_tst + (bi + 3) * N + bk);

                tr_vec0 = _mm256_load_ps(data_trn + bj * N + bk); // 4x8 blocks from train matrix
                tr_vec1 = _mm256_load_ps(data_trn + (bj + 1) * N + bk);
                tr_vec2 = _mm256_load_ps(data_trn + (bj + 2) * N + bk);
                tr_vec3 = _mm256_load_ps(data_trn + (bj + 3) * N + bk);

                sub_vec0 = _mm256_sub_ps(ts_vec0, tr_vec0); // pairwise distances between test and train blocks
                sub_vec1 = _mm256_sub_ps(ts_vec0, tr_vec1); // 1x4 per code block, total 4x8 values computed
                sub_vec2 = _mm256_sub_ps(ts_vec0, tr_vec2);
                sub_vec3 = _mm256_sub_ps(ts_vec0, tr_vec3);

                sub_vec4 = _mm256_sub_ps(ts_vec1, tr_vec0);
                sub_vec5 = _mm256_sub_ps(ts_vec1, tr_vec1);
                sub_vec6 = _mm256_sub_ps(ts_vec1, tr_vec2);
                sub_vec7 = _mm256_sub_ps(ts_vec1, tr_vec3);

                sub_vec8 = _mm256_sub_ps(ts_vec2, tr_vec0);
                sub_vec9 = _mm256_sub_ps(ts_vec2, tr_vec1);
                sub_vec10 = _mm256_sub_ps(ts_vec2, tr_vec2);
                sub_vec11 = _mm256_sub_ps(ts_vec2, tr_vec3);

                sub_vec12 = _mm256_sub_ps(ts_vec3, tr_vec0);
                sub_vec13 = _mm256_sub_ps(ts_vec3, tr_vec1);
                sub_vec14 = _mm256_sub_ps(ts_vec3, tr_vec2);
                sub_vec15 = _mm256_sub_ps(ts_vec3, tr_vec3);

                acc0 = _mm256_fmadd_ps(sub_vec0, sub_vec0, acc0); // squared distance, 4x8
                acc1 = _mm256_fmadd_ps(sub_vec1, sub_vec1, acc1);
                acc2 = _mm256_fmadd_ps(sub_vec2, sub_vec2, acc2);
                acc3 = _mm256_fmadd_ps(sub_vec3, sub_vec3, acc3);
                acc4 = _mm256_fmadd_ps(sub_vec4, sub_vec4, acc4);
                acc5 = _mm256_fmadd_ps(sub_vec5, sub_vec5, acc5);
                acc6 = _mm256_fmadd_ps(sub_vec6, sub_vec6, acc6);
                acc7 = _mm256_fmadd_ps(sub_vec7, sub_vec7, acc7);

                acc8 = _mm256_fmadd_ps(sub_vec8, sub_vec8, acc8);
                acc9 = _mm256_fmadd_ps(sub_vec9, sub_vec9, acc9);
                acc10 = _mm256_fmadd_ps(sub_vec10, sub_vec10, acc10);
                acc11 = _mm256_fmadd_ps(sub_vec11, sub_vec11, acc11);
                acc12 = _mm256_fmadd_ps(sub_vec12, sub_vec12, acc12);
                acc13 = _mm256_fmadd_ps(sub_vec13, sub_vec13, acc13);
                acc14 = _mm256_fmadd_ps(sub_vec14, sub_vec14, acc14);
                acc15 = _mm256_fmadd_ps(sub_vec15, sub_vec15, acc15);

            }

            dist_arr[bj].value = sum8(acc0); // horizontal add on 4x8 -> 8 values (sums) to be stored.
            dist_arr[bj + 1].value = sum8(acc1); // both sum8 and this one seem to work
            dist_arr[bj + 2].value = sum8(acc2);
            dist_arr[bj + 3].value = sum8(acc3);

            dist_arr_i[bj].value = sum8(acc4);
            dist_arr_i[bj + 1].value = sum8(acc5);
            dist_arr_i[bj + 2].value = sum8(acc6);
            dist_arr_i[bj + 3].value = sum8(acc7);

            dist_arr_ii[bj].value = sum8(acc8);
            dist_arr_ii[bj + 1].value = sum8(acc9);
            dist_arr_ii[bj + 2].value = sum8(acc10);
            dist_arr_ii[bj + 3].value = sum8(acc11);

            dist_arr_iii[bj].value = sum8(acc12);
            dist_arr_iii[bj + 1].value = sum8(acc13);
            dist_arr_iii[bj + 2].value = sum8(acc14);
            dist_arr_iii[bj + 3].value = sum8(acc15);

        }
        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_i, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iii, N, sizeof(pair_t), cmp);

#pragma GCC ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[bi * N + k] = dist_arr[k].index;
            x_tst_knn_gt->data[(bi + 1) * N + k] = dist_arr_i[k].index;
            x_tst_knn_gt->data[(bi + 2) * N + k] = dist_arr_ii[k].index;
            x_tst_knn_gt->data[(bi + 3) * N + k] = dist_arr_iii[k].index;
        }
    }

    free(dist_arr);
    free(dist_arr_i);
    free(dist_arr_ii);
    free(dist_arr_iii);

}

// naive vectorized blocking 8x8
void get_true_knn_opt18(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 8;
    pair_t *dist_arr, *dist_arr_i, *dist_arr_ii, *dist_arr_iii;
    dist_arr = malloc(N * sizeof(pair_t));
    dist_arr_i = malloc(N * sizeof(pair_t));
    dist_arr_ii = malloc(N * sizeof(pair_t));
    dist_arr_iii = malloc(N * sizeof(pair_t));

    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;
    __m256 ts_vec0, tr_vec0, ts_vec1, tr_vec1, ts_vec2, tr_vec2, ts_vec3, tr_vec3, ts_vec4, tr_vec4, ts_vec5, tr_vec5, ts_vec6, tr_vec6, ts_vec7, tr_vec7;
    __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3, vec_sum4, vec_sum5, vec_sum6, vec_sum7;
    __m256 sub_vec0, sub_vec1, sub_vec2, sub_vec3, sub_vec4, sub_vec5, sub_vec6, sub_vec7;
    __m256 sub_vec8, sub_vec9, sub_vec10, sub_vec11, sub_vec12, sub_vec13, sub_vec14;
    __m256 sub_vec15, sub_vec16, sub_vec17, sub_vec18, sub_vec19, sub_vec20, sub_vec21;
    __m256 sub_vec22, sub_vec23, sub_vec24, sub_vec25, sub_vec26, sub_vec27, sub_vec28;
    __m256 sub_vec29, sub_vec30, sub_vec31, sub_vec32, sub_vec33, sub_vec34, sub_vec35;
    __m256 sub_vec36, sub_vec37, sub_vec38, sub_vec39, sub_vec40, sub_vec41, sub_vec42;
    __m256 sub_vec43, sub_vec44, sub_vec45, sub_vec46, sub_vec47, sub_vec48, sub_vec49;
    __m256 sub_vec50, sub_vec51, sub_vec52, sub_vec53, sub_vec54, sub_vec55, sub_vec56;
    __m256 sub_vec57, sub_vec58, sub_vec59, sub_vec60, sub_vec61, sub_vec62, sub_vec63;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;
    float f16, f17, f18, f19, f20, f21, f22, f23;
    float f24, f25, f26, f27, f28, f29, f30, f31;
    float f32, f33, f34, f35, f36, f37, f38, f39;
    float f40, f41, f42, f43, f44, f45, f46, f47;
    float f48, f49, f50, f51, f52, f53, f54, f55;
    float f56, f57, f58, f59, f60, f61, f62, f63;


    for (bi = 0; bi < N_tst; bi += Nb) {
        for (bj = 0; bj < N; bj += Nb) {
            for (bk = 0; bk < d; bk += Nb) {
                tmp0 = _mm256_load_ps(distances.data + bi * N + bj); // 8x8 blocks temp values for l2-norm
                tmp1 = _mm256_load_ps(distances.data + (bi + 1) * N + bj);
                tmp2 = _mm256_load_ps(distances.data + (bi + 2) * N + bj);
                tmp3 = _mm256_load_ps(distances.data + (bi + 3) * N + bj);
                tmp4 = _mm256_load_ps(distances.data + (bi + 4) * N + bj);
                tmp5 = _mm256_load_ps(distances.data + (bi + 5) * N + bj);
                tmp6 = _mm256_load_ps(distances.data + (bi + 6) * N + bj);
                tmp7 = _mm256_load_ps(distances.data + (bi + 7) * N + bj);

                ts_vec0 = _mm256_load_ps(data_tst + bi * N + bk); // 8x8 blocks from test matrix
                ts_vec1 = _mm256_load_ps(data_tst + (bi + 1) * N + bk);
                ts_vec2 = _mm256_load_ps(data_tst + (bi + 2) * N + bk);
                ts_vec3 = _mm256_load_ps(data_tst + (bi + 3) * N + bk);
                ts_vec4 = _mm256_load_ps(data_tst + (bi + 4) * N + bk);
                ts_vec5 = _mm256_load_ps(data_tst + (bi + 5) * N + bk);
                ts_vec6 = _mm256_load_ps(data_tst + (bi + 6) * N + bk);
                ts_vec7 = _mm256_load_ps(data_tst + (bi + 7) * N + bk);

                tr_vec0 = _mm256_load_ps(data_trn + bj * N + bk); // 8x8 blocks from train matrix
                tr_vec1 = _mm256_load_ps(data_trn + (bj + 1) * N + bk);
                tr_vec2 = _mm256_load_ps(data_trn + (bj + 2) * N + bk);
                tr_vec3 = _mm256_load_ps(data_trn + (bj + 3) * N + bk);
                tr_vec4 = _mm256_load_ps(data_trn + (bj + 4) * N + bk);
                tr_vec5 = _mm256_load_ps(data_trn + (bj + 5) * N + bk);
                tr_vec6 = _mm256_load_ps(data_trn + (bj + 6) * N + bk);
                tr_vec7 = _mm256_load_ps(data_trn + (bj + 7) * N + bk);

                sub_vec0 = _mm256_sub_ps(ts_vec0, tr_vec0); // pairwise distances between test and train blocks
                sub_vec1 = _mm256_sub_ps(ts_vec0, tr_vec1); // 1x8 per code block, total 8x8 values computed
                sub_vec2 = _mm256_sub_ps(ts_vec0, tr_vec2);
                sub_vec3 = _mm256_sub_ps(ts_vec0, tr_vec3);
                sub_vec4 = _mm256_sub_ps(ts_vec0, tr_vec4);
                sub_vec5 = _mm256_sub_ps(ts_vec0, tr_vec5);
                sub_vec6 = _mm256_sub_ps(ts_vec0, tr_vec6);
                sub_vec7 = _mm256_sub_ps(ts_vec0, tr_vec7);

                sub_vec8 = _mm256_sub_ps(ts_vec1, tr_vec0);
                sub_vec9 = _mm256_sub_ps(ts_vec1, tr_vec1);
                sub_vec10 = _mm256_sub_ps(ts_vec1, tr_vec2);
                sub_vec11 = _mm256_sub_ps(ts_vec1, tr_vec3);
                sub_vec12 = _mm256_sub_ps(ts_vec1, tr_vec4);
                sub_vec13 = _mm256_sub_ps(ts_vec1, tr_vec5);
                sub_vec14 = _mm256_sub_ps(ts_vec1, tr_vec6);
                sub_vec15 = _mm256_sub_ps(ts_vec1, tr_vec7);

                sub_vec16 = _mm256_sub_ps(ts_vec2, tr_vec0);
                sub_vec17 = _mm256_sub_ps(ts_vec2, tr_vec1);
                sub_vec18 = _mm256_sub_ps(ts_vec2, tr_vec2);
                sub_vec19 = _mm256_sub_ps(ts_vec2, tr_vec3);
                sub_vec20 = _mm256_sub_ps(ts_vec2, tr_vec4);
                sub_vec21 = _mm256_sub_ps(ts_vec2, tr_vec5);
                sub_vec22 = _mm256_sub_ps(ts_vec2, tr_vec6);
                sub_vec23 = _mm256_sub_ps(ts_vec2, tr_vec7);

                sub_vec24 = _mm256_sub_ps(ts_vec3, tr_vec0);
                sub_vec25 = _mm256_sub_ps(ts_vec3, tr_vec1);
                sub_vec26 = _mm256_sub_ps(ts_vec3, tr_vec2);
                sub_vec27 = _mm256_sub_ps(ts_vec3, tr_vec3);
                sub_vec28 = _mm256_sub_ps(ts_vec3, tr_vec4);
                sub_vec29 = _mm256_sub_ps(ts_vec3, tr_vec5);
                sub_vec30 = _mm256_sub_ps(ts_vec3, tr_vec6);
                sub_vec31 = _mm256_sub_ps(ts_vec3, tr_vec7);

                sub_vec32 = _mm256_sub_ps(ts_vec4, tr_vec0);
                sub_vec33 = _mm256_sub_ps(ts_vec4, tr_vec1);
                sub_vec34 = _mm256_sub_ps(ts_vec4, tr_vec2);
                sub_vec35 = _mm256_sub_ps(ts_vec4, tr_vec3);
                sub_vec36 = _mm256_sub_ps(ts_vec4, tr_vec4);
                sub_vec37 = _mm256_sub_ps(ts_vec4, tr_vec5);
                sub_vec38 = _mm256_sub_ps(ts_vec4, tr_vec6);
                sub_vec39 = _mm256_sub_ps(ts_vec4, tr_vec7);

                sub_vec40 = _mm256_sub_ps(ts_vec5, tr_vec0);
                sub_vec41 = _mm256_sub_ps(ts_vec5, tr_vec1);
                sub_vec42 = _mm256_sub_ps(ts_vec5, tr_vec2);
                sub_vec43 = _mm256_sub_ps(ts_vec5, tr_vec3);
                sub_vec44 = _mm256_sub_ps(ts_vec5, tr_vec4);
                sub_vec45 = _mm256_sub_ps(ts_vec5, tr_vec5);
                sub_vec46 = _mm256_sub_ps(ts_vec5, tr_vec6);
                sub_vec47 = _mm256_sub_ps(ts_vec5, tr_vec7);

                sub_vec48 = _mm256_sub_ps(ts_vec6, tr_vec0);
                sub_vec49 = _mm256_sub_ps(ts_vec6, tr_vec1);
                sub_vec50 = _mm256_sub_ps(ts_vec6, tr_vec2);
                sub_vec51 = _mm256_sub_ps(ts_vec6, tr_vec3);
                sub_vec52 = _mm256_sub_ps(ts_vec6, tr_vec4);
                sub_vec53 = _mm256_sub_ps(ts_vec6, tr_vec5);
                sub_vec54 = _mm256_sub_ps(ts_vec6, tr_vec6);
                sub_vec55 = _mm256_sub_ps(ts_vec6, tr_vec7);

                sub_vec56 = _mm256_sub_ps(ts_vec7, tr_vec0);
                sub_vec57 = _mm256_sub_ps(ts_vec7, tr_vec1);
                sub_vec58 = _mm256_sub_ps(ts_vec7, tr_vec2);
                sub_vec59 = _mm256_sub_ps(ts_vec7, tr_vec3);
                sub_vec60 = _mm256_sub_ps(ts_vec7, tr_vec4);
                sub_vec61 = _mm256_sub_ps(ts_vec7, tr_vec5);
                sub_vec62 = _mm256_sub_ps(ts_vec7, tr_vec6);
                sub_vec63 = _mm256_sub_ps(ts_vec7, tr_vec7);

                sub_vec0 = _mm256_mul_ps(sub_vec0, sub_vec0); // squared distance, 8x8
                sub_vec1 = _mm256_mul_ps(sub_vec1, sub_vec1);
                sub_vec2 = _mm256_mul_ps(sub_vec2, sub_vec2);
                sub_vec3 = _mm256_mul_ps(sub_vec3, sub_vec3);
                sub_vec4 = _mm256_mul_ps(sub_vec4, sub_vec4);
                sub_vec5 = _mm256_mul_ps(sub_vec5, sub_vec5);
                sub_vec6 = _mm256_mul_ps(sub_vec6, sub_vec6);
                sub_vec7 = _mm256_mul_ps(sub_vec7, sub_vec7);

                sub_vec8 = _mm256_mul_ps(sub_vec8, sub_vec8);
                sub_vec9 = _mm256_mul_ps(sub_vec9, sub_vec9);
                sub_vec10 = _mm256_mul_ps(sub_vec10, sub_vec10);
                sub_vec11 = _mm256_mul_ps(sub_vec11, sub_vec11);
                sub_vec12 = _mm256_mul_ps(sub_vec12, sub_vec12);
                sub_vec13 = _mm256_mul_ps(sub_vec13, sub_vec13);
                sub_vec14 = _mm256_mul_ps(sub_vec14, sub_vec14);
                sub_vec15 = _mm256_mul_ps(sub_vec15, sub_vec15);

                sub_vec16 = _mm256_mul_ps(sub_vec16, sub_vec16);
                sub_vec17 = _mm256_mul_ps(sub_vec17, sub_vec17);
                sub_vec18 = _mm256_mul_ps(sub_vec18, sub_vec18);
                sub_vec19 = _mm256_mul_ps(sub_vec19, sub_vec19);
                sub_vec20 = _mm256_mul_ps(sub_vec20, sub_vec20);
                sub_vec21 = _mm256_mul_ps(sub_vec21, sub_vec21);
                sub_vec22 = _mm256_mul_ps(sub_vec22, sub_vec22);
                sub_vec23 = _mm256_mul_ps(sub_vec23, sub_vec23);

                sub_vec24 = _mm256_mul_ps(sub_vec24, sub_vec24);
                sub_vec25 = _mm256_mul_ps(sub_vec25, sub_vec25);
                sub_vec26 = _mm256_mul_ps(sub_vec26, sub_vec26);
                sub_vec27 = _mm256_mul_ps(sub_vec27, sub_vec27);
                sub_vec28 = _mm256_mul_ps(sub_vec28, sub_vec28);
                sub_vec29 = _mm256_mul_ps(sub_vec29, sub_vec29);
                sub_vec30 = _mm256_mul_ps(sub_vec30, sub_vec30);
                sub_vec31 = _mm256_mul_ps(sub_vec31, sub_vec31);

                sub_vec32 = _mm256_mul_ps(sub_vec32, sub_vec32);
                sub_vec33 = _mm256_mul_ps(sub_vec33, sub_vec33);
                sub_vec34 = _mm256_mul_ps(sub_vec34, sub_vec34);
                sub_vec35 = _mm256_mul_ps(sub_vec35, sub_vec35);
                sub_vec36 = _mm256_mul_ps(sub_vec36, sub_vec36);
                sub_vec37 = _mm256_mul_ps(sub_vec37, sub_vec37);
                sub_vec38 = _mm256_mul_ps(sub_vec38, sub_vec38);
                sub_vec39 = _mm256_mul_ps(sub_vec39, sub_vec39);

                sub_vec40 = _mm256_mul_ps(sub_vec40, sub_vec40);
                sub_vec41 = _mm256_mul_ps(sub_vec41, sub_vec41);
                sub_vec42 = _mm256_mul_ps(sub_vec42, sub_vec42);
                sub_vec43 = _mm256_mul_ps(sub_vec43, sub_vec43);
                sub_vec44 = _mm256_mul_ps(sub_vec44, sub_vec44);
                sub_vec45 = _mm256_mul_ps(sub_vec45, sub_vec45);
                sub_vec46 = _mm256_mul_ps(sub_vec46, sub_vec46);
                sub_vec47 = _mm256_mul_ps(sub_vec47, sub_vec47);

                sub_vec48 = _mm256_mul_ps(sub_vec48, sub_vec48);
                sub_vec49 = _mm256_mul_ps(sub_vec49, sub_vec49);
                sub_vec50 = _mm256_mul_ps(sub_vec50, sub_vec50);
                sub_vec51 = _mm256_mul_ps(sub_vec51, sub_vec51);
                sub_vec52 = _mm256_mul_ps(sub_vec52, sub_vec52);
                sub_vec53 = _mm256_mul_ps(sub_vec53, sub_vec53);
                sub_vec54 = _mm256_mul_ps(sub_vec54, sub_vec54);
                sub_vec55 = _mm256_mul_ps(sub_vec55, sub_vec55);

                sub_vec56 = _mm256_mul_ps(sub_vec56, sub_vec56);
                sub_vec57 = _mm256_mul_ps(sub_vec57, sub_vec57);
                sub_vec58 = _mm256_mul_ps(sub_vec58, sub_vec58);
                sub_vec59 = _mm256_mul_ps(sub_vec59, sub_vec59);
                sub_vec60 = _mm256_mul_ps(sub_vec60, sub_vec60);
                sub_vec61 = _mm256_mul_ps(sub_vec61, sub_vec61);
                sub_vec62 = _mm256_mul_ps(sub_vec62, sub_vec62);
                sub_vec63 = _mm256_mul_ps(sub_vec63, sub_vec63);

                f0 = hsum256_ps_avx(sub_vec0); // horizontal add on 8x8 -> 8 values (sums) to be stored.
                f1 = hsum256_ps_avx(sub_vec1); // both sum8 and this one seem to work
                f2 = hsum256_ps_avx(sub_vec2);
                f3 = hsum256_ps_avx(sub_vec3);
                f4 = hsum256_ps_avx(sub_vec4);
                f5 = hsum256_ps_avx(sub_vec5);
                f6 = hsum256_ps_avx(sub_vec6);
                f7 = hsum256_ps_avx(sub_vec7);

                vec_sum0 = _mm256_set_ps(f7, f6, f5, f4, f3, f2, f1, f0);
//                        f0, f1, f2, f3, f4, f5, f6,f7);

                f8 = hsum256_ps_avx(sub_vec8);
                f9 = hsum256_ps_avx(sub_vec9);
                f10 = hsum256_ps_avx(sub_vec10);
                f11 = hsum256_ps_avx(sub_vec11);
                f12 = hsum256_ps_avx(sub_vec12);
                f13 = hsum256_ps_avx(sub_vec13);
                f14 = hsum256_ps_avx(sub_vec14);
                f15 = hsum256_ps_avx(sub_vec15);
//
                vec_sum1 = _mm256_set_ps(f15, f14, f13, f12, f11, f10, f9, f8);
//                        f8, f9, f10, f11, f12, f13, f14,f15);

                f16 = hsum256_ps_avx(sub_vec16);
                f17 = hsum256_ps_avx(sub_vec17);
                f18 = hsum256_ps_avx(sub_vec18);
                f19 = hsum256_ps_avx(sub_vec19);
                f20 = hsum256_ps_avx(sub_vec20);
                f21 = hsum256_ps_avx(sub_vec21);
                f22 = hsum256_ps_avx(sub_vec22);
                f23 = hsum256_ps_avx(sub_vec23);
//
                vec_sum2 = _mm256_set_ps(f23, f22, f21, f20, f19, f18, f17, f16);
//                        f16, f17, f18, f19, f20, f21, f22,f23);

                f24 = hsum256_ps_avx(sub_vec24);
                f25 = hsum256_ps_avx(sub_vec25);
                f26 = hsum256_ps_avx(sub_vec26);
                f27 = hsum256_ps_avx(sub_vec27);
                f28 = hsum256_ps_avx(sub_vec28);
                f29 = hsum256_ps_avx(sub_vec29);
                f30 = hsum256_ps_avx(sub_vec30);
                f31 = hsum256_ps_avx(sub_vec31);
//
                vec_sum3 = _mm256_set_ps(f31, f30, f29, f28, f27, f26, f25, f24);
//                        f24, f25, f26, f27, f28, f29, f30,f31);

                f32 = hsum256_ps_avx(sub_vec32);
                f33 = hsum256_ps_avx(sub_vec33);
                f34 = hsum256_ps_avx(sub_vec34);
                f35 = hsum256_ps_avx(sub_vec35);
                f36 = hsum256_ps_avx(sub_vec36);
                f37 = hsum256_ps_avx(sub_vec37);
                f38 = hsum256_ps_avx(sub_vec38);
                f39 = hsum256_ps_avx(sub_vec39);
//
                vec_sum4 = _mm256_set_ps(f39, f38, f37, f36, f35, f34, f33, f32);
//                        f32, f33, f34, f35, f36, f37, f38,f39);

                f40 = hsum256_ps_avx(sub_vec40);
                f41 = hsum256_ps_avx(sub_vec41);
                f42 = hsum256_ps_avx(sub_vec42);
                f43 = hsum256_ps_avx(sub_vec43);
                f44 = hsum256_ps_avx(sub_vec44);
                f45 = hsum256_ps_avx(sub_vec45);
                f46 = hsum256_ps_avx(sub_vec46);
                f47 = hsum256_ps_avx(sub_vec47);
//
                vec_sum5 = _mm256_set_ps(f47, f46, f45, f44, f43, f42, f41, f40);
//                        f40, f41, f42,f43, f44, f45, f46, f47);

                f48 = hsum256_ps_avx(sub_vec48);
                f49 = hsum256_ps_avx(sub_vec49);
                f50 = hsum256_ps_avx(sub_vec50);
                f51 = hsum256_ps_avx(sub_vec51);
                f52 = hsum256_ps_avx(sub_vec52);
                f53 = hsum256_ps_avx(sub_vec53);
                f54 = hsum256_ps_avx(sub_vec54);
                f55 = hsum256_ps_avx(sub_vec55);
//
                vec_sum6 = _mm256_set_ps(f55, f54, f53, f52, f51, f50, f49, f48);
//                        f48, f49, f50, f51, f52, f53, f54, f55);

                f56 = hsum256_ps_avx(sub_vec56);
                f57 = hsum256_ps_avx(sub_vec57);
                f58 = hsum256_ps_avx(sub_vec58);
                f59 = hsum256_ps_avx(sub_vec59);
                f60 = hsum256_ps_avx(sub_vec60);
                f61 = hsum256_ps_avx(sub_vec61);
                f62 = hsum256_ps_avx(sub_vec62);
                f63 = hsum256_ps_avx(sub_vec63);

                vec_sum7 = _mm256_set_ps(f63, f62, f61, f60, f59, f58, f57, f56);
//                        f56,f57, f58, f59, f60, f61, f62, f63);

                tmp0 = _mm256_add_ps(tmp0, vec_sum0); // accumulate
                tmp1 = _mm256_add_ps(tmp1, vec_sum1);
                tmp2 = _mm256_add_ps(tmp2, vec_sum2);
                tmp3 = _mm256_add_ps(tmp3, vec_sum3);
                tmp4 = _mm256_add_ps(tmp4, vec_sum4);
                tmp5 = _mm256_add_ps(tmp5, vec_sum5);
                tmp6 = _mm256_add_ps(tmp6, vec_sum6);
                tmp7 = _mm256_add_ps(tmp7, vec_sum7);

                _mm256_store_ps(distances.data + (bi) * N + bj, tmp0); //store the partial l2-norm
                _mm256_store_ps(distances.data + (bi + 1) * N + bj, tmp1);
                _mm256_store_ps(distances.data + (bi + 2) * N + bj, tmp2);
                _mm256_store_ps(distances.data + (bi + 3) * N + bj, tmp3);
                _mm256_store_ps(distances.data + (bi + 4) * N + bj, tmp4);
                _mm256_store_ps(distances.data + (bi + 5) * N + bj, tmp5);
                _mm256_store_ps(distances.data + (bi + 6) * N + bj, tmp6);
                _mm256_store_ps(distances.data + (bi + 7) * N + bj, tmp7);

            }
        }
    }

    for (int i_tst = 0; i_tst < N_tst; i_tst += 4) {
#pragma GCC ivdep
        for (int i_trn = 0; i_trn < N; i_trn++) {
            dist_arr[i_trn].value = distances.data[i_tst * N + i_trn];
            dist_arr_i[i_trn].value = distances.data[(i_tst + 1) * N + i_trn];
            dist_arr_ii[i_trn].value = distances.data[(i_tst + 2) * N + i_trn];
            dist_arr_iii[i_trn].value = distances.data[(i_tst + 3) * N + i_trn];
            dist_arr[i_trn].index = i_trn;
            dist_arr_i[i_trn].index = i_trn;
            dist_arr_ii[i_trn].index = i_trn;
            dist_arr_iii[i_trn].index = i_trn;
        }

        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_i, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iii, N, sizeof(pair_t), cmp);
#pragma GCC ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
            x_tst_knn_gt->data[(i_tst + 1) * N + k] = dist_arr_i[k].index;
            x_tst_knn_gt->data[(i_tst + 2) * N + k] = dist_arr_ii[k].index;
            x_tst_knn_gt->data[(i_tst + 3) * N + k] = dist_arr_iii[k].index;
        }
    }
    free(dist_arr);
    free(dist_arr_i);
    free(dist_arr_ii);
    free(dist_arr_iii);
    destroy(&distances);
}

// vectorized blocking 8x8 - horizontal sum outside
void get_true_knn_opt19(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 8;
    pair_t *dist_arr, *dist_arr_i, *dist_arr_ii, *dist_arr_iii;
    dist_arr = malloc(N * sizeof(pair_t));
    dist_arr_i = malloc(N * sizeof(pair_t));
    dist_arr_ii = malloc(N * sizeof(pair_t));
    dist_arr_iii = malloc(N * sizeof(pair_t));

    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;
    __m256 ts_vec0, tr_vec0, ts_vec1, tr_vec1, ts_vec2, tr_vec2, ts_vec3, tr_vec3, ts_vec4, tr_vec4, ts_vec5, tr_vec5, ts_vec6, tr_vec6, ts_vec7, tr_vec7;
    __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3, vec_sum4, vec_sum5, vec_sum6, vec_sum7;
    __m256 sub_vec0, sub_vec1, sub_vec2, sub_vec3, sub_vec4, sub_vec5, sub_vec6, sub_vec7;
    __m256 sub_vec8, sub_vec9, sub_vec10, sub_vec11, sub_vec12, sub_vec13, sub_vec14;
    __m256 sub_vec15, sub_vec16, sub_vec17, sub_vec18, sub_vec19, sub_vec20, sub_vec21;
    __m256 sub_vec22, sub_vec23, sub_vec24, sub_vec25, sub_vec26, sub_vec27, sub_vec28;
    __m256 sub_vec29, sub_vec30, sub_vec31, sub_vec32, sub_vec33, sub_vec34, sub_vec35;
    __m256 sub_vec36, sub_vec37, sub_vec38, sub_vec39, sub_vec40, sub_vec41, sub_vec42;
    __m256 sub_vec43, sub_vec44, sub_vec45, sub_vec46, sub_vec47, sub_vec48, sub_vec49;
    __m256 sub_vec50, sub_vec51, sub_vec52, sub_vec53, sub_vec54, sub_vec55, sub_vec56;
    __m256 sub_vec57, sub_vec58, sub_vec59, sub_vec60, sub_vec61, sub_vec62, sub_vec63;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;
    float f16, f17, f18, f19, f20, f21, f22, f23;
    float f24, f25, f26, f27, f28, f29, f30, f31;
    float f32, f33, f34, f35, f36, f37, f38, f39;
    float f40, f41, f42, f43, f44, f45, f46, f47;
    float f48, f49, f50, f51, f52, f53, f54, f55;
    float f56, f57, f58, f59, f60, f61, f62, f63;

    __m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m256 acc8, acc9, acc10, acc11, acc12, acc13, acc14;
    __m256 acc15, acc16, acc17, acc18, acc19, acc20, acc21;
    __m256 acc22, acc23, acc24, acc25, acc26, acc27, acc28;
    __m256 acc29, acc30, acc31, acc32, acc33, acc34, acc35;
    __m256 acc36, acc37, acc38, acc39, acc40, acc41, acc42;
    __m256 acc43, acc44, acc45, acc46, acc47, acc48, acc49;
    __m256 acc50, acc51, acc52, acc53, acc54, acc55, acc56;
    __m256 acc57, acc58, acc59, acc60, acc61, acc62, acc63;


    for (bi = 0; bi < N_tst; bi += Nb) {
        for (bj = 0; bj < N; bj += Nb) {
            // accumulators
            acc0 = _mm256_setzero_ps();
            acc1 = _mm256_setzero_ps();
            acc2 = _mm256_setzero_ps();
            acc3 = _mm256_setzero_ps();
            acc4 = _mm256_setzero_ps();
            acc5 = _mm256_setzero_ps();
            acc6 = _mm256_setzero_ps();
            acc7 = _mm256_setzero_ps();

            acc8 = _mm256_setzero_ps();
            acc9 = _mm256_setzero_ps();
            acc10 = _mm256_setzero_ps();
            acc11 = _mm256_setzero_ps();
            acc12 = _mm256_setzero_ps();
            acc13 = _mm256_setzero_ps();
            acc14 = _mm256_setzero_ps();
            acc15 = _mm256_setzero_ps();

            acc16 = _mm256_setzero_ps();
            acc17 = _mm256_setzero_ps();
            acc18 = _mm256_setzero_ps();
            acc19 = _mm256_setzero_ps();
            acc20 = _mm256_setzero_ps();
            acc21 = _mm256_setzero_ps();
            acc22 = _mm256_setzero_ps();
            acc23 = _mm256_setzero_ps();

            acc24 = _mm256_setzero_ps();
            acc25 = _mm256_setzero_ps();
            acc26 = _mm256_setzero_ps();
            acc27 = _mm256_setzero_ps();
            acc28 = _mm256_setzero_ps();
            acc29 = _mm256_setzero_ps();
            acc30 = _mm256_setzero_ps();
            acc31 = _mm256_setzero_ps();

            acc32 = _mm256_setzero_ps();
            acc33 = _mm256_setzero_ps();
            acc34 = _mm256_setzero_ps();
            acc35 = _mm256_setzero_ps();
            acc36 = _mm256_setzero_ps();
            acc37 = _mm256_setzero_ps();
            acc38 = _mm256_setzero_ps();
            acc39 = _mm256_setzero_ps();

            acc40 = _mm256_setzero_ps();
            acc41 = _mm256_setzero_ps();
            acc42 = _mm256_setzero_ps();
            acc43 = _mm256_setzero_ps();
            acc44 = _mm256_setzero_ps();
            acc45 = _mm256_setzero_ps();
            acc46 = _mm256_setzero_ps();
            acc47 = _mm256_setzero_ps();

            acc48 = _mm256_setzero_ps();
            acc49 = _mm256_setzero_ps();
            acc50 = _mm256_setzero_ps();
            acc51 = _mm256_setzero_ps();
            acc52 = _mm256_setzero_ps();
            acc53 = _mm256_setzero_ps();
            acc54 = _mm256_setzero_ps();
            acc55 = _mm256_setzero_ps();

            acc56 = _mm256_setzero_ps();
            acc57 = _mm256_setzero_ps();
            acc58 = _mm256_setzero_ps();
            acc59 = _mm256_setzero_ps();
            acc60 = _mm256_setzero_ps();
            acc61 = _mm256_setzero_ps();
            acc62 = _mm256_setzero_ps();
            acc63 = _mm256_setzero_ps();

            for (bk = 0; bk < d; bk += Nb) {
                tmp0 = _mm256_load_ps(distances.data + bi * N + bj); // 8x8 blocks temp values for l2-norm
                tmp1 = _mm256_load_ps(distances.data + (bi + 1) * N + bj);
                tmp2 = _mm256_load_ps(distances.data + (bi + 2) * N + bj);
                tmp3 = _mm256_load_ps(distances.data + (bi + 3) * N + bj);
                tmp4 = _mm256_load_ps(distances.data + (bi + 4) * N + bj);
                tmp5 = _mm256_load_ps(distances.data + (bi + 5) * N + bj);
                tmp6 = _mm256_load_ps(distances.data + (bi + 6) * N + bj);
                tmp7 = _mm256_load_ps(distances.data + (bi + 7) * N + bj);

                ts_vec0 = _mm256_load_ps(data_tst + bi * N + bk); // 8x8 blocks from test matrix
                ts_vec1 = _mm256_load_ps(data_tst + (bi + 1) * N + bk);
                ts_vec2 = _mm256_load_ps(data_tst + (bi + 2) * N + bk);
                ts_vec3 = _mm256_load_ps(data_tst + (bi + 3) * N + bk);
                ts_vec4 = _mm256_load_ps(data_tst + (bi + 4) * N + bk);
                ts_vec5 = _mm256_load_ps(data_tst + (bi + 5) * N + bk);
                ts_vec6 = _mm256_load_ps(data_tst + (bi + 6) * N + bk);
                ts_vec7 = _mm256_load_ps(data_tst + (bi + 7) * N + bk);

                tr_vec0 = _mm256_load_ps(data_trn + bj * N + bk); // 8x8 blocks from train matrix
                tr_vec1 = _mm256_load_ps(data_trn + (bj + 1) * N + bk);
                tr_vec2 = _mm256_load_ps(data_trn + (bj + 2) * N + bk);
                tr_vec3 = _mm256_load_ps(data_trn + (bj + 3) * N + bk);
                tr_vec4 = _mm256_load_ps(data_trn + (bj + 4) * N + bk);
                tr_vec5 = _mm256_load_ps(data_trn + (bj + 5) * N + bk);
                tr_vec6 = _mm256_load_ps(data_trn + (bj + 6) * N + bk);
                tr_vec7 = _mm256_load_ps(data_trn + (bj + 7) * N + bk);

                sub_vec0 = _mm256_sub_ps(ts_vec0, tr_vec0); // pairwise distances between test and train blocks
                sub_vec1 = _mm256_sub_ps(ts_vec0, tr_vec1); // 1x8 per code block, total 8x8 values computed
                sub_vec2 = _mm256_sub_ps(ts_vec0, tr_vec2);
                sub_vec3 = _mm256_sub_ps(ts_vec0, tr_vec3);
                sub_vec4 = _mm256_sub_ps(ts_vec0, tr_vec4);
                sub_vec5 = _mm256_sub_ps(ts_vec0, tr_vec5);
                sub_vec6 = _mm256_sub_ps(ts_vec0, tr_vec6);
                sub_vec7 = _mm256_sub_ps(ts_vec0, tr_vec7);

                sub_vec8 = _mm256_sub_ps(ts_vec1, tr_vec0);
                sub_vec9 = _mm256_sub_ps(ts_vec1, tr_vec1);
                sub_vec10 = _mm256_sub_ps(ts_vec1, tr_vec2);
                sub_vec11 = _mm256_sub_ps(ts_vec1, tr_vec3);
                sub_vec12 = _mm256_sub_ps(ts_vec1, tr_vec4);
                sub_vec13 = _mm256_sub_ps(ts_vec1, tr_vec5);
                sub_vec14 = _mm256_sub_ps(ts_vec1, tr_vec6);
                sub_vec15 = _mm256_sub_ps(ts_vec1, tr_vec7);

                sub_vec16 = _mm256_sub_ps(ts_vec2, tr_vec0);
                sub_vec17 = _mm256_sub_ps(ts_vec2, tr_vec1);
                sub_vec18 = _mm256_sub_ps(ts_vec2, tr_vec2);
                sub_vec19 = _mm256_sub_ps(ts_vec2, tr_vec3);
                sub_vec20 = _mm256_sub_ps(ts_vec2, tr_vec4);
                sub_vec21 = _mm256_sub_ps(ts_vec2, tr_vec5);
                sub_vec22 = _mm256_sub_ps(ts_vec2, tr_vec6);
                sub_vec23 = _mm256_sub_ps(ts_vec2, tr_vec7);

                sub_vec24 = _mm256_sub_ps(ts_vec3, tr_vec0);
                sub_vec25 = _mm256_sub_ps(ts_vec3, tr_vec1);
                sub_vec26 = _mm256_sub_ps(ts_vec3, tr_vec2);
                sub_vec27 = _mm256_sub_ps(ts_vec3, tr_vec3);
                sub_vec28 = _mm256_sub_ps(ts_vec3, tr_vec4);
                sub_vec29 = _mm256_sub_ps(ts_vec3, tr_vec5);
                sub_vec30 = _mm256_sub_ps(ts_vec3, tr_vec6);
                sub_vec31 = _mm256_sub_ps(ts_vec3, tr_vec7);

                sub_vec32 = _mm256_sub_ps(ts_vec4, tr_vec0);
                sub_vec33 = _mm256_sub_ps(ts_vec4, tr_vec1);
                sub_vec34 = _mm256_sub_ps(ts_vec4, tr_vec2);
                sub_vec35 = _mm256_sub_ps(ts_vec4, tr_vec3);
                sub_vec36 = _mm256_sub_ps(ts_vec4, tr_vec4);
                sub_vec37 = _mm256_sub_ps(ts_vec4, tr_vec5);
                sub_vec38 = _mm256_sub_ps(ts_vec4, tr_vec6);
                sub_vec39 = _mm256_sub_ps(ts_vec4, tr_vec7);

                sub_vec40 = _mm256_sub_ps(ts_vec5, tr_vec0);
                sub_vec41 = _mm256_sub_ps(ts_vec5, tr_vec1);
                sub_vec42 = _mm256_sub_ps(ts_vec5, tr_vec2);
                sub_vec43 = _mm256_sub_ps(ts_vec5, tr_vec3);
                sub_vec44 = _mm256_sub_ps(ts_vec5, tr_vec4);
                sub_vec45 = _mm256_sub_ps(ts_vec5, tr_vec5);
                sub_vec46 = _mm256_sub_ps(ts_vec5, tr_vec6);
                sub_vec47 = _mm256_sub_ps(ts_vec5, tr_vec7);

                sub_vec48 = _mm256_sub_ps(ts_vec6, tr_vec0);
                sub_vec49 = _mm256_sub_ps(ts_vec6, tr_vec1);
                sub_vec50 = _mm256_sub_ps(ts_vec6, tr_vec2);
                sub_vec51 = _mm256_sub_ps(ts_vec6, tr_vec3);
                sub_vec52 = _mm256_sub_ps(ts_vec6, tr_vec4);
                sub_vec53 = _mm256_sub_ps(ts_vec6, tr_vec5);
                sub_vec54 = _mm256_sub_ps(ts_vec6, tr_vec6);
                sub_vec55 = _mm256_sub_ps(ts_vec6, tr_vec7);

                sub_vec56 = _mm256_sub_ps(ts_vec7, tr_vec0);
                sub_vec57 = _mm256_sub_ps(ts_vec7, tr_vec1);
                sub_vec58 = _mm256_sub_ps(ts_vec7, tr_vec2);
                sub_vec59 = _mm256_sub_ps(ts_vec7, tr_vec3);
                sub_vec60 = _mm256_sub_ps(ts_vec7, tr_vec4);
                sub_vec61 = _mm256_sub_ps(ts_vec7, tr_vec5);
                sub_vec62 = _mm256_sub_ps(ts_vec7, tr_vec6);
                sub_vec63 = _mm256_sub_ps(ts_vec7, tr_vec7);

                sub_vec0 = _mm256_mul_ps(sub_vec0, sub_vec0); // squared distance, 8x8
                sub_vec1 = _mm256_mul_ps(sub_vec1, sub_vec1);
                sub_vec2 = _mm256_mul_ps(sub_vec2, sub_vec2);
                sub_vec3 = _mm256_mul_ps(sub_vec3, sub_vec3);
                sub_vec4 = _mm256_mul_ps(sub_vec4, sub_vec4);
                sub_vec5 = _mm256_mul_ps(sub_vec5, sub_vec5);
                sub_vec6 = _mm256_mul_ps(sub_vec6, sub_vec6);
                sub_vec7 = _mm256_mul_ps(sub_vec7, sub_vec7);

                sub_vec8 = _mm256_mul_ps(sub_vec8, sub_vec8);
                sub_vec9 = _mm256_mul_ps(sub_vec9, sub_vec9);
                sub_vec10 = _mm256_mul_ps(sub_vec10, sub_vec10);
                sub_vec11 = _mm256_mul_ps(sub_vec11, sub_vec11);
                sub_vec12 = _mm256_mul_ps(sub_vec12, sub_vec12);
                sub_vec13 = _mm256_mul_ps(sub_vec13, sub_vec13);
                sub_vec14 = _mm256_mul_ps(sub_vec14, sub_vec14);
                sub_vec15 = _mm256_mul_ps(sub_vec15, sub_vec15);

                sub_vec16 = _mm256_mul_ps(sub_vec16, sub_vec16);
                sub_vec17 = _mm256_mul_ps(sub_vec17, sub_vec17);
                sub_vec18 = _mm256_mul_ps(sub_vec18, sub_vec18);
                sub_vec19 = _mm256_mul_ps(sub_vec19, sub_vec19);
                sub_vec20 = _mm256_mul_ps(sub_vec20, sub_vec20);
                sub_vec21 = _mm256_mul_ps(sub_vec21, sub_vec21);
                sub_vec22 = _mm256_mul_ps(sub_vec22, sub_vec22);
                sub_vec23 = _mm256_mul_ps(sub_vec23, sub_vec23);

                sub_vec24 = _mm256_mul_ps(sub_vec24, sub_vec24);
                sub_vec25 = _mm256_mul_ps(sub_vec25, sub_vec25);
                sub_vec26 = _mm256_mul_ps(sub_vec26, sub_vec26);
                sub_vec27 = _mm256_mul_ps(sub_vec27, sub_vec27);
                sub_vec28 = _mm256_mul_ps(sub_vec28, sub_vec28);
                sub_vec29 = _mm256_mul_ps(sub_vec29, sub_vec29);
                sub_vec30 = _mm256_mul_ps(sub_vec30, sub_vec30);
                sub_vec31 = _mm256_mul_ps(sub_vec31, sub_vec31);

                sub_vec32 = _mm256_mul_ps(sub_vec32, sub_vec32);
                sub_vec33 = _mm256_mul_ps(sub_vec33, sub_vec33);
                sub_vec34 = _mm256_mul_ps(sub_vec34, sub_vec34);
                sub_vec35 = _mm256_mul_ps(sub_vec35, sub_vec35);
                sub_vec36 = _mm256_mul_ps(sub_vec36, sub_vec36);
                sub_vec37 = _mm256_mul_ps(sub_vec37, sub_vec37);
                sub_vec38 = _mm256_mul_ps(sub_vec38, sub_vec38);
                sub_vec39 = _mm256_mul_ps(sub_vec39, sub_vec39);

                sub_vec40 = _mm256_mul_ps(sub_vec40, sub_vec40);
                sub_vec41 = _mm256_mul_ps(sub_vec41, sub_vec41);
                sub_vec42 = _mm256_mul_ps(sub_vec42, sub_vec42);
                sub_vec43 = _mm256_mul_ps(sub_vec43, sub_vec43);
                sub_vec44 = _mm256_mul_ps(sub_vec44, sub_vec44);
                sub_vec45 = _mm256_mul_ps(sub_vec45, sub_vec45);
                sub_vec46 = _mm256_mul_ps(sub_vec46, sub_vec46);
                sub_vec47 = _mm256_mul_ps(sub_vec47, sub_vec47);

                sub_vec48 = _mm256_mul_ps(sub_vec48, sub_vec48);
                sub_vec49 = _mm256_mul_ps(sub_vec49, sub_vec49);
                sub_vec50 = _mm256_mul_ps(sub_vec50, sub_vec50);
                sub_vec51 = _mm256_mul_ps(sub_vec51, sub_vec51);
                sub_vec52 = _mm256_mul_ps(sub_vec52, sub_vec52);
                sub_vec53 = _mm256_mul_ps(sub_vec53, sub_vec53);
                sub_vec54 = _mm256_mul_ps(sub_vec54, sub_vec54);
                sub_vec55 = _mm256_mul_ps(sub_vec55, sub_vec55);

                sub_vec56 = _mm256_mul_ps(sub_vec56, sub_vec56);
                sub_vec57 = _mm256_mul_ps(sub_vec57, sub_vec57);
                sub_vec58 = _mm256_mul_ps(sub_vec58, sub_vec58);
                sub_vec59 = _mm256_mul_ps(sub_vec59, sub_vec59);
                sub_vec60 = _mm256_mul_ps(sub_vec60, sub_vec60);
                sub_vec61 = _mm256_mul_ps(sub_vec61, sub_vec61);
                sub_vec62 = _mm256_mul_ps(sub_vec62, sub_vec62);
                sub_vec63 = _mm256_mul_ps(sub_vec63, sub_vec63);

                acc0 = _mm256_add_ps(acc0, sub_vec0); // accumulate pairwise distances on 8x8
                acc1 = _mm256_add_ps(acc1, sub_vec1);
                acc2 = _mm256_add_ps(acc2, sub_vec2);
                acc3 = _mm256_add_ps(acc3, sub_vec3);
                acc4 = _mm256_add_ps(acc4, sub_vec4);
                acc5 = _mm256_add_ps(acc5, sub_vec5);
                acc6 = _mm256_add_ps(acc6, sub_vec6);
                acc7 = _mm256_add_ps(acc7, sub_vec7);

                acc8 = _mm256_add_ps(acc8, sub_vec8);
                acc9 = _mm256_add_ps(acc9, sub_vec9);
                acc10 = _mm256_add_ps(acc10, sub_vec10);
                acc11 = _mm256_add_ps(acc11, sub_vec11);
                acc12 = _mm256_add_ps(acc12, sub_vec12);
                acc13 = _mm256_add_ps(acc13, sub_vec13);
                acc14 = _mm256_add_ps(acc14, sub_vec14);
                acc15 = _mm256_add_ps(acc15, sub_vec15);

                acc16 = _mm256_add_ps(acc16, sub_vec16);
                acc17 = _mm256_add_ps(acc17, sub_vec17);
                acc18 = _mm256_add_ps(acc18, sub_vec18);
                acc19 = _mm256_add_ps(acc19, sub_vec19);
                acc20 = _mm256_add_ps(acc20, sub_vec20);
                acc21 = _mm256_add_ps(acc21, sub_vec21);
                acc22 = _mm256_add_ps(acc22, sub_vec22);
                acc23 = _mm256_add_ps(acc23, sub_vec23);

                acc24 = _mm256_add_ps(acc24, sub_vec24);
                acc25 = _mm256_add_ps(acc25, sub_vec25);
                acc26 = _mm256_add_ps(acc26, sub_vec26);
                acc27 = _mm256_add_ps(acc27, sub_vec27);
                acc28 = _mm256_add_ps(acc28, sub_vec28);
                acc29 = _mm256_add_ps(acc29, sub_vec29);
                acc30 = _mm256_add_ps(acc30, sub_vec30);
                acc31 = _mm256_add_ps(acc31, sub_vec31);

                acc32 = _mm256_add_ps(acc32, sub_vec32);
                acc33 = _mm256_add_ps(acc33, sub_vec33);
                acc34 = _mm256_add_ps(acc34, sub_vec34);
                acc35 = _mm256_add_ps(acc35, sub_vec35);
                acc36 = _mm256_add_ps(acc36, sub_vec36);
                acc37 = _mm256_add_ps(acc37, sub_vec37);
                acc38 = _mm256_add_ps(acc38, sub_vec38);
                acc39 = _mm256_add_ps(acc39, sub_vec39);

                acc40 = _mm256_add_ps(acc40, sub_vec40);
                acc41 = _mm256_add_ps(acc41, sub_vec41);
                acc42 = _mm256_add_ps(acc42, sub_vec42);
                acc43 = _mm256_add_ps(acc43, sub_vec43);
                acc44 = _mm256_add_ps(acc44, sub_vec44);
                acc45 = _mm256_add_ps(acc45, sub_vec45);
                acc46 = _mm256_add_ps(acc46, sub_vec46);
                acc47 = _mm256_add_ps(acc47, sub_vec47);

                acc48 = _mm256_add_ps(acc48, sub_vec48);
                acc49 = _mm256_add_ps(acc49, sub_vec49);
                acc50 = _mm256_add_ps(acc50, sub_vec50);
                acc51 = _mm256_add_ps(acc51, sub_vec51);
                acc52 = _mm256_add_ps(acc52, sub_vec52);
                acc53 = _mm256_add_ps(acc53, sub_vec53);
                acc54 = _mm256_add_ps(acc54, sub_vec54);
                acc55 = _mm256_add_ps(acc55, sub_vec55);

                acc56 = _mm256_add_ps(acc56, sub_vec56);
                acc57 = _mm256_add_ps(acc57, sub_vec57);
                acc58 = _mm256_add_ps(acc58, sub_vec58);
                acc59 = _mm256_add_ps(acc59, sub_vec59);
                acc60 = _mm256_add_ps(acc60, sub_vec60);
                acc61 = _mm256_add_ps(acc61, sub_vec61);
                acc62 = _mm256_add_ps(acc62, sub_vec62);
                acc63 = _mm256_add_ps(acc63, sub_vec63);
            }
            f0 = sum8(acc0);
            f1 = sum8(acc1);
            f2 = sum8(acc2);
            f3 = sum8(acc3);
            f4 = sum8(acc4);
            f5 = sum8(acc5);
            f6 = sum8(acc6);
            f7 = sum8(acc7);

            vec_sum0 = _mm256_set_ps(f7, f6, f5, f4, f3, f2, f1, f0);

            f8 = sum8(acc8);
            f9 = sum8(acc9);
            f10 = sum8(acc10);
            f11 = sum8(acc11);
            f12 = sum8(acc12);
            f13 = sum8(acc13);
            f14 = sum8(acc14);
            f15 = sum8(acc15);

            vec_sum1 = _mm256_set_ps(f15, f14, f13, f12, f11, f10, f9, f8);

            f16 = sum8(acc16);
            f17 = sum8(acc17);
            f18 = sum8(acc18);
            f19 = sum8(acc19);
            f20 = sum8(acc20);
            f21 = sum8(acc21);
            f22 = sum8(acc22);
            f23 = sum8(acc23);

            vec_sum2 = _mm256_set_ps(f23, f22, f21, f20, f19, f18, f17, f16);

            f24 = sum8(acc24);
            f25 = sum8(acc25);
            f26 = sum8(acc26);
            f27 = sum8(acc27);
            f28 = sum8(acc28);
            f29 = sum8(acc29);
            f30 = sum8(acc30);
            f31 = sum8(acc31);

            vec_sum3 = _mm256_set_ps(f31, f30, f29, f28, f27, f26, f25, f24);

            f32 = sum8(acc32);
            f33 = sum8(acc33);
            f34 = sum8(acc34);
            f35 = sum8(acc35);
            f36 = sum8(acc36);
            f37 = sum8(acc37);
            f38 = sum8(acc38);
            f39 = sum8(acc39);

            vec_sum4 = _mm256_set_ps(f39, f38, f37, f36, f35, f34, f33, f32);

            f40 = sum8(acc40);
            f41 = sum8(acc41);
            f42 = sum8(acc42);
            f43 = sum8(acc43);
            f44 = sum8(acc44);
            f45 = sum8(acc45);
            f46 = sum8(acc46);
            f47 = sum8(acc47);

            vec_sum5 = _mm256_set_ps(f47, f46, f45, f44, f43, f42, f41, f40);

            f48 = sum8(acc48);
            f49 = sum8(acc49);
            f50 = sum8(acc50);
            f51 = sum8(acc51);
            f52 = sum8(acc52);
            f53 = sum8(acc53);
            f54 = sum8(acc54);
            f55 = sum8(acc55);

            vec_sum6 = _mm256_set_ps(f55, f54, f53, f52, f51, f50, f49, f48);

            f56 = sum8(acc56);
            f57 = sum8(acc57);
            f58 = sum8(acc58);
            f59 = sum8(acc59);
            f60 = sum8(acc60);
            f61 = sum8(acc61);
            f62 = sum8(acc62);
            f63 = sum8(acc63);

            vec_sum7 = _mm256_set_ps(f63, f62, f61, f60, f59, f58, f57, f56);

            tmp0 = _mm256_load_ps(distances.data + bi * N + bj);
            tmp1 = _mm256_load_ps(distances.data + (bi+1) * N + bj);
            tmp2 = _mm256_load_ps(distances.data + (bi+2) * N + bj);
            tmp3 = _mm256_load_ps(distances.data + (bi+3) * N + bj);
            tmp4 = _mm256_load_ps(distances.data + (bi+4) * N + bj);
            tmp5 = _mm256_load_ps(distances.data + (bi+5) * N + bj);
            tmp6 = _mm256_load_ps(distances.data + (bi+6) * N + bj);
            tmp7 = _mm256_load_ps(distances.data + (bi+7) * N + bj);

            tmp0 = _mm256_add_ps(tmp0, vec_sum0);
            tmp1 = _mm256_add_ps(tmp1, vec_sum1);
            tmp2 = _mm256_add_ps(tmp2, vec_sum2);
            tmp3 = _mm256_add_ps(tmp3, vec_sum3);
            tmp4 = _mm256_add_ps(tmp4, vec_sum4);
            tmp5 = _mm256_add_ps(tmp5, vec_sum5);
            tmp6 = _mm256_add_ps(tmp6, vec_sum6);
            tmp7 = _mm256_add_ps(tmp7, vec_sum7);

            _mm256_store_ps(distances.data + (bi) * N + bj, tmp0);
            _mm256_store_ps(distances.data + (bi+1) * N + bj, tmp1);
            _mm256_store_ps(distances.data + (bi+2) * N + bj, tmp2);
            _mm256_store_ps(distances.data + (bi+3) * N + bj, tmp3);
            _mm256_store_ps(distances.data + (bi+4) * N + bj, tmp4);
            _mm256_store_ps(distances.data + (bi+5) * N + bj, tmp5);
            _mm256_store_ps(distances.data + (bi+6) * N + bj, tmp6);
            _mm256_store_ps(distances.data + (bi+7) * N + bj, tmp7);

        }
    }

    for (int i_tst = 0; i_tst < N_tst; i_tst += 4) {
#pragma GCC ivdep
        for (int i_trn = 0; i_trn < N; i_trn++) {
            dist_arr[i_trn].value = distances.data[i_tst * N + i_trn];
            dist_arr_i[i_trn].value = distances.data[(i_tst + 1) * N + i_trn];
            dist_arr_ii[i_trn].value = distances.data[(i_tst + 2) * N + i_trn];
            dist_arr_iii[i_trn].value = distances.data[(i_tst + 3) * N + i_trn];
            dist_arr[i_trn].index = i_trn;
            dist_arr_i[i_trn].index = i_trn;
            dist_arr_ii[i_trn].index = i_trn;
            dist_arr_iii[i_trn].index = i_trn;
        }

        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_i, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iii, N, sizeof(pair_t), cmp);
#pragma GCC ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
            x_tst_knn_gt->data[(i_tst + 1) * N + k] = dist_arr_i[k].index;
            x_tst_knn_gt->data[(i_tst + 2) * N + k] = dist_arr_ii[k].index;
            x_tst_knn_gt->data[(i_tst + 3) * N + k] = dist_arr_iii[k].index;
        }
    }
    free(dist_arr);
    free(dist_arr_i);
    free(dist_arr_ii);
    free(dist_arr_iii);
    destroy(&distances);
}

// vectorized blocking 8x8 - horizontal sum outside + fmas
void get_true_knn_opt20(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 8;
    pair_t *dist_arr, *dist_arr_i, *dist_arr_ii, *dist_arr_iii;
    dist_arr = malloc(N * sizeof(pair_t));
    dist_arr_i = malloc(N * sizeof(pair_t));
    dist_arr_ii = malloc(N * sizeof(pair_t));
    dist_arr_iii = malloc(N * sizeof(pair_t));

    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float *data_trn = x_trn->data;
    float *data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;
    __m256 ts_vec0, tr_vec0, ts_vec1, tr_vec1, ts_vec2, tr_vec2, ts_vec3, tr_vec3, ts_vec4, tr_vec4, ts_vec5, tr_vec5, ts_vec6, tr_vec6, ts_vec7, tr_vec7;
    __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3, vec_sum4, vec_sum5, vec_sum6, vec_sum7;
    __m256 sub_vec0, sub_vec1, sub_vec2, sub_vec3, sub_vec4, sub_vec5, sub_vec6, sub_vec7;
    __m256 sub_vec8, sub_vec9, sub_vec10, sub_vec11, sub_vec12, sub_vec13, sub_vec14;
    __m256 sub_vec15, sub_vec16, sub_vec17, sub_vec18, sub_vec19, sub_vec20, sub_vec21;
    __m256 sub_vec22, sub_vec23, sub_vec24, sub_vec25, sub_vec26, sub_vec27, sub_vec28;
    __m256 sub_vec29, sub_vec30, sub_vec31, sub_vec32, sub_vec33, sub_vec34, sub_vec35;
    __m256 sub_vec36, sub_vec37, sub_vec38, sub_vec39, sub_vec40, sub_vec41, sub_vec42;
    __m256 sub_vec43, sub_vec44, sub_vec45, sub_vec46, sub_vec47, sub_vec48, sub_vec49;
    __m256 sub_vec50, sub_vec51, sub_vec52, sub_vec53, sub_vec54, sub_vec55, sub_vec56;
    __m256 sub_vec57, sub_vec58, sub_vec59, sub_vec60, sub_vec61, sub_vec62, sub_vec63;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;
    float f16, f17, f18, f19, f20, f21, f22, f23;
    float f24, f25, f26, f27, f28, f29, f30, f31;
    float f32, f33, f34, f35, f36, f37, f38, f39;
    float f40, f41, f42, f43, f44, f45, f46, f47;
    float f48, f49, f50, f51, f52, f53, f54, f55;
    float f56, f57, f58, f59, f60, f61, f62, f63;

    __m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m256 acc8, acc9, acc10, acc11, acc12, acc13, acc14;
    __m256 acc15, acc16, acc17, acc18, acc19, acc20, acc21;
    __m256 acc22, acc23, acc24, acc25, acc26, acc27, acc28;
    __m256 acc29, acc30, acc31, acc32, acc33, acc34, acc35;
    __m256 acc36, acc37, acc38, acc39, acc40, acc41, acc42;
    __m256 acc43, acc44, acc45, acc46, acc47, acc48, acc49;
    __m256 acc50, acc51, acc52, acc53, acc54, acc55, acc56;
    __m256 acc57, acc58, acc59, acc60, acc61, acc62, acc63;


    for (bi = 0; bi < N_tst; bi += Nb) {
        for (bj = 0; bj < N; bj += Nb) {
            // accumulators
            acc0 = _mm256_setzero_ps();
            acc1 = _mm256_setzero_ps();
            acc2 = _mm256_setzero_ps();
            acc3 = _mm256_setzero_ps();
            acc4 = _mm256_setzero_ps();
            acc5 = _mm256_setzero_ps();
            acc6 = _mm256_setzero_ps();
            acc7 = _mm256_setzero_ps();

            acc8 = _mm256_setzero_ps();
            acc9 = _mm256_setzero_ps();
            acc10 = _mm256_setzero_ps();
            acc11 = _mm256_setzero_ps();
            acc12 = _mm256_setzero_ps();
            acc13 = _mm256_setzero_ps();
            acc14 = _mm256_setzero_ps();
            acc15 = _mm256_setzero_ps();

            acc16 = _mm256_setzero_ps();
            acc17 = _mm256_setzero_ps();
            acc18 = _mm256_setzero_ps();
            acc19 = _mm256_setzero_ps();
            acc20 = _mm256_setzero_ps();
            acc21 = _mm256_setzero_ps();
            acc22 = _mm256_setzero_ps();
            acc23 = _mm256_setzero_ps();

            acc24 = _mm256_setzero_ps();
            acc25 = _mm256_setzero_ps();
            acc26 = _mm256_setzero_ps();
            acc27 = _mm256_setzero_ps();
            acc28 = _mm256_setzero_ps();
            acc29 = _mm256_setzero_ps();
            acc30 = _mm256_setzero_ps();
            acc31 = _mm256_setzero_ps();

            acc32 = _mm256_setzero_ps();
            acc33 = _mm256_setzero_ps();
            acc34 = _mm256_setzero_ps();
            acc35 = _mm256_setzero_ps();
            acc36 = _mm256_setzero_ps();
            acc37 = _mm256_setzero_ps();
            acc38 = _mm256_setzero_ps();
            acc39 = _mm256_setzero_ps();

            acc40 = _mm256_setzero_ps();
            acc41 = _mm256_setzero_ps();
            acc42 = _mm256_setzero_ps();
            acc43 = _mm256_setzero_ps();
            acc44 = _mm256_setzero_ps();
            acc45 = _mm256_setzero_ps();
            acc46 = _mm256_setzero_ps();
            acc47 = _mm256_setzero_ps();

            acc48 = _mm256_setzero_ps();
            acc49 = _mm256_setzero_ps();
            acc50 = _mm256_setzero_ps();
            acc51 = _mm256_setzero_ps();
            acc52 = _mm256_setzero_ps();
            acc53 = _mm256_setzero_ps();
            acc54 = _mm256_setzero_ps();
            acc55 = _mm256_setzero_ps();

            acc56 = _mm256_setzero_ps();
            acc57 = _mm256_setzero_ps();
            acc58 = _mm256_setzero_ps();
            acc59 = _mm256_setzero_ps();
            acc60 = _mm256_setzero_ps();
            acc61 = _mm256_setzero_ps();
            acc62 = _mm256_setzero_ps();
            acc63 = _mm256_setzero_ps();

            for (bk = 0; bk < d; bk += Nb) {
                tmp0 = _mm256_load_ps(distances.data + bi * N + bj); // 8x8 blocks temp values for l2-norm
                tmp1 = _mm256_load_ps(distances.data + (bi + 1) * N + bj);
                tmp2 = _mm256_load_ps(distances.data + (bi + 2) * N + bj);
                tmp3 = _mm256_load_ps(distances.data + (bi + 3) * N + bj);
                tmp4 = _mm256_load_ps(distances.data + (bi + 4) * N + bj);
                tmp5 = _mm256_load_ps(distances.data + (bi + 5) * N + bj);
                tmp6 = _mm256_load_ps(distances.data + (bi + 6) * N + bj);
                tmp7 = _mm256_load_ps(distances.data + (bi + 7) * N + bj);

                ts_vec0 = _mm256_load_ps(data_tst + bi * N + bk); // 8x8 blocks from test matrix
                ts_vec1 = _mm256_load_ps(data_tst + (bi + 1) * N + bk);
                ts_vec2 = _mm256_load_ps(data_tst + (bi + 2) * N + bk);
                ts_vec3 = _mm256_load_ps(data_tst + (bi + 3) * N + bk);
                ts_vec4 = _mm256_load_ps(data_tst + (bi + 4) * N + bk);
                ts_vec5 = _mm256_load_ps(data_tst + (bi + 5) * N + bk);
                ts_vec6 = _mm256_load_ps(data_tst + (bi + 6) * N + bk);
                ts_vec7 = _mm256_load_ps(data_tst + (bi + 7) * N + bk);

                tr_vec0 = _mm256_load_ps(data_trn + bj * N + bk); // 8x8 blocks from train matrix
                tr_vec1 = _mm256_load_ps(data_trn + (bj + 1) * N + bk);
                tr_vec2 = _mm256_load_ps(data_trn + (bj + 2) * N + bk);
                tr_vec3 = _mm256_load_ps(data_trn + (bj + 3) * N + bk);
                tr_vec4 = _mm256_load_ps(data_trn + (bj + 4) * N + bk);
                tr_vec5 = _mm256_load_ps(data_trn + (bj + 5) * N + bk);
                tr_vec6 = _mm256_load_ps(data_trn + (bj + 6) * N + bk);
                tr_vec7 = _mm256_load_ps(data_trn + (bj + 7) * N + bk);

                sub_vec0 = _mm256_sub_ps(ts_vec0, tr_vec0); // pairwise distances between test and train blocks
                sub_vec1 = _mm256_sub_ps(ts_vec0, tr_vec1); // 1x8 per code block, total 8x8 values computed
                sub_vec2 = _mm256_sub_ps(ts_vec0, tr_vec2);
                sub_vec3 = _mm256_sub_ps(ts_vec0, tr_vec3);
                sub_vec4 = _mm256_sub_ps(ts_vec0, tr_vec4);
                sub_vec5 = _mm256_sub_ps(ts_vec0, tr_vec5);
                sub_vec6 = _mm256_sub_ps(ts_vec0, tr_vec6);
                sub_vec7 = _mm256_sub_ps(ts_vec0, tr_vec7);

                sub_vec8 = _mm256_sub_ps(ts_vec1, tr_vec0);
                sub_vec9 = _mm256_sub_ps(ts_vec1, tr_vec1);
                sub_vec10 = _mm256_sub_ps(ts_vec1, tr_vec2);
                sub_vec11 = _mm256_sub_ps(ts_vec1, tr_vec3);
                sub_vec12 = _mm256_sub_ps(ts_vec1, tr_vec4);
                sub_vec13 = _mm256_sub_ps(ts_vec1, tr_vec5);
                sub_vec14 = _mm256_sub_ps(ts_vec1, tr_vec6);
                sub_vec15 = _mm256_sub_ps(ts_vec1, tr_vec7);

                sub_vec16 = _mm256_sub_ps(ts_vec2, tr_vec0);
                sub_vec17 = _mm256_sub_ps(ts_vec2, tr_vec1);
                sub_vec18 = _mm256_sub_ps(ts_vec2, tr_vec2);
                sub_vec19 = _mm256_sub_ps(ts_vec2, tr_vec3);
                sub_vec20 = _mm256_sub_ps(ts_vec2, tr_vec4);
                sub_vec21 = _mm256_sub_ps(ts_vec2, tr_vec5);
                sub_vec22 = _mm256_sub_ps(ts_vec2, tr_vec6);
                sub_vec23 = _mm256_sub_ps(ts_vec2, tr_vec7);

                sub_vec24 = _mm256_sub_ps(ts_vec3, tr_vec0);
                sub_vec25 = _mm256_sub_ps(ts_vec3, tr_vec1);
                sub_vec26 = _mm256_sub_ps(ts_vec3, tr_vec2);
                sub_vec27 = _mm256_sub_ps(ts_vec3, tr_vec3);
                sub_vec28 = _mm256_sub_ps(ts_vec3, tr_vec4);
                sub_vec29 = _mm256_sub_ps(ts_vec3, tr_vec5);
                sub_vec30 = _mm256_sub_ps(ts_vec3, tr_vec6);
                sub_vec31 = _mm256_sub_ps(ts_vec3, tr_vec7);

                sub_vec32 = _mm256_sub_ps(ts_vec4, tr_vec0);
                sub_vec33 = _mm256_sub_ps(ts_vec4, tr_vec1);
                sub_vec34 = _mm256_sub_ps(ts_vec4, tr_vec2);
                sub_vec35 = _mm256_sub_ps(ts_vec4, tr_vec3);
                sub_vec36 = _mm256_sub_ps(ts_vec4, tr_vec4);
                sub_vec37 = _mm256_sub_ps(ts_vec4, tr_vec5);
                sub_vec38 = _mm256_sub_ps(ts_vec4, tr_vec6);
                sub_vec39 = _mm256_sub_ps(ts_vec4, tr_vec7);

                sub_vec40 = _mm256_sub_ps(ts_vec5, tr_vec0);
                sub_vec41 = _mm256_sub_ps(ts_vec5, tr_vec1);
                sub_vec42 = _mm256_sub_ps(ts_vec5, tr_vec2);
                sub_vec43 = _mm256_sub_ps(ts_vec5, tr_vec3);
                sub_vec44 = _mm256_sub_ps(ts_vec5, tr_vec4);
                sub_vec45 = _mm256_sub_ps(ts_vec5, tr_vec5);
                sub_vec46 = _mm256_sub_ps(ts_vec5, tr_vec6);
                sub_vec47 = _mm256_sub_ps(ts_vec5, tr_vec7);

                sub_vec48 = _mm256_sub_ps(ts_vec6, tr_vec0);
                sub_vec49 = _mm256_sub_ps(ts_vec6, tr_vec1);
                sub_vec50 = _mm256_sub_ps(ts_vec6, tr_vec2);
                sub_vec51 = _mm256_sub_ps(ts_vec6, tr_vec3);
                sub_vec52 = _mm256_sub_ps(ts_vec6, tr_vec4);
                sub_vec53 = _mm256_sub_ps(ts_vec6, tr_vec5);
                sub_vec54 = _mm256_sub_ps(ts_vec6, tr_vec6);
                sub_vec55 = _mm256_sub_ps(ts_vec6, tr_vec7);

                sub_vec56 = _mm256_sub_ps(ts_vec7, tr_vec0);
                sub_vec57 = _mm256_sub_ps(ts_vec7, tr_vec1);
                sub_vec58 = _mm256_sub_ps(ts_vec7, tr_vec2);
                sub_vec59 = _mm256_sub_ps(ts_vec7, tr_vec3);
                sub_vec60 = _mm256_sub_ps(ts_vec7, tr_vec4);
                sub_vec61 = _mm256_sub_ps(ts_vec7, tr_vec5);
                sub_vec62 = _mm256_sub_ps(ts_vec7, tr_vec6);
                sub_vec63 = _mm256_sub_ps(ts_vec7, tr_vec7);

                acc0 = _mm256_fmadd_ps(sub_vec0, sub_vec0, acc0); // squared distance, 8x8
                acc1 = _mm256_fmadd_ps(sub_vec1, sub_vec1, acc1);
                acc2 = _mm256_fmadd_ps(sub_vec2, sub_vec2, acc2);
                acc3 = _mm256_fmadd_ps(sub_vec3, sub_vec3, acc3);
                acc4 = _mm256_fmadd_ps(sub_vec4, sub_vec4, acc4);
                acc5 = _mm256_fmadd_ps(sub_vec5, sub_vec5, acc5);
                acc6 = _mm256_fmadd_ps(sub_vec6, sub_vec6, acc6);
                acc7 = _mm256_fmadd_ps(sub_vec7, sub_vec7, acc7);

                acc8 = _mm256_fmadd_ps(sub_vec8, sub_vec8, acc8);
                acc9 = _mm256_fmadd_ps(sub_vec9, sub_vec9, acc9);
                acc10 = _mm256_fmadd_ps(sub_vec10, sub_vec10, acc10);
                acc11 = _mm256_fmadd_ps(sub_vec11, sub_vec11, acc11);
                acc12 = _mm256_fmadd_ps(sub_vec12, sub_vec12, acc12);
                acc13 = _mm256_fmadd_ps(sub_vec13, sub_vec13, acc13);
                acc14 = _mm256_fmadd_ps(sub_vec14, sub_vec14, acc14);
                acc15 = _mm256_fmadd_ps(sub_vec15, sub_vec15, acc15);

                acc16 = _mm256_fmadd_ps(sub_vec16, sub_vec16, acc16);
                acc17 = _mm256_fmadd_ps(sub_vec17, sub_vec17, acc17);
                acc18 = _mm256_fmadd_ps(sub_vec18, sub_vec18, acc18);
                acc19 = _mm256_fmadd_ps(sub_vec19, sub_vec19, acc19);
                acc20 = _mm256_fmadd_ps(sub_vec20, sub_vec20, acc20);
                acc21 = _mm256_fmadd_ps(sub_vec21, sub_vec21, acc21);
                acc22 = _mm256_fmadd_ps(sub_vec22, sub_vec22, acc22);
                acc23 = _mm256_fmadd_ps(sub_vec23, sub_vec23, acc23);

                acc24 = _mm256_fmadd_ps(sub_vec24, sub_vec24, acc24);
                acc25 = _mm256_fmadd_ps(sub_vec25, sub_vec25, acc25);
                acc26 = _mm256_fmadd_ps(sub_vec26, sub_vec26, acc26);
                acc27 = _mm256_fmadd_ps(sub_vec27, sub_vec27, acc27);
                acc28 = _mm256_fmadd_ps(sub_vec28, sub_vec28, acc28);
                acc29 = _mm256_fmadd_ps(sub_vec29, sub_vec29, acc29);
                acc30 = _mm256_fmadd_ps(sub_vec30, sub_vec30, acc30);
                acc31 = _mm256_fmadd_ps(sub_vec31, sub_vec31, acc31);

                acc32 = _mm256_fmadd_ps(sub_vec32, sub_vec32, acc32);
                acc33 = _mm256_fmadd_ps(sub_vec33, sub_vec33, acc33);
                acc34 = _mm256_fmadd_ps(sub_vec34, sub_vec34, acc34);
                acc35 = _mm256_fmadd_ps(sub_vec35, sub_vec35, acc35);
                acc36 = _mm256_fmadd_ps(sub_vec36, sub_vec36, acc36);
                acc37 = _mm256_fmadd_ps(sub_vec37, sub_vec37, acc37);
                acc38 = _mm256_fmadd_ps(sub_vec38, sub_vec38, acc38);
                acc39 = _mm256_fmadd_ps(sub_vec39, sub_vec39, acc39);

                acc40 = _mm256_fmadd_ps(sub_vec40, sub_vec40, acc40);
                acc41 = _mm256_fmadd_ps(sub_vec41, sub_vec41, acc41);
                acc42 = _mm256_fmadd_ps(sub_vec42, sub_vec42, acc42);
                acc43 = _mm256_fmadd_ps(sub_vec43, sub_vec43, acc43);
                acc44 = _mm256_fmadd_ps(sub_vec44, sub_vec44, acc44);
                acc45 = _mm256_fmadd_ps(sub_vec45, sub_vec45, acc45);
                acc46 = _mm256_fmadd_ps(sub_vec46, sub_vec46, acc46);
                acc47 = _mm256_fmadd_ps(sub_vec47, sub_vec47, acc47);

                acc48 = _mm256_fmadd_ps(sub_vec48, sub_vec48, acc48);
                acc49 = _mm256_fmadd_ps(sub_vec49, sub_vec49, acc49);
                acc50 = _mm256_fmadd_ps(sub_vec50, sub_vec50, acc50);
                acc51 = _mm256_fmadd_ps(sub_vec51, sub_vec51, acc51);
                acc52 = _mm256_fmadd_ps(sub_vec52, sub_vec52, acc52);
                acc53 = _mm256_fmadd_ps(sub_vec53, sub_vec53, acc53);
                acc54 = _mm256_fmadd_ps(sub_vec54, sub_vec54, acc54);
                acc55 = _mm256_fmadd_ps(sub_vec55, sub_vec55, acc55);

                acc56 = _mm256_fmadd_ps(sub_vec56, sub_vec56, acc56);
                acc57 = _mm256_fmadd_ps(sub_vec57, sub_vec57, acc57);
                acc58 = _mm256_fmadd_ps(sub_vec58, sub_vec58, acc58);
                acc59 = _mm256_fmadd_ps(sub_vec59, sub_vec59, acc59);
                acc60 = _mm256_fmadd_ps(sub_vec60, sub_vec60, acc60);
                acc61 = _mm256_fmadd_ps(sub_vec61, sub_vec61, acc61);
                acc62 = _mm256_fmadd_ps(sub_vec62, sub_vec62, acc62);
                acc63 = _mm256_fmadd_ps(sub_vec63, sub_vec63, acc63);

            }
            f0 = sum8(acc0);
            f1 = sum8(acc1);
            f2 = sum8(acc2);
            f3 = sum8(acc3);
            f4 = sum8(acc4);
            f5 = sum8(acc5);
            f6 = sum8(acc6);
            f7 = sum8(acc7);

            vec_sum0 = _mm256_set_ps(f7, f6, f5, f4, f3, f2, f1, f0);

            f8 = sum8(acc8);
            f9 = sum8(acc9);
            f10 = sum8(acc10);
            f11 = sum8(acc11);
            f12 = sum8(acc12);
            f13 = sum8(acc13);
            f14 = sum8(acc14);
            f15 = sum8(acc15);

            vec_sum1 = _mm256_set_ps(f15, f14, f13, f12, f11, f10, f9, f8);

            f16 = sum8(acc16);
            f17 = sum8(acc17);
            f18 = sum8(acc18);
            f19 = sum8(acc19);
            f20 = sum8(acc20);
            f21 = sum8(acc21);
            f22 = sum8(acc22);
            f23 = sum8(acc23);

            vec_sum2 = _mm256_set_ps(f23, f22, f21, f20, f19, f18, f17, f16);

            f24 = sum8(acc24);
            f25 = sum8(acc25);
            f26 = sum8(acc26);
            f27 = sum8(acc27);
            f28 = sum8(acc28);
            f29 = sum8(acc29);
            f30 = sum8(acc30);
            f31 = sum8(acc31);

            vec_sum3 = _mm256_set_ps(f31, f30, f29, f28, f27, f26, f25, f24);

            f32 = sum8(acc32);
            f33 = sum8(acc33);
            f34 = sum8(acc34);
            f35 = sum8(acc35);
            f36 = sum8(acc36);
            f37 = sum8(acc37);
            f38 = sum8(acc38);
            f39 = sum8(acc39);

            vec_sum4 = _mm256_set_ps(f39, f38, f37, f36, f35, f34, f33, f32);

            f40 = sum8(acc40);
            f41 = sum8(acc41);
            f42 = sum8(acc42);
            f43 = sum8(acc43);
            f44 = sum8(acc44);
            f45 = sum8(acc45);
            f46 = sum8(acc46);
            f47 = sum8(acc47);

            vec_sum5 = _mm256_set_ps(f47, f46, f45, f44, f43, f42, f41, f40);

            f48 = sum8(acc48);
            f49 = sum8(acc49);
            f50 = sum8(acc50);
            f51 = sum8(acc51);
            f52 = sum8(acc52);
            f53 = sum8(acc53);
            f54 = sum8(acc54);
            f55 = sum8(acc55);

            vec_sum6 = _mm256_set_ps(f55, f54, f53, f52, f51, f50, f49, f48);

            f56 = sum8(acc56);
            f57 = sum8(acc57);
            f58 = sum8(acc58);
            f59 = sum8(acc59);
            f60 = sum8(acc60);
            f61 = sum8(acc61);
            f62 = sum8(acc62);
            f63 = sum8(acc63);

            vec_sum7 = _mm256_set_ps(f63, f62, f61, f60, f59, f58, f57, f56);

            tmp0 = _mm256_load_ps(distances.data + bi * N + bj);
            tmp1 = _mm256_load_ps(distances.data + (bi+1) * N + bj);
            tmp2 = _mm256_load_ps(distances.data + (bi+2) * N + bj);
            tmp3 = _mm256_load_ps(distances.data + (bi+3) * N + bj);
            tmp4 = _mm256_load_ps(distances.data + (bi+4) * N + bj);
            tmp5 = _mm256_load_ps(distances.data + (bi+5) * N + bj);
            tmp6 = _mm256_load_ps(distances.data + (bi+6) * N + bj);
            tmp7 = _mm256_load_ps(distances.data + (bi+7) * N + bj);

            tmp0 = _mm256_add_ps(tmp0, vec_sum0);
            tmp1 = _mm256_add_ps(tmp1, vec_sum1);
            tmp2 = _mm256_add_ps(tmp2, vec_sum2);
            tmp3 = _mm256_add_ps(tmp3, vec_sum3);
            tmp4 = _mm256_add_ps(tmp4, vec_sum4);
            tmp5 = _mm256_add_ps(tmp5, vec_sum5);
            tmp6 = _mm256_add_ps(tmp6, vec_sum6);
            tmp7 = _mm256_add_ps(tmp7, vec_sum7);

            _mm256_store_ps(distances.data + (bi) * N + bj, tmp0);
            _mm256_store_ps(distances.data + (bi+1) * N + bj, tmp1);
            _mm256_store_ps(distances.data + (bi+2) * N + bj, tmp2);
            _mm256_store_ps(distances.data + (bi+3) * N + bj, tmp3);
            _mm256_store_ps(distances.data + (bi+4) * N + bj, tmp4);
            _mm256_store_ps(distances.data + (bi+5) * N + bj, tmp5);
            _mm256_store_ps(distances.data + (bi+6) * N + bj, tmp6);
            _mm256_store_ps(distances.data + (bi+7) * N + bj, tmp7);

        }
    }

    for (int i_tst = 0; i_tst < N_tst; i_tst += 4) {
#pragma GCC ivdep
        for (int i_trn = 0; i_trn < N; i_trn++) {
            dist_arr[i_trn].value = distances.data[i_tst * N + i_trn];
            dist_arr_i[i_trn].value = distances.data[(i_tst + 1) * N + i_trn];
            dist_arr_ii[i_trn].value = distances.data[(i_tst + 2) * N + i_trn];
            dist_arr_iii[i_trn].value = distances.data[(i_tst + 3) * N + i_trn];
            dist_arr[i_trn].index = i_trn;
            dist_arr_i[i_trn].index = i_trn;
            dist_arr_ii[i_trn].index = i_trn;
            dist_arr_iii[i_trn].index = i_trn;
        }

        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_i, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iii, N, sizeof(pair_t), cmp);
#pragma GCC ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
            x_tst_knn_gt->data[(i_tst + 1) * N + k] = dist_arr_i[k].index;
            x_tst_knn_gt->data[(i_tst + 2) * N + k] = dist_arr_ii[k].index;
            x_tst_knn_gt->data[(i_tst + 3) * N + k] = dist_arr_iii[k].index;
        }
    }
    free(dist_arr);
    free(dist_arr_i);
    free(dist_arr_ii);
    free(dist_arr_iii);
    destroy(&distances);
}

// vectorized blocking 8x8 - horizontal sum outside + fmas + reducing size of the AoS to 8 rows
void get_true_knn_opt21(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst) {
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 8;
    pair_t *__restrict__ dist_arr, *__restrict__ dist_arr_i, *__restrict__ dist_arr_ii, *__restrict__ dist_arr_iii, *__restrict__ dist_arr_iiii, *__restrict__ dist_arr_iiiii, *__restrict__ dist_arr_iiiiii, *__restrict__ dist_arr_iiiiiii;
    dist_arr = malloc(N * sizeof(pair_t));
    dist_arr_i = malloc(N * sizeof(pair_t));
    dist_arr_ii = malloc(N * sizeof(pair_t));
    dist_arr_iii = malloc(N * sizeof(pair_t));
    dist_arr_iiii = malloc(N * sizeof(pair_t));
    dist_arr_iiiii = malloc(N * sizeof(pair_t));
    dist_arr_iiiiii = malloc(N * sizeof(pair_t));
    dist_arr_iiiiiii = malloc(N * sizeof(pair_t));

    float *__restrict__ data_trn = x_trn->data;
    float *__restrict__ data_tst = x_tst->data;
    int i, j, k, bi, bj, bk;
    __m256 ts_vec0, tr_vec0, ts_vec1, tr_vec1, ts_vec2, tr_vec2, ts_vec3, tr_vec3, ts_vec4, tr_vec4, ts_vec5, tr_vec5, ts_vec6, tr_vec6, ts_vec7, tr_vec7;
    __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3, vec_sum4, vec_sum5, vec_sum6, vec_sum7;
    __m256 sub_vec0, sub_vec1, sub_vec2, sub_vec3, sub_vec4, sub_vec5, sub_vec6, sub_vec7;
    __m256 sub_vec8, sub_vec9, sub_vec10, sub_vec11, sub_vec12, sub_vec13, sub_vec14;
    __m256 sub_vec15, sub_vec16, sub_vec17, sub_vec18, sub_vec19, sub_vec20, sub_vec21;
    __m256 sub_vec22, sub_vec23, sub_vec24, sub_vec25, sub_vec26, sub_vec27, sub_vec28;
    __m256 sub_vec29, sub_vec30, sub_vec31, sub_vec32, sub_vec33, sub_vec34, sub_vec35;
    __m256 sub_vec36, sub_vec37, sub_vec38, sub_vec39, sub_vec40, sub_vec41, sub_vec42;
    __m256 sub_vec43, sub_vec44, sub_vec45, sub_vec46, sub_vec47, sub_vec48, sub_vec49;
    __m256 sub_vec50, sub_vec51, sub_vec52, sub_vec53, sub_vec54, sub_vec55, sub_vec56;
    __m256 sub_vec57, sub_vec58, sub_vec59, sub_vec60, sub_vec61, sub_vec62, sub_vec63;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;
    float f16, f17, f18, f19, f20, f21, f22, f23;
    float f24, f25, f26, f27, f28, f29, f30, f31;
    float f32, f33, f34, f35, f36, f37, f38, f39;
    float f40, f41, f42, f43, f44, f45, f46, f47;
    float f48, f49, f50, f51, f52, f53, f54, f55;
    float f56, f57, f58, f59, f60, f61, f62, f63;

    __m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m256 acc8, acc9, acc10, acc11, acc12, acc13, acc14;
    __m256 acc15, acc16, acc17, acc18, acc19, acc20, acc21;
    __m256 acc22, acc23, acc24, acc25, acc26, acc27, acc28;
    __m256 acc29, acc30, acc31, acc32, acc33, acc34, acc35;
    __m256 acc36, acc37, acc38, acc39, acc40, acc41, acc42;
    __m256 acc43, acc44, acc45, acc46, acc47, acc48, acc49;
    __m256 acc50, acc51, acc52, acc53, acc54, acc55, acc56;
    __m256 acc57, acc58, acc59, acc60, acc61, acc62, acc63;


    for (bi = 0; bi < N_tst; bi += Nb) {

        for (bj = 0; bj < N; bj += Nb) {
#pragma unroll (8)
            for (j = 0; j < 8; j++){
                dist_arr[bj + j].index = bj + j;
                dist_arr_i[bj + j].index = bj + j;
                dist_arr_ii[bj + j].index = bj + j;
                dist_arr_iii[bj + j].index = bj + j;
                dist_arr_iiii[bj + j].index = bj + j;
                dist_arr_iiiii[bj + j].index = bj + j;
                dist_arr_iiiiii[bj + j].index = bj + j;
                dist_arr_iiiiiii[bj + j].index = bj + j;
            }
            // accumulators
            acc0 = _mm256_setzero_ps();
            acc1 = _mm256_setzero_ps();
            acc2 = _mm256_setzero_ps();
            acc3 = _mm256_setzero_ps();
            acc4 = _mm256_setzero_ps();
            acc5 = _mm256_setzero_ps();
            acc6 = _mm256_setzero_ps();
            acc7 = _mm256_setzero_ps();

            acc8 = _mm256_setzero_ps();
            acc9 = _mm256_setzero_ps();
            acc10 = _mm256_setzero_ps();
            acc11 = _mm256_setzero_ps();
            acc12 = _mm256_setzero_ps();
            acc13 = _mm256_setzero_ps();
            acc14 = _mm256_setzero_ps();
            acc15 = _mm256_setzero_ps();

            acc16 = _mm256_setzero_ps();
            acc17 = _mm256_setzero_ps();
            acc18 = _mm256_setzero_ps();
            acc19 = _mm256_setzero_ps();
            acc20 = _mm256_setzero_ps();
            acc21 = _mm256_setzero_ps();
            acc22 = _mm256_setzero_ps();
            acc23 = _mm256_setzero_ps();

            acc24 = _mm256_setzero_ps();
            acc25 = _mm256_setzero_ps();
            acc26 = _mm256_setzero_ps();
            acc27 = _mm256_setzero_ps();
            acc28 = _mm256_setzero_ps();
            acc29 = _mm256_setzero_ps();
            acc30 = _mm256_setzero_ps();
            acc31 = _mm256_setzero_ps();

            acc32 = _mm256_setzero_ps();
            acc33 = _mm256_setzero_ps();
            acc34 = _mm256_setzero_ps();
            acc35 = _mm256_setzero_ps();
            acc36 = _mm256_setzero_ps();
            acc37 = _mm256_setzero_ps();
            acc38 = _mm256_setzero_ps();
            acc39 = _mm256_setzero_ps();

            acc40 = _mm256_setzero_ps();
            acc41 = _mm256_setzero_ps();
            acc42 = _mm256_setzero_ps();
            acc43 = _mm256_setzero_ps();
            acc44 = _mm256_setzero_ps();
            acc45 = _mm256_setzero_ps();
            acc46 = _mm256_setzero_ps();
            acc47 = _mm256_setzero_ps();

            acc48 = _mm256_setzero_ps();
            acc49 = _mm256_setzero_ps();
            acc50 = _mm256_setzero_ps();
            acc51 = _mm256_setzero_ps();
            acc52 = _mm256_setzero_ps();
            acc53 = _mm256_setzero_ps();
            acc54 = _mm256_setzero_ps();
            acc55 = _mm256_setzero_ps();

            acc56 = _mm256_setzero_ps();
            acc57 = _mm256_setzero_ps();
            acc58 = _mm256_setzero_ps();
            acc59 = _mm256_setzero_ps();
            acc60 = _mm256_setzero_ps();
            acc61 = _mm256_setzero_ps();
            acc62 = _mm256_setzero_ps();
            acc63 = _mm256_setzero_ps();

#pragma ivdep
            for (bk = 0; bk < d; bk += Nb) {

                ts_vec0 = _mm256_load_ps(data_tst + bi * N + bk); // 8x8 blocks from test matrix
                ts_vec1 = _mm256_load_ps(data_tst + (bi + 1) * N + bk);
                ts_vec2 = _mm256_load_ps(data_tst + (bi + 2) * N + bk);
                ts_vec3 = _mm256_load_ps(data_tst + (bi + 3) * N + bk);
                ts_vec4 = _mm256_load_ps(data_tst + (bi + 4) * N + bk);
                ts_vec5 = _mm256_load_ps(data_tst + (bi + 5) * N + bk);
                ts_vec6 = _mm256_load_ps(data_tst + (bi + 6) * N + bk);
                ts_vec7 = _mm256_load_ps(data_tst + (bi + 7) * N + bk);

                tr_vec0 = _mm256_load_ps(data_trn + bj * N + bk); // 8x8 blocks from train matrix
                tr_vec1 = _mm256_load_ps(data_trn + (bj + 1) * N + bk);
                tr_vec2 = _mm256_load_ps(data_trn + (bj + 2) * N + bk);
                tr_vec3 = _mm256_load_ps(data_trn + (bj + 3) * N + bk);
                tr_vec4 = _mm256_load_ps(data_trn + (bj + 4) * N + bk);
                tr_vec5 = _mm256_load_ps(data_trn + (bj + 5) * N + bk);
                tr_vec6 = _mm256_load_ps(data_trn + (bj + 6) * N + bk);
                tr_vec7 = _mm256_load_ps(data_trn + (bj + 7) * N + bk);

                sub_vec0 = _mm256_sub_ps(ts_vec0, tr_vec0); // pairwise distances between test and train blocks
                sub_vec1 = _mm256_sub_ps(ts_vec0, tr_vec1); // 1x8 per code block, total 8x8 values computed
                sub_vec2 = _mm256_sub_ps(ts_vec0, tr_vec2);
                sub_vec3 = _mm256_sub_ps(ts_vec0, tr_vec3);
                sub_vec4 = _mm256_sub_ps(ts_vec0, tr_vec4);
                sub_vec5 = _mm256_sub_ps(ts_vec0, tr_vec5);
                sub_vec6 = _mm256_sub_ps(ts_vec0, tr_vec6);
                sub_vec7 = _mm256_sub_ps(ts_vec0, tr_vec7);

                sub_vec8 = _mm256_sub_ps(ts_vec1, tr_vec0);
                sub_vec9 = _mm256_sub_ps(ts_vec1, tr_vec1);
                sub_vec10 = _mm256_sub_ps(ts_vec1, tr_vec2);
                sub_vec11 = _mm256_sub_ps(ts_vec1, tr_vec3);
                sub_vec12 = _mm256_sub_ps(ts_vec1, tr_vec4);
                sub_vec13 = _mm256_sub_ps(ts_vec1, tr_vec5);
                sub_vec14 = _mm256_sub_ps(ts_vec1, tr_vec6);
                sub_vec15 = _mm256_sub_ps(ts_vec1, tr_vec7);

                sub_vec16 = _mm256_sub_ps(ts_vec2, tr_vec0);
                sub_vec17 = _mm256_sub_ps(ts_vec2, tr_vec1);
                sub_vec18 = _mm256_sub_ps(ts_vec2, tr_vec2);
                sub_vec19 = _mm256_sub_ps(ts_vec2, tr_vec3);
                sub_vec20 = _mm256_sub_ps(ts_vec2, tr_vec4);
                sub_vec21 = _mm256_sub_ps(ts_vec2, tr_vec5);
                sub_vec22 = _mm256_sub_ps(ts_vec2, tr_vec6);
                sub_vec23 = _mm256_sub_ps(ts_vec2, tr_vec7);

                sub_vec24 = _mm256_sub_ps(ts_vec3, tr_vec0);
                sub_vec25 = _mm256_sub_ps(ts_vec3, tr_vec1);
                sub_vec26 = _mm256_sub_ps(ts_vec3, tr_vec2);
                sub_vec27 = _mm256_sub_ps(ts_vec3, tr_vec3);
                sub_vec28 = _mm256_sub_ps(ts_vec3, tr_vec4);
                sub_vec29 = _mm256_sub_ps(ts_vec3, tr_vec5);
                sub_vec30 = _mm256_sub_ps(ts_vec3, tr_vec6);
                sub_vec31 = _mm256_sub_ps(ts_vec3, tr_vec7);

                sub_vec32 = _mm256_sub_ps(ts_vec4, tr_vec0);
                sub_vec33 = _mm256_sub_ps(ts_vec4, tr_vec1);
                sub_vec34 = _mm256_sub_ps(ts_vec4, tr_vec2);
                sub_vec35 = _mm256_sub_ps(ts_vec4, tr_vec3);
                sub_vec36 = _mm256_sub_ps(ts_vec4, tr_vec4);
                sub_vec37 = _mm256_sub_ps(ts_vec4, tr_vec5);
                sub_vec38 = _mm256_sub_ps(ts_vec4, tr_vec6);
                sub_vec39 = _mm256_sub_ps(ts_vec4, tr_vec7);

                sub_vec40 = _mm256_sub_ps(ts_vec5, tr_vec0);
                sub_vec41 = _mm256_sub_ps(ts_vec5, tr_vec1);
                sub_vec42 = _mm256_sub_ps(ts_vec5, tr_vec2);
                sub_vec43 = _mm256_sub_ps(ts_vec5, tr_vec3);
                sub_vec44 = _mm256_sub_ps(ts_vec5, tr_vec4);
                sub_vec45 = _mm256_sub_ps(ts_vec5, tr_vec5);
                sub_vec46 = _mm256_sub_ps(ts_vec5, tr_vec6);
                sub_vec47 = _mm256_sub_ps(ts_vec5, tr_vec7);

                sub_vec48 = _mm256_sub_ps(ts_vec6, tr_vec0);
                sub_vec49 = _mm256_sub_ps(ts_vec6, tr_vec1);
                sub_vec50 = _mm256_sub_ps(ts_vec6, tr_vec2);
                sub_vec51 = _mm256_sub_ps(ts_vec6, tr_vec3);
                sub_vec52 = _mm256_sub_ps(ts_vec6, tr_vec4);
                sub_vec53 = _mm256_sub_ps(ts_vec6, tr_vec5);
                sub_vec54 = _mm256_sub_ps(ts_vec6, tr_vec6);
                sub_vec55 = _mm256_sub_ps(ts_vec6, tr_vec7);

                sub_vec56 = _mm256_sub_ps(ts_vec7, tr_vec0);
                sub_vec57 = _mm256_sub_ps(ts_vec7, tr_vec1);
                sub_vec58 = _mm256_sub_ps(ts_vec7, tr_vec2);
                sub_vec59 = _mm256_sub_ps(ts_vec7, tr_vec3);
                sub_vec60 = _mm256_sub_ps(ts_vec7, tr_vec4);
                sub_vec61 = _mm256_sub_ps(ts_vec7, tr_vec5);
                sub_vec62 = _mm256_sub_ps(ts_vec7, tr_vec6);
                sub_vec63 = _mm256_sub_ps(ts_vec7, tr_vec7);

                acc0 = _mm256_fmadd_ps(sub_vec0, sub_vec0, acc0); // squared distance, 8x8
                acc1 = _mm256_fmadd_ps(sub_vec1, sub_vec1, acc1);
                acc2 = _mm256_fmadd_ps(sub_vec2, sub_vec2, acc2);
                acc3 = _mm256_fmadd_ps(sub_vec3, sub_vec3, acc3);
                acc4 = _mm256_fmadd_ps(sub_vec4, sub_vec4, acc4);
                acc5 = _mm256_fmadd_ps(sub_vec5, sub_vec5, acc5);
                acc6 = _mm256_fmadd_ps(sub_vec6, sub_vec6, acc6);
                acc7 = _mm256_fmadd_ps(sub_vec7, sub_vec7, acc7);

                acc8 = _mm256_fmadd_ps(sub_vec8, sub_vec8, acc8);
                acc9 = _mm256_fmadd_ps(sub_vec9, sub_vec9, acc9);
                acc10 = _mm256_fmadd_ps(sub_vec10, sub_vec10, acc10);
                acc11 = _mm256_fmadd_ps(sub_vec11, sub_vec11, acc11);
                acc12 = _mm256_fmadd_ps(sub_vec12, sub_vec12, acc12);
                acc13 = _mm256_fmadd_ps(sub_vec13, sub_vec13, acc13);
                acc14 = _mm256_fmadd_ps(sub_vec14, sub_vec14, acc14);
                acc15 = _mm256_fmadd_ps(sub_vec15, sub_vec15, acc15);

                acc16 = _mm256_fmadd_ps(sub_vec16, sub_vec16, acc16);
                acc17 = _mm256_fmadd_ps(sub_vec17, sub_vec17, acc17);
                acc18 = _mm256_fmadd_ps(sub_vec18, sub_vec18, acc18);
                acc19 = _mm256_fmadd_ps(sub_vec19, sub_vec19, acc19);
                acc20 = _mm256_fmadd_ps(sub_vec20, sub_vec20, acc20);
                acc21 = _mm256_fmadd_ps(sub_vec21, sub_vec21, acc21);
                acc22 = _mm256_fmadd_ps(sub_vec22, sub_vec22, acc22);
                acc23 = _mm256_fmadd_ps(sub_vec23, sub_vec23, acc23);

                acc24 = _mm256_fmadd_ps(sub_vec24, sub_vec24, acc24);
                acc25 = _mm256_fmadd_ps(sub_vec25, sub_vec25, acc25);
                acc26 = _mm256_fmadd_ps(sub_vec26, sub_vec26, acc26);
                acc27 = _mm256_fmadd_ps(sub_vec27, sub_vec27, acc27);
                acc28 = _mm256_fmadd_ps(sub_vec28, sub_vec28, acc28);
                acc29 = _mm256_fmadd_ps(sub_vec29, sub_vec29, acc29);
                acc30 = _mm256_fmadd_ps(sub_vec30, sub_vec30, acc30);
                acc31 = _mm256_fmadd_ps(sub_vec31, sub_vec31, acc31);

                acc32 = _mm256_fmadd_ps(sub_vec32, sub_vec32, acc32);
                acc33 = _mm256_fmadd_ps(sub_vec33, sub_vec33, acc33);
                acc34 = _mm256_fmadd_ps(sub_vec34, sub_vec34, acc34);
                acc35 = _mm256_fmadd_ps(sub_vec35, sub_vec35, acc35);
                acc36 = _mm256_fmadd_ps(sub_vec36, sub_vec36, acc36);
                acc37 = _mm256_fmadd_ps(sub_vec37, sub_vec37, acc37);
                acc38 = _mm256_fmadd_ps(sub_vec38, sub_vec38, acc38);
                acc39 = _mm256_fmadd_ps(sub_vec39, sub_vec39, acc39);

                acc40 = _mm256_fmadd_ps(sub_vec40, sub_vec40, acc40);
                acc41 = _mm256_fmadd_ps(sub_vec41, sub_vec41, acc41);
                acc42 = _mm256_fmadd_ps(sub_vec42, sub_vec42, acc42);
                acc43 = _mm256_fmadd_ps(sub_vec43, sub_vec43, acc43);
                acc44 = _mm256_fmadd_ps(sub_vec44, sub_vec44, acc44);
                acc45 = _mm256_fmadd_ps(sub_vec45, sub_vec45, acc45);
                acc46 = _mm256_fmadd_ps(sub_vec46, sub_vec46, acc46);
                acc47 = _mm256_fmadd_ps(sub_vec47, sub_vec47, acc47);

                acc48 = _mm256_fmadd_ps(sub_vec48, sub_vec48, acc48);
                acc49 = _mm256_fmadd_ps(sub_vec49, sub_vec49, acc49);
                acc50 = _mm256_fmadd_ps(sub_vec50, sub_vec50, acc50);
                acc51 = _mm256_fmadd_ps(sub_vec51, sub_vec51, acc51);
                acc52 = _mm256_fmadd_ps(sub_vec52, sub_vec52, acc52);
                acc53 = _mm256_fmadd_ps(sub_vec53, sub_vec53, acc53);
                acc54 = _mm256_fmadd_ps(sub_vec54, sub_vec54, acc54);
                acc55 = _mm256_fmadd_ps(sub_vec55, sub_vec55, acc55);

                acc56 = _mm256_fmadd_ps(sub_vec56, sub_vec56, acc56);
                acc57 = _mm256_fmadd_ps(sub_vec57, sub_vec57, acc57);
                acc58 = _mm256_fmadd_ps(sub_vec58, sub_vec58, acc58);
                acc59 = _mm256_fmadd_ps(sub_vec59, sub_vec59, acc59);
                acc60 = _mm256_fmadd_ps(sub_vec60, sub_vec60, acc60);
                acc61 = _mm256_fmadd_ps(sub_vec61, sub_vec61, acc61);
                acc62 = _mm256_fmadd_ps(sub_vec62, sub_vec62, acc62);
                acc63 = _mm256_fmadd_ps(sub_vec63, sub_vec63, acc63);

            }

            dist_arr[bj].value = sum8(acc0);
            dist_arr[bj + 1].value  = sum8(acc1);
            dist_arr[bj + 2].value = sum8(acc2);
            dist_arr[bj + 3].value  = sum8(acc3);
            dist_arr[bj + 4].value  = sum8(acc4);
            dist_arr[bj + 5].value  = sum8(acc5);
            dist_arr[bj + 6].value  = sum8(acc6);
            dist_arr[bj + 7].value  = sum8(acc7);
            
            dist_arr_i[bj].value  = sum8(acc8);
            dist_arr_i[bj + 1].value  = sum8(acc9);
            dist_arr_i[bj + 2].value  =  sum8(acc10);
            dist_arr_i[bj + 3].value  =  sum8(acc11);
            dist_arr_i[bj + 4].value  =  sum8(acc12);
            dist_arr_i[bj + 5].value  =  sum8(acc13);
            dist_arr_i[bj + 6].value  =  sum8(acc14);
            dist_arr_i[bj + 7].value  =  sum8(acc15);
            
            dist_arr_ii[bj].value = sum8(acc16);
            dist_arr_ii[bj + 1].value = sum8(acc17);
            dist_arr_ii[bj + 2].value = sum8(acc18);
            dist_arr_ii[bj + 3].value = sum8(acc19);
            dist_arr_ii[bj + 4].value = sum8(acc20);
            dist_arr_ii[bj + 5].value = sum8(acc21);
            dist_arr_ii[bj + 6].value = sum8(acc22);
            dist_arr_ii[bj + 7].value = sum8(acc23);
            
            dist_arr_iii[bj].value = sum8(acc24);
            dist_arr_iii[bj + 1].value = sum8(acc25);
            dist_arr_iii[bj + 2].value = sum8(acc26);
            dist_arr_iii[bj + 3].value = sum8(acc27);
            dist_arr_iii[bj + 4].value = sum8(acc28);
            dist_arr_iii[bj + 5].value = sum8(acc29);
            dist_arr_iii[bj + 6].value = sum8(acc30);
            dist_arr_iii[bj + 7].value = sum8(acc31);
            
            dist_arr_iiii[bj].value = sum8(acc32);
            dist_arr_iiii[bj + 1].value = sum8(acc33);
            dist_arr_iiii[bj + 2].value = sum8(acc34);
            dist_arr_iiii[bj + 3].value = sum8(acc35);
            dist_arr_iiii[bj + 4].value = sum8(acc36);
            dist_arr_iiii[bj + 5].value = sum8(acc37);
            dist_arr_iiii[bj + 6].value = sum8(acc38);
            dist_arr_iiii[bj + 7].value = sum8(acc39);
            
            dist_arr_iiiii[bj].value = sum8(acc40);
            dist_arr_iiiii[bj + 1].value = sum8(acc41);
            dist_arr_iiiii[bj + 2].value = sum8(acc42);
            dist_arr_iiiii[bj + 3].value = sum8(acc43);
            dist_arr_iiiii[bj + 4].value = sum8(acc44);
            dist_arr_iiiii[bj + 5].value = sum8(acc45);
            dist_arr_iiiii[bj + 6].value = sum8(acc46);
            dist_arr_iiiii[bj + 7].value = sum8(acc47);

            dist_arr_iiiiii[bj].value = sum8(acc48);
            dist_arr_iiiiii[bj + 1].value = sum8(acc49);
            dist_arr_iiiiii[bj + 2].value = sum8(acc50);
            dist_arr_iiiiii[bj + 3].value = sum8(acc51);
            dist_arr_iiiiii[bj + 4].value = sum8(acc52);
            dist_arr_iiiiii[bj + 5].value = sum8(acc53);
            dist_arr_iiiiii[bj + 6].value = sum8(acc54);
            dist_arr_iiiiii[bj + 7].value = sum8(acc55);

            
            dist_arr_iiiiiii[bj].value = sum8(acc56);
            dist_arr_iiiiiii[bj + 1].value = sum8(acc57);
            dist_arr_iiiiiii[bj + 2].value = sum8(acc58);
            dist_arr_iiiiiii[bj + 3].value = sum8(acc59);
            dist_arr_iiiiiii[bj + 4].value = sum8(acc60);
            dist_arr_iiiiiii[bj + 5].value = sum8(acc61);
            dist_arr_iiiiiii[bj + 6].value = sum8(acc62);
            dist_arr_iiiiiii[bj + 7].value = sum8(acc63);

        }
        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_i, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iiii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iiiii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iiiiii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iiiiiii, N, sizeof(pair_t), cmp);

#pragma GCC ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[bi * N + k] = dist_arr[k].index;
            x_tst_knn_gt->data[(bi + 1) * N + k] = dist_arr_i[k].index;
            x_tst_knn_gt->data[(bi + 2) * N + k] = dist_arr_ii[k].index;
            x_tst_knn_gt->data[(bi + 3) * N + k] = dist_arr_iii[k].index;
            x_tst_knn_gt->data[(bi + 4) * N + k] = dist_arr_iiii[k].index;
            x_tst_knn_gt->data[(bi + 5) * N + k] = dist_arr_iiiii[k].index;
            x_tst_knn_gt->data[(bi + 6) * N + k] = dist_arr_iiiiii[k].index;
            x_tst_knn_gt->data[(bi + 7) * N + k] = dist_arr_iiiiiii[k].index;
        }
    }

    free(dist_arr);
    free(dist_arr_i);
    free(dist_arr_ii);
    free(dist_arr_iii);
    free(dist_arr_iiii);
    free(dist_arr_iiiii);
    free(dist_arr_iiiiii);
    free(dist_arr_iiiiiii);
}

// vectorized blocking 8x16 - horizontal sum outside
void get_true_knn_opt22(int_mat *x_tst_knn_gt, mat *x_trn, mat *x_tst){
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    int Nb = 8;
    pair_t *dist_arr;
    dist_arr = malloc(N * sizeof(pair_t));
    mat distances;
    build(&distances, N_tst, N);
    initialize_mat(&distances, 0.0);
    float* data_trn = x_trn->data;
    float* data_tst = x_tst->data;
    int k, bi, bj, bk;
    __m256 vec0_0, vec1_0, vec2_0, vec3_0, vec4_0, vec5_0, vec6_0, vec7_0, vec8_0, vec9_0, vec10_0, vec11_0, vec12_0, vec13_0, vec14_0, vec15_0;
    __m256 vec0_1, vec1_1, vec2_1, vec3_1, vec4_1, vec5_1, vec6_1, vec7_1, vec8_1, vec9_1, vec10_1, vec11_1, vec12_1, vec13_1, vec14_1, vec15_1;
    __m256 tmp_vec0, tmp_vec1, tmp_vec2, tmp_vec3, tmp_vec4, tmp_vec5, tmp_vec6, tmp_vec7;

    __m256 sub_vec0_0, sub_vec1_0, sub_vec2_0, sub_vec3_0, sub_vec4_0, sub_vec5_0, sub_vec6_0, sub_vec7_0;
    __m256 sub_vec8_0, sub_vec9_0, sub_vec10_0, sub_vec11_0, sub_vec12_0, sub_vec13_0, sub_vec14_0;
    __m256 sub_vec15_0, sub_vec16_0, sub_vec17_0, sub_vec18_0, sub_vec19_0, sub_vec20_0, sub_vec21_0;
    __m256 sub_vec22_0, sub_vec23_0, sub_vec24_0, sub_vec25_0, sub_vec26_0, sub_vec27_0, sub_vec28_0;
    __m256 sub_vec29_0, sub_vec30_0, sub_vec31_0, sub_vec32_0, sub_vec33_0, sub_vec34_0, sub_vec35_0;
    __m256 sub_vec36_0, sub_vec37_0, sub_vec38_0, sub_vec39_0, sub_vec40_0, sub_vec41_0, sub_vec42_0;
    __m256 sub_vec43_0, sub_vec44_0, sub_vec45_0, sub_vec46_0, sub_vec47_0, sub_vec48_0, sub_vec49_0;
    __m256 sub_vec50_0, sub_vec51_0, sub_vec52_0, sub_vec53_0, sub_vec54_0, sub_vec55_0, sub_vec56_0;
    __m256 sub_vec57_0, sub_vec58_0, sub_vec59_0, sub_vec60_0, sub_vec61_0, sub_vec62_0, sub_vec63_0;

    __m256 sub_vec0_1, sub_vec1_1, sub_vec2_1, sub_vec3_1, sub_vec4_1, sub_vec5_1, sub_vec6_1, sub_vec7_1;
    __m256 sub_vec8_1, sub_vec9_1, sub_vec10_1, sub_vec11_1, sub_vec12_1, sub_vec13_1, sub_vec14_1;
    __m256 sub_vec15_1, sub_vec16_1, sub_vec17_1, sub_vec18_1, sub_vec19_1, sub_vec20_1, sub_vec21_1;
    __m256 sub_vec22_1, sub_vec23_1, sub_vec24_1, sub_vec25_1, sub_vec26_1, sub_vec27_1, sub_vec28_1;
    __m256 sub_vec29_1, sub_vec30_1, sub_vec31_1, sub_vec32_1, sub_vec33_1, sub_vec34_1, sub_vec35_1;
    __m256 sub_vec36_1, sub_vec37_1, sub_vec38_1, sub_vec39_1, sub_vec40_1, sub_vec41_1, sub_vec42_1;
    __m256 sub_vec43_1, sub_vec44_1, sub_vec45_1, sub_vec46_1, sub_vec47_1, sub_vec48_1, sub_vec49_1;
    __m256 sub_vec50_1, sub_vec51_1, sub_vec52_1, sub_vec53_1, sub_vec54_1, sub_vec55_1, sub_vec56_1;
    __m256 sub_vec57_1, sub_vec58_1, sub_vec59_1, sub_vec60_1, sub_vec61_1, sub_vec62_1, sub_vec63_1;

    __m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    __m256 acc8, acc9, acc10, acc11, acc12, acc13, acc14;
    __m256 acc15, acc16, acc17, acc18, acc19, acc20, acc21;
    __m256 acc22, acc23, acc24, acc25, acc26, acc27, acc28;
    __m256 acc29, acc30, acc31, acc32, acc33, acc34, acc35;
    __m256 acc36, acc37, acc38, acc39, acc40, acc41, acc42;
    __m256 acc43, acc44, acc45, acc46, acc47, acc48, acc49;
    __m256 acc50, acc51, acc52, acc53, acc54, acc55, acc56;
    __m256 acc57, acc58, acc59, acc60, acc61, acc62, acc63;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;
    float f16, f17, f18, f19, f20, f21, f22, f23;
    float f24, f25, f26, f27, f28, f29, f30, f31;
    float f32, f33, f34, f35, f36, f37, f38, f39;
    float f40, f41, f42, f43, f44, f45, f46, f47;
    float f48, f49, f50, f51, f52, f53, f54, f55;
    float f56, f57, f58, f59, f60, f61, f62, f63;

    __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3, vec_sum4, vec_sum5, vec_sum6, vec_sum7;


    for(bi=0; bi<N_tst; bi+=Nb) {
        for (bj = 0; bj < N; bj += Nb) {
            // accumulators
            acc0 = _mm256_setzero_ps();
            acc1 = _mm256_setzero_ps();
            acc2 = _mm256_setzero_ps();
            acc3 = _mm256_setzero_ps();
            acc4 = _mm256_setzero_ps();
            acc5 = _mm256_setzero_ps();
            acc6 = _mm256_setzero_ps();
            acc7 = _mm256_setzero_ps();

            acc8 = _mm256_setzero_ps();
            acc9 = _mm256_setzero_ps();
            acc10 = _mm256_setzero_ps();
            acc11 = _mm256_setzero_ps();
            acc12 = _mm256_setzero_ps();
            acc13 = _mm256_setzero_ps();
            acc14 = _mm256_setzero_ps();
            acc15 = _mm256_setzero_ps();

            acc16 = _mm256_setzero_ps();
            acc17 = _mm256_setzero_ps();
            acc18 = _mm256_setzero_ps();
            acc19 = _mm256_setzero_ps();
            acc20 = _mm256_setzero_ps();
            acc21 = _mm256_setzero_ps();
            acc22 = _mm256_setzero_ps();
            acc23 = _mm256_setzero_ps();

            acc24 = _mm256_setzero_ps();
            acc25 = _mm256_setzero_ps();
            acc26 = _mm256_setzero_ps();
            acc27 = _mm256_setzero_ps();
            acc28 = _mm256_setzero_ps();
            acc29 = _mm256_setzero_ps();
            acc30 = _mm256_setzero_ps();
            acc31 = _mm256_setzero_ps();

            acc32 = _mm256_setzero_ps();
            acc33 = _mm256_setzero_ps();
            acc34 = _mm256_setzero_ps();
            acc35 = _mm256_setzero_ps();
            acc36 = _mm256_setzero_ps();
            acc37 = _mm256_setzero_ps();
            acc38 = _mm256_setzero_ps();
            acc39 = _mm256_setzero_ps();

            acc40 = _mm256_setzero_ps();
            acc41 = _mm256_setzero_ps();
            acc42 = _mm256_setzero_ps();
            acc43 = _mm256_setzero_ps();
            acc44 = _mm256_setzero_ps();
            acc45 = _mm256_setzero_ps();
            acc46 = _mm256_setzero_ps();
            acc47 = _mm256_setzero_ps();

            acc48 = _mm256_setzero_ps();
            acc49 = _mm256_setzero_ps();
            acc50 = _mm256_setzero_ps();
            acc51 = _mm256_setzero_ps();
            acc52 = _mm256_setzero_ps();
            acc53 = _mm256_setzero_ps();
            acc54 = _mm256_setzero_ps();
            acc55 = _mm256_setzero_ps();

            acc56 = _mm256_setzero_ps();
            acc57 = _mm256_setzero_ps();
            acc58 = _mm256_setzero_ps();
            acc59 = _mm256_setzero_ps();
            acc60 = _mm256_setzero_ps();
            acc61 = _mm256_setzero_ps();
            acc62 = _mm256_setzero_ps();
            acc63 = _mm256_setzero_ps();

            for (bk = 0; bk < d; bk += 2*Nb) {
                vec0_0 = _mm256_loadu_ps(data_tst + bi * N + bk);
                vec0_1 = _mm256_loadu_ps(data_tst + bi * N + bk + 8);
                vec1_0 = _mm256_loadu_ps(data_trn + bj * N + bk);
                vec1_1 = _mm256_loadu_ps(data_trn + bj * N + bk + 8);
                vec2_0 = _mm256_loadu_ps(data_tst + (bi + 1) * N + bk);
                vec2_1 = _mm256_loadu_ps(data_tst + (bi + 1) * N + bk + 8);
                vec3_0 = _mm256_loadu_ps(data_trn + (bj + 1) * N + bk);
                vec3_1 = _mm256_loadu_ps(data_trn + (bj + 1) * N + bk + 8);
                vec4_0 = _mm256_loadu_ps(data_tst + (bi + 2) * N + bk);
                vec4_1 = _mm256_loadu_ps(data_tst + (bi + 2) * N + bk + 8);
                vec5_0 = _mm256_loadu_ps(data_trn + (bj + 2) * N + bk);
                vec5_1 = _mm256_loadu_ps(data_trn + (bj + 2) * N + bk + 8);
                vec6_0 = _mm256_loadu_ps(data_tst + (bi + 3) * N + bk);
                vec6_1 = _mm256_loadu_ps(data_tst + (bi + 3) * N + bk + 8);
                vec7_0 = _mm256_loadu_ps(data_trn + (bj + 3) * N + bk);
                vec7_1 = _mm256_loadu_ps(data_trn + (bj + 3) * N + bk + 8);
                vec8_0 = _mm256_loadu_ps(data_tst + (bi + 4) * N + bk);
                vec8_1 = _mm256_loadu_ps(data_tst + (bi + 4) * N + bk + 8);
                vec9_0 = _mm256_loadu_ps(data_trn + (bj + 4) * N + bk);
                vec9_1 = _mm256_loadu_ps(data_trn + (bj + 4) * N + bk + 8);
                vec10_0 = _mm256_loadu_ps(data_tst + (bi + 5) * N + bk);
                vec10_1 = _mm256_loadu_ps(data_tst + (bi + 5) * N + bk + 8);
                vec11_0 = _mm256_loadu_ps(data_trn + (bj + 5) * N + bk);
                vec11_1 = _mm256_loadu_ps(data_trn + (bj + 5) * N + bk + 8);
                vec12_0 = _mm256_loadu_ps(data_tst + (bi + 6) * N + bk);
                vec12_1 = _mm256_loadu_ps(data_tst + (bi + 6) * N + bk + 8);
                vec13_0 = _mm256_loadu_ps(data_trn + (bj + 6) * N + bk);
                vec13_1 = _mm256_loadu_ps(data_trn + (bj + 6) * N + bk + 8);
                vec14_0 = _mm256_loadu_ps(data_tst + (bi + 7) * N + bk);
                vec14_1 = _mm256_loadu_ps(data_tst + (bi + 7) * N + bk + 8);
                vec15_0 = _mm256_loadu_ps(data_trn + (bj + 7) * N + bk);
                vec15_1 = _mm256_loadu_ps(data_trn + (bj + 7) * N + bk + 8);

                sub_vec0_0 = _mm256_sub_ps(vec0_0, vec1_0);
                sub_vec1_0 = _mm256_sub_ps(vec0_0, vec3_0);
                sub_vec2_0 = _mm256_sub_ps(vec0_0, vec5_0);
                sub_vec3_0 = _mm256_sub_ps(vec0_0, vec7_0);
                sub_vec4_0 = _mm256_sub_ps(vec0_0, vec9_0);
                sub_vec5_0 = _mm256_sub_ps(vec0_0, vec11_0);
                sub_vec6_0 = _mm256_sub_ps(vec0_0, vec13_0);
                sub_vec7_0 = _mm256_sub_ps(vec0_0, vec15_0);

                sub_vec8_0 = _mm256_sub_ps(vec2_0, vec1_0);
                sub_vec9_0 = _mm256_sub_ps(vec2_0, vec3_0);
                sub_vec10_0 = _mm256_sub_ps(vec2_0, vec5_0);
                sub_vec11_0 = _mm256_sub_ps(vec2_0, vec7_0);
                sub_vec12_0 = _mm256_sub_ps(vec2_0, vec9_0);
                sub_vec13_0 = _mm256_sub_ps(vec2_0, vec11_0);
                sub_vec14_0 = _mm256_sub_ps(vec2_0, vec13_0);
                sub_vec15_0 = _mm256_sub_ps(vec2_0, vec15_0);

                sub_vec16_0 = _mm256_sub_ps(vec4_0, vec1_0);
                sub_vec17_0 = _mm256_sub_ps(vec4_0, vec3_0);
                sub_vec18_0 = _mm256_sub_ps(vec4_0, vec5_0);
                sub_vec19_0 = _mm256_sub_ps(vec4_0, vec7_0);
                sub_vec20_0 = _mm256_sub_ps(vec4_0, vec9_0);
                sub_vec21_0 = _mm256_sub_ps(vec4_0, vec11_0);
                sub_vec22_0 = _mm256_sub_ps(vec4_0, vec13_0);
                sub_vec23_0 = _mm256_sub_ps(vec4_0, vec15_0);

                sub_vec24_0 = _mm256_sub_ps(vec6_0, vec1_0);
                sub_vec25_0 = _mm256_sub_ps(vec6_0, vec3_0);
                sub_vec26_0 = _mm256_sub_ps(vec6_0, vec5_0);
                sub_vec27_0 = _mm256_sub_ps(vec6_0, vec7_0);
                sub_vec28_0 = _mm256_sub_ps(vec6_0, vec9_0);
                sub_vec29_0 = _mm256_sub_ps(vec6_0, vec11_0);
                sub_vec30_0 = _mm256_sub_ps(vec6_0, vec13_0);
                sub_vec31_0 = _mm256_sub_ps(vec6_0, vec15_0);

                sub_vec32_0 = _mm256_sub_ps(vec8_0, vec1_0);
                sub_vec33_0 = _mm256_sub_ps(vec8_0, vec3_0);
                sub_vec34_0 = _mm256_sub_ps(vec8_0, vec5_0);
                sub_vec35_0 = _mm256_sub_ps(vec8_0, vec7_0);
                sub_vec36_0 = _mm256_sub_ps(vec8_0, vec9_0);
                sub_vec37_0 = _mm256_sub_ps(vec8_0, vec11_0);
                sub_vec38_0 = _mm256_sub_ps(vec8_0, vec13_0);
                sub_vec39_0 = _mm256_sub_ps(vec8_0, vec15_0);

                sub_vec40_0 = _mm256_sub_ps(vec10_0, vec1_0);
                sub_vec41_0 = _mm256_sub_ps(vec10_0, vec3_0);
                sub_vec42_0 = _mm256_sub_ps(vec10_0, vec5_0);
                sub_vec43_0 = _mm256_sub_ps(vec10_0, vec7_0);
                sub_vec44_0 = _mm256_sub_ps(vec10_0, vec9_0);
                sub_vec45_0 = _mm256_sub_ps(vec10_0, vec11_0);
                sub_vec46_0 = _mm256_sub_ps(vec10_0, vec13_0);
                sub_vec47_0 = _mm256_sub_ps(vec10_0, vec15_0);

                sub_vec48_0 = _mm256_sub_ps(vec12_0, vec1_0);
                sub_vec49_0 = _mm256_sub_ps(vec12_0, vec3_0);
                sub_vec50_0 = _mm256_sub_ps(vec12_0, vec5_0);
                sub_vec51_0 = _mm256_sub_ps(vec12_0, vec7_0);
                sub_vec52_0 = _mm256_sub_ps(vec12_0, vec9_0);
                sub_vec53_0 = _mm256_sub_ps(vec12_0, vec11_0);
                sub_vec54_0 = _mm256_sub_ps(vec12_0, vec13_0);
                sub_vec55_0 = _mm256_sub_ps(vec12_0, vec15_0);

                sub_vec56_0 = _mm256_sub_ps(vec14_0, vec1_0);
                sub_vec57_0 = _mm256_sub_ps(vec14_0, vec3_0);
                sub_vec58_0 = _mm256_sub_ps(vec14_0, vec5_0);
                sub_vec59_0 = _mm256_sub_ps(vec14_0, vec7_0);
                sub_vec60_0 = _mm256_sub_ps(vec14_0, vec9_0);
                sub_vec61_0 = _mm256_sub_ps(vec14_0, vec11_0);
                sub_vec62_0 = _mm256_sub_ps(vec14_0, vec13_0);
                sub_vec63_0 = _mm256_sub_ps(vec14_0, vec15_0);

                sub_vec0_1 = _mm256_sub_ps(vec0_1, vec1_1);
                sub_vec1_1 = _mm256_sub_ps(vec0_1, vec3_1);
                sub_vec2_1 = _mm256_sub_ps(vec0_1, vec5_1);
                sub_vec3_1 = _mm256_sub_ps(vec0_1, vec7_1);
                sub_vec4_1 = _mm256_sub_ps(vec0_1, vec9_1);
                sub_vec5_1 = _mm256_sub_ps(vec0_1, vec11_1);
                sub_vec6_1 = _mm256_sub_ps(vec0_1, vec13_1);
                sub_vec7_1 = _mm256_sub_ps(vec0_1, vec15_1);

                sub_vec8_1 = _mm256_sub_ps(vec2_1, vec1_1);
                sub_vec9_1 = _mm256_sub_ps(vec2_1, vec3_1);
                sub_vec10_1 = _mm256_sub_ps(vec2_1, vec5_1);
                sub_vec11_1 = _mm256_sub_ps(vec2_1, vec7_1);
                sub_vec12_1 = _mm256_sub_ps(vec2_1, vec9_1);
                sub_vec13_1 = _mm256_sub_ps(vec2_1, vec11_1);
                sub_vec14_1 = _mm256_sub_ps(vec2_1, vec13_1);
                sub_vec15_1 = _mm256_sub_ps(vec2_1, vec15_1);

                sub_vec16_1 = _mm256_sub_ps(vec4_1, vec1_1);
                sub_vec17_1 = _mm256_sub_ps(vec4_1, vec3_1);
                sub_vec18_1 = _mm256_sub_ps(vec4_1, vec5_1);
                sub_vec19_1 = _mm256_sub_ps(vec4_1, vec7_1);
                sub_vec20_1 = _mm256_sub_ps(vec4_1, vec9_1);
                sub_vec21_1 = _mm256_sub_ps(vec4_1, vec11_1);
                sub_vec22_1 = _mm256_sub_ps(vec4_1, vec13_1);
                sub_vec23_1 = _mm256_sub_ps(vec4_1, vec15_1);

                sub_vec24_1 = _mm256_sub_ps(vec6_1, vec1_1);
                sub_vec25_1 = _mm256_sub_ps(vec6_1, vec3_1);
                sub_vec26_1 = _mm256_sub_ps(vec6_1, vec5_1);
                sub_vec27_1 = _mm256_sub_ps(vec6_1, vec7_1);
                sub_vec28_1 = _mm256_sub_ps(vec6_1, vec9_1);
                sub_vec29_1 = _mm256_sub_ps(vec6_1, vec11_1);
                sub_vec30_1 = _mm256_sub_ps(vec6_1, vec13_1);
                sub_vec31_1 = _mm256_sub_ps(vec6_1, vec15_1);

                sub_vec32_1 = _mm256_sub_ps(vec8_1, vec1_1);
                sub_vec33_1 = _mm256_sub_ps(vec8_1, vec3_1);
                sub_vec34_1 = _mm256_sub_ps(vec8_1, vec5_1);
                sub_vec35_1 = _mm256_sub_ps(vec8_1, vec7_1);
                sub_vec36_1 = _mm256_sub_ps(vec8_1, vec9_1);
                sub_vec37_1 = _mm256_sub_ps(vec8_1, vec11_1);
                sub_vec38_1 = _mm256_sub_ps(vec8_1, vec13_1);
                sub_vec39_1 = _mm256_sub_ps(vec8_1, vec15_1);

                sub_vec40_1 = _mm256_sub_ps(vec10_1, vec1_1);
                sub_vec41_1 = _mm256_sub_ps(vec10_1, vec3_1);
                sub_vec42_1 = _mm256_sub_ps(vec10_1, vec5_1);
                sub_vec43_1 = _mm256_sub_ps(vec10_1, vec7_1);
                sub_vec44_1 = _mm256_sub_ps(vec10_1, vec9_1);
                sub_vec45_1 = _mm256_sub_ps(vec10_1, vec11_1);
                sub_vec46_1 = _mm256_sub_ps(vec10_1, vec13_1);
                sub_vec47_1 = _mm256_sub_ps(vec10_1, vec15_1);

                sub_vec48_1 = _mm256_sub_ps(vec12_1, vec1_1);
                sub_vec49_1 = _mm256_sub_ps(vec12_1, vec3_1);
                sub_vec50_1 = _mm256_sub_ps(vec12_1, vec5_1);
                sub_vec51_1 = _mm256_sub_ps(vec12_1, vec7_1);
                sub_vec52_1 = _mm256_sub_ps(vec12_1, vec9_1);
                sub_vec53_1 = _mm256_sub_ps(vec12_1, vec11_1);
                sub_vec54_1 = _mm256_sub_ps(vec12_1, vec13_1);
                sub_vec55_1 = _mm256_sub_ps(vec12_1, vec15_1);

                sub_vec56_1 = _mm256_sub_ps(vec14_1, vec1_1);
                sub_vec57_1 = _mm256_sub_ps(vec14_1, vec3_1);
                sub_vec58_1 = _mm256_sub_ps(vec14_1, vec5_1);
                sub_vec59_1 = _mm256_sub_ps(vec14_1, vec7_1);
                sub_vec60_1 = _mm256_sub_ps(vec14_1, vec9_1);
                sub_vec61_1 = _mm256_sub_ps(vec14_1, vec11_1);
                sub_vec62_1 = _mm256_sub_ps(vec14_1, vec13_1);
                sub_vec63_1 = _mm256_sub_ps(vec14_1, vec15_1);

                sub_vec0_0 = _mm256_mul_ps(sub_vec0_0, sub_vec0_0);
                sub_vec1_0 = _mm256_mul_ps(sub_vec1_0, sub_vec1_0);
                sub_vec2_0 = _mm256_mul_ps(sub_vec2_0, sub_vec2_0);
                sub_vec3_0 = _mm256_mul_ps(sub_vec3_0, sub_vec3_0);
                sub_vec4_0 = _mm256_mul_ps(sub_vec4_0, sub_vec4_0);
                sub_vec5_0 = _mm256_mul_ps(sub_vec5_0, sub_vec5_0);
                sub_vec6_0 = _mm256_mul_ps(sub_vec6_0, sub_vec6_0);
                sub_vec7_0 = _mm256_mul_ps(sub_vec7_0, sub_vec7_0);

                sub_vec8_0 = _mm256_mul_ps(sub_vec8_0, sub_vec8_0);
                sub_vec9_0 = _mm256_mul_ps(sub_vec9_0, sub_vec9_0);
                sub_vec10_0 = _mm256_mul_ps(sub_vec10_0, sub_vec10_0);
                sub_vec11_0 = _mm256_mul_ps(sub_vec11_0, sub_vec11_0);
                sub_vec12_0 = _mm256_mul_ps(sub_vec12_0, sub_vec12_0);
                sub_vec13_0 = _mm256_mul_ps(sub_vec13_0, sub_vec13_0);
                sub_vec14_0 = _mm256_mul_ps(sub_vec14_0, sub_vec14_0);
                sub_vec15_0 = _mm256_mul_ps(sub_vec15_0, sub_vec15_0);

                sub_vec16_0 = _mm256_mul_ps(sub_vec16_0, sub_vec16_0);
                sub_vec17_0 = _mm256_mul_ps(sub_vec17_0, sub_vec17_0);
                sub_vec18_0 = _mm256_mul_ps(sub_vec18_0, sub_vec18_0);
                sub_vec19_0 = _mm256_mul_ps(sub_vec19_0, sub_vec19_0);
                sub_vec20_0 = _mm256_mul_ps(sub_vec20_0, sub_vec20_0);
                sub_vec21_0 = _mm256_mul_ps(sub_vec21_0, sub_vec21_0);
                sub_vec22_0 = _mm256_mul_ps(sub_vec22_0, sub_vec22_0);
                sub_vec23_0 = _mm256_mul_ps(sub_vec23_0, sub_vec23_0);

                sub_vec24_0 = _mm256_mul_ps(sub_vec24_0, sub_vec24_0);
                sub_vec25_0 = _mm256_mul_ps(sub_vec25_0, sub_vec25_0);
                sub_vec26_0 = _mm256_mul_ps(sub_vec26_0, sub_vec26_0);
                sub_vec27_0 = _mm256_mul_ps(sub_vec27_0, sub_vec27_0);
                sub_vec28_0 = _mm256_mul_ps(sub_vec28_0, sub_vec28_0);
                sub_vec29_0 = _mm256_mul_ps(sub_vec29_0, sub_vec29_0);
                sub_vec30_0 = _mm256_mul_ps(sub_vec30_0, sub_vec30_0);
                sub_vec31_0 = _mm256_mul_ps(sub_vec31_0, sub_vec31_0);

                sub_vec32_0 = _mm256_mul_ps(sub_vec32_0, sub_vec32_0);
                sub_vec33_0 = _mm256_mul_ps(sub_vec33_0, sub_vec33_0);
                sub_vec34_0 = _mm256_mul_ps(sub_vec34_0, sub_vec34_0);
                sub_vec35_0 = _mm256_mul_ps(sub_vec35_0, sub_vec35_0);
                sub_vec36_0 = _mm256_mul_ps(sub_vec36_0, sub_vec36_0);
                sub_vec37_0 = _mm256_mul_ps(sub_vec37_0, sub_vec37_0);
                sub_vec38_0 = _mm256_mul_ps(sub_vec38_0, sub_vec38_0);
                sub_vec39_0 = _mm256_mul_ps(sub_vec39_0, sub_vec39_0);

                sub_vec40_0 = _mm256_mul_ps(sub_vec40_0, sub_vec40_0);
                sub_vec41_0 = _mm256_mul_ps(sub_vec41_0, sub_vec41_0);
                sub_vec42_0 = _mm256_mul_ps(sub_vec42_0, sub_vec42_0);
                sub_vec43_0 = _mm256_mul_ps(sub_vec43_0, sub_vec43_0);
                sub_vec44_0 = _mm256_mul_ps(sub_vec44_0, sub_vec44_0);
                sub_vec45_0 = _mm256_mul_ps(sub_vec45_0, sub_vec45_0);
                sub_vec46_0 = _mm256_mul_ps(sub_vec46_0, sub_vec46_0);
                sub_vec47_0 = _mm256_mul_ps(sub_vec47_0, sub_vec47_0);

                sub_vec48_0 = _mm256_mul_ps(sub_vec48_0, sub_vec48_0);
                sub_vec49_0 = _mm256_mul_ps(sub_vec49_0, sub_vec49_0);
                sub_vec50_0 = _mm256_mul_ps(sub_vec50_0, sub_vec50_0);
                sub_vec51_0 = _mm256_mul_ps(sub_vec51_0, sub_vec51_0);
                sub_vec52_0 = _mm256_mul_ps(sub_vec52_0, sub_vec52_0);
                sub_vec53_0 = _mm256_mul_ps(sub_vec53_0, sub_vec53_0);
                sub_vec54_0 = _mm256_mul_ps(sub_vec54_0, sub_vec54_0);
                sub_vec55_0 = _mm256_mul_ps(sub_vec55_0, sub_vec55_0);

                sub_vec56_0 = _mm256_mul_ps(sub_vec56_0, sub_vec56_0);
                sub_vec57_0 = _mm256_mul_ps(sub_vec57_0, sub_vec57_0);
                sub_vec58_0 = _mm256_mul_ps(sub_vec58_0, sub_vec58_0);
                sub_vec59_0 = _mm256_mul_ps(sub_vec59_0, sub_vec59_0);
                sub_vec60_0 = _mm256_mul_ps(sub_vec60_0, sub_vec60_0);
                sub_vec61_0 = _mm256_mul_ps(sub_vec61_0, sub_vec61_0);
                sub_vec62_0 = _mm256_mul_ps(sub_vec62_0, sub_vec62_0);
                sub_vec63_0 = _mm256_mul_ps(sub_vec63_0, sub_vec63_0);

                sub_vec0_1 = _mm256_mul_ps(sub_vec0_1, sub_vec0_1);
                sub_vec1_1 = _mm256_mul_ps(sub_vec1_1, sub_vec1_1);
                sub_vec2_1 = _mm256_mul_ps(sub_vec2_1, sub_vec2_1);
                sub_vec3_1 = _mm256_mul_ps(sub_vec3_1, sub_vec3_1);
                sub_vec4_1 = _mm256_mul_ps(sub_vec4_1, sub_vec4_1);
                sub_vec5_1 = _mm256_mul_ps(sub_vec5_1, sub_vec5_1);
                sub_vec6_1 = _mm256_mul_ps(sub_vec6_1, sub_vec6_1);
                sub_vec7_1 = _mm256_mul_ps(sub_vec7_1, sub_vec7_1);

                sub_vec8_1 = _mm256_mul_ps(sub_vec8_1, sub_vec8_1);
                sub_vec9_1 = _mm256_mul_ps(sub_vec9_1, sub_vec9_1);
                sub_vec10_1 = _mm256_mul_ps(sub_vec10_1, sub_vec10_1);
                sub_vec11_1 = _mm256_mul_ps(sub_vec11_1, sub_vec11_1);
                sub_vec12_1 = _mm256_mul_ps(sub_vec12_1, sub_vec12_1);
                sub_vec13_1 = _mm256_mul_ps(sub_vec13_1, sub_vec13_1);
                sub_vec14_1 = _mm256_mul_ps(sub_vec14_1, sub_vec14_1);
                sub_vec15_1 = _mm256_mul_ps(sub_vec15_1, sub_vec15_1);

                sub_vec16_1 = _mm256_mul_ps(sub_vec16_1, sub_vec16_1);
                sub_vec17_1 = _mm256_mul_ps(sub_vec17_1, sub_vec17_1);
                sub_vec18_1 = _mm256_mul_ps(sub_vec18_1, sub_vec18_1);
                sub_vec19_1 = _mm256_mul_ps(sub_vec19_1, sub_vec19_1);
                sub_vec20_1 = _mm256_mul_ps(sub_vec20_1, sub_vec20_1);
                sub_vec21_1 = _mm256_mul_ps(sub_vec21_1, sub_vec21_1);
                sub_vec22_1 = _mm256_mul_ps(sub_vec22_1, sub_vec22_1);
                sub_vec23_1 = _mm256_mul_ps(sub_vec23_1, sub_vec23_1);

                sub_vec24_1 = _mm256_mul_ps(sub_vec24_1, sub_vec24_1);
                sub_vec25_1 = _mm256_mul_ps(sub_vec25_1, sub_vec25_1);
                sub_vec26_1 = _mm256_mul_ps(sub_vec26_1, sub_vec26_1);
                sub_vec27_1 = _mm256_mul_ps(sub_vec27_1, sub_vec27_1);
                sub_vec28_1 = _mm256_mul_ps(sub_vec28_1, sub_vec28_1);
                sub_vec29_1 = _mm256_mul_ps(sub_vec29_1, sub_vec29_1);
                sub_vec30_1 = _mm256_mul_ps(sub_vec30_1, sub_vec30_1);
                sub_vec31_1 = _mm256_mul_ps(sub_vec31_1, sub_vec31_1);

                sub_vec32_1 = _mm256_mul_ps(sub_vec32_1, sub_vec32_1);
                sub_vec33_1 = _mm256_mul_ps(sub_vec33_1, sub_vec33_1);
                sub_vec34_1 = _mm256_mul_ps(sub_vec34_1, sub_vec34_1);
                sub_vec35_1 = _mm256_mul_ps(sub_vec35_1, sub_vec35_1);
                sub_vec36_1 = _mm256_mul_ps(sub_vec36_1, sub_vec36_1);
                sub_vec37_1 = _mm256_mul_ps(sub_vec37_1, sub_vec37_1);
                sub_vec38_1 = _mm256_mul_ps(sub_vec38_1, sub_vec38_1);
                sub_vec39_1 = _mm256_mul_ps(sub_vec39_1, sub_vec39_1);

                sub_vec40_1 = _mm256_mul_ps(sub_vec40_1, sub_vec40_1);
                sub_vec41_1 = _mm256_mul_ps(sub_vec41_1, sub_vec41_1);
                sub_vec42_1 = _mm256_mul_ps(sub_vec42_1, sub_vec42_1);
                sub_vec43_1 = _mm256_mul_ps(sub_vec43_1, sub_vec43_1);
                sub_vec44_1 = _mm256_mul_ps(sub_vec44_1, sub_vec44_1);
                sub_vec45_1 = _mm256_mul_ps(sub_vec45_1, sub_vec45_1);
                sub_vec46_1 = _mm256_mul_ps(sub_vec46_1, sub_vec46_1);
                sub_vec47_1 = _mm256_mul_ps(sub_vec47_1, sub_vec47_1);

                sub_vec48_1 = _mm256_mul_ps(sub_vec48_1, sub_vec48_1);
                sub_vec49_1 = _mm256_mul_ps(sub_vec49_1, sub_vec49_1);
                sub_vec50_1 = _mm256_mul_ps(sub_vec50_1, sub_vec50_1);
                sub_vec51_1 = _mm256_mul_ps(sub_vec51_1, sub_vec51_1);
                sub_vec52_1 = _mm256_mul_ps(sub_vec52_1, sub_vec52_1);
                sub_vec53_1 = _mm256_mul_ps(sub_vec53_1, sub_vec53_1);
                sub_vec54_1 = _mm256_mul_ps(sub_vec54_1, sub_vec54_1);
                sub_vec55_1 = _mm256_mul_ps(sub_vec55_1, sub_vec55_1);

                sub_vec56_1 = _mm256_mul_ps(sub_vec56_1, sub_vec56_1);
                sub_vec57_1 = _mm256_mul_ps(sub_vec57_1, sub_vec57_1);
                sub_vec58_1 = _mm256_mul_ps(sub_vec58_1, sub_vec58_1);
                sub_vec59_1 = _mm256_mul_ps(sub_vec59_1, sub_vec59_1);
                sub_vec60_1 = _mm256_mul_ps(sub_vec60_1, sub_vec60_1);
                sub_vec61_1 = _mm256_mul_ps(sub_vec61_1, sub_vec61_1);
                sub_vec62_1 = _mm256_mul_ps(sub_vec62_1, sub_vec62_1);
                sub_vec63_1 = _mm256_mul_ps(sub_vec63_1, sub_vec63_1);

                sub_vec0_0 = _mm256_add_ps(sub_vec0_0, sub_vec0_1);
                sub_vec1_0 = _mm256_add_ps(sub_vec1_0, sub_vec1_1);
                sub_vec2_0 = _mm256_add_ps(sub_vec2_0, sub_vec2_1);
                sub_vec3_0 = _mm256_add_ps(sub_vec3_0, sub_vec3_1);
                sub_vec4_0 = _mm256_add_ps(sub_vec4_0, sub_vec4_1);
                sub_vec5_0 = _mm256_add_ps(sub_vec5_0, sub_vec5_1);
                sub_vec6_0 = _mm256_add_ps(sub_vec6_0, sub_vec6_1);
                sub_vec7_0 = _mm256_add_ps(sub_vec7_0, sub_vec7_1);

                sub_vec8_0 = _mm256_add_ps(sub_vec8_0, sub_vec8_1);
                sub_vec9_0 = _mm256_add_ps(sub_vec9_0, sub_vec9_1);
                sub_vec10_0 = _mm256_add_ps(sub_vec10_0, sub_vec10_1);
                sub_vec11_0 = _mm256_add_ps(sub_vec11_0, sub_vec11_1);
                sub_vec12_0 = _mm256_add_ps(sub_vec12_0, sub_vec12_1);
                sub_vec13_0 = _mm256_add_ps(sub_vec13_0, sub_vec13_1);
                sub_vec14_0 = _mm256_add_ps(sub_vec14_0, sub_vec14_1);
                sub_vec15_0 = _mm256_add_ps(sub_vec15_0, sub_vec15_1);

                sub_vec16_0 = _mm256_add_ps(sub_vec16_0, sub_vec16_1);
                sub_vec17_0 = _mm256_add_ps(sub_vec17_0, sub_vec17_1);
                sub_vec18_0 = _mm256_add_ps(sub_vec18_0, sub_vec18_1);
                sub_vec19_0 = _mm256_add_ps(sub_vec19_0, sub_vec19_1);
                sub_vec20_0 = _mm256_add_ps(sub_vec20_0, sub_vec20_1);
                sub_vec21_0 = _mm256_add_ps(sub_vec21_0, sub_vec21_1);
                sub_vec22_0 = _mm256_add_ps(sub_vec22_0, sub_vec22_1);
                sub_vec23_0 = _mm256_add_ps(sub_vec23_0, sub_vec23_1);

                sub_vec24_0 = _mm256_add_ps(sub_vec24_0, sub_vec24_1);
                sub_vec25_0 = _mm256_add_ps(sub_vec25_0, sub_vec25_1);
                sub_vec26_0 = _mm256_add_ps(sub_vec26_0, sub_vec26_1);
                sub_vec27_0 = _mm256_add_ps(sub_vec27_0, sub_vec27_1);
                sub_vec28_0 = _mm256_add_ps(sub_vec28_0, sub_vec28_1);
                sub_vec29_0 = _mm256_add_ps(sub_vec29_0, sub_vec29_1);
                sub_vec30_0 = _mm256_add_ps(sub_vec30_0, sub_vec30_1);
                sub_vec31_0 = _mm256_add_ps(sub_vec31_0, sub_vec31_1);

                sub_vec32_0 = _mm256_add_ps(sub_vec32_0, sub_vec32_1);
                sub_vec33_0 = _mm256_add_ps(sub_vec33_0, sub_vec33_1);
                sub_vec34_0 = _mm256_add_ps(sub_vec34_0, sub_vec34_1);
                sub_vec35_0 = _mm256_add_ps(sub_vec35_0, sub_vec35_1);
                sub_vec36_0 = _mm256_add_ps(sub_vec36_0, sub_vec36_1);
                sub_vec37_0 = _mm256_add_ps(sub_vec37_0, sub_vec37_1);
                sub_vec38_0 = _mm256_add_ps(sub_vec38_0, sub_vec38_1);
                sub_vec39_0 = _mm256_add_ps(sub_vec39_0, sub_vec39_1);

                sub_vec40_0 = _mm256_add_ps(sub_vec40_0, sub_vec40_1);
                sub_vec41_0 = _mm256_add_ps(sub_vec41_0, sub_vec41_1);
                sub_vec42_0 = _mm256_add_ps(sub_vec42_0, sub_vec42_1);
                sub_vec43_0 = _mm256_add_ps(sub_vec43_0, sub_vec43_1);
                sub_vec44_0 = _mm256_add_ps(sub_vec44_0, sub_vec44_1);
                sub_vec45_0 = _mm256_add_ps(sub_vec45_0, sub_vec45_1);
                sub_vec46_0 = _mm256_add_ps(sub_vec46_0, sub_vec46_1);
                sub_vec47_0 = _mm256_add_ps(sub_vec47_0, sub_vec47_1);

                sub_vec48_0 = _mm256_add_ps(sub_vec48_0, sub_vec48_1);
                sub_vec49_0 = _mm256_add_ps(sub_vec49_0, sub_vec49_1);
                sub_vec50_0 = _mm256_add_ps(sub_vec50_0, sub_vec50_1);
                sub_vec51_0 = _mm256_add_ps(sub_vec51_0, sub_vec51_1);
                sub_vec52_0 = _mm256_add_ps(sub_vec52_0, sub_vec52_1);
                sub_vec53_0 = _mm256_add_ps(sub_vec53_0, sub_vec53_1);
                sub_vec54_0 = _mm256_add_ps(sub_vec54_0, sub_vec54_1);
                sub_vec55_0 = _mm256_add_ps(sub_vec55_0, sub_vec55_1);

                sub_vec56_0 = _mm256_add_ps(sub_vec56_0, sub_vec56_1);
                sub_vec57_0 = _mm256_add_ps(sub_vec57_0, sub_vec57_1);
                sub_vec58_0 = _mm256_add_ps(sub_vec58_0, sub_vec58_1);
                sub_vec59_0 = _mm256_add_ps(sub_vec59_0, sub_vec59_1);
                sub_vec60_0 = _mm256_add_ps(sub_vec60_0, sub_vec60_1);
                sub_vec61_0 = _mm256_add_ps(sub_vec61_0, sub_vec61_1);
                sub_vec62_0 = _mm256_add_ps(sub_vec62_0, sub_vec62_1);
                sub_vec63_0 = _mm256_add_ps(sub_vec63_0, sub_vec63_1);


                acc0 = _mm256_add_ps(acc0, sub_vec0_0);
                acc1 = _mm256_add_ps(acc1, sub_vec1_0);
                acc2 = _mm256_add_ps(acc2, sub_vec2_0);
                acc3 = _mm256_add_ps(acc3, sub_vec3_0);
                acc4 = _mm256_add_ps(acc4, sub_vec4_0);
                acc5 = _mm256_add_ps(acc5, sub_vec5_0);
                acc6 = _mm256_add_ps(acc6, sub_vec6_0);
                acc7 = _mm256_add_ps(acc7, sub_vec7_0);

                acc8 = _mm256_add_ps(acc8, sub_vec8_0);
                acc9 = _mm256_add_ps(acc9, sub_vec9_0);
                acc10 = _mm256_add_ps(acc10, sub_vec10_0);
                acc11 = _mm256_add_ps(acc11, sub_vec11_0);
                acc12 = _mm256_add_ps(acc12, sub_vec12_0);
                acc13 = _mm256_add_ps(acc13, sub_vec13_0);
                acc14 = _mm256_add_ps(acc14, sub_vec14_0);
                acc15 = _mm256_add_ps(acc15, sub_vec15_0);

                acc16 = _mm256_add_ps(acc16, sub_vec16_0);
                acc17 = _mm256_add_ps(acc17, sub_vec17_0);
                acc18 = _mm256_add_ps(acc18, sub_vec18_0);
                acc19 = _mm256_add_ps(acc19, sub_vec19_0);
                acc20 = _mm256_add_ps(acc20, sub_vec20_0);
                acc21 = _mm256_add_ps(acc21, sub_vec21_0);
                acc22 = _mm256_add_ps(acc22, sub_vec22_0);
                acc23 = _mm256_add_ps(acc23, sub_vec23_0);

                acc24 = _mm256_add_ps(acc24, sub_vec24_0);
                acc25 = _mm256_add_ps(acc25, sub_vec25_0);
                acc26 = _mm256_add_ps(acc26, sub_vec26_0);
                acc27 = _mm256_add_ps(acc27, sub_vec27_0);
                acc28 = _mm256_add_ps(acc28, sub_vec28_0);
                acc29 = _mm256_add_ps(acc29, sub_vec29_0);
                acc30 = _mm256_add_ps(acc30, sub_vec30_0);
                acc31 = _mm256_add_ps(acc31, sub_vec31_0);

                acc32 = _mm256_add_ps(acc32, sub_vec32_0);
                acc33 = _mm256_add_ps(acc33, sub_vec33_0);
                acc34 = _mm256_add_ps(acc34, sub_vec34_0);
                acc35 = _mm256_add_ps(acc35, sub_vec35_0);
                acc36 = _mm256_add_ps(acc36, sub_vec36_0);
                acc37 = _mm256_add_ps(acc37, sub_vec37_0);
                acc38 = _mm256_add_ps(acc38, sub_vec38_0);
                acc39 = _mm256_add_ps(acc39, sub_vec39_0);

                acc40 = _mm256_add_ps(acc40, sub_vec40_0);
                acc41 = _mm256_add_ps(acc41, sub_vec41_0);
                acc42 = _mm256_add_ps(acc42, sub_vec42_0);
                acc43 = _mm256_add_ps(acc43, sub_vec43_0);
                acc44 = _mm256_add_ps(acc44, sub_vec44_0);
                acc45 = _mm256_add_ps(acc45, sub_vec45_0);
                acc46 = _mm256_add_ps(acc46, sub_vec46_0);
                acc47 = _mm256_add_ps(acc47, sub_vec47_0);

                acc48 = _mm256_add_ps(acc48, sub_vec48_0);
                acc49 = _mm256_add_ps(acc49, sub_vec49_0);
                acc50 = _mm256_add_ps(acc50, sub_vec50_0);
                acc51 = _mm256_add_ps(acc51, sub_vec51_0);
                acc52 = _mm256_add_ps(acc52, sub_vec52_0);
                acc53 = _mm256_add_ps(acc53, sub_vec53_0);
                acc54 = _mm256_add_ps(acc54, sub_vec54_0);
                acc55 = _mm256_add_ps(acc55, sub_vec55_0);

                acc56 = _mm256_add_ps(acc56, sub_vec56_0);
                acc57 = _mm256_add_ps(acc57, sub_vec57_0);
                acc58 = _mm256_add_ps(acc58, sub_vec58_0);
                acc59 = _mm256_add_ps(acc59, sub_vec59_0);
                acc60 = _mm256_add_ps(acc60, sub_vec60_0);
                acc61 = _mm256_add_ps(acc61, sub_vec61_0);
                acc62 = _mm256_add_ps(acc62, sub_vec62_0);
                acc63 = _mm256_add_ps(acc63, sub_vec63_0);

            }

            f0 = sum8(acc0);
            f1 = sum8(acc1);
            f2 = sum8(acc2);
            f3 = sum8(acc3);
            f4 = sum8(acc4);
            f5 = sum8(acc5);
            f6 = sum8(acc6);
            f7 = sum8(acc7);

            vec_sum0 = _mm256_set_ps(f7, f6, f5, f4, f3, f2, f1, f0);

            f8 = sum8(acc8);
            f9 = sum8(acc9);
            f10 = sum8(acc10);
            f11 = sum8(acc11);
            f12 = sum8(acc12);
            f13 = sum8(acc13);
            f14 = sum8(acc14);
            f15 = sum8(acc15);

            vec_sum1 = _mm256_set_ps(f15, f14, f13, f12, f11, f10, f9, f8);

            f16 = sum8(acc16);
            f17 = sum8(acc17);
            f18 = sum8(acc18);
            f19 = sum8(acc19);
            f20 = sum8(acc20);
            f21 = sum8(acc21);
            f22 = sum8(acc22);
            f23 = sum8(acc23);

            vec_sum2 = _mm256_set_ps(f23, f22, f21, f20, f19, f18, f17, f16);

            f24 = sum8(acc24);
            f25 = sum8(acc25);
            f26 = sum8(acc26);
            f27 = sum8(acc27);
            f28 = sum8(acc28);
            f29 = sum8(acc29);
            f30 = sum8(acc30);
            f31 = sum8(acc31);

            vec_sum3 = _mm256_set_ps(f31, f30, f29, f28, f27, f26, f25, f24);

            f32 = sum8(acc32);
            f33 = sum8(acc33);
            f34 = sum8(acc34);
            f35 = sum8(acc35);
            f36 = sum8(acc36);
            f37 = sum8(acc37);
            f38 = sum8(acc38);
            f39 = sum8(acc39);

            vec_sum4 = _mm256_set_ps(f39, f38, f37, f36, f35, f34, f33, f32);

            f40 = sum8(acc40);
            f41 = sum8(acc41);
            f42 = sum8(acc42);
            f43 = sum8(acc43);
            f44 = sum8(acc44);
            f45 = sum8(acc45);
            f46 = sum8(acc46);
            f47 = sum8(acc47);

            vec_sum5 = _mm256_set_ps(f47, f46, f45, f44, f43, f42, f41, f40);

            f48 = sum8(acc48);
            f49 = sum8(acc49);
            f50 = sum8(acc50);
            f51 = sum8(acc51);
            f52 = sum8(acc52);
            f53 = sum8(acc53);
            f54 = sum8(acc54);
            f55 = sum8(acc55);

            vec_sum6 = _mm256_set_ps(f55, f54, f53, f52, f51, f50, f49, f48);

            f56 = sum8(acc56);
            f57 = sum8(acc57);
            f58 = sum8(acc58);
            f59 = sum8(acc59);
            f60 = sum8(acc60);
            f61 = sum8(acc61);
            f62 = sum8(acc62);
            f63 = sum8(acc63);

            vec_sum7 = _mm256_set_ps(f63, f62, f61, f60, f59, f58, f57, f56);

            tmp0 = _mm256_loadu_ps(distances.data + bi * N + bj);
            tmp1 = _mm256_loadu_ps(distances.data + (bi+1) * N + bj);
            tmp2 = _mm256_loadu_ps(distances.data + (bi+2) * N + bj);
            tmp3 = _mm256_loadu_ps(distances.data + (bi+3) * N + bj);
            tmp4 = _mm256_loadu_ps(distances.data + (bi+4) * N + bj);
            tmp5 = _mm256_loadu_ps(distances.data + (bi+5) * N + bj);
            tmp6 = _mm256_loadu_ps(distances.data + (bi+6) * N + bj);
            tmp7 = _mm256_loadu_ps(distances.data + (bi+7) * N + bj);

            tmp0 = _mm256_add_ps(tmp0, vec_sum0);
            tmp1 = _mm256_add_ps(tmp1, vec_sum1);
            tmp2 = _mm256_add_ps(tmp2, vec_sum2);
            tmp3 = _mm256_add_ps(tmp3, vec_sum3);
            tmp4 = _mm256_add_ps(tmp4, vec_sum4);
            tmp5 = _mm256_add_ps(tmp5, vec_sum5);
            tmp6 = _mm256_add_ps(tmp6, vec_sum6);
            tmp7 = _mm256_add_ps(tmp7, vec_sum7);

            _mm256_storeu_ps(distances.data + (bi) * N + bj, tmp0);
            _mm256_storeu_ps(distances.data + (bi+1) * N + bj, tmp1);
            _mm256_storeu_ps(distances.data + (bi+2) * N + bj, tmp2);
            _mm256_storeu_ps(distances.data + (bi+3) * N + bj, tmp3);
            _mm256_storeu_ps(distances.data + (bi+4) * N + bj, tmp4);
            _mm256_storeu_ps(distances.data + (bi+5) * N + bj, tmp5);
            _mm256_storeu_ps(distances.data + (bi+6) * N + bj, tmp6);
            _mm256_storeu_ps(distances.data + (bi+7) * N + bj, tmp7);
        }
    }

    for (int i_tst = 0; i_tst < N_tst; i_tst++){
        for (int i_trn = 0; i_trn < N; i_trn++){
            dist_arr[i_trn].value = distances.data[i_tst*N+i_trn];
            dist_arr[i_trn].index = i_trn;
        }

        quadsort(dist_arr,N,sizeof(pair_t),cmp);
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_arr[k].index;
        }
    }
    free(dist_arr);
    destroy(&distances);
}