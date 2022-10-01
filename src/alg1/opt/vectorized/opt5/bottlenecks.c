//
// Created by lucasck on 17/06/22.
//
#include <stdio.h>
#include "mat.h"
#include <math.h>
#include <string.h>
#include "utils.h"
#include "tsc_x86.h"
#include <immintrin.h>
#include "quadsort.h"

#define NUM_RUNS 100
float sum8(__m256 x) {
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    const __m128 loQuad = _mm256_castps256_ps128(x);
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    const __m128 loDual = sumQuad;
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    const __m128 lo = sumDual;
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

// 4x8 vectorized blocking
void get_true_knn(int_mat *x_tst_knn_gt, mat *x_trn, mat *x_tst){
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

    __m256 vec_sum0, vec_sum1, vec_sum2, vec_sum3;
    __m256 sub_vec0, sub_vec1, sub_vec2, sub_vec3, sub_vec4, sub_vec5, sub_vec6, sub_vec7;
    __m256 sub_vec8, sub_vec9, sub_vec10, sub_vec11, sub_vec12, sub_vec13, sub_vec14, sub_vec15;

    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;

    myInt64 start_l2norm, start_argsort, start, cycles_l2norm, cycles_argsort, cycles;
    cycles_l2norm = cycles_argsort = cycles = 0;

    start = start_tsc();
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

            start_l2norm = start_tsc();
            for (bk = 0; bk < d; bk += 8) {

                ts_vec0 = _mm256_loadu_ps(data_tst + bi * N + bk); // 4x8 blocks from test matrix
                ts_vec1 = _mm256_loadu_ps(data_tst + (bi + 1) * N + bk);
                ts_vec2 = _mm256_loadu_ps(data_tst + (bi + 2) * N + bk);
                ts_vec3 = _mm256_loadu_ps(data_tst + (bi + 3) * N + bk);

                tr_vec0 = _mm256_loadu_ps(data_trn + bj * N + bk); // 4x8 blocks from train matrix
                tr_vec1 = _mm256_loadu_ps(data_trn + (bj + 1) * N + bk);
                tr_vec2 = _mm256_loadu_ps(data_trn + (bj + 2) * N + bk);
                tr_vec3 = _mm256_loadu_ps(data_trn + (bj + 3) * N + bk);

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
            dist_arr[bj + 1].value = sum8(acc1);
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
            cycles_l2norm += stop_tsc(start_l2norm);
        }
        start_argsort = start_tsc();
        quadsort(dist_arr, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_i, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_ii, N, sizeof(pair_t), cmp);
        quadsort(dist_arr_iii, N, sizeof(pair_t), cmp);
        cycles_argsort += stop_tsc(start_argsort);

#pragma GCC ivdep
        for (k = 0; k < N; k++) {
            x_tst_knn_gt->data[bi * N + k] = dist_arr[k].index;
            x_tst_knn_gt->data[(bi + 1) * N + k] = dist_arr_i[k].index;
            x_tst_knn_gt->data[(bi + 2) * N + k] = dist_arr_ii[k].index;
            x_tst_knn_gt->data[(bi + 3) * N + k] = dist_arr_iii[k].index;
        }
    }
    cycles = stop_tsc(start);
    cycles = cycles - cycles_argsort - cycles_l2norm;
    printf("%lld,", cycles);
    printf("%lld,", cycles_argsort);
    printf("%lld,", cycles_l2norm);
    free(dist_arr);
    free(dist_arr_i);
    free(dist_arr_ii);
    free(dist_arr_iii);

}









void compute_single_unweighted_knn_class_shapley(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmpj, tmpjj, tmp0j, tmp1j, tmp3j, tmp0jj, tmp1jj, tmp3jj;
    int* row_j = malloc(N_tst*sizeof(int));
    int* row_jj = malloc(N_tst*sizeof(int));
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;
    int x_tst_knn_gt_jj_i, x_tst_knn_gt_jj_i_plus_1, x_tst_knn_gt_jj_last_i;
    int y_tst_j, y_tst_jj;
    float N_inv = 1.0/N;
    float K_inv = 1.0/K;
    float i_inv, min_K, factor;

    for (int j=0; j < N_tst;j+=2){
        y_tst_j = y_tst[j];
        y_tst_jj = y_tst[j+1];

        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        x_tst_knn_gt_jj_last_i = int_mat_get(x_tst_knn_gt, j+1, N-1);

        tmpj = (y_trn[x_tst_knn_gt_j_last_i] != y_tst_j) ? 0.0:N_inv;
        tmpjj = (y_trn[x_tst_knn_gt_jj_last_i] != y_tst_jj) ? 0.0:N_inv;

        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmpj);
        mat_set(sp_gt, j+1, x_tst_knn_gt_jj_last_i, tmpjj);

        get_int_row(row_j, x_tst_knn_gt, j);
        get_int_row(row_jj, x_tst_knn_gt, j+1);
        x_tst_knn_gt_j_i_plus_1 = row_j[N-1];
        x_tst_knn_gt_jj_i_plus_1 = row_jj[N-1];

        for (int i=N-2; i>-1; i--){
            i_inv = 1.0 / (i + 1);
            min_K = (K > i+1) ? i+1 : K;
            factor = i_inv * min_K * K_inv;

            x_tst_knn_gt_j_i = row_j[i];
            x_tst_knn_gt_jj_i = row_jj[i];

            tmp0j = (y_trn[x_tst_knn_gt_j_i] != y_tst_j) ? 0.0:1.0;
            tmp1j = (y_trn[x_tst_knn_gt_j_i_plus_1] != y_tst_j) ? 0.0:1.0;
            tmp3j = mat_get(sp_gt, j, x_tst_knn_gt_j_i_plus_1) + (tmp0j - tmp1j) * factor;

            tmp0jj = (y_trn[x_tst_knn_gt_jj_i] != y_tst_jj) ? 0.0:1.0;
            tmp1jj = (y_trn[x_tst_knn_gt_jj_i_plus_1] != y_tst_jj) ? 0.0:1.0;
            tmp3jj = mat_get(sp_gt, j+1, x_tst_knn_gt_jj_i_plus_1) + (tmp0jj - tmp1jj) * factor;

            x_tst_knn_gt_j_i_plus_1 = x_tst_knn_gt_j_i;
            x_tst_knn_gt_jj_i_plus_1 = x_tst_knn_gt_jj_i;

            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3j);
            mat_set(sp_gt, j+1, x_tst_knn_gt_jj_i, tmp3jj);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 5){

        printf("No/not enough arguments given, please input N M d K");
        return 0;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int d = atoi(argv[3]);
    int K = atoi(argv[4]);


    printf("remaining_runtime,argsort,l2-norm,compute_shapley\n");

    for(int iter = 0; iter < NUM_RUNS; ++iter ) {
        int_mat x_tst_knn_gt;
        mat sp_gt;
        mat x_trn;
        mat x_tst;
        int* y_trn = malloc(N*sizeof(int));
        int* y_tst = malloc(M*sizeof(int));

        build(&sp_gt, M, N);
        build(&x_trn, N, d);
        build(&x_tst, M, d);
        build_int_mat(&x_tst_knn_gt, M, N);

        myInt64 start, cycles;
        initialize_rand_array(y_trn, N);
        initialize_rand_array(y_tst, M);
        initialize_rand_mat(&x_trn);
        initialize_rand_mat(&x_tst);

        initialize_mat(&sp_gt, 0.0);

        get_true_knn(&x_tst_knn_gt, &x_trn, &x_tst);
        start = start_tsc();
        compute_single_unweighted_knn_class_shapley(&sp_gt, y_trn, y_tst, &x_tst_knn_gt, K);
        cycles = stop_tsc(start);
        printf("%lld", cycles);
        printf("\n");
        destroy(&x_trn);
        destroy(&x_tst);
        destroy(&sp_gt);
        destroy_int_mat(&x_tst_knn_gt);
        free(y_trn);
        free(y_tst);
    }

    return 0;
}
