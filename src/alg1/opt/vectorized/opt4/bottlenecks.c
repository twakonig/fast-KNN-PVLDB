//
// Created by lucasck on 05/06/22.
//
#include <stdio.h>
#include "mat.h"
#include <math.h>
#include <string.h>
#include "utils.h"
#include "tsc_x86.h"
#include <immintrin.h>
#include "quadsort.h"
// #ifndef T 130
#define NUM_RUNS 30

float l2norm_opt(float a[], float b[], size_t strt1, size_t strt2, size_t len){
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

    int offset1 = strt1 * len;
    int offset2 = strt2 * len;

    for (size_t i = 0; i < len; i+=32) {
        // load data
        a_vec0 = _mm256_loadu_ps(a+i+offset1);
        b_vec0 = _mm256_loadu_ps(b+i+offset2);
        a_vec1 = _mm256_loadu_ps(a+i+8+offset1);
        b_vec1 = _mm256_loadu_ps(b+i+8+offset2);
        a_vec2 = _mm256_loadu_ps(a+i+16+offset1);
        b_vec2 = _mm256_loadu_ps(b+i+16+offset2);
        a_vec3 = _mm256_loadu_ps(a+i+24+offset1);
        b_vec3 = _mm256_loadu_ps(b+i+24+offset2);
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


// get knn, returns mat of sorted data entries (knn alg)
void get_true_knn(int_mat *x_tst_knn_gt, mat *x_trn, mat *x_tst){
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    pair_t *distances, *distances_i, *distances_ii, *distances_iii;
    distances = malloc(N * sizeof(pair_t));
    distances_i = malloc(N * sizeof(pair_t));
    distances_ii = malloc(N * sizeof(pair_t));
    distances_iii = malloc(N * sizeof(pair_t));

    float* data_trn = x_trn->data;
    float* data_tst = x_tst->data;

    int i_tstd, i_tstdd, i_tstddd, i_tstdddd, i_tstN, i_tstNN, i_tstNNN, i_tstNNNN;
    myInt64 start_l2norm, start_argsort, start, cycles_l2norm, cycles_argsort, cycles;
    cycles_l2norm = cycles_argsort = cycles = 0;

    start = start_tsc();
    for (int i_tst = 0; i_tst < N_tst; i_tst+=4){

        i_tstN = i_tst * N;
        i_tstNN = (i_tst+1) * N;
        i_tstNNN = (i_tst+2) * N;
        i_tstNNNN = (i_tst+3) * N;

#pragma GCC ivdep
        for (int i_trn = 0; i_trn < N; i_trn++){
            start_l2norm = start_tsc();
            distances[i_trn].value = l2norm_opt(data_trn, data_tst, i_trn, i_tst, d);
            distances_i[i_trn].value = l2norm_opt(data_trn, data_tst, i_trn, (i_tst + 1), d);
            distances_ii[i_trn].value = l2norm_opt(data_trn, data_tst, i_trn, (i_tst + 2),  d);
            distances_iii[i_trn].value = l2norm_opt(data_trn, data_tst, i_trn, (i_tst + 3), d);
            cycles_l2norm += stop_tsc(start_l2norm);
            distances[i_trn].index = i_trn;
            distances_i[i_trn].index = i_trn;
            distances_ii[i_trn].index = i_trn;
            distances_iii[i_trn].index = i_trn;
        }
        start_argsort = start_tsc();
        quadsort(distances, N, sizeof(pair_t), cmp);
        quadsort(distances_i, N, sizeof(pair_t), cmp);
        quadsort(distances_ii, N, sizeof(pair_t), cmp);
        quadsort(distances_iii, N, sizeof(pair_t), cmp);
        cycles_argsort += stop_tsc(start_argsort);
#pragma GCC ivdep
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tstNNNN + k] = distances_iii[k].index;
            x_tst_knn_gt->data[i_tstNNN + k] = distances_ii[k].index;
            x_tst_knn_gt->data[i_tstNN + k] = distances_i[k].index;
            x_tst_knn_gt->data[i_tstN + k] = distances[k].index;
        }
    }
    cycles = stop_tsc(start);
    cycles = cycles - cycles_argsort - cycles_l2norm;
    printf("%lld,", cycles);
    printf("%lld,", cycles_argsort);
    printf("%lld,", cycles_l2norm);
    free(distances);
    free(distances_i);
    free(distances_ii);
    free(distances_iii);
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
