#include <stdio.h>
#include "../../../../../include/mat.h"
#include <math.h>
#include <string.h>
#include "../../../../../include/ksort.h"
#include "../../../../../include/quadsort.h"
#include "../../../../../include/utils.h"
#include <immintrin.h>
#define pair_lt(a, b) ((a).value < (b).value)

KSORT_INIT(pair, pair_t, pair_lt)
KSORT_INIT_GENERIC(float)

// START L2-NORM

float l2norm_opt(float *arr1, float *arr2, size_t strt1, size_t strt2, size_t len){
    float res = 0.0;
    int aij = strt1 * len;
    int bij = strt2 * len;
    float res_tmp0, res_tmp1, res_tmp2, res_tmp3, res_tmp4, res_tmp5, res_tmp6, res_tmp7;
    float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    for (size_t i = 0; i < len-7; i+=8) {
        // separate accumulators
        tmp0 = arr1[i+aij] - arr2[i+bij];
        tmp1 = arr1[i+1+aij] - arr2[i+1+bij];
        tmp2 = arr1[i+2+aij] - arr2[i+2+bij];
        tmp3 = arr1[i+3+aij] - arr2[i+3+bij];
        tmp4 = arr1[i+4+aij] - arr2[i+4+bij];
        tmp5 = arr1[i+5+aij] - arr2[i+5+bij];
        tmp6 = arr1[i+6+aij] - arr2[i+6+bij];
        tmp7 = arr1[i+7+aij] - arr2[i+7+bij];

        // intermediate store
        res_tmp0 = tmp0 * tmp0;
        res_tmp1 = tmp1 * tmp1;
        res_tmp2 = tmp2 * tmp2;
        res_tmp3 = tmp3 * tmp3;
        res_tmp4 = tmp4 * tmp4;
        res_tmp5 = tmp5 * tmp5;
        res_tmp6 = tmp6 * tmp6;
        res_tmp7 = tmp7 * tmp7;

        // collect intermediate results in THIS ORDER (separate accumulators: failed)
        res = res + res_tmp0 + res_tmp1 + res_tmp2 + res_tmp3 + res_tmp4 + res_tmp5 + res_tmp6 + res_tmp7;
    }
    return res;
}

// END L2-NORM

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

    for (int i_tst = 0; i_tst < N_tst; i_tst+=4){
        i_tstN = i_tst * N;
        i_tstNN = (i_tst+1) * N;
        i_tstNNN = (i_tst+2) * N;
        i_tstNNNN = (i_tst+3) * N;

#pragma GCC ivdep
        for (int i_trn = 0; i_trn < N; i_trn++){
            distances[i_trn].value = l2norm_opt(data_trn, data_tst, i_trn, i_tst, d);
            distances_i[i_trn].value = l2norm_opt(data_trn, data_tst, i_trn, (i_tst + 1), d);
            distances_ii[i_trn].value = l2norm_opt(data_trn, data_tst, i_trn, (i_tst + 2),  d);
            distances_iii[i_trn].value = l2norm_opt(data_trn, data_tst, i_trn, (i_tst + 3), d);
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



void compute_single_unweighted_knn_class_shapley(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmp0, tmp1, tmp2, tmp3;
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;

    for (int j=0; j < N_tst;j++){
        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        tmp0 = (y_trn[x_tst_knn_gt_j_last_i] == y_tst[j]) ? 1.0/N : 0.0;
        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmp0);
        for (int i=N-2; i>-1; i--){
            x_tst_knn_gt_j_i = int_mat_get(x_tst_knn_gt, j, i);
            x_tst_knn_gt_j_i_plus_1 = int_mat_get(x_tst_knn_gt, j, \
             i+1);
            tmp0 = (y_trn[x_tst_knn_gt_j_i] == y_tst[j]) ? 1.0:0.0;
            tmp1 = (y_trn[x_tst_knn_gt_j_i_plus_1] == y_tst[j]) ? 1.0:0.0;
            tmp2 = (K > i+1) ? i+1 : K;
            tmp3 = mat_get(sp_gt, j, x_tst_knn_gt_j_i_plus_1) + \
                    (tmp0 - tmp1) / K * tmp2 / (i + 1);
            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3);
        }
    }
}
