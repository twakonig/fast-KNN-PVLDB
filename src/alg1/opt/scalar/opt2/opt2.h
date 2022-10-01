#include <stdio.h>
#include "mat.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include "quadsort.h"

// START L2-NORM

float l2norm_opt(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float res_tmp0, res_tmp1, res_tmp2, res_tmp3, res_tmp4, res_tmp5, res_tmp6, res_tmp7;
    float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    for (size_t i = 0; i < len-7; i+=8) {
        // separate accumulators
        tmp0 = arr1[i] - arr2[i];
        tmp1 = arr1[i+1] - arr2[i+1];
        tmp2 = arr1[i+2] - arr2[i+2];
        tmp3 = arr1[i+3] - arr2[i+3];
        tmp4 = arr1[i+4] - arr2[i+4];
        tmp5 = arr1[i+5] - arr2[i+5];
        tmp6 = arr1[i+6] - arr2[i+6];
        tmp7 = arr1[i+7] - arr2[i+7];

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
    float* data_trn = x_trn->data;
    float* data_tst = x_tst->data;


    for (int i_tst = 0; i_tst < N_tst; i_tst++){
        pair_t* dist_gt = malloc(N * sizeof(pair_t));
        for (int i_trn = 0; i_trn < N; i_trn++){
            float trn_row[d];
            float tst_row[d];
            for (int j = 0; j < d; j++) {
                trn_row[j] = data_trn[i_trn * d + j];
                tst_row[j] = data_tst[i_tst * d + j];
            }
            dist_gt[i_trn].index = i_trn;
            dist_gt[i_trn].value = l2norm_opt(trn_row, tst_row, d);
        }
        quadsort(dist_gt, N, sizeof(pair_t), cmp);
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = dist_gt[k].index;
        }
        free(dist_gt);
    }
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
