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

// get knn, returns mat of sorted data entries (knn alg)
void get_true_knn(mat *x_tst_knn_gt, mat *x_trn, mat *x_tst){
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    float* data_trn = x_trn->data;
    float* data_tst = x_tst->data;
    pair_t* distances;
    distances = malloc(N * sizeof(pair_t));

    myInt64 start_l2norm, start_argsort, start, cycles_l2norm, cycles_argsort, cycles;
    cycles_l2norm = cycles_argsort = cycles = 0;

    build(x_tst_knn_gt, N_tst, N);
    start = start_tsc();
    for (int i_tst = 0; i_tst < N_tst; i_tst++){
//        pair_t* dist_gt[N];
//        int idx_arr[N];
        for (int i_trn = 0; i_trn < N; i_trn++){
            float trn_row[d];
            float tst_row[d];
            for (int j = 0; j < d; j++) {
                trn_row[j] = data_trn[i_trn * d + j];
                tst_row[j] = data_tst[i_tst * d + j];
            }
            distances[i_trn].index = i_trn;
            start_l2norm = start_tsc();
            distances[i_trn].value = l2norm_opt(trn_row, tst_row, d);
            cycles_l2norm += stop_tsc(start_l2norm);
        }
        start_argsort = start_tsc();
        quadsort(distances, N, sizeof(pair_t), cmp);
        cycles_argsort += stop_tsc(start_argsort);
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = distances[k].index;
        }
    }
    cycles = stop_tsc(start);
    cycles = cycles - cycles_argsort - cycles_l2norm;
    printf("%lld,", cycles);
    printf("%lld,", cycles_argsort);
    printf("%lld,", cycles_l2norm);
    free(distances);
}

void compute_single_unweighted_knn_class_shapley(mat* sp_gt, const int* y_trn, const int* y_tst, mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmp0, tmp1, tmp2, tmp3;
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;

    for (int j=0; j < N_tst;j++){
        x_tst_knn_gt_j_last_i = (int) mat_get(x_tst_knn_gt, j, N-1);
        tmp0 = (y_trn[x_tst_knn_gt_j_last_i] == y_tst[j]) ? 1.0/N : 0.0;
        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmp0);
        for (int i=N-2; i>-1; i--){
            x_tst_knn_gt_j_i = (int) mat_get(x_tst_knn_gt, j, i);
            x_tst_knn_gt_j_i_plus_1 = (int) mat_get(x_tst_knn_gt, j, \
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
        mat x_tst_knn_gt;
        mat sp_gt;
        mat x_trn;
        mat x_tst;
        int* y_trn = malloc(N*sizeof(int));
        int* y_tst = malloc(M*sizeof(int));

        build(&sp_gt, M, N);
        build(&x_trn, N, d);
        build(&x_tst, M, d);
        build(&x_tst_knn_gt, M, N);

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
        destroy(&x_tst_knn_gt);
        free(y_trn);
        free(y_tst);
    }

    return 0;
}
