#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "mat.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

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

//permutation function
void shuffle(int *arr, int n) {
    int i, j, tmp;

    for (i = n - 1; i >= 0; i--) {
        j = rand() % (i + 1);
        tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

float unweighted_knn_utility(int* y_trn, int y_tst, int size, int K){
    float sum = 0.0;
    for (int i = 0; i < size; i++){
        sum += (y_trn[i] == y_tst) ? 1.0 : 0.0;
    }
    return sum / K;
}

void compute_col_mean(mat* m, mat* res, int i_tst){
    float mean;
    for (int i = 0; i < res->n2; i++){ // N
        mean = 0.0;
        for (int j = 0; j < res->n1; j++){ // T
            mean += mat_get(res, j, i);
        }
        mean /= res->n1;
        mat_set(m, i_tst, i, mean);
    }

}

// get knn, returns mat of sorted data entries (knn alg)
void knn_mc_approximation(mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T){

    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;

    mat sp_approx_all;

    int* n_trn = (int *) malloc(N * sizeof(int));
    float* value_now = (float *) malloc(N * sizeof(float));
    float* x_tst_row = (float *) malloc(d * sizeof(float));


    build(&sp_approx_all, T, N);

    float* row = (float*) malloc(d * sizeof(float));
    int* y_trn_slice = (int *) malloc(K * sizeof(int));

    int tmp, parent_index, curr_index, tar_index, n_trn_k;
    float norm_idx, norm_parentidx, d_elem, d_root, norm1, norm2;

    // init heap
    int* heap = malloc(K * sizeof(int));

    // populate n_trn
    for(int i = 0; i < N; i++){
        n_trn[i] = i;
    }

    for (int i_tst = 0; i_tst < N_tst; i_tst++){
        int y_tst_i_tst = y_tst[i_tst];
        for (int t = 0; t < T; t++){

            shuffle(n_trn, N);
            get_row(x_tst_row, x_tst, i_tst);

            // k = 0
            int n_trn_0 = n_trn[0];
            heap[0] = n_trn_0;

            int y_trn_elem = y_trn[n_trn_0];
            value_now[0] = (float)((y_trn_elem == y_tst_i_tst))/K;

            for (int k = 1; k < K; k++){
                // INSERT HEAP -- BEGIN
                heap[k] = n_trn[k];

                // UP -- BEGIN
                parent_index = (k-1)/2;

                get_row(row, x_trn, heap[k]);
                norm_idx = l2norm_opt(x_tst_row, row, d);
                get_row(row, x_trn, heap[parent_index]);
                norm_parentidx = l2norm_opt(x_tst_row, row, d);

                curr_index = k;
                while(norm_idx > norm_parentidx){
                    tmp = heap[curr_index];
                    heap[curr_index] = heap[parent_index];
                    heap[parent_index] = tmp;

                    curr_index = parent_index;
                    if (curr_index == 0) {
                        break;
                    }

                    parent_index = (curr_index-1)/2;

                    get_row(row, x_trn, heap[curr_index]);
                    norm_idx = l2norm_opt(x_tst_row, row, d);
                    get_row(row, x_trn, heap[parent_index]);
                    norm_parentidx = l2norm_opt(x_tst_row, row, d);
                }
                // UP -- END
                for (int m = 0; m < k + 1; m++){
                    y_trn_slice[m] = y_trn[heap[m]];
                }
                value_now[k] = unweighted_knn_utility(y_trn_slice, y_tst_i_tst, k + 1, K);
            }

            for (int k = K; k < N; k++){
                n_trn_k = n_trn[k];
                get_row(row, x_trn, n_trn_k);
                d_elem = l2norm_opt(row, x_tst_row, d);
                get_row(row, x_trn, heap[0]);
                d_root = l2norm_opt(row, x_tst_row, d);
                if (d_elem < d_root) {
                    heap[0] = n_trn_k;
                    // DOWN -- BEGIN
                    if (K >= 2){
                        if(K == 2){
                            tar_index = 1;
                        } else{
                            get_row(row, x_trn, heap[1]);
                            norm1 = l2norm_opt(x_tst_row, row, d);
                            get_row(row, x_trn, heap[2]);
                            norm2 = l2norm_opt(x_tst_row, row, d);
                            if (norm1 < norm2) {
                                tar_index = 2;
                            } else {
                                tar_index = 1;
                            }
                        }

                        get_row(row, x_trn, heap[0]);
                        norm1 = l2norm_opt(x_tst_row, row, d);
                        get_row(row, x_trn, heap[tar_index]);
                        norm2 = l2norm_opt(x_tst_row, row, d);

                        curr_index = 0;
                        while(norm1 < norm2){
                            tmp = heap[curr_index];
                            heap[curr_index] = heap[tar_index];
                            heap[tar_index] = tmp;

                            curr_index = tar_index;
                            if(2*curr_index + 2 > K){
                                break;
                            }
                            else if(2*curr_index + 2 == K){
                                tar_index = 2*curr_index+1;
                            } else{
                                get_row(row, x_trn, heap[2*curr_index+1]);
                                norm1 = l2norm_opt(x_tst_row, row, d);
                                get_row(row, x_trn, heap[2*curr_index+2]);
                                norm2 = l2norm_opt(x_tst_row, row, d);
                                if (norm1 < norm2) {
                                    tar_index = 2*curr_index+2;
                                } else {
                                    tar_index = 2*curr_index+1;
                                }
                            }

                            get_row(row, x_trn, heap[curr_index]);
                            norm1 = l2norm_opt(x_tst_row, row, d);
                            get_row(row, x_trn, heap[tar_index]);
                            norm2 = l2norm_opt(x_tst_row, row, d);
                        }
                    }
                    // DOWN -- END
                    for (int m = 0; m < K; m++){
                        y_trn_slice[m] = y_trn[heap[m]];
                    }
                    value_now[k] = unweighted_knn_utility(y_trn_slice, y_tst_i_tst, K, K);
                }
                else
                    value_now[k] = value_now[k-1];
            }
            // INSERT HEAP -- END

            // compute the marginal contribution of the k-th user's data
            mat_set(&sp_approx_all, t, n_trn[0], value_now[0]);

            for (int l = 1; l < N; l++){
                mat_set(&sp_approx_all, t, n_trn[l], value_now[l] - value_now[l-1]);
            }
        }
        compute_col_mean(sp_approx, &sp_approx_all, i_tst);
    }
    destroy(&sp_approx_all);
    free(heap);
    free(row);
    free(value_now);
    free(x_tst_row);
    free(n_trn);
}
