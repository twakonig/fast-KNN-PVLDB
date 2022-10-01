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

// START COLMEAN
void compute_col_mean(mat* m, mat* res, int i_tst){
    float mean0;
    float mean1;
    float mean2;
    float mean3;
    float mean4;
    float mean5;
    float mean6;
    float mean7;

    float mean8;
    float mean9;
    float mean10;
    float mean11;
    float mean12;
    float mean13;
    float mean14;
    float mean15;


    int n2 = res->n2;
    int n1 = res->n1;
    float *data = res->data;
    float *m_data = m->data;
    for (int i = 0; i < n2-15; i+=16){ // N
        mean0 = 0.0;
        mean1 = 0.0;
        mean2 = 0.0;
        mean3 = 0.0;
        mean4 = 0.0;
        mean5 = 0.0;
        mean6 = 0.0;
        mean7 = 0.0;

        mean8 = 0.0;
        mean9 = 0.0;
        mean10 = 0.0;
        mean11 = 0.0;
        mean12 = 0.0;
        mean13 = 0.0;
        mean14 = 0.0;
        mean15 = 0.0;
        for (int j = 0; j < n1-7; j+=8){ // T
            mean0 += data[j*n2+i];
            mean0 += data[(j+1)*n2+i];
            mean0 += data[(j+2)*n2+i];
            mean0 += data[(j+3)*n2+i];
            mean0 += data[(j+4)*n2+i];
            mean0 += data[(j+5)*n2+i];
            mean0 += data[(j+6)*n2+i];
            mean0 += data[(j+7)*n2+i];

            mean1 += data[j*n2+i+1];
            mean1 += data[(j+1)*n2+i+1];
            mean1 += data[(j+2)*n2+i+1];
            mean1 += data[(j+3)*n2+i+1];
            mean1 += data[(j+4)*n2+i+1];
            mean1 += data[(j+5)*n2+i+1];
            mean1 += data[(j+6)*n2+i+1];
            mean1 += data[(j+7)*n2+i+1];

            mean2 += data[j*n2+i+2];
            mean2 += data[(j+1)*n2+i+2];
            mean2 += data[(j+2)*n2+i+2];
            mean2 += data[(j+3)*n2+i+2];
            mean2 += data[(j+4)*n2+i+2];
            mean2 += data[(j+5)*n2+i+2];
            mean2 += data[(j+6)*n2+i+2];
            mean2 += data[(j+7)*n2+i+2];

            mean3 += data[j*n2+i+3];
            mean3 += data[(j+1)*n2+i+3];
            mean3 += data[(j+2)*n2+i+3];
            mean3 += data[(j+3)*n2+i+3];
            mean3 += data[(j+4)*n2+i+3];
            mean3 += data[(j+5)*n2+i+3];
            mean3 += data[(j+6)*n2+i+3];
            mean3 += data[(j+7)*n2+i+3];

            mean4 += data[j*n2+i+4];
            mean4 += data[(j+1)*n2+i+4];
            mean4 += data[(j+2)*n2+i+4];
            mean4 += data[(j+3)*n2+i+4];
            mean4 += data[(j+4)*n2+i+4];
            mean4 += data[(j+5)*n2+i+4];
            mean4 += data[(j+6)*n2+i+4];
            mean4 += data[(j+7)*n2+i+4];

            mean5 += data[j*n2+i+5];
            mean5 += data[(j+1)*n2+i+5];
            mean5 += data[(j+2)*n2+i+5];
            mean5 += data[(j+3)*n2+i+5];
            mean5 += data[(j+4)*n2+i+5];
            mean5 += data[(j+5)*n2+i+5];
            mean5 += data[(j+6)*n2+i+5];
            mean5 += data[(j+7)*n2+i+5];

            mean6 += data[j*n2+i+6];
            mean6 += data[(j+1)*n2+i+6];
            mean6 += data[(j+2)*n2+i+6];
            mean6 += data[(j+3)*n2+i+6];
            mean6 += data[(j+4)*n2+i+6];
            mean6 += data[(j+5)*n2+i+6];
            mean6 += data[(j+6)*n2+i+6];
            mean6 += data[(j+7)*n2+i+6];

            mean7 += data[j*n2+i+7];
            mean7 += data[(j+1)*n2+i+7];
            mean7 += data[(j+2)*n2+i+7];
            mean7 += data[(j+3)*n2+i+7];
            mean7 += data[(j+4)*n2+i+7];
            mean7 += data[(j+5)*n2+i+7];
            mean7 += data[(j+6)*n2+i+7];
            mean7 += data[(j+7)*n2+i+7];

            //---------------------------
            mean8 += data[j*n2+i+8];
            mean8 += data[(j+1)*n2+i+8];
            mean8 += data[(j+2)*n2+i+8];
            mean8 += data[(j+3)*n2+i+8];
            mean8 += data[(j+4)*n2+i+8];
            mean8 += data[(j+5)*n2+i+8];
            mean8 += data[(j+6)*n2+i+8];
            mean8 += data[(j+7)*n2+i+8];

            mean9 += data[j*n2+i+9];
            mean9 += data[(j+1)*n2+i+9];
            mean9 += data[(j+2)*n2+i+9];
            mean9 += data[(j+3)*n2+i+9];
            mean9 += data[(j+4)*n2+i+9];
            mean9 += data[(j+5)*n2+i+9];
            mean9 += data[(j+6)*n2+i+9];
            mean9 += data[(j+7)*n2+i+9];

            mean10 += data[j*n2+i+10];
            mean10 += data[(j+1)*n2+i+10];
            mean10 += data[(j+2)*n2+i+10];
            mean10 += data[(j+3)*n2+i+10];
            mean10 += data[(j+4)*n2+i+10];
            mean10 += data[(j+5)*n2+i+10];
            mean10 += data[(j+6)*n2+i+10];
            mean10 += data[(j+7)*n2+i+10];

            mean11 += data[j*n2+i+11];
            mean11 += data[(j+1)*n2+i+11];
            mean11 += data[(j+2)*n2+i+11];
            mean11 += data[(j+3)*n2+i+11];
            mean11 += data[(j+4)*n2+i+11];
            mean11 += data[(j+5)*n2+i+11];
            mean11 += data[(j+6)*n2+i+11];
            mean11 += data[(j+7)*n2+i+11];

            mean12 += data[j*n2+i+12];
            mean12 += data[(j+1)*n2+i+12];
            mean12 += data[(j+2)*n2+i+12];
            mean12 += data[(j+3)*n2+i+12];
            mean12 += data[(j+4)*n2+i+12];
            mean12 += data[(j+5)*n2+i+12];
            mean12 += data[(j+6)*n2+i+12];
            mean12 += data[(j+7)*n2+i+12];

            mean13 += data[j*n2+i+13];
            mean13 += data[(j+1)*n2+i+13];
            mean13 += data[(j+2)*n2+i+13];
            mean13 += data[(j+3)*n2+i+13];
            mean13 += data[(j+4)*n2+i+13];
            mean13 += data[(j+5)*n2+i+13];
            mean13 += data[(j+6)*n2+i+13];
            mean13 += data[(j+7)*n2+i+13];

            mean14 += data[j*n2+i+14];
            mean14 += data[(j+1)*n2+i+14];
            mean14 += data[(j+2)*n2+i+14];
            mean14 += data[(j+3)*n2+i+14];
            mean14 += data[(j+4)*n2+i+14];
            mean14 += data[(j+5)*n2+i+14];
            mean14 += data[(j+6)*n2+i+14];
            mean14 += data[(j+7)*n2+i+14];

            mean15 += data[j*n2+i+15];
            mean15 += data[(j+1)*n2+i+15];
            mean15 += data[(j+2)*n2+i+15];
            mean15 += data[(j+3)*n2+i+15];
            mean15 += data[(j+4)*n2+i+15];
            mean15 += data[(j+5)*n2+i+15];
            mean15 += data[(j+6)*n2+i+15];
            mean15 += data[(j+7)*n2+i+15];

        }
        mean0 /= n1;
        mean1 /= n1;
        mean2 /= n1;
        mean3 /= n1;
        mean4 /= n1;
        mean5 /= n1;
        mean6 /= n1;
        mean7 /= n1;

        mean8 /= n1;
        mean9 /= n1;
        mean10 /= n1;
        mean11 /= n1;
        mean12 /= n1;
        mean13 /= n1;
        mean14 /= n1;
        mean15 /= n1;
        //mat_set(m, i_tst, i, mean);
        m_data[i_tst*m->n2+i] = mean0;
        m_data[i_tst*m->n2+i+1] = mean1;
        m_data[i_tst*m->n2+i+2] = mean2;
        m_data[i_tst*m->n2+i+3] = mean3;
        m_data[i_tst*m->n2+i+4] = mean4;
        m_data[i_tst*m->n2+i+5] = mean5;
        m_data[i_tst*m->n2+i+6] = mean6;
        m_data[i_tst*m->n2+i+7] = mean7;

        m_data[i_tst*m->n2+i+8] = mean8;
        m_data[i_tst*m->n2+i+9] = mean9;
        m_data[i_tst*m->n2+i+10] = mean10;
        m_data[i_tst*m->n2+i+11] = mean11;
        m_data[i_tst*m->n2+i+12] = mean12;
        m_data[i_tst*m->n2+i+13] = mean13;
        m_data[i_tst*m->n2+i+14] = mean14;
        m_data[i_tst*m->n2+i+15] = mean15;
    }

}
// END COLMEAN

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
