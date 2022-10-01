#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "mat.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

// L2-NORM START

float l2norm_opt(float a[], float b[], size_t len){
    float* vresult = malloc(8*sizeof(float));
    float res_scalar = 0.0;
    __m256 a_vec0, b_vec0, a_vec1, b_vec1, a_vec2, b_vec2, a_vec3, b_vec3;
    __m256 sub_vec0, res_vec0, sub_vec1, res_vec1, sub_vec2, res_vec2, sub_vec3, res_vec3;
    __m256 tmp_vec0, tmp_vec1, res_vec;

    res_vec0 = _mm256_setzero_ps();
    res_vec1 = _mm256_setzero_ps();
    res_vec2 = _mm256_setzero_ps();
    res_vec3 = _mm256_setzero_ps();

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
    // sum up vector elements
    for (int k = 0; k < 8; k++) {
        res_scalar += vresult[k];
    }
    free(vresult);
    return res_scalar;
}

// L2-NORM END

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

// COL-MEAN START

void compute_col_mean(mat* m, mat* res, int i_tst){
    __m256 res_vec, res_vec2, row0, row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13, row14, row15;
    __m256 res_vecb, res_vec2b, row0b, row1b, row2b, row3b, row4b, row5b, row6b, row7b, row8b, row9b, row10b, row11b, row12b, row13b, row14b, row15b;
    __m256 subvec0, subvec1, subvec2, subvec3, subvec4, subvec5, subvec6, subvec7, subvec8, subvec9, subvec10, subvec11, subvec12, subvec13;
    __m256 subvec0b, subvec1b, subvec2b, subvec3b, subvec4b, subvec5b, subvec6b, subvec7b, subvec8b, subvec9b, subvec10b, subvec11b, subvec12b, subvec13b;
    int n2 = res->n2;
    int n1 = res->n1;
    float n_inv = 1.0/n1;
    float *data = res->data;
    float *m_data = m->data;
    __m256 n = _mm256_set1_ps(n_inv);
    for (int i = 0; i < res->n2-31; i+=32){ // N
        res_vec = _mm256_set1_ps(0.0);
        res_vec2 = _mm256_set1_ps(0.0);

        res_vecb = _mm256_set1_ps(0.0);
        res_vec2b = _mm256_set1_ps(0.0);
        for (int j = 0; j < res->n1-7; j+=8){ // T
            // load
            row0 = _mm256_loadu_ps(&data[j*n2+i]);
            row1 = _mm256_loadu_ps(&data[(j+1)*n2+i]);
            row2 = _mm256_loadu_ps(&data[(j+2)*n2+i]);
            row3 = _mm256_loadu_ps(&data[(j+3)*n2+i]);
            row4 = _mm256_loadu_ps(&data[(j+4)*n2+i]);
            row5 = _mm256_loadu_ps(&data[(j+5)*n2+i]);
            row6 = _mm256_loadu_ps(&data[(j+6)*n2+i]);
            row7 = _mm256_loadu_ps(&data[(j+7)*n2+i]);

            row8 =  _mm256_loadu_ps(&data[j*n2+i+8]);
            row9 =  _mm256_loadu_ps(&data[(j+1)*n2+i+8]);
            row10 = _mm256_loadu_ps(&data[(j+2)*n2+i+8]);
            row11 = _mm256_loadu_ps(&data[(j+3)*n2+i+8]);
            row12 = _mm256_loadu_ps(&data[(j+4)*n2+i+8]);
            row13 = _mm256_loadu_ps(&data[(j+5)*n2+i+8]);
            row14 = _mm256_loadu_ps(&data[(j+6)*n2+i+8]);
            row15 = _mm256_loadu_ps(&data[(j+7)*n2+i+8]);

            row0b = _mm256_loadu_ps(&data[j*n2+i+16]);
            row1b = _mm256_loadu_ps(&data[(j+1)*n2+i+16]);
            row2b = _mm256_loadu_ps(&data[(j+2)*n2+i+16]);
            row3b = _mm256_loadu_ps(&data[(j+3)*n2+i+16]);
            row4b = _mm256_loadu_ps(&data[(j+4)*n2+i+16]);
            row5b = _mm256_loadu_ps(&data[(j+5)*n2+i+16]);
            row6b = _mm256_loadu_ps(&data[(j+6)*n2+i+16]);
            row7b = _mm256_loadu_ps(&data[(j+7)*n2+i+16]);

            row8b =  _mm256_loadu_ps(&data[j*n2+i+24]);
            row9b =  _mm256_loadu_ps(&data[(j+1)*n2+i+24]);
            row10b = _mm256_loadu_ps(&data[(j+2)*n2+i+24]);
            row11b = _mm256_loadu_ps(&data[(j+3)*n2+i+24]);
            row12b = _mm256_loadu_ps(&data[(j+4)*n2+i+24]);
            row13b = _mm256_loadu_ps(&data[(j+5)*n2+i+24]);
            row14b = _mm256_loadu_ps(&data[(j+6)*n2+i+24]);
            row15b = _mm256_loadu_ps(&data[(j+7)*n2+i+24]);

            // calculate
            subvec0 = _mm256_add_ps(row0, row1);
            subvec1 = _mm256_add_ps(row2, row3);
            subvec2 = _mm256_add_ps(row4, row5);
            subvec3 = _mm256_add_ps(row6, row7);

            subvec4 = _mm256_add_ps(row8, row9);
            subvec5 = _mm256_add_ps(row10, row11);
            subvec6 = _mm256_add_ps(row12, row13);
            subvec7 = _mm256_add_ps(row14, row15);

            subvec0b = _mm256_add_ps(row0b, row1b);
            subvec1b = _mm256_add_ps(row2b, row3b);
            subvec2b = _mm256_add_ps(row4b, row5b);
            subvec3b = _mm256_add_ps(row6b, row7b);

            subvec4b = _mm256_add_ps(row8b, row9b);
            subvec5b = _mm256_add_ps(row10b, row11b);
            subvec6b = _mm256_add_ps(row12b, row13b);
            subvec7b = _mm256_add_ps(row14b, row15b);

            subvec8 = _mm256_add_ps(subvec0, subvec1);
            subvec9 = _mm256_add_ps(subvec2, subvec3);

            subvec10 = _mm256_add_ps(subvec4, subvec5);
            subvec11 = _mm256_add_ps(subvec6, subvec7);

            subvec8b = _mm256_add_ps(subvec0b, subvec1b);
            subvec9b = _mm256_add_ps(subvec2b, subvec3b);

            subvec10b = _mm256_add_ps(subvec4b, subvec5b);
            subvec11b = _mm256_add_ps(subvec6b, subvec7b);

            subvec12 = _mm256_add_ps(subvec8, subvec9);
            subvec13 = _mm256_add_ps(subvec10, subvec11);

            subvec12b = _mm256_add_ps(subvec8b, subvec9b);
            subvec13b = _mm256_add_ps(subvec10b, subvec11b);

            res_vec = _mm256_add_ps(res_vec, subvec12);
            res_vec2 = _mm256_add_ps(res_vec2, subvec13);

            res_vecb = _mm256_add_ps(res_vecb, subvec12b);
            res_vec2b = _mm256_add_ps(res_vec2b, subvec13b);
        }
        res_vec = _mm256_mul_ps(res_vec, n);
        res_vec2 = _mm256_mul_ps(res_vec2, n);

        res_vecb = _mm256_mul_ps(res_vecb, n);
        res_vec2b = _mm256_mul_ps(res_vec2b, n);

        _mm256_storeu_ps(&m_data[i_tst*m->n2+i], res_vec);
        _mm256_storeu_ps(&m_data[i_tst*m->n2+i+8], res_vec2);

        _mm256_storeu_ps(&m_data[i_tst*m->n2+i+16], res_vecb);
        _mm256_storeu_ps(&m_data[i_tst*m->n2+i+24], res_vec2b);
    }
}

// COL-MEAN END

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

    free(heap);
    free(row);
    free(value_now);
    free(x_tst_row);
    free(n_trn);
    destroy(&sp_approx_all);
}
