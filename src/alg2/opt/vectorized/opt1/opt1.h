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

// HEAP START
typedef struct{
    int K;
    mat* x_trn; // pointer to train matrix
    int counter;
    bool changed;
    int current_sz;
    int* heap;
    float* x_tst;
} maxheap;

void build_heap(maxheap* h, int K, mat* x_trn, float* rowvec){
    h->K = K;
    h->x_trn = x_trn;
    h->counter = -1;
    h->current_sz = 0;
    h->heap = malloc(K * sizeof(int));
    h->x_tst = malloc(x_trn->n2 * sizeof(float));
    memcpy(h->x_tst, rowvec, x_trn->n2 * sizeof(float));
    h->changed = false;
}

void nuke_heap(maxheap* h) {
    free(h->heap);
    free(h->x_tst);
}

void up(maxheap* h, int index) {
    if (index == 0) {
        return;
    }
    int parent_index = (index-1)/2;

    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    get_row(row, h->x_trn, h->heap[index]);
    float norm_idx = l2norm_opt(h->x_tst, row, h->x_trn->n2);
    get_row(row, h->x_trn, h->heap[parent_index]);
    float norm_parentidx = l2norm_opt(h->x_tst, row, h->x_trn->n2);
    free(row);
    if (norm_idx > norm_parentidx) {
        int temp = h->heap[index];
        h->heap[index] = h->heap[parent_index];
        h->heap[parent_index] = temp;
        up(h, parent_index);
    }

    return;
}

void down(maxheap* h, int index) {

    int tar_index;
    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    if (2*index + 1 > h->counter){
        free(row);
        return;
    }
    if (2*index + 1 < h->counter) {
        get_row(row, h->x_trn, h->heap[2*index+1]);
        float norm1 = l2norm_opt(h->x_tst, row, h->x_trn->n2);
        get_row(row, h->x_trn, h->heap[2*index+2]);
        float norm2 = l2norm_opt(h->x_tst, row, h->x_trn->n2);
        // free(row);
        if (norm1 < norm2) {
            tar_index = 2*index+2;
        } else {
            tar_index = 2*index+1;
        }
    } else {
        tar_index = 2*index+1;
    }
    get_row(row, h->x_trn, h->heap[index]);
    float norm1 = l2norm_opt(h->x_tst, row, h->x_trn->n2);
    get_row(row, h->x_trn, h->heap[tar_index]);
    float norm2 = l2norm_opt(h->x_tst, row, h->x_trn->n2);


    if (norm1 < norm2) {
        int temp = h->heap[index];
        h->heap[index] = h->heap[tar_index];
        h->heap[tar_index] = temp;
        down(h, tar_index);
    }
    return;
}

// elem is index of permuted matrix from main
void insert(maxheap* h, int elem) {
    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    get_row(row, h->x_trn, elem);
    float d_elem = l2norm_opt(row, h->x_tst, h->x_trn->n2);

    if (h->counter <= (h->K - 2)) {
        h->heap[h->current_sz] = elem;
        h->current_sz += 1;
        h->counter += 1;
        up(h, h->counter);
        h->changed = 1;
    }
    else {
        get_row(row,h->x_trn, h->heap[0]);
        float d_root = l2norm_opt(row, h->x_tst, h->x_trn->n2);
        if (d_elem < d_root) {
            h->heap[0] = elem;
            down(h, 0);
            h->changed = 1;
        }
        else
            h->changed = 0;
    }
    free(row);
}

// HEAP END



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

    maxheap heap;
    mat sp_approx_all;

    int* n_trn = (int *) malloc(N * sizeof(int));
    float* value_now = (float *) malloc(N * sizeof(float));
    float* x_tst_row = (float *) malloc(d * sizeof(float));

    build(&sp_approx_all, T, N);

    // populate n_trn
    for(int i = 0; i < N; i++){
        n_trn[i] = i;
    }


    for (int i_tst = 0; i_tst < N_tst; i_tst++){
        for (int t = 0; t < T; t++){

            // populate value_now with zeros
            for(int i = 0; i < N; i++){
                value_now[i] = 0.0;
            }

            shuffle(n_trn, N);
            get_row(x_tst_row, x_tst, i_tst);
            build_heap(&heap, K, x_trn, x_tst_row);

            for (int k = 0; k < N; k++){
                insert(&heap, n_trn[k]);
                if (heap.changed){
                    int* y_trn_slice = (int *) malloc((heap.counter + 1) * sizeof(int));

                    for (int m = 0; m < heap.counter + 1; m++){
                        y_trn_slice[m] = y_trn[heap.heap[m]];
                    }
                    value_now[k] = unweighted_knn_utility(y_trn_slice, y_tst[i_tst], heap.counter + 1, K);
                    free(y_trn_slice);
                }else{
                    value_now[k] = value_now[k-1];
                }
            }

            // compute the marginal contribution of the k-th user's data
            mat_set(&sp_approx_all, t, n_trn[0], value_now[0]);
            for (int l = 1; l < N; l++){
                mat_set(&sp_approx_all, t, n_trn[l], value_now[l] - value_now[l-1]);
            }
            nuke_heap(&heap);
        }

        compute_col_mean(sp_approx, &sp_approx_all, i_tst);
    }
    free(value_now);
    free(x_tst_row);
    free(n_trn);
    destroy(&sp_approx_all);
}
