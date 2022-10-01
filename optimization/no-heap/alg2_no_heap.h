#pragma once
#include <immintrin.h>
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "common.h"


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

//permutation function
// Renamed with a "_" added at the end, to avoid double declaration
void shuffle(int *arr, int n) {
     int i, j, tmp;

     for (i = n - 1; i >= 0; i--) {
         j = rand() % (i + 1);
         tmp = arr[j];
         arr[j] = arr[i];
         arr[i] = tmp;
     }
}

// Renamed with a "_" added at the end, to avoid double declaration
float unweighted_knn_utility_(int* y_trn, int y_tst, int size, int K){
    float sum = 0.0;
    for (int i = 0; i < size; i++){
        sum += (y_trn[i] == y_tst) ? 1.0 : 0.0;
    }
    return sum / K;
}

// Renamed with a "_" added at the end, to avoid double declaration
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
void knn_mc_approximation_noheap_baseline(mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T){
    //printf("\n");
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;

    int curr_element;

    float head_dist;

    tensor sp_approx_all;
    mat tensor_slice;

    int* n_trn = (int *) malloc(N * sizeof(int));
    float* value_now = (float *) malloc(N * sizeof(float));
    float* x_tst_row = (float *) malloc(d * sizeof(float));

    float* row = (float *) malloc(d * sizeof(float));
    int y_trn_slice;
    float new_elem_dist;


    build_tensor(&sp_approx_all, N_tst, T, N);
    build(&tensor_slice, T, N);

    // populate n_trn
    for(int i = 0; i < N; i++){
        n_trn[i] = i;
    }


    for (int i_tst = 0; i_tst < N_tst; i_tst++){
        for (int t = 0; t < T; t++){

            shuffle(n_trn, N);
            get_row(x_tst_row, x_tst, i_tst);

            // First element will be the current head.
            curr_element = n_trn[0];
            get_row(row, x_trn, curr_element);
            head_dist = l2norm(x_tst_row, row, x_trn->n2);
            y_trn_slice = y_trn[curr_element];
            value_now[0] = (y_tst[i_tst] == y_trn_slice) ? 1.0 : 0.0;

            for (int k = 1; k < N; k++){

                get_row(row, x_trn, n_trn[k]);
                new_elem_dist = l2norm(x_tst_row, row, x_trn->n2);

                if (new_elem_dist < head_dist) {
                    curr_element = n_trn[k];
                    head_dist = new_elem_dist;


                    y_trn_slice = y_trn[curr_element];

                    value_now[k] = (y_tst[i_tst] == y_trn_slice) ? 1.0 : 0.0;
                } else {
                    value_now[k] = value_now[k-1];
                }
            }

            // compute the marginal contribution of the k-th user's data
            tensor_set(&sp_approx_all, i_tst, t, n_trn[0], value_now[0]);
            for (int l = 1; l < N; l++){
                tensor_set(&sp_approx_all, i_tst, t, n_trn[l], value_now[l] - value_now[l-1]);
            }
        }

        get_mat_from_tensor(&sp_approx_all, &tensor_slice, i_tst);
        compute_col_mean(sp_approx, &tensor_slice, i_tst);
    }
    free(row);
    free(value_now);
    free(x_tst_row);
    free(n_trn);
    destroy(&tensor_slice);
    destroy_tensor(&sp_approx_all);
}


void register_scalar_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &knn_mc_approximation_noheap_baseline;


}