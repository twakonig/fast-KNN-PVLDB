
#pragma once
#include <immintrin.h>
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "common.h"
#include "alg2_no_heap.h"





// get knn, returns mat of sorted data entries (knn alg)
void knn_mc_approximation_noheap_scalar_baseline(mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T){
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




void register_simd_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &knn_mc_approximation_noheap_scalar_baseline;


}
