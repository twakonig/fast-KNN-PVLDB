#include <stdio.h>
#include <stdlib.h>
#include "../include/mat.h"
#include "heap_flops.h"
#include <math.h>
#include <string.h>
#include <time.h>

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


float unweighted_knn_utility(int* y_trn, int y_tst, int size, int K, long long unsigned int *flops){
    float sum = 0.0;
    for (int i = 0; i < size; i++){
        sum += (y_trn[i] == y_tst) ? 1.0 : 0.0;
    }
    *flops += size +1;
    return sum / K;
}

void compute_col_mean(mat* m, mat* res, int i_tst, long long unsigned int *flops){
    float mean;
    for (int i = 0; i < res->n2; i++){ // N
        mean = 0.0;
        for (int j = 0; j < res->n1; j++){ // T
            mean += mat_get(res, j, i);
            *flops += 1;
        }
        mean /= res->n1;
        *flops += 1;
        mat_set(m, i_tst, i, mean);
    }

}

// get knn, returns mat of sorted data entries (knn alg)
void knn_mc_approximation(mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T, long long unsigned int *flops){

    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;

    tensor sp_approx_all;
    maxheap heap;
    mat tensor_slice;

    int* n_trn = (int *) malloc(N * sizeof(int));
    float* value_now = (float *) malloc(N * sizeof(float));
    float* x_tst_row = (float *) malloc(d * sizeof(float));


    build_tensor(&sp_approx_all, N_tst, T, N);
    build(&tensor_slice, T, N);

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
                insert(&heap, n_trn[k], flops);
                if (heap.changed){
                    int* y_trn_slice = (int *) malloc((heap.counter + 1) * sizeof(int));

                    for (int m = 0; m < heap.counter + 1; m++){
                        y_trn_slice[m] = y_trn[heap.heap[m]];
                    }
                    value_now[k] = unweighted_knn_utility(y_trn_slice, y_tst[i_tst], heap.counter + 1, K, flops);
                    free(y_trn_slice);
                }else{
                    value_now[k] = value_now[k-1];
                }
            }
            
            // compute the marginal contribution of the k-th user's data
            tensor_set(&sp_approx_all, i_tst, t, n_trn[0], value_now[0]);
            for (int l = 1; l < N; l++){
                tensor_set(&sp_approx_all, i_tst, t, n_trn[l], value_now[l] - value_now[l-1]);
                *flops += 1;
            }
            nuke_heap(&heap);
        }

        get_mat_from_tensor(&sp_approx_all, &tensor_slice, i_tst);
        compute_col_mean(sp_approx, &tensor_slice, i_tst, flops);
    }
    free(value_now);
    free(x_tst_row);
    free(n_trn);
    destroy(&tensor_slice);
    destroy_tensor(&sp_approx_all);
}
