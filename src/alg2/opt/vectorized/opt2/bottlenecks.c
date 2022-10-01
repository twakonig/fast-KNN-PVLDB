//
// Created by lucasck on 05/06/22.
//

//#include "alg2.h"
#include "mat.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "tsc_x86.h"
#include "utils.h"
#include <immintrin.h>
// warmup iterations
#define NUM_WARMUP 100
// num. of iterations (measurements) per n
#define NUM_RUNS 30

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
    res_vec = _mm256_setzero_ps();

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
void knn_mc_approximation_bottlenecks(mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T){

    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    mat sp_approx_all;

    int* n_trn = (int *) malloc(N * sizeof(int));
    float* value_now = (float *) malloc(N * sizeof(float));
    float* x_tst_row = (float *) malloc(d * sizeof(float));
    float* row = (float*) malloc(d * sizeof(float));
    int* y_trn_slice = (int *) malloc(K * sizeof(int));
    int* heap = malloc(K * sizeof(int));

    build(&sp_approx_all, T, N);

    myInt64 start_shuffle, start_build_heap, start_inner, start, start_insert, start_up, start_down, start_l2_norm, start_l2_norm_up, start_l2_norm_down, start_col_mean, start_utility;
    myInt64 cycles_shuffle, cycles_build_heap, cycles_inner, cycles, cycles_insert, cycles_insert_l2_norm, cycles_insert_l2_norm_up, cycles_insert_l2_norm_down,  cycles_insert_up, cycles_insert_down, cycles_col_mean, cycles_utility;
    cycles_shuffle = cycles_build_heap = cycles_inner = cycles = cycles_insert = cycles_col_mean = cycles_utility = cycles_insert_l2_norm = cycles_insert_l2_norm_up = cycles_insert_l2_norm_down = cycles_insert_up = cycles_insert_down = 0;

    int tmp, parent_index, curr_index, tar_index, n_trn_k;
    float norm_idx, norm_parentidx, d_elem, d_root, norm1, norm2;

    // populate n_trn
    for(int i = 0; i < N; i++){
        n_trn[i] = i;
    }

    start = start_tsc();
    for (int i_tst = 0; i_tst < N_tst; i_tst++){
        start_inner = start_tsc();
        int y_tst_i_tst = y_tst[i_tst];
        for (int t = 0; t < T; t++){

            // populate value_now with zeros
            for(int i = 0; i < N; i++){
                value_now[i] = 0.0;
            }

            start_shuffle = start_tsc();
            shuffle(n_trn, N);
            cycles_shuffle += stop_tsc(start_shuffle);

            get_row(x_tst_row, x_tst, i_tst);

            // k = 0
            start_build_heap = start_tsc();
            int n_trn_0 = n_trn[0];
            heap[0] = n_trn_0;
            int y_trn_elem = y_trn[n_trn_0];
            value_now[0] = (float)((y_trn_elem == y_tst_i_tst))/K;
            cycles_build_heap += stop_tsc(start_build_heap);

            start_insert = start_tsc();


            for (int k = 1; k < K; k++){
                start_up = start_tsc();
                // INSERT HEAP -- BEGIN
                heap[k] = n_trn[k];

                // UP -- BEGIN
                parent_index = (k-1)/2;

                get_row(row, x_trn, heap[k]);
                start_l2_norm_up = start_tsc();
                norm_idx = l2norm_opt(x_tst_row, row, d);
                cycles_insert_l2_norm_up += stop_tsc(start_l2_norm_up);
                get_row(row, x_trn, heap[parent_index]);
                start_l2_norm_up = start_tsc();
                norm_parentidx = l2norm_opt(x_tst_row, row, d);
                cycles_insert_l2_norm_up += stop_tsc(start_l2_norm_up);

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
                    start_l2_norm_up = start_tsc();
                    norm_idx = l2norm_opt(x_tst_row, row, d);
                    cycles_insert_l2_norm_up += stop_tsc(start_l2_norm_up);
                    get_row(row, x_trn, heap[parent_index]);
                    start_l2_norm_up = start_tsc();
                    norm_parentidx = l2norm_opt(x_tst_row, row, d);
                    cycles_insert_l2_norm_up += stop_tsc(start_l2_norm_up);
                }
                // UP -- END
                cycles_insert_up += stop_tsc(start_up);
                for (int m = 0; m < k + 1; m++){
                    y_trn_slice[m] = y_trn[heap[m]];
                }
                start_utility = start_tsc();
                value_now[k] = unweighted_knn_utility(y_trn_slice, y_tst_i_tst, k + 1, K);
                cycles_utility += stop_tsc(start_utility);
            }



            for (int k = K; k < N; k++){
                n_trn_k = n_trn[k];
                get_row(row, x_trn, n_trn_k);
                start_l2_norm = start_tsc();
                d_elem = l2norm_opt(row, x_tst_row, d);
                cycles_insert_l2_norm += stop_tsc(start_l2_norm);
                get_row(row, x_trn, heap[0]);
                start_l2_norm = start_tsc();
                d_root = l2norm_opt(row, x_tst_row, d);
                cycles_insert_l2_norm += stop_tsc(start_l2_norm);

                if (d_elem < d_root) {
                    heap[0] = n_trn_k;
                    // DOWN -- BEGIN
                    start_down = start_tsc();
                    if (K >= 2){
                        if(K == 2){
                            tar_index = 1;
                        } else{
                            get_row(row, x_trn, heap[1]);
                            start_l2_norm_down = start_tsc();
                            norm1 = l2norm_opt(x_tst_row, row, d);
                            cycles_insert_l2_norm_down += stop_tsc(start_l2_norm_down);
                            start_l2_norm_down = start_tsc();
                            get_row(row, x_trn, heap[2]);
                            norm2 = l2norm_opt(x_tst_row, row, d);
                            cycles_insert_l2_norm_down += stop_tsc(start_l2_norm_down);
                            if (norm1 < norm2) {
                                tar_index = 2;
                            } else {
                                tar_index = 1;
                            }
                        }

                        get_row(row, x_trn, heap[0]);
                        start_l2_norm_down = start_tsc();
                        norm1 = l2norm_opt(x_tst_row, row, d);
                        cycles_insert_l2_norm_down += stop_tsc(start_l2_norm_down);
                        get_row(row, x_trn, heap[tar_index]);

                        start_l2_norm_down = start_tsc();
                        norm2 = l2norm_opt(x_tst_row, row, d);
                        cycles_insert_l2_norm_down += stop_tsc(start_l2_norm_down);
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
                                start_l2_norm_down = start_tsc();
                                norm1 = l2norm_opt(x_tst_row, row, d);
                                cycles_insert_l2_norm_down += stop_tsc(start_l2_norm_down);
                                get_row(row, x_trn, heap[2*curr_index+2]);
                                start_l2_norm_down = start_tsc();
                                norm2 = l2norm_opt(x_tst_row, row, d);
                                cycles_insert_l2_norm_down += stop_tsc(start_l2_norm_down);
                                if (norm1 < norm2) {
                                    tar_index = 2*curr_index+2;
                                } else {
                                    tar_index = 2*curr_index+1;
                                }
                            }

                            get_row(row, x_trn, heap[curr_index]);
                            start_l2_norm_down = start_tsc();
                            norm1 = l2norm_opt(x_tst_row, row, d);
                            cycles_insert_l2_norm_down += stop_tsc(start_l2_norm_down);
                            get_row(row, x_trn, heap[tar_index]);
                            start_l2_norm_down = start_tsc();
                            norm2 = l2norm_opt(x_tst_row, row, d);
                            cycles_insert_l2_norm_down += stop_tsc(start_l2_norm_down);
                        }
                    }
                    // DOWN -- END
                    cycles_insert_down += stop_tsc(start_down);
                    for (int m = 0; m < K; m++){
                        y_trn_slice[m] = y_trn[heap[m]];
                    }
                    start_utility = start_tsc();
                    value_now[k] = unweighted_knn_utility(y_trn_slice, y_tst_i_tst, K, K);
                    cycles_utility += stop_tsc(start_utility);
                }
                else
                    value_now[k] = value_now[k-1];
            }
            // INSERT HEAP -- END
            cycles_insert += stop_tsc(start_insert);
            // compute the marginal contribution of the k-th user's data
            mat_set(&sp_approx_all, t, n_trn[0], value_now[0]);

            for (int l = 1; l < N; l++){
                mat_set(&sp_approx_all, t, n_trn[l], value_now[l] - value_now[l-1]);
            }
        }

        cycles_inner += stop_tsc(start_inner);
        start_col_mean = start_tsc();
        compute_col_mean(sp_approx, &sp_approx_all, i_tst);
        cycles_col_mean += stop_tsc(start_col_mean);

    }
    cycles = stop_tsc(start);
    cycles_insert_down = cycles_insert_down - cycles_insert_l2_norm_down;
    cycles_insert_up = cycles_insert_up - cycles_insert_l2_norm_up;
    cycles_insert_l2_norm += cycles_insert_l2_norm_up + cycles_insert_l2_norm_down;
    cycles_insert = cycles_insert - cycles_insert_down - cycles_insert_l2_norm - cycles_insert_up;
    cycles_inner = cycles_inner - cycles_insert_down - \
                   cycles_insert_l2_norm - cycles_insert_up -\
                   cycles_insert - cycles_build_heap - cycles_shuffle - \
                   cycles_utility;
    cycles = cycles - cycles_inner - cycles_insert_down - \
                   cycles_insert_l2_norm - cycles_insert_up -  \
                   cycles_insert - cycles_build_heap - cycles_shuffle - \
                   cycles_utility - cycles_col_mean;

    printf("%lld,", cycles);
    printf("%lld,", cycles_build_heap);
    printf("%lld,", cycles_insert);
    printf("%lld,", cycles_insert_l2_norm);
    printf("%lld,", cycles_insert_up);
    printf("%lld,", cycles_insert_down);
    printf("%lld,", cycles_shuffle);
    printf("%lld,", cycles_utility);
    printf("%lld,", cycles_col_mean);
    printf("%lld", cycles_inner);
    printf("\n");

    destroy(&sp_approx_all);
    free(heap);
    free(row);
    free(value_now);
    free(x_tst_row);
    free(n_trn);
    free(y_trn_slice);
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {

        printf("No/not enough arguments given, please input N M d K");
        return 0;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int d = atoi(argv[3]);
    int K = atoi(argv[4]);
    int T = 128;
    printf("remaining_runtime,build_heap,insert,l2-norm,insert_up,insert_down,shuffle,utility,col_mean,get_knn_inner\n");
    for (int iter = 0; iter < NUM_RUNS; ++iter){
        mat sp_approx;
        mat x_trn;
        mat x_tst;
        int* y_trn = malloc(N*sizeof(int));
        int* y_tst = malloc(M*sizeof(int));
        build(&sp_approx, M, N);
        build(&x_trn, N, d);
        build(&x_tst, M, d);
        initialize_rand_array(y_trn, N);
        initialize_rand_array(y_tst, M);
        initialize_rand_mat(&x_trn);
        initialize_rand_mat(&x_tst);

        initialize_mat(&sp_approx, 0.0);

        for(int i = 0; i < sp_approx.n1; i++){
            for(int j = 0; j < sp_approx.n2; j++){
                mat_set(&sp_approx, i, j, 0.0);
            }
        }

        srand(42); // fix seed for RNG
        knn_mc_approximation_bottlenecks(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, K, T);
        free(y_trn);
        free(y_tst);
        destroy(&sp_approx);
        destroy(&x_trn);
        destroy(&x_tst);

    }
    return 0;
}
