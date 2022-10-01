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
        j = (myInt64) rand() * (myInt64) (i + 1) >> 32;
        tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
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
void knn_mc_approximation_bottlenecks(mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T){

    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    float sum;
    mat sp_approx_all;

    int* n_trn = (int *) malloc(N * sizeof(int));
    float* value_now = (float *) malloc(N * sizeof(float));
    float* x_tst_row = (float *) malloc(d * sizeof(float));
    float* row = (float*) malloc(d * sizeof(float));
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

                start_utility = start_tsc();
                sum = 0.0;
                for (int i = 0; i < k+1; i++){
                    sum += (y_trn[heap[i]] == y_tst_i_tst);
                }
                value_now[k] = sum / K;
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

                    start_utility = start_tsc();
                    sum = 0.0;
                    for (int i = 0; i < K; i++){
                        sum += (y_trn[heap[i]] == y_tst_i_tst);
                    }
                    value_now[k] = sum / K;
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