//
// Created by lucasck on 03/06/22.
//
//#include "alg2.h"
#include "mat.h"
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "tsc_x86.h"
// warmup iterations
#define NUM_WARMUP 100
// num. of iterations (measurements) per n
#define NUM_RUNS 30


typedef struct {
    myInt64 cycles_l2_norm, cycles_l2_norm_up, cycles_l2_norm_down, cycles_up, cycles_down;
} cycles_insert_struct;

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

myInt64 up_bottlenecks(maxheap* h, int index, myInt64 start, myInt64 end) {
    if (index == 0) {
        return end;
    }
    int parent_index = (index-1)/2;

    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    get_row(row, h->x_trn, h->heap[index]);
    start = start_tsc();
    float norm_idx = l2norm_opt(h->x_tst, row, h->x_trn->n2);
    end += stop_tsc(start);
    get_row(row, h->x_trn, h->heap[parent_index]);
    start = start_tsc();
    float norm_parentidx = l2norm_opt(h->x_tst, row, h->x_trn->n2);
    end += stop_tsc(start);

    free(row);
    if (norm_idx > norm_parentidx) {
        int temp = h->heap[index];
        h->heap[index] = h->heap[parent_index];
        h->heap[parent_index] = temp;
        up_bottlenecks(h, parent_index, start, end);
    }

    return end;
}

myInt64 down_bottlenecks(maxheap* h, int index, myInt64 start, myInt64 end) {

    int tar_index;
    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    if (2*index + 1 > h->counter){
        free(row);
        return end;
    }
    if (2*index + 1 < h->counter) {
        get_row(row, h->x_trn, h->heap[2*index+1]);
        start = start_tsc();
        float norm1 = l2norm_opt(h->x_tst, row, h->x_trn->n2);
        end += stop_tsc(start);
        get_row(row, h->x_trn, h->heap[2*index+2]);
        start = start_tsc();
        float norm2 = l2norm_opt(h->x_tst, row, h->x_trn->n2);
        end += stop_tsc(start);
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
    start = start_tsc();
    float norm1 = l2norm_opt(h->x_tst, row, h->x_trn->n2);
    end += stop_tsc(start);
    get_row(row, h->x_trn, h->heap[tar_index]);
    start = start_tsc();
    float norm2 = l2norm_opt(h->x_tst, row, h->x_trn->n2);
    end += stop_tsc(start);

    if (norm1 < norm2) {
        int temp = h->heap[index];
        h->heap[index] = h->heap[tar_index];
        h->heap[tar_index] = temp;
        down_bottlenecks(h, tar_index, start, end);
    }
    free(row);
    return end;
}

// elem is index of permuted matrix from main
cycles_insert_struct measure_insert(maxheap* h, int elem) {
    cycles_insert_struct cycles_insert;
    cycles_insert.cycles_l2_norm = cycles_insert.cycles_l2_norm_down = cycles_insert.cycles_l2_norm_up = cycles_insert.cycles_down = cycles_insert.cycles_up = 0;
    myInt64 start_l2_norm, start_down, start_up;

    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    get_row(row, h->x_trn, elem);

    start_l2_norm = start_tsc();
    float d_elem = l2norm_opt(row, h->x_tst, h->x_trn->n2);
    cycles_insert.cycles_l2_norm += stop_tsc(start_l2_norm);

    if (h->counter <= (h->K - 2)) {
        h->heap[h->current_sz] = elem;
        h->current_sz += 1;
        h->counter += 1;

        start_up = start_tsc();
        cycles_insert.cycles_l2_norm_up += up_bottlenecks(h, h->counter, start_up, (myInt64) 0);
        cycles_insert.cycles_up += stop_tsc(start_up);
        h->changed = 1;
    }
    else {
        get_row(row,h->x_trn, h->heap[0]);

        start_l2_norm = start_tsc();
        float d_root = l2norm_opt(row, h->x_tst, h->x_trn->n2);
        cycles_insert.cycles_l2_norm = stop_tsc(start_l2_norm);

        if (d_elem < d_root) {
            h->heap[0] = elem;

            start_down = start_tsc();
            cycles_insert.cycles_l2_norm_down += down_bottlenecks(h, 0, start_down, (myInt64) 0);
            cycles_insert.cycles_down += stop_tsc(start_down);

            h->changed = 1;
        }
        else
            h->changed = 0;
    }
    free(row);
    return cycles_insert;
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
void knn_mc_approximation_bottlenecks(mat* sp_approx, mat* x_trn, int *y_trn, mat* x_tst, int* y_tst, int K, int T){

    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;

    maxheap heap;
    mat sp_approx_all;
    cycles_insert_struct res;

    int* n_trn = (int *) malloc(N * sizeof(int));
    float* value_now = (float *) malloc(N * sizeof(float));
    float* x_tst_row = (float *) malloc(d * sizeof(float));


    build(&sp_approx_all, T, N);

    myInt64 start_shuffle, start_build_heap, start_inner, start, start_insert, start_col_mean, start_utility;
    myInt64 cycles_shuffle, cycles_build_heap, cycles_inner, cycles, cycles_insert, cycles_insert_l2_norm, cycles_insert_l2_norm_up, cycles_insert_l2_norm_down, cycles_insert_up, cycles_insert_down, cycles_col_mean, cycles_utility;
    cycles_shuffle = cycles_build_heap = cycles_inner = cycles = cycles_insert = cycles_col_mean = cycles_utility = cycles_insert_l2_norm = cycles_insert_l2_norm_down = cycles_insert_l2_norm_up = cycles_insert_up = cycles_insert_down = 0;


    // populate n_trn
    for(int i = 0; i < N; i++){
        n_trn[i] = i;
    }

    start = start_tsc();
    for (int i_tst = 0; i_tst < N_tst; i_tst++){
        start_inner = start_tsc();
        for (int t = 0; t < T; t++){

            // populate value_now with zeros
            for(int i = 0; i < N; i++){
                value_now[i] = 0.0;
            }

            start_shuffle = start_tsc();
            shuffle(n_trn, N);
            cycles_shuffle += stop_tsc(start_shuffle);

            get_row(x_tst_row, x_tst, i_tst);

            start_build_heap = start_tsc();
            build_heap(&heap, K, x_trn, x_tst_row);
            cycles_build_heap += stop_tsc(start_build_heap);

            for (int k = 0; k < N; k++){
                start_insert = start_tsc();
                res = measure_insert(&heap, n_trn[k]);
                cycles_insert += stop_tsc(start_insert);
                cycles_insert_l2_norm += res.cycles_l2_norm;
                cycles_insert_l2_norm_up += res.cycles_l2_norm_up;
                cycles_insert_l2_norm_down += res.cycles_l2_norm_down;
                cycles_insert_up += res.cycles_up;
                cycles_insert_down += res.cycles_down;

                if (heap.changed){
                    int* y_trn_slice = (int *) malloc((heap.counter + 1) * sizeof(int));

                    for (int m = 0; m < heap.counter + 1; m++){
                        y_trn_slice[m] = y_trn[heap.heap[m]];
                    }
                    start_utility = start_tsc();
                    value_now[k] = unweighted_knn_utility(y_trn_slice, y_tst[i_tst], heap.counter + 1, K);
                    cycles_utility += stop_tsc(start_utility);
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
        cycles_inner += stop_tsc(start_inner);

        start_col_mean = start_tsc();
        compute_col_mean(sp_approx, &sp_approx_all, i_tst);
        cycles_col_mean += stop_tsc(start_col_mean);
    }
    cycles = stop_tsc(start);
    cycles_insert_down -= cycles_insert_l2_norm_down;
    cycles_insert_up -= cycles_insert_l2_norm_up;
    cycles_insert_l2_norm += cycles_insert_l2_norm_up + cycles_insert_l2_norm_down;
    cycles_insert = cycles_insert - cycles_insert_down - cycles_insert_l2_norm - cycles_insert_up;
    cycles_inner = cycles_inner - cycles_insert_down - \
                   cycles_insert_l2_norm - cycles_insert_up - \
                   cycles_insert - cycles_build_heap - cycles_shuffle - \
                   cycles_utility;
    cycles = cycles - cycles_inner - cycles_insert_down - \
                   cycles_insert_l2_norm - cycles_insert_up - \
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
    free(value_now);
    free(x_tst_row);
    free(n_trn);
    destroy(&sp_approx_all);
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