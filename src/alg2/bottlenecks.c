#include "alg2.h"
#include "read_input.h"
// #include "heap_bottlenecks.h"
// #ifndef T 130
#define PATH "../data/"
#define FEATURE_TRAIN "features_training.npy"
#define LABEL_TRAIN "labels_training.npy"
#define FEATURE_TEST "features_testing.npy"
#define LABEL_TEST "labels_testing.npy"
#define PATH_FEATURE_TRAIN PATH FEATURE_TRAIN
#define PATH_FEATURE_TEST PATH FEATURE_TEST
#define PATH_LABEL_TRAIN PATH LABEL_TRAIN
#define PATH_LABEL_TEST PATH LABEL_TEST
#include "tsc_x86.h"
// warmup iterations
#define NUM_WARMUP 100
// num. of iterations (measurements) per n
#define NUM_RUNS 10


typedef struct {
    myInt64 cycles_l2norm, cycles_up, cycles_down;
} cycles_insert_struct;



myInt64 up_bottleneck(maxheap* h, int index, myInt64 start, myInt64 end) {
    if (index == 0) {
        return end;
    }
    int parent_index = (index-1)/2;

    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    get_row(row, h->x_trn, h->heap[index]);

    start = start_tsc();
    float norm_idx = l2norm(h->x_tst, row, h->x_trn->n2);
    end += stop_tsc(start);
    get_row(row, h->x_trn, h->heap[parent_index]);

    start = start_tsc();
    float norm_parentidx = l2norm(h->x_tst, row, h->x_trn->n2);
    end += stop_tsc(start);

    free(row);
    if (norm_idx > norm_parentidx) {
        int temp = h->heap[index];
        h->heap[index] = h->heap[parent_index];
        h->heap[parent_index] = temp;
        up_bottleneck(h, parent_index, start, end);
    }
    return end;
}

myInt64 down_bottleneck(maxheap* h, int index, myInt64 start, myInt64 end) {
    int tar_index;
    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    if (2*index + 1 > h->counter){
        free(row);
        return end;
    }
    if (2*index + 1 < h->counter) {
        get_row(row, h->x_trn, h->heap[2*index+1]);
        start = start_tsc();
        float norm1 = l2norm(h->x_tst, row, h->x_trn->n2);
        end += stop_tsc(start);

        get_row(row, h->x_trn, h->heap[2*index+2]);
        start = start_tsc();
        float norm2 = l2norm(h->x_tst, row, h->x_trn->n2);
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
    float norm1 = l2norm(h->x_tst, row, h->x_trn->n2);
    end = stop_tsc(start);

    get_row(row, h->x_trn, h->heap[tar_index]);
    start = start_tsc();
    float norm2 = l2norm(h->x_tst, row, h->x_trn->n2);
    end += stop_tsc(start);
    
    if (norm1 < norm2) {
        int temp = h->heap[index];
        h->heap[index] = h->heap[tar_index];
        h->heap[tar_index] = temp;
        down_bottleneck(h, tar_index, start, end);
    }
    free(row);
    return end;
}


// elem is index of permuted matrix from main
cycles_insert_struct measure_insert(maxheap* h, int elem) {
    cycles_insert_struct cycles_insert;
    cycles_insert.cycles_l2norm = cycles_insert.cycles_down = cycles_insert.cycles_up = 0;
    myInt64 start_l2norm, start_down, start_up;

    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    get_row(row, h->x_trn, elem);

    start_l2norm = start_tsc();
    float d_elem = l2norm(row, h->x_tst, h->x_trn->n2);
    cycles_insert.cycles_l2norm += stop_tsc(start_l2norm);

    if (h->counter <= (h->K - 2)) {
        h->heap[h->current_sz] = elem;
        h->current_sz += 1;
        h->counter += 1;

        start_up = start_tsc();
        cycles_insert.cycles_l2norm += up_bottleneck(h, h->counter, start_up, (myInt64) 0);
        cycles_insert.cycles_up += stop_tsc(start_up);
        h->changed = 1;
    }
    else {
        get_row(row,h->x_trn, h->heap[0]);

        start_l2norm = start_tsc();
        float d_root = l2norm(row, h->x_tst, h->x_trn->n2);
        cycles_insert.cycles_l2norm = stop_tsc(start_l2norm);

        if (d_elem < d_root) {
            h->heap[0] = elem;

            start_down = start_tsc();
            cycles_insert.cycles_l2norm += down_bottleneck(h, 0, start_down, (myInt64) 0); 
            cycles_insert.cycles_down += stop_tsc(start_down);

            h->changed = 1;
        }
        else
            h->changed = 0;
    }
    free(row);
    return cycles_insert;
}



// get knn, returns mat of sorted data entries (knn alg)
void knn_mc_approximation_bottlenecks(mat *sp_approx, mat *x_trn, int *y_trn, mat *x_tst, int *y_tst, int K, int T)
{

    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;

    tensor sp_approx_all;
    maxheap heap;
    mat tensor_slice;
    cycles_insert_struct res;

    int *n_trn = (int *)malloc(N * sizeof(int));
    float *value_now = (float *) malloc(N * sizeof(float));
    float *x_tst_row = (float *) malloc(d * sizeof(float));

    build_tensor(&sp_approx_all, N_tst, T, N);
    build(&tensor_slice, T, N);

    myInt64 start_shuffle, start_build_heap, start_inner, start, start_insert, start_col_mean, start_utility;
    myInt64 cycles_shuffle, cycles_build_heap, cycles_inner, cycles, cycles_insert, cycles_insert_l2norm, cycles_insert_up, cycles_insert_down, cycles_col_mean, cycles_utility;
    cycles_shuffle = cycles_build_heap = cycles_inner = cycles = cycles_insert = cycles_col_mean = cycles_utility = cycles_insert_l2norm = cycles_insert_up = cycles_insert_down = 0;



    // populate n_trn
    for (int i = 0; i < N; i++){
        n_trn[i] = i;
    }

    start = start_tsc();
    for (int i_tst = 0; i_tst < N_tst; i_tst++){
        start_inner = start_tsc();
        for (int t = 0; t < T; t++){

            // populate value_now with zeros
            for (int i = 0; i < N; i++){
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
                cycles_insert_l2norm += res.cycles_l2norm;
                cycles_insert_up += res.cycles_up;
                cycles_insert_down += res.cycles_down;

                if (heap.changed){
                    int *y_trn_slice = (int *)malloc((heap.counter + 1) * sizeof(int));

                    for (int m = 0; m < heap.counter + 1; m++){
                        y_trn_slice[m] = y_trn[heap.heap[m]];
                    }

                    start_utility = start_tsc();
                    value_now[k] = unweighted_knn_utility(y_trn_slice, y_tst[i_tst], heap.counter + 1, K);
                    cycles_utility += stop_tsc(start_utility);

                    free(y_trn_slice);
                }
                else{
                    value_now[k] = value_now[k - 1];
                }
            }
            
            // compute the marginal contribution of the k-th user's data
            tensor_set(&sp_approx_all, i_tst, t, n_trn[0], value_now[0]);
            for (int l = 1; l < N; l++){
                tensor_set(&sp_approx_all, i_tst, t, n_trn[l], value_now[l] - value_now[l - 1]);
            }
            nuke_heap(&heap);
        }
        cycles_inner += stop_tsc(start_inner);

        start_col_mean = start_tsc();
        get_mat_from_tensor(&sp_approx_all, &tensor_slice, i_tst);
        compute_col_mean(sp_approx, &tensor_slice, i_tst);
        cycles_col_mean += stop_tsc(start_col_mean);

    }

    cycles = stop_tsc(start);
    cycles_insert = cycles_insert - cycles_insert_down - cycles_insert_l2norm - cycles_insert_up;
    cycles_inner = cycles_inner - cycles_insert_down - \
                   cycles_insert_l2norm - cycles_insert_up - \
                   cycles_insert - cycles_build_heap - cycles_shuffle - \
                   cycles_utility;
    cycles = cycles - cycles_inner - cycles_insert_down - \
                   cycles_insert_l2norm - cycles_insert_up - \
                   cycles_insert - cycles_build_heap - cycles_shuffle - \
                   cycles_utility - cycles_col_mean;

    printf("%lld,", cycles);
    printf("%lld,", cycles_build_heap);
    printf("%lld,", cycles_insert);
    printf("%lld,", cycles_insert_l2norm);
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
    destroy_tensor(&sp_approx_all);
    destroy(&tensor_slice);
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
 
    printf("remaining_runtime,build_heap,insert,l2-norm,insert_up,insert_down,shuffle,utility,col_mean,get_knn_inner\n");
    for (int iter = 0; iter < NUM_RUNS; ++iter){
        mat sp_approx;
        mat x_trn;
        mat x_tst;
        int* y_trn = malloc(N*sizeof(int));
        int* y_tst = malloc(M*sizeof(int));

        build(&x_trn, N, d);
        build(&x_tst, M, d);
        build(&sp_approx, M, N);
        read_input_features(x_trn.data, PATH_FEATURE_TRAIN, N, d);
        read_input_labels(y_trn, PATH_LABEL_TRAIN, N, 1);
        read_input_features(x_tst.data, PATH_FEATURE_TEST, M, d);
        read_input_labels(y_tst, PATH_LABEL_TEST, M, 1);

        for(int i = 0; i < sp_approx.n1; i++){
            for(int j = 0; j < sp_approx.n2; j++){
                mat_set(&sp_approx, i, j, 0.0);
            }
        }

        srand(42); // fix seed for RNG
        knn_mc_approximation_bottlenecks(&sp_approx, &x_trn, y_trn, &x_tst, y_tst, K, 130);
        free(y_trn);
        free(y_tst);
        destroy(&sp_approx);
        destroy(&x_trn);
        destroy(&x_tst);
        
    }
    return 0;
}
