// #include "alg1.h"
#include "mat.h"
#include "utils.h"
#include "ksort.h"
#include "read_input.h"
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
#include <immintrin.h>
#include "quadsort.h"
// warmup iterations
#define NUM_WARMUP 100
// num. of iterations (measurements) per n
#define NUM_RUNS 30
#define pair_lt(a, b) ((a).value < (b).value)

KSORT_INIT(pair, pair_t, pair_lt)
KSORT_INIT_GENERIC(float)

// Merge two subarrays L and M into arr
void merge(float arr[], int idx[], int p, int q, int r) {

  // Create L ← A[p..q] and M ← A[q+1..r]
  int n1 = q - p + 1;
  int n2 = r - q;

  float L[n1], M[n2];
  int L_idx[n1], M_idx[n2];

  for (int i = 0; i < n1; i++) {
    L[i] = arr[p + i];
    L_idx[i] = idx[p + i];
  }
  for (int j = 0; j < n2; j++) {
    M[j] = arr[q + 1 + j];
    M_idx[j] = idx[q + 1 + j];
  }
  // Maintain current index of sub-arrays and main array
  int i, j, k;
  i = 0;
  j = 0;
  k = p;

  // Until we reach either end of either L or M, pick larger among
  // elements L and M and place them in the correct position at A[p..r]
  while (i < n1 && j < n2) {
    if (L[i] <= M[j]) {
      arr[k] = L[i];
      idx[k] = L_idx[i];
      i++;
    } else {
      arr[k] = M[j];
      idx[k] = M_idx[j];
      j++;
    }
    k++;
  }

  // When we run out of elements in either L or M,
  // pick up the remaining elements and put in A[p..r]
  while (i < n1) {
    arr[k] = L[i];
    idx[k] = L_idx[i];
    i++;
    k++;
  }

  while (j < n2) {
    arr[k] = M[j];
    idx[k] = M_idx[j];
    j++;
    k++;
  }
}

// Divide the array into two subarrays, sort them and merge them
void mergeSort(float arr[], int idx[], int l, int r) {
  if (l < r) {

    // m is the point where the array is divided into two subarrays
    int m = l + (r - l) / 2;

    mergeSort(arr, idx, l, m);
    mergeSort(arr, idx, m + 1, r);

    // Merge the sorted subarrays
    merge(arr, idx, l, m, r);
  }
}


// returns array of sorted indices according to distance
void argsort(int idx[], float arr[], int len) {
    float tmp[len];
    memcpy(tmp, arr, len * sizeof(float));
    // initialize array of indices
    for (int i = 0; i < len; i++) {
        idx[i] = i;
    }
    mergeSort(tmp, idx, 0, len - 1);
}

//// get knn, returns mat of sorted data entries (knn alg)
//void get_true_knn_bottlenecks(int_mat *x_tst_knn_gt, mat *x_trn, mat *x_tst){
//    int N = x_trn->n1;
//    int N_tst = x_tst->n1;
//    int d = x_tst->n2;
//    myInt64 start_l2norm, start_argsort, start, cycles_l2norm, cycles_argsort, cycles;
//    cycles_l2norm = cycles_argsort = cycles = 0;
//    float* data_trn = x_trn->data;
//    float* data_tst = x_tst->data;
//
//
//    start = start_tsc();
//    for (int i_tst = 0; i_tst < N_tst; i_tst++){
//
//        for (int i_trn = 0; i_trn < N; i_trn++){
//
//            float trn_row[d];
//            float tst_row[d];
//            for (int j = 0; j < d/4; j+=4) {
//                trn_row[j] = data_trn[i_trn * d + j];
//                trn_row[j+1] = data_trn[i_trn * d + j + 1];
//                trn_row[j+2] = data_trn[i_trn * d + j + 2];
//                trn_row[j+3] = data_trn[i_trn * d + j + 3];
//                tst_row[j] = data_tst[i_tst * d + j];
//                tst_row[j + 1] = data_tst[i_tst * d + j + 1];
//                tst_row[j + 2] = data_tst[i_tst * d + j + 2];
//                tst_row[j + 3] = data_tst[i_tst * d + j + 3];
//            }
//
//            distances[i_trn].index = i_trn;
//            start_l2norm = start_tsc();
//            distances[i_trn].value = l2norm(trn_row, tst_row, d);
//            cycles_l2norm += stop_tsc(start_l2norm);
//        }
//        start_argsort = start_tsc();
//        ks_mergesort(pair, N, distances, 0);
//        cycles_argsort += stop_tsc(start_argsort);
//        for (int k = 0; k < N; k++) {
//            x_tst_knn_gt->data[i_tst * N + k] = distances[k].index;
//        }
//
//    }
//    cycles = stop_tsc(start);
//    cycles = cycles - cycles_argsort - cycles_l2norm;
//    printf("%lld,", cycles);
//    printf("%lld,", cycles_argsort);
//    printf("%lld,", cycles_l2norm);
//
//    free(distances);
//}

void compute_single_unweighted_knn_class_shapley(mat* sp_gt, int* y_trn, int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmp0, tmp1, tmp2, tmp3;
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;

    for (int j=0; j < N_tst;j++){
        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        tmp0 = (y_trn[x_tst_knn_gt_j_last_i] == y_tst[j]) ? 1.0/N : 0.0;
        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmp0);
        for (int i=N-2; i>-1; i--){
            x_tst_knn_gt_j_i = int_mat_get(x_tst_knn_gt, j, i);
            x_tst_knn_gt_j_i_plus_1 = int_mat_get(x_tst_knn_gt, j, \
             i+1);
            tmp0 = (y_trn[x_tst_knn_gt_j_i] == y_tst[j]) ? 1.0:0.0;
            tmp1 = (y_trn[x_tst_knn_gt_j_i_plus_1] == y_tst[j]) ? 1.0:0.0;
            tmp2 = (K > i+1) ? i+1 : K;
            tmp3 = mat_get(sp_gt, j, x_tst_knn_gt_j_i_plus_1) + \
                    (tmp0 - tmp1) / K * tmp2 / (i + 1);
            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3);
        }
    }
}

// get knn, returns mat of sorted data entries (knn alg)
void get_true_knn_bottlenecks_base(int_mat *x_tst_knn_gt, mat *x_trn, mat *x_tst){
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;
    myInt64 start_l2norm, start_argsort, start, cycles_l2norm, cycles_argsort, cycles;
    cycles_l2norm = cycles_argsort = cycles = 0;
    float dist_gt[N];
    int idx_arr[N];
    float* data_trn = x_trn->data;
    float* data_tst = x_tst->data;


    start = start_tsc();
    for (int i_tst = 0; i_tst < N_tst; i_tst++){

        for (int i_trn = 0; i_trn < N; i_trn++){

            float trn_row[d];
            float tst_row[d];
            for (int j = 0; j < d; j++) {
                trn_row[j] = data_trn[i_trn * d + j];
                tst_row[j] = data_tst[i_tst * d + j];
            }

            idx_arr[i_trn] = i_trn;
            start_l2norm = start_tsc();
            dist_gt[i_trn] = l2norm(trn_row, tst_row, d);
            cycles_l2norm += stop_tsc(start_l2norm);
        }
        start_argsort = start_tsc();
         argsort(idx_arr, dist_gt, N);
        cycles_argsort += stop_tsc(start_argsort);
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = idx_arr[k];

        }

    }
    cycles = stop_tsc(start);
    cycles = cycles - cycles_argsort - cycles_l2norm;
    printf("%lld,", cycles);
    printf("%lld,", cycles_argsort);
    printf("%lld,", cycles_l2norm);
}


int main(int argc, char** argv) {
    if (argc != 5){

      printf("No/not enough arguments given, please input N M d K");
      return 0;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int d = atoi(argv[3]);
    int K = atoi(argv[4]);


    printf("remaining_runtime,argsort,l2-norm,compute_shapley\n");

    for(int iter = 0; iter < NUM_RUNS; ++iter ) {
        int_mat x_tst_knn_gt;
        mat sp_gt;
        mat x_trn;
        mat x_tst;
        int* y_trn = malloc(N*sizeof(int));
        int* y_tst = malloc(M*sizeof(int));

        build(&x_trn, N, d);
        build(&x_tst, M, d);
        build(&sp_gt, M, N);
        build_int_mat(&x_tst_knn_gt, N, N);

        myInt64 start, cycles;
        read_input_features(x_trn.data, PATH_FEATURE_TRAIN, N, d);
        read_input_labels(y_trn, PATH_LABEL_TRAIN, N, 1);
        read_input_features(x_tst.data, PATH_FEATURE_TEST, M, d);
        read_input_labels(y_tst, PATH_LABEL_TEST, M, 1);

        for(int i = 0; i < sp_gt.n1; i++){
            for(int j = 0; j < sp_gt.n2; j++){
                mat_set(&sp_gt, i, j, 0.0);
            }
        }

        get_true_knn_bottlenecks_base(&x_tst_knn_gt, &x_trn, &x_tst);
        start = start_tsc();
        compute_single_unweighted_knn_class_shapley(&sp_gt, y_trn, y_tst, &x_tst_knn_gt, K);
        cycles = stop_tsc(start);
        printf("%lld", cycles);
        printf("\n");
        destroy(&x_trn);
        destroy(&x_tst);
        destroy(&sp_gt);
        destroy_int_mat(&x_tst_knn_gt);
        free(y_trn);
        free(y_tst);
    }

    return 0;
}
