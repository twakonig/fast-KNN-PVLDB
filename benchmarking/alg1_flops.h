#ifndef ALG1_ALG1_FLOPS_H
#define ALG1_ALG1_FLOPS_H
#include <stdio.h>
//#include "mat.h"
#include <math.h>
#include <string.h>
//#include "utils.h"

#include "../include/mat.h"
#include "../include/utils.h"


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

// get knn, returns mat of sorted data entries (knn alg)
void get_true_knn(int_mat *x_tst_knn_gt, mat *x_trn, mat *x_tst, long long unsigned int *flops){
    int N = x_trn->n1;
    int N_tst = x_tst->n1;
    int d = x_tst->n2;

    float* data_trn = x_trn->data;
    float* data_tst = x_tst->data;

    for (int i_tst = 0; i_tst < N_tst; i_tst++){
        float dist_gt[N];
        int idx_arr[N];

        for (int i_trn = 0; i_trn < N; i_trn++){
            float trn_row[d];
            float tst_row[d];
            for (int j = 0; j < d; j++) {
                trn_row[j] = data_trn[i_trn * d + j];
                tst_row[j] = data_tst[i_tst * d + j];
            }
            dist_gt[i_trn] = l2norm(trn_row, tst_row, d);
            *flops += 2*d+1; //Add 2n +1 flops
        }
        argsort(idx_arr, dist_gt, N);
        for (int k = 0; k < N; k++) {
            x_tst_knn_gt->data[i_tst * N + k] = idx_arr[k];
        }
    }
}

void compute_single_unweighted_knn_class_shapley(mat* sp_gt, int* y_trn, int* y_tst, int_mat* x_tst_knn_gt, int K, long long unsigned int *flops) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmp0, tmp1, tmp2, tmp3;
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;

    for (int j=0; j < N_tst;j++){
        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        tmp0 = (y_trn[x_tst_knn_gt_j_last_i] == y_tst[j]) ? 1.0/N : 0.0;
        if (y_trn[x_tst_knn_gt_j_last_i] == y_tst[j]) {
            *flops += 1; // update flop count
        }
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
            *flops += 6; // update flop count
            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3);
        }
    }
}



#endif //ALG1_ALG1_H
