#pragma once
#include "../include/utils.h"
#include "../include/tsc_x86.h"
#include "common.h"

// 1st function registered must be base in terms of cycles
// 2nd function registered must be base in terms of correctness

// base implementation, not optimized
// BASE CYCLES
void compute_shapley_base(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmp0, tmp1, tmp2, tmp3;
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;

    for (int j=0; j < N_tst;j++){
        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        tmp0 = (y_trn[x_tst_knn_gt_j_last_i] == y_tst[j]) ? 1.0/N : 0.0;
        //printf("%lf\n", tmp0);
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


// y_tst[j] only depends on outer loop: intermediate storage as y_tst_j
void compute_shapley_opt1(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmp0, tmp1, tmp2, tmp3;
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;
    int y_tst_j;

    for (int j=0; j < N_tst;j++){
        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        // scalar replacement
        y_tst_j = y_tst[j];
        tmp0 = (y_trn[x_tst_knn_gt_j_last_i] == y_tst_j) ? 1.0/N : 0.0;
        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmp0);

        for (int i=N-2; i>-1; i--){
            x_tst_knn_gt_j_i = int_mat_get(x_tst_knn_gt, j, i);
            x_tst_knn_gt_j_i_plus_1 = int_mat_get(x_tst_knn_gt, j, \
             i+1);
            tmp0 = (y_trn[x_tst_knn_gt_j_i] == y_tst_j) ? 1.0:0.0;
            tmp1 = (y_trn[x_tst_knn_gt_j_i_plus_1] == y_tst_j) ? 1.0:0.0;
            tmp2 = (K > i+1) ? i+1 : K;
            tmp3 = mat_get(sp_gt, j, x_tst_knn_gt_j_i_plus_1) + \
                    (tmp0 - tmp1) / K * tmp2 / (i + 1);
            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3);
        }
    }
}


// get rid of unneccesary divisions
void compute_shapley_opt2(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmp, tmp0, tmp1, tmp2, tmp3;
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;
    int y_tst_j;
    float N_inv = 1.0/N;
    float K_inv = 1.0/K;

    for (int j=0; j < N_tst;j++){
        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        // scalar replacement
        y_tst_j = y_tst[j];
        tmp = (y_trn[x_tst_knn_gt_j_last_i] == y_tst_j) ? N_inv:0.0;
        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmp);

        for (int i=N-2; i>-1; i--){
            x_tst_knn_gt_j_i = int_mat_get(x_tst_knn_gt, j, i);
            x_tst_knn_gt_j_i_plus_1 = int_mat_get(x_tst_knn_gt, j, \
             i+1);
            tmp0 = (y_trn[x_tst_knn_gt_j_i] == y_tst_j) ? 1.0:0.0;
            tmp1 = (y_trn[x_tst_knn_gt_j_i_plus_1] == y_tst_j) ? 1.0:0.0;
            tmp2 = (K > i+1) ? i+1 : K;
            tmp3 = mat_get(sp_gt, j, x_tst_knn_gt_j_i_plus_1) + \
                    (tmp0 - tmp1) * K_inv * tmp2 / (i + 1);
            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3);
        }
    }
}

// change if clause to !=, because == case occurs very rarely
void compute_shapley_opt3(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmp, tmp0, tmp1, tmp2, tmp3;
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;
    int y_tst_j;
    float N_inv = 1.0/N;
    float K_inv = 1.0/K;

    for (int j=0; j < N_tst;j++){
        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        // scalar replacement
        y_tst_j = y_tst[j];
        tmp = (y_trn[x_tst_knn_gt_j_last_i] != y_tst_j) ? 0.0:N_inv;
        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmp);

        for (int i=N-2; i>-1; i--){
            x_tst_knn_gt_j_i = int_mat_get(x_tst_knn_gt, j, i);
            x_tst_knn_gt_j_i_plus_1 = int_mat_get(x_tst_knn_gt, j, \
             i+1);
            tmp0 = (y_trn[x_tst_knn_gt_j_i] != y_tst_j) ? 0.0:1.0;
            tmp1 = (y_trn[x_tst_knn_gt_j_i_plus_1] != y_tst_j) ? 0.0:1.0;
            tmp2 = (K > i+1) ? i+1 : K;
            tmp3 = mat_get(sp_gt, j, x_tst_knn_gt_j_i_plus_1) + \
                    (tmp0 - tmp1) * K_inv * tmp2 / (i + 1);
            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3);
        }
    }
}

// pre-load row in outer loop
// load only one element of row per inner loop iteration (x_tst_knn_gt_i)
void compute_shapley_opt4(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmp, tmp0, tmp1, tmp2, tmp3;
    int* row_j = malloc(N_tst*sizeof(int));
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;
    int y_tst_j;
    float N_inv = 1.0/N;
    float K_inv = 1.0/K;

    for (int j=0; j < N_tst;j++){
        y_tst_j = y_tst[j];
        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        tmp = (y_trn[x_tst_knn_gt_j_last_i] != y_tst_j) ? 0.0:N_inv;
        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmp);
        // preload row(j) (N_tst elems)
        get_int_row(row_j, x_tst_knn_gt, j);
        x_tst_knn_gt_j_i_plus_1 = row_j[N-1];

        for (int i=N-2; i>-1; i--){
            x_tst_knn_gt_j_i = row_j[i];
            tmp0 = (y_trn[x_tst_knn_gt_j_i] != y_tst_j) ? 0.0:1.0;
            tmp1 = (y_trn[x_tst_knn_gt_j_i_plus_1] != y_tst_j) ? 0.0:1.0;
            tmp2 = (K > i+1) ? i+1 : K;
            tmp3 = mat_get(sp_gt, j, x_tst_knn_gt_j_i_plus_1) + (tmp0 - tmp1) * K_inv * tmp2 / (i + 1);
            x_tst_knn_gt_j_i_plus_1 = x_tst_knn_gt_j_i;
            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3);

        }
    }
}


// unroll j by factor 2 (operate on 2 rows)
void compute_shapley_opt5(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmpj, tmpjj, tmp0j, tmp1j, tmp2j, tmp3j, tmp0jj, tmp1jj, tmp2jj, tmp3jj;
    int* row_j = malloc(N_tst*sizeof(int));
    int* row_jj = malloc(N_tst*sizeof(int));
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;
    int x_tst_knn_gt_jj_i, x_tst_knn_gt_jj_i_plus_1, x_tst_knn_gt_jj_last_i;
    int y_tst_j, y_tst_jj;
    float N_inv = 1.0/N;
    float K_inv = 1.0/K;

    for (int j=0; j < N_tst;j+=2){
        y_tst_j = y_tst[j];
        y_tst_jj = y_tst[j+1];

        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        x_tst_knn_gt_jj_last_i = int_mat_get(x_tst_knn_gt, j+1, N-1);

        tmpj = (y_trn[x_tst_knn_gt_j_last_i] != y_tst_j) ? 0.0:N_inv;
        tmpjj = (y_trn[x_tst_knn_gt_jj_last_i] != y_tst_jj) ? 0.0:N_inv;

        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmpj);
        mat_set(sp_gt, j+1, x_tst_knn_gt_jj_last_i, tmpjj);

        get_int_row(row_j, x_tst_knn_gt, j);
        get_int_row(row_jj, x_tst_knn_gt, j+1);
        x_tst_knn_gt_j_i_plus_1 = row_j[N-1];
        x_tst_knn_gt_jj_i_plus_1 = row_jj[N-1];

        for (int i=N-2; i>-1; i--){

            x_tst_knn_gt_j_i = row_j[i];
            x_tst_knn_gt_jj_i = row_jj[i];

            tmp0j = (y_trn[x_tst_knn_gt_j_i] != y_tst_j) ? 0.0:1.0;
            tmp1j = (y_trn[x_tst_knn_gt_j_i_plus_1] != y_tst_j) ? 0.0:1.0;
            tmp2j = (K > i+1) ? i+1 : K;
            tmp3j = mat_get(sp_gt, j, x_tst_knn_gt_j_i_plus_1) + (tmp0j - tmp1j) * K_inv * tmp2j / (i + 1);

            tmp0jj = (y_trn[x_tst_knn_gt_jj_i] != y_tst_jj) ? 0.0:1.0;
            tmp1jj = (y_trn[x_tst_knn_gt_jj_i_plus_1] != y_tst_jj) ? 0.0:1.0;
            tmp2jj = (K > i+1) ? i+1 : K;
            tmp3jj = mat_get(sp_gt, j+1, x_tst_knn_gt_jj_i_plus_1) + (tmp0jj - tmp1jj) * K_inv * tmp2jj / (i + 1);

            x_tst_knn_gt_j_i_plus_1 = x_tst_knn_gt_j_i;
            x_tst_knn_gt_jj_i_plus_1 = x_tst_knn_gt_jj_i;

            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3j);
            mat_set(sp_gt, j+1, x_tst_knn_gt_jj_i, tmp3jj);
        }
    }
}

// pull factors out of computation
// BEST VERSION WITH -O3 -march=native (see speedup_shapley.txt)
/* -------------------------- USE THIS VERSION ---------------------------*/
void compute_shapley_opt6(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmpj, tmpjj, tmp0j, tmp1j, tmp3j, tmp0jj, tmp1jj, tmp3jj;
    int* row_j = malloc(N_tst*sizeof(int));
    int* row_jj = malloc(N_tst*sizeof(int));
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;
    int x_tst_knn_gt_jj_i, x_tst_knn_gt_jj_i_plus_1, x_tst_knn_gt_jj_last_i;
    int y_tst_j, y_tst_jj;
    float N_inv = 1.0/N;
    float K_inv = 1.0/K;
    float i_inv, min_K, factor;

    for (int j=0; j < N_tst;j+=2){
        y_tst_j = y_tst[j];
        y_tst_jj = y_tst[j+1];

        x_tst_knn_gt_j_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        x_tst_knn_gt_jj_last_i = int_mat_get(x_tst_knn_gt, j+1, N-1);

        tmpj = (y_trn[x_tst_knn_gt_j_last_i] != y_tst_j) ? 0.0:N_inv;
        tmpjj = (y_trn[x_tst_knn_gt_jj_last_i] != y_tst_jj) ? 0.0:N_inv;

        mat_set(sp_gt, j, x_tst_knn_gt_j_last_i, tmpj);
        mat_set(sp_gt, j+1, x_tst_knn_gt_jj_last_i, tmpjj);

        get_int_row(row_j, x_tst_knn_gt, j);
        get_int_row(row_jj, x_tst_knn_gt, j+1);
        x_tst_knn_gt_j_i_plus_1 = row_j[N-1];
        x_tst_knn_gt_jj_i_plus_1 = row_jj[N-1];

        for (int i=N-2; i>-1; i--){
            i_inv = 1.0 / (i + 1);
            min_K = (K > i+1) ? i+1 : K;
            factor = i_inv * min_K * K_inv;

            x_tst_knn_gt_j_i = row_j[i];
            x_tst_knn_gt_jj_i = row_jj[i];

            tmp0j = (y_trn[x_tst_knn_gt_j_i] != y_tst_j) ? 0.0:1.0;
            tmp1j = (y_trn[x_tst_knn_gt_j_i_plus_1] != y_tst_j) ? 0.0:1.0;
            tmp3j = mat_get(sp_gt, j, x_tst_knn_gt_j_i_plus_1) + (tmp0j - tmp1j) * factor;

            tmp0jj = (y_trn[x_tst_knn_gt_jj_i] != y_tst_jj) ? 0.0:1.0;
            tmp1jj = (y_trn[x_tst_knn_gt_jj_i_plus_1] != y_tst_jj) ? 0.0:1.0;
            tmp3jj = mat_get(sp_gt, j+1, x_tst_knn_gt_jj_i_plus_1) + (tmp0jj - tmp1jj) * factor;

            x_tst_knn_gt_j_i_plus_1 = x_tst_knn_gt_j_i;
            x_tst_knn_gt_jj_i_plus_1 = x_tst_knn_gt_jj_i;

            mat_set(sp_gt, j, x_tst_knn_gt_j_i, tmp3j);
            mat_set(sp_gt, j+1, x_tst_knn_gt_jj_i, tmp3jj);
        }
    }
}

// manual function inlining (good speedup for version without flags)
void compute_shapley_opt6_inlined(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    float tmpj, tmpjj, tmp0j, tmp1j, tmp3j, tmp0jj, tmp1jj, tmp3jj;
    int* row_j = malloc(N_tst*sizeof(int));
    int* row_jj = malloc(N_tst*sizeof(int));
    int x_tst_knn_gt_j_i, x_tst_knn_gt_j_i_plus_1, x_tst_knn_gt_j_last_i;
    int x_tst_knn_gt_jj_i, x_tst_knn_gt_jj_i_plus_1, x_tst_knn_gt_jj_last_i;
    int y_tst_j, y_tst_jj;
    float N_inv = 1.0/N;
    float K_inv = 1.0/K;
    float i_inv, min_K, factor;

    for (int j=0; j < N_tst;j+=2){
        y_tst_j = y_tst[j];
        y_tst_jj = y_tst[j+1];

        x_tst_knn_gt_j_last_i = x_tst_knn_gt->data[j * N + N-1];
        x_tst_knn_gt_jj_last_i = x_tst_knn_gt->data[(j+1) * N + N-1];

        tmpj = (y_trn[x_tst_knn_gt_j_last_i] != y_tst_j) ? 0.0:N_inv;
        tmpjj = (y_trn[x_tst_knn_gt_jj_last_i] != y_tst_jj) ? 0.0:N_inv;

        sp_gt->data[j * sp_gt->n2 + x_tst_knn_gt_j_last_i] = tmpj;
        sp_gt->data[(j+1) * sp_gt->n2 + x_tst_knn_gt_jj_last_i] = tmpjj;

        for(int i = 0; i < N; i++){
            row_j[i] = x_tst_knn_gt->data[j * N + i];
            row_jj[i] = x_tst_knn_gt->data[(j+1) * N + i];
        }

        x_tst_knn_gt_j_i_plus_1 = row_j[N-1];
        x_tst_knn_gt_jj_i_plus_1 = row_jj[N-1];

        for (int i=N-2; i>-1; i--){
            i_inv = 1.0 / (i + 1);
            min_K = (K > i+1) ? i+1 : K;
            factor = i_inv * min_K * K_inv;

            x_tst_knn_gt_j_i = row_j[i];
            x_tst_knn_gt_jj_i = row_jj[i];

            tmp0j = (y_trn[x_tst_knn_gt_j_i] != y_tst_j) ? 0.0:1.0;
            tmp1j = (y_trn[x_tst_knn_gt_j_i_plus_1] != y_tst_j) ? 0.0:1.0;
            tmp3j = sp_gt->data[j * sp_gt->n2 + x_tst_knn_gt_j_i_plus_1] + (tmp0j - tmp1j) * factor;

            tmp0jj = (y_trn[x_tst_knn_gt_jj_i] != y_tst_jj) ? 0.0:1.0;
            tmp1jj = (y_trn[x_tst_knn_gt_jj_i_plus_1] != y_tst_jj) ? 0.0:1.0;
            tmp3jj = sp_gt->data[(j+1) * sp_gt->n2 + x_tst_knn_gt_jj_i_plus_1] + (tmp0jj - tmp1jj) * factor;

            x_tst_knn_gt_j_i_plus_1 = x_tst_knn_gt_j_i;
            x_tst_knn_gt_jj_i_plus_1 = x_tst_knn_gt_jj_i;

            sp_gt->data[j * sp_gt->n2 + x_tst_knn_gt_j_i] = tmp3j;
            sp_gt->data[(j+1) * sp_gt->n2 + x_tst_knn_gt_jj_i] = tmp3jj;
        }
    }
}

// work on 4 rows simultaneously (unroll outer loop by factor 4) change order of matset
void compute_shapley_opt7(mat* sp_gt, const int* y_trn, const int* y_tst, int_mat* x_tst_knn_gt, int K) {
    int N = x_tst_knn_gt->n2;
    int N_tst = x_tst_knn_gt->n1;
    // tmp vars vor inner loop (t_type_row)
    float t00, t01, t02, t03, t10, t11, t12, t13, t30, t31, t32, t33;
    // tmp vars for outer loop
    float tmp0, tmp1, tmp2, tmp3;
    int* row_0 = malloc(N_tst*sizeof(int));
    int* row_1 = malloc(N_tst*sizeof(int));
    int* row_2 = malloc(N_tst*sizeof(int));
    int* row_3 = malloc(N_tst*sizeof(int));
    // index refers to row
    int x_tst_knn_gt_0_i, x_tst_knn_gt_0_i_plus_1, x_tst_knn_gt_0_last_i;
    int x_tst_knn_gt_1_i, x_tst_knn_gt_1_i_plus_1, x_tst_knn_gt_1_last_i;
    int x_tst_knn_gt_2_i, x_tst_knn_gt_2_i_plus_1, x_tst_knn_gt_2_last_i;
    int x_tst_knn_gt_3_i, x_tst_knn_gt_3_i_plus_1, x_tst_knn_gt_3_last_i;
    int y_tst_0, y_tst_1, y_tst_2, y_tst_3;
    // constants
    float N_inv = 1.0/N;
    float K_inv = 1.0/K;
    float i_inv, min_K, factor;

    for (int j = 0; j < N_tst; j+=4){
        y_tst_0 = y_tst[j];
        y_tst_1 = y_tst[j+1];
        y_tst_2 = y_tst[j+2];
        y_tst_3 = y_tst[j+3];

        x_tst_knn_gt_0_last_i = int_mat_get(x_tst_knn_gt, j, N-1);
        x_tst_knn_gt_1_last_i = int_mat_get(x_tst_knn_gt, j+1, N-1);
        x_tst_knn_gt_2_last_i = int_mat_get(x_tst_knn_gt, j+2, N-1);
        x_tst_knn_gt_3_last_i = int_mat_get(x_tst_knn_gt, j+3, N-1);

        tmp0 = (y_trn[x_tst_knn_gt_0_last_i] != y_tst_0) ? 0.0:N_inv;
        tmp1 = (y_trn[x_tst_knn_gt_1_last_i] != y_tst_1) ? 0.0:N_inv;
        tmp2 = (y_trn[x_tst_knn_gt_2_last_i] != y_tst_2) ? 0.0:N_inv;
        tmp3 = (y_trn[x_tst_knn_gt_3_last_i] != y_tst_3) ? 0.0:N_inv;

        mat_set(sp_gt, j, x_tst_knn_gt_0_last_i, tmp0);
        mat_set(sp_gt, j+1, x_tst_knn_gt_1_last_i, tmp1);
        mat_set(sp_gt, j+2, x_tst_knn_gt_2_last_i, tmp2);
        mat_set(sp_gt, j+3, x_tst_knn_gt_3_last_i, tmp3);

        get_int_row(row_0, x_tst_knn_gt, j);
        get_int_row(row_1, x_tst_knn_gt, j+1);
        get_int_row(row_2, x_tst_knn_gt, j+2);
        get_int_row(row_3, x_tst_knn_gt, j+3);

        x_tst_knn_gt_0_i_plus_1 = row_0[N-1];
        x_tst_knn_gt_1_i_plus_1 = row_1[N-1];
        x_tst_knn_gt_2_i_plus_1 = row_2[N-1];
        x_tst_knn_gt_3_i_plus_1 = row_3[N-1];

        for (int i=N-2; i>-1; i--){
            i_inv = 1.0 / (i + 1);
            min_K = (K > i+1) ? i+1 : K;
            factor = i_inv * min_K * K_inv;

            x_tst_knn_gt_0_i = row_0[i];
            x_tst_knn_gt_1_i = row_1[i];
            x_tst_knn_gt_2_i = row_2[i];
            x_tst_knn_gt_3_i = row_3[i];

            // row offset: 0
            t00 = (y_trn[x_tst_knn_gt_0_i] != y_tst_0) ? 0.0:1.0;
            t10 = (y_trn[x_tst_knn_gt_0_i_plus_1] != y_tst_0) ? 0.0:1.0;
            t30 = mat_get(sp_gt, j, x_tst_knn_gt_0_i_plus_1) + (t00 - t10) * factor;

            // row offset: 1
            t01 = (y_trn[x_tst_knn_gt_1_i] != y_tst_1) ? 0.0:1.0;
            t11 = (y_trn[x_tst_knn_gt_1_i_plus_1] != y_tst_1) ? 0.0:1.0;
            t31 = mat_get(sp_gt, j+1, x_tst_knn_gt_1_i_plus_1) + (t01 - t11) * factor;

            // row offset: 2
            t02 = (y_trn[x_tst_knn_gt_2_i] != y_tst_2) ? 0.0:1.0;
            t12 = (y_trn[x_tst_knn_gt_2_i_plus_1] != y_tst_2) ? 0.0:1.0;
            t32 = mat_get(sp_gt, j+2, x_tst_knn_gt_2_i_plus_1) + (t02 - t12) * factor;

            // row offset: 3
            t03 = (y_trn[x_tst_knn_gt_3_i] != y_tst_3) ? 0.0:1.0;
            t13 = (y_trn[x_tst_knn_gt_3_i_plus_1] != y_tst_3) ? 0.0:1.0;
            t33 = mat_get(sp_gt, j+3, x_tst_knn_gt_3_i_plus_1) + (t03 - t13) * factor;

            mat_set(sp_gt, j, x_tst_knn_gt_0_i, t30);
            mat_set(sp_gt, j+1, x_tst_knn_gt_1_i, t31);
            mat_set(sp_gt, j+2, x_tst_knn_gt_2_i, t32);
            mat_set(sp_gt, j+3, x_tst_knn_gt_3_i, t33);

            x_tst_knn_gt_0_i_plus_1 = x_tst_knn_gt_0_i;
            x_tst_knn_gt_1_i_plus_1 = x_tst_knn_gt_1_i;
            x_tst_knn_gt_2_i_plus_1 = x_tst_knn_gt_2_i;
            x_tst_knn_gt_3_i_plus_1 = x_tst_knn_gt_3_i;
        }
    }
}


// 1st function registered must be base in terms of cycles
// start testing from 4 onwards
void register_comp_shapley(svfctptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &compute_shapley_base;
    userFuncs[1] = &compute_shapley_opt1;
    userFuncs[2] = &compute_shapley_opt2;
    userFuncs[3] = &compute_shapley_opt3;
    userFuncs[4] = &compute_shapley_opt4;
    userFuncs[5] = &compute_shapley_opt5;
    userFuncs[6] = &compute_shapley_opt6;
    userFuncs[7] = &compute_shapley_opt6_inlined;
    userFuncs[8] = &compute_shapley_opt7;
}
