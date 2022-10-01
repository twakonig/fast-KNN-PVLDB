#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    float value;
    int index;
} pair_t;

struct pair_idx {
    float value;
    int index;
};

typedef struct{
    int n1;
    int n2;
    float *data;
} mat;

typedef struct{
    int n1;
    int n2;
    int *data;
} int_mat;

typedef struct{
    int n1;
    int n2;
    pair_t *data;
} knn;

typedef struct{
    int n1;
    int n2;
    int n3;
    float *data;
} tensor;

void destroy(mat* m)
{
    free(m->data);
}

void destroy_int_mat(int_mat* m)
{
    free(m->data);
}

void destroy_knn(knn *m)
{
    free(m->data);
}

void destroy_tensor(tensor* m)
{
    free(m->data);
}

void build(mat* m, int n1, int n2){
    m->n1 = n1;
    m->n2 = n2;
    m->data = (float *) malloc(n1 * n2 * sizeof(float));
}

void build_int_mat(int_mat* m, int n1, int n2){
    m->n1 = n1;
    m->n2 = n2;
    m->data = (int *) malloc(n1 * n2 * sizeof(int));
}

void build_knn(knn* m, int n1, int n2){
    m->n1 = n1;
    m->n2 = n2;
    m->data = malloc(n1 * n2 * sizeof(pair_t));

    for (int i = 0; i < n1; i++){
        for (int j = 0; j < n2; j++){
            m->data[i * n2 + j].index = j;
        }
    }
}

void build_tensor(tensor* m, int n1, int n2, int n3){
    m->n1 = n1;
    m->n2 = n2;
    m->n3 = n3;
    m->data = (float *) malloc(n1 * n2 * n3 * sizeof(float));
}

int knn_len(knn* m){
    return m->n1 * m->n2;
}

int mat_len(mat* m) {
    return m->n1 * m->n2;
}

int tensor_len(tensor* m){
    return m->n1 * m->n2 * m->n3;
}

float mat_get(mat *m, int i, int j) {
    int ij = i * m->n2 + j;
    return m->data[ij];
}

int int_mat_get(int_mat *m, int i, int j) {
    int ij = i * m->n2 + j;
    return m->data[ij];
}

int knn_get(knn *m, int i, int j) {
    int ij = i * m->n2 + j;
    return m->data[ij].value;
}

float tensor_get(tensor *m, int i, int j, int k) {
    int ijk = i * m->n3 * m->n2 + j * m->n3 + k;
    return m->data[ijk];
}

void mat_set(mat *m, int i, int j, float val) {
    int ij = i * m->n2 + j;
    m->data[ij] = val;
}

void int_mat_set(int_mat *m, int i, int j, int val) {
    int ij = i * m->n2 + j;
    m->data[ij] = val;
}

void knn_set(knn *m, int i, int j, int val) {
    int ij = i * m->n2 + j;
    m->data[ij].value = val;
}

void tensor_set(tensor* m, int i, int j, int k, float val){
    int ijk = i * m->n3 * m->n2 + j * m->n3 + k;
    m->data[ijk] = val;
}

void get_row(float* vec, mat *m, int ind) {
    int len = m->n2;

    for(int i = 0; i < len; i++){
        vec[i] = m->data[ind * m->n2 + i];
    }
}

void get_int_row(int* vec, int_mat *m, int ind) {
    int len = m->n2;

    for(int i = 0; i < len; i++){
        vec[i] = m->data[ind * m->n2 + i];
    }
}

void get_mat_from_tensor(tensor* t, mat* m, int ind) {
    for(int i = 0; i < m->n1; i++) {
        for(int j = 0; j < m->n2; j++) {
            mat_set(m, i, j, tensor_get(t, ind, i, j));
        }
    }
}

/* fRand retrieved from:
    https://stackoverflow.com/questions/2704521/generate-random-float-numbers-in-c
*/
float fRand(float fMin, float fMax)
{
    float f = (float) (rand()) / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void initialize_rand_mat(mat* m) {
    // random matrix initialization
    for(int i = 0; i < m->n1; i++){
        for(int j = 0; j < m->n2; j++){
            mat_set(m, i, j, fRand(-1.0, 1.0));
        }
    }
}

//------------------JUST FOR TESTING PURPOSES--------------------
//permutation function
void shuffle_ints(int *arr, int n) {
    int i, j, tmp;

    for (i = n - 1; i >= 0; i--) {
        j = rand() % (i + 1);
        tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

void initialize_rand_int_mat(int_mat* m, int N) {
    int* row = malloc(N*sizeof(int));
    for(int k = 0; k < N; k++) {
        row[k] = k;
    }
    // simulate knn_gt matrix of ints
    for(int i = 0; i < m->n1; i++){
        shuffle_ints(row, N);
        for(int j = 0; j < m->n2; j++){
            // store permutation into matrix
            int_mat_set(m, i, j, row[j]);
        }
    }
    free(row);
}
//---------------------------------------------------------------


void initialize_mat(mat* m, float value) {
    // random matrix initialization
    for(int i = 0; i < m->n1; i++){
        for(int j = 0; j < m->n2; j++){
            mat_set(m, i, j, value);
        }
    }
}

// random array initialization; floats between 0 and 1
void initialize_rand_struct(pair_t a[], int n) {
    for(int i = 0; i < n; i++){
        a[i].value = fRand(-1.0, 1.0);
        a[i].index = i;
    }
}


