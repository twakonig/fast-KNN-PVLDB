#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mat.h"
#include <string.h>
#include <stdbool.h>
#include "utils.h"

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

void up(maxheap* h, int index) {
    if (index == 0) {
        return;
    }
    int parent_index = (index-1)/2;

    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    get_row(row, h->x_trn, h->heap[index]);
    float norm_idx = l2norm(h->x_tst, row, h->x_trn->n2);
    get_row(row, h->x_trn, h->heap[parent_index]);
    float norm_parentidx = l2norm(h->x_tst, row, h->x_trn->n2);
    free(row);
    if (norm_idx > norm_parentidx) {
        int temp = h->heap[index];
        h->heap[index] = h->heap[parent_index];
        h->heap[parent_index] = temp;
        up(h, parent_index);
    }

    return;
}

void down(maxheap* h, int index) {

    int tar_index;
    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    if (2*index + 1 > h->counter){
        free(row);
        return;
    }
    if (2*index + 1 < h->counter) {
        get_row(row, h->x_trn, h->heap[2*index+1]);
        float norm1 = l2norm(h->x_tst, row, h->x_trn->n2);
        get_row(row, h->x_trn, h->heap[2*index+2]);
        float norm2 = l2norm(h->x_tst, row, h->x_trn->n2);
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
    float norm1 = l2norm(h->x_tst, row, h->x_trn->n2);
    get_row(row, h->x_trn, h->heap[tar_index]);
    float norm2 = l2norm(h->x_tst, row, h->x_trn->n2);


    if (norm1 < norm2) {
        int temp = h->heap[index];
        h->heap[index] = h->heap[tar_index];
        h->heap[tar_index] = temp;
        down(h, tar_index);
    }
    return;
}

// elem is index of permuted matrix from main
void insert(maxheap* h, int elem) {
    float* row = (float*) malloc(h->x_trn->n2 * sizeof(float));
    get_row(row, h->x_trn, elem);
    float d_elem = l2norm(row, h->x_tst, h->x_trn->n2);

    if (h->counter <= (h->K - 2)) {
        h->heap[h->current_sz] = elem;
        h->current_sz += 1;
        h->counter += 1;
        up(h, h->counter);
        h->changed = 1;
    }
    else {
        get_row(row,h->x_trn, h->heap[0]);
        float d_root = l2norm(row, h->x_tst, h->x_trn->n2);
        if (d_elem < d_root) {
            h->heap[0] = elem;
            down(h, 0);
            h->changed = 1;
        }
        else
            h->changed = 0;
    }
    free(row);
}
