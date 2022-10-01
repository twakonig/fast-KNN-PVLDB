//
// Created by theresa on 28.04.22.
//

#ifndef ALG1_UTILS_H
#define ALG1_UTILS_H

#include <stdio.h>
#include <math.h>
#include "mat.h"

float l2norm(float arr1[], float arr2[], size_t len){
    float res = 0.0;

    for (size_t i = 0; i < len; i++) {
        res += pow(arr1[i] - arr2[i], 2);
    }

    return sqrt(res);
}

int cmp(const void *a, const void *b){
    pair_t *a1 = (pair_t *)a;
    pair_t *a2 = (pair_t *)b;
    return (a1->value > a2->value) - (a2->value > a1->value);
}

void initialize_rand_array(int* a, int n) {
    // random array initialization
    for(int i = 0; i < n; i++){
        a[i] = (int) (rand() % 10);
    }
}

void initialize_rand_int_array(int* a, int n) {
    for(int i = 0; i < n; i++){
        a[i] = rand() % 100;
    }
}

// random array initialization; floats between 0 and 1
void initialize_rand_float_array(float* a, int n) {

    for(int i = 0; i < n; i++){
        a[i] = (float)(rand()) / RAND_MAX;
    }
}

void initialize_incr_int_array(int* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
}

void print_float_array(float* a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.6f\n", a[i]);
    }
}

void print_int_array(int* a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d\n", a[i]);
    }
}

#endif //ALG1_UTILS_H
