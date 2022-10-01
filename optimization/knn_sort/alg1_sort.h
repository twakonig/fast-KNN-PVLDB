//
// Created by lucasck on 18/05/22.
// Modified by jokics on 19/05/22.
//

#include "../../src/alg1/alg1.h"
#include "../../include/tsc_x86.h"
#include "sort_opt.h"
#include "../../include/quadsort.h"
#include "../../include/ksort.h"
#include "../../include/utils.h"
#define pair_lt(a, b) ((a).value < (b).value)

KSORT_INIT(pair, pair_t, pair_lt)
KSORT_INIT_GENERIC(float)
// int cmp(const void *a, const void *b){
//     struct pair *a1 = (struct pair *)a;
//     struct pair *a2 = (struct pair *)b;
//     if ((*a1).value > (*a2).value) return 1;
//     else if ((*a1).value < (*a2).value) return -1;
//     else return 0;
// }
// stdlib qsort
void sort_qsort(pair_t distances[], size_t N){
    qsort(distances, N, sizeof(pair_t), cmp);
}

// klib mergesort
void sort_klib_mergesort(pair_t distances[], size_t N){
    ks_mergesort(pair, N, distances, 0);
}

// quadsort
void sort_quadsort(pair_t distances[], size_t N){
    quadsort(distances, N, sizeof(pair_t), cmp);
}
