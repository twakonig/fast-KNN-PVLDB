//
// Created by lucasck on 24/05/22.
//

#include "../../include/quadsort.h"
#include "../../include/mat.h"
//#include "../../include/utils.h"

//__attribute__((always_inline)) int compare(const void *__restrict__ a, const void *__restrict__ b){
//    return ((*(int*)a > *(int*)b) - (*(int*)b > *(int*)a));
//}

int cmp(const void *a, const void *b){
    pair_t *a1 = (pair_t *)a;
    pair_t *a2 = (pair_t *)b;
    return (a1->value > a2->value) - (a2->value > a1->value);
}

void _quadsort(pair_t* distances, size_t N){
    quadsort(distances, N, sizeof(pair_t), cmp);
}

int main(int argc, char **argv){
    size_t N = atoi(argv[1]);
    pair_t* a = malloc(N * sizeof(pair_t));
    pair_t* b = malloc(N * sizeof(pair_t));
    initialize_rand_struct(a, N);
    for(int i=0; i < N; i++) {
    b[i].value = a[i].value;
    b[i].index = a[i].index;
    }
//    for(int i=0; i < N; i++){
//        printf("%f %d \t", a[i].value, a[i].index);
//    }
    printf("\n");
    printf("\n");
    _quadsort(a, N);
    printf("quadsort: \n");
    for(int i=0; i < N; i++){
        printf("%d ", a[i].index);
    }
    printf("\n");
    qsort(b, N, sizeof(pair_t), cmp);
    printf("qsort: \n");
    for(int i=0; i < N; i++){
        printf("%d ", b[i].index);
    }
    int error = 0;
    for(int i=0; i < N; i++) {
        if(a[i].index != b[i].index){
            error = 1;
        }
    }
    if (error == 1){
        printf("\n ERROR, TEST FAIL");
    }else {
        printf("\n TEST PASS");
    }
    free(b);
    free(a);
    return 0;
}