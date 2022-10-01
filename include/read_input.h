#pragma once

#include <stdio.h>
#include <string.h>
#include "cnpy.h"
#include <assert.h>

void read_input_features(float* out, char* file, int N, int M){
    /* Open file */

    cnpy_array in;
    if (cnpy_open(file, true, &in) != CNPY_SUCCESS) {
        cnpy_perror("Input file not opened");
    }

    if(in.dtype != CNPY_F4){
        assert(false);
    }
    // assert(in.dtype == CNPY_F8);
    // assert(in.n_dim == 2);

    size_t index[] = {0, 0};
    for (index[0] = 0; index[0] < (size_t) N; index[0] += 1) {
        for (index[1] = 0; index[1] < (size_t) M; index[1] += 1) {
            float x;
            x = cnpy_get_f4(in, index);
            out[index[0] * M + index[1]] = x;
        }
    }
}

void read_input_labels(int* out, char* file, int N, int M){
    /* Open file */
    cnpy_array in;
    if (cnpy_open(file, true, &in) != CNPY_SUCCESS) {
        cnpy_perror("Input file not opened");
    }
    if(in.dtype != CNPY_U1){
        assert(false);
    }

    // assert(in.dtype == CNPY_F8);
    // assert(in.n_dim == 2);

    size_t index[] = {0, 0};
    for (index[0] = 0; index[0] < (size_t) N; index[0] += 1) {
        for (index[1] = 0; index[1] < (size_t) M; index[1] += 1) {
            int x;
            x = cnpy_get_u1(in, index);
            out[index[0] * M + index[1]] = x;
        }
    }
}