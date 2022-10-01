#pragma once
#include <immintrin.h>
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "common.h"


// base implementation, not optimized
void compute_col_mean_scalar_baseline(mat* m, mat* res, int i_tst){
    float mean;
    for (int i = 0; i < res->n2; i++){ // N
        mean = 0.0;
        for (int j = 0; j < res->n1; j++){ // T
            mean += mat_get(res, j, i);
        }
        mean /= res->n1;
        mat_set(m, i_tst, i, mean);
    }
}


void compute_col_mean_simd1(mat* m, mat* res, int i_tst){
    __m256 res_vec, row0, row1, row2, row3, row4, row5, row6, row7;
    __m256 subvec0, subvec1, subvec2, subvec3, subvec4, subvec5, subvec6;
    int n2 = res->n2;
    int n1 = res->n1;
    float n_inv = 1.0/n1;
    float *data = res->data;
    float *m_data = m->data;
    __m256 n = _mm256_setr_ps(n_inv, n_inv, n_inv, n_inv, n_inv, n_inv, n_inv, n_inv);
    for (int i = 0; i < res->n2; i+=8){ // N
        res_vec = _mm256_setr_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (int j = 0; j < res->n1; j+=8){ // T
            // load
            row0 = _mm256_loadu_ps(&data[j*n2+i]);
            row1 = _mm256_loadu_ps(&data[(j+1)*n2+i]);
            row2 = _mm256_loadu_ps(&data[(j+2)*n2+i]);
            row3 = _mm256_loadu_ps(&data[(j+3)*n2+i]);
            row4 = _mm256_loadu_ps(&data[(j+4)*n2+i]);
            row5 = _mm256_loadu_ps(&data[(j+5)*n2+i]);
            row6 = _mm256_loadu_ps(&data[(j+6)*n2+i]);
            row7 = _mm256_loadu_ps(&data[(j+7)*n2+i]);

            // calculate
            subvec0 = _mm256_add_ps(row0, row1);
            subvec1 = _mm256_add_ps(row2, row3);
            subvec2 = _mm256_add_ps(row4, row5);
            subvec3 = _mm256_add_ps(row6, row7);

            subvec4 = _mm256_add_ps(subvec0, subvec1);
            subvec5 = _mm256_add_ps(subvec2, subvec3);

            subvec6 = _mm256_add_ps(subvec4, subvec5);

            res_vec = _mm256_add_ps(res_vec, subvec6);
        }
        res_vec = _mm256_mul_ps(res_vec, n);
        _mm256_storeu_ps(&m_data[i_tst*m->n2+i], res_vec);
    }
}

// more unrolling
void compute_col_mean_simd2(mat* m, mat* res, int i_tst){
    __m256 res_vec, row0, row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13, row14, row15;
    __m256 subvec0, subvec1, subvec2, subvec3, subvec4, subvec5, subvec6, subvec7, subvec8, subvec9, subvec10, subvec11, subvec12, subvec13, subvec14;
    int n2 = res->n2;
    int n1 = res->n1;
    float n_inv = 1.0/n1;
    float *data = res->data;
    float *m_data = m->data;
    __m256 n = _mm256_setr_ps(n_inv, n_inv, n_inv, n_inv, n_inv, n_inv, n_inv, n_inv);
    for (int i = 0; i < res->n2; i+=8){ // N
        res_vec = _mm256_setr_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (int j = 0; j < res->n1; j+=16){ // T
            // load
            row0 = _mm256_loadu_ps(&data[j*n2+i]);
            row1 = _mm256_loadu_ps(&data[(j+1)*n2+i]);
            row2 = _mm256_loadu_ps(&data[(j+2)*n2+i]);
            row3 = _mm256_loadu_ps(&data[(j+3)*n2+i]);
            row4 = _mm256_loadu_ps(&data[(j+4)*n2+i]);
            row5 = _mm256_loadu_ps(&data[(j+5)*n2+i]);
            row6 = _mm256_loadu_ps(&data[(j+6)*n2+i]);
            row7 = _mm256_loadu_ps(&data[(j+7)*n2+i]);

            row8 = _mm256_loadu_ps(&data[(j+8)*n2+i]);
            row9 = _mm256_loadu_ps(&data[(j+9)*n2+i]);
            row10 = _mm256_loadu_ps(&data[(j+10)*n2+i]);
            row11 = _mm256_loadu_ps(&data[(j+11)*n2+i]);
            row12 = _mm256_loadu_ps(&data[(j+12)*n2+i]);
            row13 = _mm256_loadu_ps(&data[(j+13)*n2+i]);
            row14 = _mm256_loadu_ps(&data[(j+14)*n2+i]);
            row15 = _mm256_loadu_ps(&data[(j+15)*n2+i]);

            // calculate
            subvec0 = _mm256_add_ps(row0, row1);
            subvec1 = _mm256_add_ps(row2, row3);
            subvec2 = _mm256_add_ps(row4, row5);
            subvec3 = _mm256_add_ps(row6, row7);

            subvec4 = _mm256_add_ps(row8, row9);
            subvec5 = _mm256_add_ps(row10, row11);
            subvec6 = _mm256_add_ps(row12, row13);
            subvec7 = _mm256_add_ps(row14, row15);

            subvec8 = _mm256_add_ps(subvec0, subvec1);
            subvec9 = _mm256_add_ps(subvec2, subvec3);
            subvec10 = _mm256_add_ps(subvec4, subvec5);
            subvec11 = _mm256_add_ps(subvec6, subvec7);

            subvec12 = _mm256_add_ps(subvec8, subvec9);
            subvec13 = _mm256_add_ps(subvec10, subvec11);

            subvec14 = _mm256_add_ps(subvec12, subvec13);

            res_vec = _mm256_add_ps(res_vec, subvec14);
        }
        res_vec = _mm256_mul_ps(res_vec, n);
        _mm256_storeu_ps(&m_data[i_tst*m->n2+i], res_vec);
    }
}

// more unrolling, different direction
void compute_col_mean_simd3(mat* m, mat* res, int i_tst){
    __m256 res_vec, res_vec2, row0, row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13, row14, row15;
    __m256 subvec0, subvec1, subvec2, subvec3, subvec4, subvec5, subvec6, subvec7, subvec8, subvec9, subvec10, subvec11, subvec12, subvec13;
    int n2 = res->n2;
    int n1 = res->n1;
    float n_inv = 1.0/n1;
    float *data = res->data;
    float *m_data = m->data;
    __m256 n = _mm256_set1_ps(n_inv);
    for (int i = 0; i < res->n2; i+=16){ // N
        res_vec = _mm256_set1_ps(0.0);
        res_vec2 = _mm256_set1_ps(0.0);
        for (int j = 0; j < res->n1; j+=8){ // T
            // load
            row0 = _mm256_loadu_ps(&data[j*n2+i]);
            row1 = _mm256_loadu_ps(&data[(j+1)*n2+i]);
            row2 = _mm256_loadu_ps(&data[(j+2)*n2+i]);
            row3 = _mm256_loadu_ps(&data[(j+3)*n2+i]);
            row4 = _mm256_loadu_ps(&data[(j+4)*n2+i]);
            row5 = _mm256_loadu_ps(&data[(j+5)*n2+i]);
            row6 = _mm256_loadu_ps(&data[(j+6)*n2+i]);
            row7 = _mm256_loadu_ps(&data[(j+7)*n2+i]);

            row8 =  _mm256_loadu_ps(&data[j*n2+i+8]);
            row9 =  _mm256_loadu_ps(&data[(j+1)*n2+i+8]);
            row10 = _mm256_loadu_ps(&data[(j+2)*n2+i+8]);
            row11 = _mm256_loadu_ps(&data[(j+3)*n2+i+8]);
            row12 = _mm256_loadu_ps(&data[(j+4)*n2+i+8]);
            row13 = _mm256_loadu_ps(&data[(j+5)*n2+i+8]);
            row14 = _mm256_loadu_ps(&data[(j+6)*n2+i+8]);
            row15 = _mm256_loadu_ps(&data[(j+7)*n2+i+8]);

            // calculate
            subvec0 = _mm256_add_ps(row0, row1);
            subvec1 = _mm256_add_ps(row2, row3);
            subvec2 = _mm256_add_ps(row4, row5);
            subvec3 = _mm256_add_ps(row6, row7);

            subvec4 = _mm256_add_ps(row8, row9);
            subvec5 = _mm256_add_ps(row10, row11);
            subvec6 = _mm256_add_ps(row12, row13);
            subvec7 = _mm256_add_ps(row14, row15);

            subvec8 = _mm256_add_ps(subvec0, subvec1);
            subvec9 = _mm256_add_ps(subvec2, subvec3);

            subvec10 = _mm256_add_ps(subvec4, subvec5);
            subvec11 = _mm256_add_ps(subvec6, subvec7);

            subvec12 = _mm256_add_ps(subvec8, subvec9);
            subvec13 = _mm256_add_ps(subvec10, subvec11);

            res_vec = _mm256_add_ps(res_vec, subvec12);
            res_vec2 = _mm256_add_ps(res_vec2, subvec13);
        }
        res_vec = _mm256_mul_ps(res_vec, n);
        res_vec2 = _mm256_mul_ps(res_vec2, n);

        _mm256_storeu_ps(&m_data[i_tst*m->n2+i], res_vec);
        _mm256_storeu_ps(&m_data[i_tst*m->n2+i+8], res_vec2);
    }
}

// more unrolling, same direction like 3 but moooooooorrreeeeee unnrroooooollllllliinng
void compute_col_mean_simd4(mat* m, mat* res, int i_tst){
    __m256 res_vec, res_vec2, row0, row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13, row14, row15;
    __m256 res_vecb, res_vec2b, row0b, row1b, row2b, row3b, row4b, row5b, row6b, row7b, row8b, row9b, row10b, row11b, row12b, row13b, row14b, row15b;
    __m256 subvec0, subvec1, subvec2, subvec3, subvec4, subvec5, subvec6, subvec7, subvec8, subvec9, subvec10, subvec11, subvec12, subvec13;
    __m256 subvec0b, subvec1b, subvec2b, subvec3b, subvec4b, subvec5b, subvec6b, subvec7b, subvec8b, subvec9b, subvec10b, subvec11b, subvec12b, subvec13b;
    int n2 = res->n2;
    int n1 = res->n1;
    float n_inv = 1.0/n1;
    float *data = res->data;
    float *m_data = m->data;
    __m256 n = _mm256_set1_ps(n_inv);
    for (int i = 0; i < res->n2; i+=32){ // N
        res_vec = _mm256_set1_ps(0.0);
        res_vec2 = _mm256_set1_ps(0.0);

        res_vecb = _mm256_set1_ps(0.0);
        res_vec2b = _mm256_set1_ps(0.0);
        for (int j = 0; j < res->n1; j+=8){ // T
            // load
            row0 = _mm256_loadu_ps(&data[j*n2+i]);
            row1 = _mm256_loadu_ps(&data[(j+1)*n2+i]);
            row2 = _mm256_loadu_ps(&data[(j+2)*n2+i]);
            row3 = _mm256_loadu_ps(&data[(j+3)*n2+i]);
            row4 = _mm256_loadu_ps(&data[(j+4)*n2+i]);
            row5 = _mm256_loadu_ps(&data[(j+5)*n2+i]);
            row6 = _mm256_loadu_ps(&data[(j+6)*n2+i]);
            row7 = _mm256_loadu_ps(&data[(j+7)*n2+i]);

            row8 =  _mm256_loadu_ps(&data[j*n2+i+8]);
            row9 =  _mm256_loadu_ps(&data[(j+1)*n2+i+8]);
            row10 = _mm256_loadu_ps(&data[(j+2)*n2+i+8]);
            row11 = _mm256_loadu_ps(&data[(j+3)*n2+i+8]);
            row12 = _mm256_loadu_ps(&data[(j+4)*n2+i+8]);
            row13 = _mm256_loadu_ps(&data[(j+5)*n2+i+8]);
            row14 = _mm256_loadu_ps(&data[(j+6)*n2+i+8]);
            row15 = _mm256_loadu_ps(&data[(j+7)*n2+i+8]);

            row0b = _mm256_loadu_ps(&data[j*n2+i+16]);
            row1b = _mm256_loadu_ps(&data[(j+1)*n2+i+16]);
            row2b = _mm256_loadu_ps(&data[(j+2)*n2+i+16]);
            row3b = _mm256_loadu_ps(&data[(j+3)*n2+i+16]);
            row4b = _mm256_loadu_ps(&data[(j+4)*n2+i+16]);
            row5b = _mm256_loadu_ps(&data[(j+5)*n2+i+16]);
            row6b = _mm256_loadu_ps(&data[(j+6)*n2+i+16]);
            row7b = _mm256_loadu_ps(&data[(j+7)*n2+i+16]);

            row8b =  _mm256_loadu_ps(&data[j*n2+i+24]);
            row9b =  _mm256_loadu_ps(&data[(j+1)*n2+i+24]);
            row10b = _mm256_loadu_ps(&data[(j+2)*n2+i+24]);
            row11b = _mm256_loadu_ps(&data[(j+3)*n2+i+24]);
            row12b = _mm256_loadu_ps(&data[(j+4)*n2+i+24]);
            row13b = _mm256_loadu_ps(&data[(j+5)*n2+i+24]);
            row14b = _mm256_loadu_ps(&data[(j+6)*n2+i+24]);
            row15b = _mm256_loadu_ps(&data[(j+7)*n2+i+24]);

            // calculate
            subvec0 = _mm256_add_ps(row0, row1);
            subvec1 = _mm256_add_ps(row2, row3);
            subvec2 = _mm256_add_ps(row4, row5);
            subvec3 = _mm256_add_ps(row6, row7);

            subvec4 = _mm256_add_ps(row8, row9);
            subvec5 = _mm256_add_ps(row10, row11);
            subvec6 = _mm256_add_ps(row12, row13);
            subvec7 = _mm256_add_ps(row14, row15);

            subvec0b = _mm256_add_ps(row0b, row1b);
            subvec1b = _mm256_add_ps(row2b, row3b);
            subvec2b = _mm256_add_ps(row4b, row5b);
            subvec3b = _mm256_add_ps(row6b, row7b);

            subvec4b = _mm256_add_ps(row8b, row9b);
            subvec5b = _mm256_add_ps(row10b, row11b);
            subvec6b = _mm256_add_ps(row12b, row13b);
            subvec7b = _mm256_add_ps(row14b, row15b);

            subvec8 = _mm256_add_ps(subvec0, subvec1);
            subvec9 = _mm256_add_ps(subvec2, subvec3);

            subvec10 = _mm256_add_ps(subvec4, subvec5);
            subvec11 = _mm256_add_ps(subvec6, subvec7);

            subvec8b = _mm256_add_ps(subvec0b, subvec1b);
            subvec9b = _mm256_add_ps(subvec2b, subvec3b);

            subvec10b = _mm256_add_ps(subvec4b, subvec5b);
            subvec11b = _mm256_add_ps(subvec6b, subvec7b);

            subvec12 = _mm256_add_ps(subvec8, subvec9);
            subvec13 = _mm256_add_ps(subvec10, subvec11);

            subvec12b = _mm256_add_ps(subvec8b, subvec9b);
            subvec13b = _mm256_add_ps(subvec10b, subvec11b);

            res_vec = _mm256_add_ps(res_vec, subvec12);
            res_vec2 = _mm256_add_ps(res_vec2, subvec13);

            res_vecb = _mm256_add_ps(res_vecb, subvec12b);
            res_vec2b = _mm256_add_ps(res_vec2b, subvec13b);
        }
        res_vec = _mm256_mul_ps(res_vec, n);
        res_vec2 = _mm256_mul_ps(res_vec2, n);

        res_vecb = _mm256_mul_ps(res_vecb, n);
        res_vec2b = _mm256_mul_ps(res_vec2b, n);

        _mm256_storeu_ps(&m_data[i_tst*m->n2+i], res_vec);
        _mm256_storeu_ps(&m_data[i_tst*m->n2+i+8], res_vec2);

        _mm256_storeu_ps(&m_data[i_tst*m->n2+i+16], res_vecb);
        _mm256_storeu_ps(&m_data[i_tst*m->n2+i+24], res_vec2b);
    }
}




void register_simd_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &compute_col_mean_scalar_baseline;
    userFuncs[1] = &compute_col_mean_simd1;
    userFuncs[2] = &compute_col_mean_simd2;
    userFuncs[3] = &compute_col_mean_simd3;
    userFuncs[4] = &compute_col_mean_simd4;

}
