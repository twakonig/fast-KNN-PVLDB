#pragma once
#include <immintrin.h>
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "common.h"

void compute_col_mean(mat* m, mat* res, int i_tst){
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

void compute_col_mean_opt(mat* m, mat* res, int i_tst){
    float mean0;
    float mean1;
    float mean2;
    float mean3;
    float mean4;
    float mean5;
    float mean6;
    float mean7;

    float mean8;
    float mean9;
    float mean10;
    float mean11;
    float mean12;
    float mean13;
    float mean14;
    float mean15;


    int n2 = res->n2;
    int n1 = res->n1;
    float *data = res->data;
    float *m_data = m->data;
    for (int i = 0; i < n2; i+=16){ // N
        mean0 = 0.0;
        mean1 = 0.0;
        mean2 = 0.0;
        mean3 = 0.0;
        mean4 = 0.0;
        mean5 = 0.0;
        mean6 = 0.0;
        mean7 = 0.0;

        mean8 = 0.0;
        mean9 = 0.0;
        mean10 = 0.0;
        mean11 = 0.0;
        mean12 = 0.0;
        mean13 = 0.0;
        mean14 = 0.0;
        mean15 = 0.0;
        for (int j = 0; j < n1; j+=8){ // T
            mean0 += data[j*n2+i];
            mean0 += data[(j+1)*n2+i];
            mean0 += data[(j+2)*n2+i];
            mean0 += data[(j+3)*n2+i];
            mean0 += data[(j+4)*n2+i];
            mean0 += data[(j+5)*n2+i];
            mean0 += data[(j+6)*n2+i];
            mean0 += data[(j+7)*n2+i];

            mean1 += data[j*n2+i+1];
            mean1 += data[(j+1)*n2+i+1];
            mean1 += data[(j+2)*n2+i+1];
            mean1 += data[(j+3)*n2+i+1];
            mean1 += data[(j+4)*n2+i+1];
            mean1 += data[(j+5)*n2+i+1];
            mean1 += data[(j+6)*n2+i+1];
            mean1 += data[(j+7)*n2+i+1];

            mean2 += data[j*n2+i+2];
            mean2 += data[(j+1)*n2+i+2];
            mean2 += data[(j+2)*n2+i+2];
            mean2 += data[(j+3)*n2+i+2];
            mean2 += data[(j+4)*n2+i+2];
            mean2 += data[(j+5)*n2+i+2];
            mean2 += data[(j+6)*n2+i+2];
            mean2 += data[(j+7)*n2+i+2];

            mean3 += data[j*n2+i+3];
            mean3 += data[(j+1)*n2+i+3];
            mean3 += data[(j+2)*n2+i+3];
            mean3 += data[(j+3)*n2+i+3];
            mean3 += data[(j+4)*n2+i+3];
            mean3 += data[(j+5)*n2+i+3];
            mean3 += data[(j+6)*n2+i+3];
            mean3 += data[(j+7)*n2+i+3];

            mean4 += data[j*n2+i+4];
            mean4 += data[(j+1)*n2+i+4];
            mean4 += data[(j+2)*n2+i+4];
            mean4 += data[(j+3)*n2+i+4];
            mean4 += data[(j+4)*n2+i+4];
            mean4 += data[(j+5)*n2+i+4];
            mean4 += data[(j+6)*n2+i+4];
            mean4 += data[(j+7)*n2+i+4];

            mean5 += data[j*n2+i+5];
            mean5 += data[(j+1)*n2+i+5];
            mean5 += data[(j+2)*n2+i+5];
            mean5 += data[(j+3)*n2+i+5];
            mean5 += data[(j+4)*n2+i+5];
            mean5 += data[(j+5)*n2+i+5];
            mean5 += data[(j+6)*n2+i+5];
            mean5 += data[(j+7)*n2+i+5];

            mean6 += data[j*n2+i+6];
            mean6 += data[(j+1)*n2+i+6];
            mean6 += data[(j+2)*n2+i+6];
            mean6 += data[(j+3)*n2+i+6];
            mean6 += data[(j+4)*n2+i+6];
            mean6 += data[(j+5)*n2+i+6];
            mean6 += data[(j+6)*n2+i+6];
            mean6 += data[(j+7)*n2+i+6];

            mean7 += data[j*n2+i+7];
            mean7 += data[(j+1)*n2+i+7];
            mean7 += data[(j+2)*n2+i+7];
            mean7 += data[(j+3)*n2+i+7];
            mean7 += data[(j+4)*n2+i+7];
            mean7 += data[(j+5)*n2+i+7];
            mean7 += data[(j+6)*n2+i+7];
            mean7 += data[(j+7)*n2+i+7];

            //---------------------------
            mean8 += data[j*n2+i+8];
            mean8 += data[(j+1)*n2+i+8];
            mean8 += data[(j+2)*n2+i+8];
            mean8 += data[(j+3)*n2+i+8];
            mean8 += data[(j+4)*n2+i+8];
            mean8 += data[(j+5)*n2+i+8];
            mean8 += data[(j+6)*n2+i+8];
            mean8 += data[(j+7)*n2+i+8];

            mean9 += data[j*n2+i+9];
            mean9 += data[(j+1)*n2+i+9];
            mean9 += data[(j+2)*n2+i+9];
            mean9 += data[(j+3)*n2+i+9];
            mean9 += data[(j+4)*n2+i+9];
            mean9 += data[(j+5)*n2+i+9];
            mean9 += data[(j+6)*n2+i+9];
            mean9 += data[(j+7)*n2+i+9];

            mean10 += data[j*n2+i+10];
            mean10 += data[(j+1)*n2+i+10];
            mean10 += data[(j+2)*n2+i+10];
            mean10 += data[(j+3)*n2+i+10];
            mean10 += data[(j+4)*n2+i+10];
            mean10 += data[(j+5)*n2+i+10];
            mean10 += data[(j+6)*n2+i+10];
            mean10 += data[(j+7)*n2+i+10];

            mean11 += data[j*n2+i+11];
            mean11 += data[(j+1)*n2+i+11];
            mean11 += data[(j+2)*n2+i+11];
            mean11 += data[(j+3)*n2+i+11];
            mean11 += data[(j+4)*n2+i+11];
            mean11 += data[(j+5)*n2+i+11];
            mean11 += data[(j+6)*n2+i+11];
            mean11 += data[(j+7)*n2+i+11];

            mean12 += data[j*n2+i+12];
            mean12 += data[(j+1)*n2+i+12];
            mean12 += data[(j+2)*n2+i+12];
            mean12 += data[(j+3)*n2+i+12];
            mean12 += data[(j+4)*n2+i+12];
            mean12 += data[(j+5)*n2+i+12];
            mean12 += data[(j+6)*n2+i+12];
            mean12 += data[(j+7)*n2+i+12];

            mean13 += data[j*n2+i+13];
            mean13 += data[(j+1)*n2+i+13];
            mean13 += data[(j+2)*n2+i+13];
            mean13 += data[(j+3)*n2+i+13];
            mean13 += data[(j+4)*n2+i+13];
            mean13 += data[(j+5)*n2+i+13];
            mean13 += data[(j+6)*n2+i+13];
            mean13 += data[(j+7)*n2+i+13];

            mean14 += data[j*n2+i+14];
            mean14 += data[(j+1)*n2+i+14];
            mean14 += data[(j+2)*n2+i+14];
            mean14 += data[(j+3)*n2+i+14];
            mean14 += data[(j+4)*n2+i+14];
            mean14 += data[(j+5)*n2+i+14];
            mean14 += data[(j+6)*n2+i+14];
            mean14 += data[(j+7)*n2+i+14];

            mean15 += data[j*n2+i+15];
            mean15 += data[(j+1)*n2+i+15];
            mean15 += data[(j+2)*n2+i+15];
            mean15 += data[(j+3)*n2+i+15];
            mean15 += data[(j+4)*n2+i+15];
            mean15 += data[(j+5)*n2+i+15];
            mean15 += data[(j+6)*n2+i+15];
            mean15 += data[(j+7)*n2+i+15];

        }
        mean0 /= n1;
        mean1 /= n1;
        mean2 /= n1;
        mean3 /= n1;
        mean4 /= n1;
        mean5 /= n1;
        mean6 /= n1;
        mean7 /= n1;

        mean8 /= n1;
        mean9 /= n1;
        mean10 /= n1;
        mean11 /= n1;
        mean12 /= n1;
        mean13 /= n1;
        mean14 /= n1;
        mean15 /= n1;
        //mat_set(m, i_tst, i, mean);
        m_data[i_tst*m->n2+i] = mean0;
        m_data[i_tst*m->n2+i+1] = mean1;
        m_data[i_tst*m->n2+i+2] = mean2;
        m_data[i_tst*m->n2+i+3] = mean3;
        m_data[i_tst*m->n2+i+4] = mean4;
        m_data[i_tst*m->n2+i+5] = mean5;
        m_data[i_tst*m->n2+i+6] = mean6;
        m_data[i_tst*m->n2+i+7] = mean7;

        m_data[i_tst*m->n2+i+8] = mean8;
        m_data[i_tst*m->n2+i+9] = mean9;
        m_data[i_tst*m->n2+i+10] = mean10;
        m_data[i_tst*m->n2+i+11] = mean11;
        m_data[i_tst*m->n2+i+12] = mean12;
        m_data[i_tst*m->n2+i+13] = mean13;
        m_data[i_tst*m->n2+i+14] = mean14;
        m_data[i_tst*m->n2+i+15] = mean15;
    }

}



void register_scalar_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &compute_col_mean;
    userFuncs[1] = &compute_col_mean_opt;

}