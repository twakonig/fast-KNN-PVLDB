#pragma once
#include <immintrin.h>
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "common.h"


// Renamed with a "_" added at the end, to avoid double declaration
float unweighted_knn_utility_basline(int* y_trn, int y_tst, int size, int K){
    float sum = 0.0;
    for (int i = 0; i < size; i++){
        sum += (y_trn[i] == y_tst) ? 1.0 : 0.0;
    }
    return sum / K;
}
// un-rollin', rollin', rollin'
float unweighted_knn_utility_opt1(int* y_trn, int y_tst, int size, int K){
    float sum, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14;
    float tmp1 = 0.0;
    float tmp2 = 0.0;
    float tmp3 = 0.0;
    float tmp4 = 0.0;
    float tmp5 = 0.0;
    float tmp6 = 0.0;
    float tmp7 = 0.0;
    float tmp8 = 0.0;
    float tmp9 = 0.0;
    float tmp10 = 0.0;
    float tmp11 = 0.0;
    float tmp12 = 0.0;
    float tmp13 = 0.0;
    float tmp14 = 0.0;
    float tmp15 = 0.0;
    float tmp16 = 0.0;

    for (int i = 0; i < size; i+=16){
        tmp1 += (y_trn[i] == y_tst);
        tmp2 += (y_trn[i+1] == y_tst);
        tmp3 += (y_trn[i+2] == y_tst);
        tmp4 += (y_trn[i+3] == y_tst);
        tmp5 += (y_trn[i+4] == y_tst);
        tmp6 += (y_trn[i+5] == y_tst);
        tmp7 += (y_trn[i+6] == y_tst);
        tmp8 += (y_trn[i+7] == y_tst);
        tmp9 += (y_trn[i+8] == y_tst);
        tmp10 += (y_trn[i+9] == y_tst);
        tmp11 += (y_trn[i+10] == y_tst);
        tmp12 += (y_trn[i+11] == y_tst);
        tmp13 += (y_trn[i+12] == y_tst);
        tmp14 += (y_trn[i+13] == y_tst);
        tmp15 += (y_trn[i+14] == y_tst);
        tmp16 += (y_trn[i+15] == y_tst);
    }
    sum1 = tmp1 + tmp2;
    sum2 = tmp3 + tmp4;
    sum3 = tmp5 + tmp6;
    sum4 = tmp7 + tmp8;
    sum5 = tmp9 + tmp10;
    sum6 = tmp11 + tmp12;
    sum7 = tmp13 + tmp14;
    sum8 = tmp15 + tmp16;

    sum9 = sum1 + sum2;
    sum10 = sum3 + sum4;
    sum11 = sum5 + sum6;
    sum12 = sum7 + sum8;

    sum13 = sum9 + sum10;
    sum14 = sum11 + sum12;

    sum = sum13 + sum14;

    return sum / K;
}

// they see me un-rolling, they hatin'
float unweighted_knn_utility_opt2(int* y_trn, int y_tst, int size, int K){
    float sum, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14;
    float sum15, sum16, sum17, sum18, sum19, sum20, sum21, sum22, sum23, sum24, sum25, sum26, sum27, sum28, sum29, sum30;
    float tmp1 = 0.0;
    float tmp2 = 0.0;
    float tmp3 = 0.0;
    float tmp4 = 0.0;
    float tmp5 = 0.0;
    float tmp6 = 0.0;
    float tmp7 = 0.0;
    float tmp8 = 0.0;
    float tmp9 = 0.0;
    float tmp10 = 0.0;
    float tmp11 = 0.0;
    float tmp12 = 0.0;
    float tmp13 = 0.0;
    float tmp14 = 0.0;
    float tmp15 = 0.0;
    float tmp16 = 0.0;

    float tmp17 = 0.0;
    float tmp18 = 0.0;
    float tmp19 = 0.0;
    float tmp20 = 0.0;
    float tmp21 = 0.0;
    float tmp22 = 0.0;
    float tmp23 = 0.0;
    float tmp24 = 0.0;
    float tmp25 = 0.0;
    float tmp26 = 0.0;
    float tmp27 = 0.0;
    float tmp28 = 0.0;
    float tmp29 = 0.0;
    float tmp30 = 0.0;
    float tmp31 = 0.0;
    float tmp32 = 0.0;

    for (int i = 0; i < size; i+=32){
        tmp1 += (y_trn[i] == y_tst);
        tmp2 += (y_trn[i+1] == y_tst);
        tmp3 += (y_trn[i+2] == y_tst);
        tmp4 += (y_trn[i+3] == y_tst);
        tmp5 += (y_trn[i+4] == y_tst);
        tmp6 += (y_trn[i+5] == y_tst);
        tmp7 += (y_trn[i+6] == y_tst);
        tmp8 += (y_trn[i+7] == y_tst);
        tmp9 += (y_trn[i+8] == y_tst);
        tmp10 += (y_trn[i+9] == y_tst);
        tmp11 += (y_trn[i+10] == y_tst);
        tmp12 += (y_trn[i+11] == y_tst);
        tmp13 += (y_trn[i+12] == y_tst);
        tmp14 += (y_trn[i+13] == y_tst);
        tmp15 += (y_trn[i+14] == y_tst);
        tmp16 += (y_trn[i+15] == y_tst);

        tmp17 += (y_trn[i+16] == y_tst);
        tmp18 += (y_trn[i+17] == y_tst);
        tmp19 += (y_trn[i+18] == y_tst);
        tmp20 += (y_trn[i+19] == y_tst);
        tmp21 += (y_trn[i+20] == y_tst);
        tmp22 += (y_trn[i+21] == y_tst);
        tmp23 += (y_trn[i+22] == y_tst);
        tmp24 += (y_trn[i+23] == y_tst);
        tmp25 += (y_trn[i+24] == y_tst);
        tmp26 += (y_trn[i+25] == y_tst);
        tmp27 += (y_trn[i+26] == y_tst);
        tmp28 += (y_trn[i+27] == y_tst);
        tmp29 += (y_trn[i+28] == y_tst);
        tmp30 += (y_trn[i+29] == y_tst);
        tmp31 += (y_trn[i+30] == y_tst);
        tmp32 += (y_trn[i+31] == y_tst);
    }
    sum1 = tmp1 + tmp2;
    sum2 = tmp3 + tmp4;
    sum3 = tmp5 + tmp6;
    sum4 = tmp7 + tmp8;
    sum5 = tmp9 + tmp10;
    sum6 = tmp11 + tmp12;
    sum7 = tmp13 + tmp14;
    sum8 = tmp15 + tmp16;

    sum9 = tmp17 + tmp18;
    sum10 = tmp19 + tmp20;
    sum11 = tmp21 + tmp22;
    sum12 = tmp23 + tmp24;
    sum13 = tmp25 + tmp26;
    sum14 = tmp27 + tmp28;
    sum15 = tmp29 + tmp30;
    sum16 = tmp31 + tmp32;

    sum17 = sum1 + sum2;
    sum18 = sum3 + sum4;
    sum19 = sum5 + sum6;
    sum20 = sum7 + sum8;

    sum21 = sum9 + sum10;
    sum22 = sum11 + sum12;
    sum23 = sum13 + sum14;
    sum24 = sum15 + sum16;

    sum25 = sum17 + sum18;
    sum26 = sum19 + sum20;

    sum27 = sum21 + sum22;
    sum28 = sum23 + sum24;

    sum29 = sum25 + sum26;
    sum30 = sum27+sum28;

    sum = sum29 + sum30;

    return sum / K;
}


void register_scalar_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &unweighted_knn_utility_basline;
    userFuncs[1] = &unweighted_knn_utility_opt1;
    userFuncs[2] = &unweighted_knn_utility_opt2;

}