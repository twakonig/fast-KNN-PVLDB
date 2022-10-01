#pragma once
#include <immintrin.h>
#include "../include/utils.h"
#include "../include/tsc_x86.h"
#include "common.h"
#include <assert.h>

#define EPSIL (1e-5)


/*// base implementation, not optimized
float l2norm_base(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    for (size_t i = 0; i < len; i++) {
        res += pow(arr1[i] - arr2[i], 2);
    }
    return sqrt(res);
}*/

// omitting the sqrt call
float l2normsquared_base(float arr1[], float arr2[], size_t len){
    float res = 0.0;

    for (size_t i = 0; i < len; i++) {
        res += pow(arr1[i] - arr2[i], 2);
    }
    return res;
}

// remove pow function and use of tmp variable for reused expression | ~ 9.1x
float l2normsq_opt1_fma(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float tmp;
    // 3n flops
    for (size_t i = 0; i < len; i++) {
        tmp = arr1[i] - arr2[i];
        res += tmp * tmp;
    }
    return res;
}

// twosum alg: problem: 2x speedup without flags; 0.5x speedup with flags
float l2normsq_twosum(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float tmp, res_tmp;
    float x, y, z;
    // 3n flops
    for (size_t i = 0; i < len; i++) {
        tmp = arr1[i] - arr2[i];
        res_tmp = tmp * tmp;
        // calculate res+res_tmp
        x = res + res_tmp;
        z = x - res;
        y = ((res - (x - z)) + (res_tmp - z));
        res = x + y;
    }
    return res;
}



// twosum alg: problem: 2x speedup without flags; 0.5x speedup with flags
float l2normsq_fasttwosum(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float tmp, res_tmp;
    float x, y;
    // 6n flops
    for (size_t i = 0; i < len; i++) {
        tmp = arr1[i] - arr2[i];
        res_tmp = tmp * tmp;
        // calculate res+res_tmp
        x = res + res_tmp;
        y = ((res - x) + res_tmp);
        res = x + y;
    }
    return res;
}


// loop unrolling by factor 4
// PROBLEM: separate accumulators introduce a big error => far from stable. try FastTwoSum on this
// CORRECT ORDER OF ADDING UP ACCUMULATORS
float l2normsq_unrolll4(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    //float res0 = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0;
    float res_tmp0, res_tmp1, res_tmp2, res_tmp3;
    float tmp0, tmp1, tmp2, tmp3;

    for (size_t i = 0; i < len; i+=4) {
        // separate accumulators
        //printf("a - b (cancellation?)\n");
        tmp0 = arr1[i] - arr2[i];
        //printf("%lf - %lf = %lf\n", arr1[i], arr2[i], tmp0);
        tmp1 = arr1[i+1] - arr2[i+1];
        tmp2 = arr1[i+2] - arr2[i+2];
        tmp3 = arr1[i+3] - arr2[i+3];

        // intermediate store
        // multiplication not a problem
        res_tmp0 = tmp0 * tmp0;
        res_tmp1 = tmp1 * tmp1;
        res_tmp2 = tmp2 * tmp2;
        res_tmp3 = tmp3 * tmp3;

        // assert
        //assert((res_tmp0 >= (pow(tmp0, 2) - EPSIL)) && (res_tmp0 <= (pow(tmp0, 2) + EPSIL)));
//        assert(res_tmp0 == pow(tmp0, 2));
//        assert(res_tmp1 == pow(tmp1, 2));
//        assert(res_tmp2 == pow(tmp2, 2));
//        assert(res_tmp3 == pow(tmp3, 2));
        //printf("assertion successful\n");


        // partial sum up
        //printf("fma (roundoff?): r = r + t * t\n");
        //printf("r = %lf ", res0);
        /*
        res0 += res_tmp0;
        res1 += res_tmp1;
        res2 += res_tmp2;
        res3 += res_tmp3;*/

        // collect partial sums in a different manner (correct order)
        res = res + res_tmp0 + res_tmp1 + res_tmp2 + res_tmp3;
    }
    // collect partial sums
    //res = res0 + res1 + res2 + res3;
    return res;
}

// loop unrolling by factor 4
// PROBLEM: separate accumulators introduce a big error => far from stable. try FastTwoSum on this
// CORRECT ORDER OF ADDING UP ACCUMULATORS
float l2normsq_unroll4ii(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    //float res0 = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0;
    float res_tmp0, res_tmp1, res_tmp2, res_tmp3;
    float tmp0, tmp1, tmp2, tmp3;

    for (size_t i = 0; i < len; i+=4) {
        // separate accumulators
        //printf("a - b (cancellation?)\n");
        tmp0 = arr1[i] - arr2[i];
        //printf("%lf - %lf = %lf\n", arr1[i], arr2[i], tmp0);
        tmp1 = arr1[i+1] - arr2[i+1];
        tmp2 = arr1[i+2] - arr2[i+2];
        tmp3 = arr1[i+3] - arr2[i+3];

        // intermediate store
        // multiplication not a problem
        res_tmp0 = tmp0 * tmp0;
        res_tmp1 = tmp1 * tmp1;
        res_tmp2 = tmp2 * tmp2;
        res_tmp3 = tmp3 * tmp3;


        // collect partial sums in a different manner (correct order)
        // += GIVES WRONG RESULTS
        res += res_tmp0 + res_tmp1 + res_tmp2 + res_tmp3;
    }
    // collect partial sums
    //res = res0 + res1 + res2 + res3;
    return res;
}

// loop unrolling by factor 4, use FastTwoSum
// PROBLEM: same error es without FastTwoSum
float l2normsq_unroll4_FastTwoSum(float arr1[], float arr2[], size_t len){
    float res;
    float res0 = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0;
    float tmp0, tmp1, tmp2, tmp3;
    float res_tmp0, res_tmp1, res_tmp2, res_tmp3;
    float x0, x1, x2, x3;
    float y0, y1, y2, y3;

    for (size_t i = 0; i < len; i+=4) {
        // separate accumulators
        tmp0 = arr1[i] - arr2[i];
        tmp1 = arr1[i+1] - arr2[i+1];
        tmp2 = arr1[i+2] - arr2[i+2];
        tmp3 = arr1[i+3] - arr2[i+3];

        // intermediate store
        res_tmp0 = tmp0 * tmp0;
        res_tmp1 = tmp1 * tmp1;
        res_tmp2 = tmp2 * tmp2;
        res_tmp3 = tmp3 * tmp3;

        // partial sum up using FastTwoSum (inlined)
        x0 = res0 + res_tmp0;
        y0 = ((res0 - x0) + res_tmp0);
        res0 = x0 + y0;

        x1 = res1 + res_tmp1;
        y1 = ((res1 - x1) + res_tmp1);
        res1 = x1 + y1;

        x2 = res2 + res_tmp2;
        y2 = ((res2 - x2) + res_tmp2);
        res2 = x2 + y2;

        x3 = res3 + res_tmp3;
        y3 = ((res3 - x3) + res_tmp3);
        res3 = x3 + y3;
    }
    // collect partial sums
    res = res0 + res1 + res2 + res3;
    return res;
}

/*// rewrite arithmetic expression
float l2normsq_opt1_rewr1(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float a, b, res_tmp;
    for (size_t i = 0; i < len; i++) {
        // scalar replacement
        a = arr1[i];
        b = arr2[i];
        res_tmp = a*a + b*b - 2.0*a*b;
        res += res_tmp;
    }
    return res;
}*/

// arithmetic expr + precomp
float l2normsq_opt1_rewr2(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float a, b, res_tmp;
    for (size_t i = 0; i < len; i++) {
        // scalar replacement
        a = arr1[i];
        b = arr2[i];
        res_tmp = a*a + b*b;
        res_tmp -= 2.0*a*b;
        res += res_tmp;
    }
    return res;
}

// loop unrolling by factor 8, assume len divisible by 8
float l2normsq_unrolll8(float arr1[], float arr2[], size_t len){
    float res;
    float res0 = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0, res5 = 0.0, res6 = 0.0, res7 = 0.0;
    float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    for (size_t i = 0; i < len; i+=8) {
        // separate accumulators
        tmp0 = arr1[i] - arr2[i];
        tmp1 = arr1[i+1] - arr2[i+1];
        tmp2 = arr1[i+2] - arr2[i+2];
        tmp3 = arr1[i+3] - arr2[i+3];
        tmp4 = arr1[i+4] - arr2[i+4];
        tmp5 = arr1[i+5] - arr2[i+5];
        tmp6 = arr1[i+6] - arr2[i+6];
        tmp7 = arr1[i+7] - arr2[i+7];

        // partial sum up
        res0 += tmp0 * tmp0;
        res1 += tmp1 * tmp1;
        res2 += tmp2 * tmp2;
        res3 += tmp3 * tmp3;
        res4 += tmp4 * tmp4;
        res5 += tmp5 * tmp5;
        res6 += tmp6 * tmp6;
        res7 += tmp7 * tmp7;
    }
    // collect partial sums
    res = res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7;
    return res;
}

void register_test_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    // userFuncs[0] must be function you want to compare to (BASE)!
    userFuncs[0] = &l2normsquared_base;
    userFuncs[1] = &l2normsq_opt1_fma;

}

// ERRONEOUS
/*// remove pow function and use of tmp variable for reused expression | ~ 9.1x
// try scaling
float l2normsq_opt1_mul1(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float tmp, alpha_sq, alpha_inv;
    float alpha = 0.9;
    alpha_sq = alpha * alpha;
    alpha_inv = 1.0 / alpha;

    // 3n flops
    for (size_t i = 0; i < len; i++) {
        tmp = (arr1[i] - arr2[i]) * alpha_inv;
        res += tmp * tmp;
    }
    return res * alpha_sq;
}*/

/*
// loop unrolling by factor 8, assume len divisible by 8 | ~ 15x (8 accum. good according to uops data)
float l2normsq_opt2(float arr1[], float arr2[], size_t len){
    float res;
    float res0 = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0, res5 = 0.0, res6 = 0.0, res7 = 0.0;
    float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

    for (size_t i = 0; i < len; i+=8) {
        // separate accumulators
        tmp0 = arr1[i] - arr2[i];
        tmp1 = arr1[i+1] - arr2[i+1];
        tmp2 = arr1[i+2] - arr2[i+2];
        tmp3 = arr1[i+3] - arr2[i+3];
        tmp4 = arr1[i+4] - arr2[i+4];
        tmp5 = arr1[i+5] - arr2[i+5];
        tmp6 = arr1[i+6] - arr2[i+6];
        tmp7 = arr1[i+7] - arr2[i+7];

        // partial sum up -> later: ENFORCE USE OF FMAs
        res0 += tmp0 * tmp0;
        res1 += tmp1 * tmp1;
        res2 += tmp2 * tmp2;
        res3 += tmp3 * tmp3;
        res4 += tmp4 * tmp4;
        res5 += tmp5 * tmp5;
        res6 += tmp6 * tmp6;
        res7 += tmp7 * tmp7;
    }
    // collect partial sums
    res = res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7;
    return res;
}


// loop unrolling by factor 16
float l2normsq_opt3(float arr1[], float arr2[], size_t len){
    float res;
    float res_i, res_ii, res_iii, res_iv;
    float res0 = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0, res5 = 0.0, res6 = 0.0, res7 = 0.0;
    float res8 = 0.0, res9 = 0.0, res10 = 0.0, res11 = 0.0, res12 = 0.0, res13 = 0.0, res14 = 0.0, res15 = 0.0;
    float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    float tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;

    for (size_t i = 0; i < len; i+=16) {
        // separate accumulators
        tmp0 = arr1[i] - arr2[i];
        tmp1 = arr1[i+1] - arr2[i+1];
        tmp2 = arr1[i+2] - arr2[i+2];
        tmp3 = arr1[i+3] - arr2[i+3];
        tmp4 = arr1[i+4] - arr2[i+4];
        tmp5 = arr1[i+5] - arr2[i+5];
        tmp6 = arr1[i+6] - arr2[i+6];
        tmp7 = arr1[i+7] - arr2[i+7];
        tmp8 = arr1[i+8] - arr2[i+8];
        tmp9 = arr1[i+9] - arr2[i+9];
        tmp10 = arr1[i+10] - arr2[i+10];
        tmp11 = arr1[i+11] - arr2[i+11];
        tmp12 = arr1[i+12] - arr2[i+12];
        tmp13 = arr1[i+13] - arr2[i+13];
        tmp14 = arr1[i+14] - arr2[i+14];
        tmp15 = arr1[i+15] - arr2[i+15];

        // partial sum up -> later: ENFORCE USE OF FMAs
        res0 += tmp0 * tmp0;
        res1 += tmp1 * tmp1;
        res2 += tmp2 * tmp2;
        res3 += tmp3 * tmp3;
        res4 += tmp4 * tmp4;
        res5 += tmp5 * tmp5;
        res6 += tmp6 * tmp6;
        res7 += tmp7 * tmp7;
        res8 += tmp8 * tmp8;
        res9 += tmp9 * tmp9;
        res10 += tmp10 * tmp10;
        res11 += tmp11 * tmp11;
        res12 += tmp12 * tmp12;
        res13 += tmp13 * tmp13;
        res14 += tmp14 * tmp14;
        res15 += tmp15 * tmp15;
    }
    // collect partial sums
    res_i = res0 + res1 + res2 + res3;
    res_ii = res4 + res5 + res6 + res7;
    res_iii = res8 + res9 + res10 + res11;
    res_iv = res12 + res13 + res14 + res15;
    res = res_i + res_ii + res_iii + res_iv;

    //res = res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7 + res8 + res9 + res10 + res11 + res12 + res13 + res14 + res15;
    return res;
}


// loop unrolling by factor 32
float l2normsq_opt4(float arr1[], float arr2[], size_t len){
    float res;
    float res_A, res_B;
    float res_i, res_ii, res_iii, res_iv, res_v, res_vi, res_vii, res_viii;
    float res0 = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0, res5 = 0.0, res6 = 0.0, res7 = 0.0;
    float res8 = 0.0, res9 = 0.0, res10 = 0.0, res11 = 0.0, res12 = 0.0, res13 = 0.0, res14 = 0.0, res15 = 0.0;
    float res16 = 0.0, res17 = 0.0, res18 = 0.0, res19 = 0.0, res20 = 0.0, res21 = 0.0, res22 = 0.0, res23 = 0.0;
    float res24 = 0.0, res25 = 0.0, res26 = 0.0, res27 = 0.0, res28 = 0.0, res29 = 0.0, res30 = 0.0, res31 = 0.0;
    float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    float tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
    float tmp16, tmp17, tmp18, tmp19, tmp20, tmp21, tmp22, tmp23;
    float tmp24, tmp25, tmp26, tmp27, tmp28, tmp29, tmp30, tmp31;

    for (size_t i = 0; i < len; i+=32) {
        // separate accumulators
        tmp0 = arr1[i] - arr2[i];
        tmp1 = arr1[i+1] - arr2[i+1];
        tmp2 = arr1[i+2] - arr2[i+2];
        tmp3 = arr1[i+3] - arr2[i+3];
        tmp4 = arr1[i+4] - arr2[i+4];
        tmp5 = arr1[i+5] - arr2[i+5];
        tmp6 = arr1[i+6] - arr2[i+6];
        tmp7 = arr1[i+7] - arr2[i+7];
        tmp8 = arr1[i+8] - arr2[i+8];
        tmp9 = arr1[i+9] - arr2[i+9];
        tmp10 = arr1[i+10] - arr2[i+10];
        tmp11 = arr1[i+11] - arr2[i+11];
        tmp12 = arr1[i+12] - arr2[i+12];
        tmp13 = arr1[i+13] - arr2[i+13];
        tmp14 = arr1[i+14] - arr2[i+14];
        tmp15 = arr1[i+15] - arr2[i+15];
        tmp16 = arr1[i+16] - arr2[i+16];
        tmp17 = arr1[i+17] - arr2[i+17];
        tmp18 = arr1[i+18] - arr2[i+18];
        tmp19 = arr1[i+19] - arr2[i+19];
        tmp20 = arr1[i+20] - arr2[i+20];
        tmp21 = arr1[i+21] - arr2[i+21];
        tmp22 = arr1[i+22] - arr2[i+22];
        tmp23 = arr1[i+23] - arr2[i+23];
        tmp24 = arr1[i+24] - arr2[i+24];
        tmp25 = arr1[i+25] - arr2[i+25];
        tmp26 = arr1[i+26] - arr2[i+26];
        tmp27 = arr1[i+27] - arr2[i+27];
        tmp28 = arr1[i+28] - arr2[i+28];
        tmp29 = arr1[i+29] - arr2[i+29];
        tmp30 = arr1[i+30] - arr2[i+30];
        tmp31 = arr1[i+31] - arr2[i+31];

        // partial sum up
        res0 += tmp0 * tmp0;
        res1 += tmp1 * tmp1;
        res2 += tmp2 * tmp2;
        res3 += tmp3 * tmp3;
        res4 += tmp4 * tmp4;
        res5 += tmp5 * tmp5;
        res6 += tmp6 * tmp6;
        res7 += tmp7 * tmp7;
        res8 += tmp8 * tmp8;
        res9 += tmp9 * tmp9;
        res10 += tmp10 * tmp10;
        res11 += tmp11 * tmp11;
        res12 += tmp12 * tmp12;
        res13 += tmp13 * tmp13;
        res14 += tmp14 * tmp14;
        res15 += tmp15 * tmp15;
        res16 += tmp16 * tmp16;
        res17 += tmp17 * tmp17;
        res18 += tmp18 * tmp18;
        res19 += tmp19 * tmp19;
        res20 += tmp20 * tmp20;
        res21 += tmp21 * tmp21;
        res22 += tmp22 * tmp22;
        res23 += tmp23 * tmp23;
        res24 += tmp24 * tmp24;
        res25 += tmp25 * tmp25;
        res26 += tmp26 * tmp26;
        res27 += tmp27 * tmp27;
        res28 += tmp28 * tmp28;
        res29 += tmp29 * tmp29;
        res30 += tmp30 * tmp30;
        res31 += tmp31 * tmp31;
    }
    // collect partial sums
    res_i = res0 + res1 + res2 + res3;
    res_ii = res4 + res5 + res6 + res7;
    res_iii = res8 + res9 + res10 + res11;
    res_iv = res12 + res13 + res14 + res15;
    res_A = res_i + res_ii + res_iii + res_iv;
    res_v = res16 + res17 + res18 + res19;
    res_vi = res20 + res21 + res22 + res23;
    res_vii = res24 + res25 + res26 + res27;
    res_viii = res28 + res29 + res30 + res31;
    res_B = res_v + res_vi + res_vii + res_viii;
    res = res_A + res_B;

    return res;
}*/