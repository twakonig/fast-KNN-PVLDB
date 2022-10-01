#pragma once
#include <immintrin.h>
#include "../include/utils.h"
#include "../include/tsc_x86.h"
#include "common.h"

/*---------FILE CONTENT------------
 * all scalar implementations of l2norm function
 * best performing version, to be used for timing: l2normsq_unroll8
 *
 * Optimization techniques used: replacement of library calls, removal of sqrt (not needed), loop unrolling,
 * separate accumulators, scalar replacement
 *
 * !!MAKE SURE NOT TO USE -ffast-math FLAG -> PROBLEMS WITH FLOATING POINT ARITHMETIC!!
 * */

// 1st function registered must be base in terms of cycles
// 2nd function registered must be base in terms of correctness

// base implementation, not optimized
// BASE CYCLES
float l2norm_base(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    for (size_t i = 0; i < len; i++) {
        res += pow(arr1[i] - arr2[i], 2);
    }
    return sqrt(res);
}

// base implementation, not optimized => SQUARED, for validate function
// BASE CORRECTNESS
float l2norm_sqrd_base(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    for (size_t i = 0; i < len; i++) {
        res += pow(arr1[i] - arr2[i], 2);
    }
    return res;
}

// remove pow function and use of tmp variable for reused expression
// 3n flops
// Note: tried with scalar replacement -> caused a slowdown
float l2normsq_simple(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float tmp;
    for (size_t i = 0; i < len; i++) {
        tmp = arr1[i] - arr2[i];
        res += tmp * tmp;
    }
    return res;
}

//-----------------------FUNCTIONS FOR NUMERIC STABILITY OF SUMMATION-------------------------
// twosumalg: problem: 2x speedup without flags; 0.5x speedup with flags
// 9n flops
float l2normsq_TwoSum(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float tmp, res_tmp;
    float x, y, z;
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

// faster than twosum
// 6n flops
float l2normsq_FastTwoSum(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float tmp, res_tmp;
    float x, y;
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
//------------------------------------------------------------------------------------------

// loop unrolling by factor 4
// CORRECT ORDER OF ADDING UP ACCUMULATORS IS VITAL
// (FMA accumulation like: res0 += res_tmp0; yields numerical issues (fp arithmetic)!!!)
float l2normsq_uroll4(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float res_tmp0, res_tmp1, res_tmp2, res_tmp3;
    float tmp0, tmp1, tmp2, tmp3;

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

        // collect intermediate results in THIS ORDER (separate accumulators: failed)
        res = res + res_tmp0 + res_tmp1 + res_tmp2 + res_tmp3;
    }
    return res;
}


// loop unrolling by factor 8
//-----------------------------BEST PERFORMING SCALAR VERSION----------------------------------
// CORRECT ORDER OF ADDING UP ACCUMULATORS IS VITAL
// (FMA accumulation like: res0 += res_tmp0; yields numerical issues (fp arithmetic)!!!)
float l2normsq_uroll8(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float res_tmp0, res_tmp1, res_tmp2, res_tmp3, res_tmp4, res_tmp5, res_tmp6, res_tmp7;
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

        // intermediate store
        res_tmp0 = tmp0 * tmp0;
        res_tmp1 = tmp1 * tmp1;
        res_tmp2 = tmp2 * tmp2;
        res_tmp3 = tmp3 * tmp3;
        res_tmp4 = tmp4 * tmp4;
        res_tmp5 = tmp5 * tmp5;
        res_tmp6 = tmp6 * tmp6;
        res_tmp7 = tmp7 * tmp7;

        // collect intermediate results in THIS ORDER (separate accumulators: failed)
        res = res + res_tmp0 + res_tmp1 + res_tmp2 + res_tmp3 + res_tmp4 + res_tmp5 + res_tmp6 + res_tmp7;
    }
    return res;
}

// loop unrolling by factor 16
// CORRECT ORDER OF ADDING UP ACCUMULATORS IS VITAL
// (FMA accumulation like: res0 += res_tmp0; yields numerical issues (fp arithmetic)!!!)
float l2normsq_uroll16(float arr1[], float arr2[], size_t len){
    float res = 0.0;
    float res_tmp0, res_tmp1, res_tmp2, res_tmp3, res_tmp4, res_tmp5, res_tmp6, res_tmp7;
    float res_tmp8, res_tmp9, res_tmp10, res_tmp11, res_tmp12, res_tmp13, res_tmp14, res_tmp15;
    float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;

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

        // intermediate store
        res_tmp0 = tmp0 * tmp0;
        res_tmp1 = tmp1 * tmp1;
        res_tmp2 = tmp2 * tmp2;
        res_tmp3 = tmp3 * tmp3;
        res_tmp4 = tmp4 * tmp4;
        res_tmp5 = tmp5 * tmp5;
        res_tmp6 = tmp6 * tmp6;
        res_tmp7 = tmp7 * tmp7;
        res_tmp8 = tmp8 * tmp8;
        res_tmp9 = tmp9 * tmp9;
        res_tmp10 = tmp10 * tmp10;
        res_tmp11 = tmp11 * tmp11;
        res_tmp12 = tmp12 * tmp12;
        res_tmp13 = tmp13 * tmp13;
        res_tmp14 = tmp14 * tmp14;
        res_tmp15 = tmp15 * tmp15;

        // collect intermediate results in THIS ORDER (separate accumulators: failed)
        res = res + res_tmp0 + res_tmp1 + res_tmp2 + res_tmp3 + res_tmp4 + res_tmp5 + res_tmp6 + res_tmp7 +
                            res_tmp8 + res_tmp9 + res_tmp10 + res_tmp11 + res_tmp12 + res_tmp13 + res_tmp14 + res_tmp15;
        // alternative
        /*
        res += res_tmp0;
        res += res_tmp1;
        res += res_tmp2;
        res += res_tmp3;
        res += res_tmp4;
        res += res_tmp5;
        res += res_tmp6;
        res += res_tmp7;
        */
    }
    return res;
}

/*
// loop unrolling by factor 32
// not needed -> best performance with factor 8
float l2norm_opt4(float arr1[], float arr2[], size_t len){
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
}
*/

// 1st function registered must be base in terms of cycles
// 2nd function registered must be base in terms of correctness
void register_scalar_functions(functionptr* userFuncs) {
    // be careful not to register more functions than 'nfuncs' entered as command line argument
    userFuncs[0] = &l2norm_base;
    userFuncs[1] = &l2norm_sqrd_base;
    userFuncs[2] = &l2normsq_simple;
    userFuncs[3] = &l2normsq_uroll4;
    userFuncs[4] = &l2normsq_uroll8;
    userFuncs[5] = &l2normsq_uroll16;

}