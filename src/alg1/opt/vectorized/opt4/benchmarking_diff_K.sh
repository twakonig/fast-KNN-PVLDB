#!/bin/bash

# delete prior .csv files if existent

if [[ -d "results_diff_K" ]]
then
    rm -rf "results_diff_K"/*.csv
else
    mkdir "results_diff_K"
fi

make

# input size
N=512

# KNN parameter
max_K=10

numruns=10


# Add first line
echo "k, cycles" >> results_diff_K/avx_alg1opt4_O3-vec_cycles_diff_K.csv

for ((k = 2; k <= $max_K; k++))
do
  echo "Cycles (-O3 -ffast-math -march=native) for Alg1 opt4, K = $k"
  ./O3-vec-flags-cycles $N $k 0 $numruns >> results_diff_K/avx_alg1opt4_O3-vec_cycles_diff_K.csv
done
