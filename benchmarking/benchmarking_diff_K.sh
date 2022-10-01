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
K=(2, 3, 4, 5, 6, 7, 8, 9, 10)
T=(14984, 7112, 4184, 2768, 1976, 1480, 1160, 928, 768)

numruns=10


# Add first line
echo "k, cycles" >> results_diff_K/alg1_O3-vec_cycles_diff_K.csv

for ((k = 2; k <= $max_K; k++))
do
  echo "Cycles (-O3 -ffast-math -march=native) for Alg1, K = $k"
  ./O3-vec-flags-cycles 1 $N $k 0 $numruns >> results_diff_K/alg1_O3-vec_cycles_diff_K.csv
done

# Add first line
echo "k, cycles" >> results_diff_K/alg2_O3-vec_cycles_diff_K.csv

for i in "${!K[@]}";
do
  echo "Cycles (-O3 -ffast-math -march=native) for Alg2, K = ${K[$i]}"
  ./O3-vec-flags-cycles-alg2-diff-K $N ${K[$i]} ${T[$i]} 0 $numruns >> results_diff_K/alg2_O3-vec_cycles_diff_K.csv
done
