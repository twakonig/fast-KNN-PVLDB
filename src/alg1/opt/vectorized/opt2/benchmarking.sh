#!/bin/bash

# delete prior .csv files if existent

if [[ -d "results" ]]
then
    rm -rf "results"/*.csv
else
    mkdir "results"
fi

make

# KNN parameter
K=3

maxnum=8192
numruns_init=30
numruns=$numruns_init

# store output in .csv files for postprocessing in python
# runtime in cycles for all 3 modes, compilation without flags

# Add first line
# echo "n, cycles" >> results/avx_alg1opt2_cycles.csv
#
# for ((i = 32; i <= $maxnum; i=2*i))
# do
#   if [ $i == 512 ]
#   then
#     numruns=3
#   fi
#   echo "Cycles (no flags) for Alg1 opt2, n = $i"
#   ./no-flags-cycles $i $K 0 $numruns >> results/avx_alg1opt2_cycles.csv
# done

# runtime in cycles for all 3 modes, compilation with -O3 -fno-tree-vectorize
numruns=$numruns_init

# Add first line
echo "n, cycles" >> results/avx_alg1opt2_O3-novec_cycles.csv

for ((i = 32; i <= $maxnum; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
  echo "Cycles (-O3 -non-tree-vectorize) for Alg1 opt2, n = $i"
  ./O3-novec-flags-cycles $i $K 0 $numruns >> results/avx_alg1opt2_O3-novec_cycles.csv
done

# runtime in cycles for all 3 modes, compilation with -O3 -ffast-math -march=native
numruns=$numruns_init

# Add first line
echo "n, cycles" >> results/avx_alg1opt2_O3-vec_cycles.csv

for ((i = 32; i <= $maxnum; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
  echo "Cycles (-O3 -ffast-math -march=native) for Alg1 opt2, n = $i"
  ./O3-vec-flags-cycles $i $K 0 $numruns >> results/avx_alg1opt2_O3-vec_cycles.csv
done
