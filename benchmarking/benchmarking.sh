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

maxnum_alg1=4096
maxnum_alg2=2048
numruns_init=30
numruns=$numruns_init

# store output in .csv files for postprocessing in python
# runtime in cycles for all 3 modes, compilation without flags

# Add first line
echo "n, cycles" >> results/alg1_cycles.csv
echo "n, cycles" >> results/alg2_cycles.csv

for ((i = 32; i <= $maxnum_alg1; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
    echo "Cycles (no flags) for Alg1, n = $i"
    ./no-flags-cycles 1 $i $K 0 $numruns >> results/alg1_cycles.csv
done

numruns=$numruns_init

for ((i = 32; i <= $maxnum_alg2; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
    echo "Cycles (no flags) for Alg2, n = $i"
    ./no-flags-cycles 2 $i $K 0 $numruns >> results/alg2_cycles.csv
done

# runtime in cycles for all 3 modes, compilation with -O3 -fno-tree-vectorize
numruns=$numruns_init

# Add first line
echo "n, cycles" >> results/alg1_O3-novec_cycles.csv
echo "n, cycles" >> results/alg2_O3-novec_cycles.csv

for ((i = 32; i <= $maxnum_alg1; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
  echo "Cycles (-O3 -non-tree-vectorize) for Alg1, n = $i"
  ./O3-novec-flags-cycles 1 $i $K 0 $numruns >> results/alg1_O3-novec_cycles.csv
done

numruns=$numruns_init

for ((i = 32; i <= $maxnum_alg2; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
    echo "Cycles (-O3 -fno-tree-vectorize) for Alg2, n = $i"
    ./O3-novec-flags-cycles 2 $i $K 0 $numruns >> results/alg2_O3-novec_cycles.csv
done

# runtime in cycles for all 3 modes, compilation with -O3 -ffast-math -march=native
numruns=$numruns_init

# Add first line
echo "n, cycles" >> results/alg1_O3-vec_cycles.csv
echo "n, cycles" >> results/alg2_O3-vec_cycles.csv

for ((i = 32; i <= $maxnum_alg1; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
  echo "Cycles (-O3 -ffast-math -march=native) for Alg1, n = $i"
  ./O3-vec-flags-cycles 1 $i $K 0 $numruns >> results/alg1_O3-vec_cycles.csv
done

numruns=$numruns_init

for ((i = 32; i <= $maxnum_alg2; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
  echo "Cycles (-O3 -ffast-math -march=native) for Alg2, n = $i"
  ./O3-vec-flags-cycles 2 $i $K 0 $numruns >> results/alg2_O3-vec_cycles.csv
done

# flops for all 3 modes, compilation without flags

# Add first line
# echo "n, flops" >> results/alg1_flops.csv
# echo "n, flops" >> results/alg1KNN_flops.csv
# echo "n, flops" >> results/alg2_flops.csv
#
# echo "Flops (no flags) for Alg1 with KNN"
# ./no-flags-flops 1 $maxnum 0 >> results/alg1_flops.csv
#
# for ((i = 16; i <= $maxnum; i=2*i))
# do
#   echo "Flops (no flags) for Alg2, n = $i"
#   ./no-flags-flops 2 $i 0 >> results/alg2_flops.csv
# done
#

# plotting

# mkdir "plots"
# python3 plot.py
