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

maxnum=2048
numruns_init=30
numruns=$numruns_init

# store output in .csv files for postprocessing in python
# runtime in cycles for all 3 modes, compilation without flags

# Add first line
# echo "n, cycles" >> results/scalar_alg2opt1_cycles.csv
#
# for ((i = 32; i <= $maxnum; i=2*i))
# do
#   if [ $i == 512 ]
#   then
#     numruns=3
#   fi
#   echo "Cycles (no flags, -non-tree-vectorize) for Alg2 opt1, n = $i"
#   ./no-flags-cycles $i $K 0 $numruns >> results/scalar_alg2opt1_cycles.csv
# done

# runtime in cycles for all 3 modes, compilation with -O3 -fno-tree-vectorize
numruns=$numruns_init

# Add first line
echo "n, cycles" >> results/scalar_alg2opt1_O3_cycles.csv

for ((i = 32; i <= $maxnum; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
  echo "Cycles (-O3 -non-tree-vectorize) for Alg2 opt1, n = $i"
  ./O3-flags-cycles $i $K 0 $numruns >> results/scalar_alg2opt1_O3_cycles.csv
done

# runtime in cycles for all 3 modes, compilation with -O3 -ffast-math -march=native
numruns=$numruns_init

# Add first line
echo "n, cycles" >> results/scalar_alg2opt1_O3-ffm_cycles.csv

for ((i = 32; i <= $maxnum; i=2*i))
do
  if [ $i == 512 ]
    then
      numruns=3
  fi
  echo "Cycles (-O3 -non-tree-vectorize -ffast-math -march=native) for Alg2 opt1, n = $i"
  ./O3-ffm-flags-cycles $i $K 0 $numruns >> results/scalar_alg2opt1_O3-ffm_cycles.csv
done
