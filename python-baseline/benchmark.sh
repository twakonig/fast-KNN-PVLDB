#!/bin/bash

# delete prior .csv files if existent

if [[ -d "results" ]]
then
    rm -rf "results"/*.csv
else
    mkdir "results"
fi


# KNN parameter
max_N=2048

# kNN parameter
K=3

# number of runs
numruns=10

# store output in .csv files for postprocessing in python

# Add first line
echo "n, cycles" >> results/alg1_cycles.csv
echo "n, cycles" >> results/alg2_cycles.csv

for ((n = 32; n <= $max_N; n=2*n))
do
  echo "Cycles for Alg1, n = $n"
  python3 alg1.py $n $n $n $K >> results/alg1_cycles.csv
done

for ((n = 32; n <= $max_N; n=2*n))
do
  echo "Cycles for Alg2, n = $n"
  python3 alg2.py $n $n $n $K >> results/alg2_cycles.csv
done
