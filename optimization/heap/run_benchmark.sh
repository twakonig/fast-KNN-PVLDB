#!/bin/bash

# delete output files if they already exist
rm -f benchmark_result.txt

# create text files
touch benchmark_result.txt
textfile="benchmark_result.txt"

# -------------------set input variables-------------------------
seed=0
# number of functions you want to test
nfuncs=2
# number of nearest neighbours
K=3
# ---------------------------------------------------------------
N_MIN=16
N_MAX=256

# executables
flags=("./benchmark-no-flags-heap-opt" "./benchmark-O2-heap-opt" "./benchmark-O3-heap-opt")

# for scalar implementation
printf "N \t\t cycles base  \t\t\t heap_opt1" >> $textfile

# call the functions for testing and timing, inputting N from the command line

for (( N=N_MIN; N<=N_MAX; N*=2)); do
  printf '\n' >> $textfile
  for flag in "${flags[@]}"; do
    # console inspection of progress
    echo -n -e "N = $N, K = $K, flag = $flag"
    $flag "$N" "$K" $seed $nfuncs >> $textfile
    printf '\n'
    printf '\n' >> $textfile
  done
done
