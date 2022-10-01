#!/bin/bash

# delete output files if they already exist
rm -f speedup_shapley.txt

# create text files
touch speedup_shapley.txt

textfile0="speedup_shapley.txt"

# -------------------set input variables-------------------------
seed=0
# number of functions you want to test
nfuncs0=9
# mode 0: scalar implementation; mode 1: SIMD implementation
mode0=0
# ---------------------------------------------------------------
N_MIN=32
N_MAX=2048
REPS=3

#---------------------------------run scalar version-----------------------------------
# executables
flags0=("./no-flags" "./O3-vec")

# for scalar implementation
printf "N \t cycles base \t opt__1 \t opt__2 \t opt__3 \t opt__4 \t opt__5 \t opt__6 \t opt__6inl \t opt__7\n" >> $textfile0

# call the functions for testing and timing, inputting N from the command line
for (( N=N_MIN; N<=N_MAX; N*=2)); do
  printf '\n' >> $textfile0
  for flag in "${flags0[@]}"; do
    # console inspection of progress
    echo -n -e "N = $N, flag = $flag, REPS = $REPS"
    $flag $N $seed $nfuncs0 $mode0 $REPS >> $textfile0
    printf '\n'
  done
done

