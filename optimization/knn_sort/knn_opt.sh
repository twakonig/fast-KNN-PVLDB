#!/bin/bash

# delete output files if they already exist
rm -f speedup_table_knn_ffast.txt

# create text files
touch speedup_table_knn_ffast.txt
textfile="speedup_table_knn_ffast.txt"

# -------------------set input variables-------------------------
seed=0
# number of functions you want to test
nfuncs=5
# ---------------------------------------------------------------
N_MIN=20
N_MAX=160

# executables
#flags=("./no-flags-l2" "./O3-novec-l2" "./O3-vec-l2" "./O3-SIMD")
flags=("./no-flags-knn" "./O2-knn" "./O2-knn-native" "./O2-knn-native-novec" "./O3-knn" "./O3-knn-native" "./O3-knn-native-novec")
#"./O3-vec-sort" "./O3-SIMD")

# for scalar implementation
#printf "N \t cycles base \t opt__1 \t opt__1sr \t opt__2 \t opt__2sr \t opt__3 \t opt__3sr \t opt__4 \t opt__4sr\n" >> speedup_table.txt
# for SIMD implementationn
printf "N \t\t\tcycles base\t\t\topt__1\t\topt__2\t\topt__3\t\topt__4" >> $textfile
#\t vec__2 \t vec__3
# call the functions for testing and timing, inputting N from the command line

for (( N=N_MIN; N<=N_MAX; N*=2)); do
  printf '\n' >> $textfile
  for flag in "${flags[@]}"; do
    # console inspection of progress
    echo -n -e "N = $N, flag = $flag"
    $flag "$N" $seed $nfuncs >> $textfile
    printf '\n'
    printf '\n' >> $textfile
  done
done
