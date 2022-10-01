#!/bin/bash

# delete output files if they already exist
rm -f speedup_table_sq.txt
rm -f speedup_table.txt
rm -f speedup_table_SIMD.txt

# create text files
touch speedup_table_sq.txt
touch speedup_table.txt
touch speedup_table_SIMD.txt

textfile0="speedup_table.txt"
textfile1="speedup_table_SIMD.txt"
textfile2="speedup_table_sq.txt"

# -------------------set input variables-------------------------

N_MIN=32
N_MAX=4096
REPS=100

##---------------------------------run scalar version-----------------------------------

seed=0

# number of functions you want to test
nfuncs0=3
# mode 0: scalar implementation; mode 1: SIMD implementation
mode0=0

# executables
flags0=("./no-flags-knn-utility" "./O3-novec-knn-utility" "./O3-vec-knn-utility")


# for scalar implementation
printf "N \t cycles base \t opt__1 \t opt__1sr \t opt__2 \t opt__2sr \t opt__3 \t opt__3sr \t opt__4 \t opt__4sr\n" >> $textfile0

# call the functions for testing and timing, inputting N from the command line
for (( N=N_MIN; N<=N_MAX; N*=2)); do
  printf '\n' >> $textfile0
  for flag in "${flags0[@]}"; do
    # console inspection of progress
    echo -n -e "N = $N, flag = $flag"
    $flag $N $seed $nfuncs0 $mode0 $REPS >> $textfile0
    printf '\n'
  done
done


##---------------------------------run SIMD version-----------------------------------
## number of functions you want to test
#nfuncs1=5
## mode 0: scalar implementation; mode 1: SIMD implementation
#mode1=1
#flags1=("./no-flags-knn-utility" "./O3-vec-knn-utility" "./O3-SIMD")
#
#
## for SIMD implementation
#printf "N \t cycles base \t vec__1 \t vec__2 \t vec__3 \t vec__4 \t vec__5\n" >> $textfile1
#
## call the functions for testing and timing, inputting N from the command line
#for (( N=N_MIN; N<=N_MAX; N*=2)); do
#  printf '\n' >> $textfile1
#  for flag in "${flags1[@]}"; do
#    # console inspection of progress
#    echo -n -e "N = $N, flag = $flag"
#    $flag $N $seed $nfuncs1 $mode1 $REPS >> $textfile1
#    printf '\n'
#  done
#done

#---------------------------------run EXPERIMENT version-----------------------------------

#nfuncs2=3
#mode2=2
## executables
#flags2=("./no-flags-knn-utility" "./O3-novec-knn-utility" "./O3-vec-knn-utility")
#
#
## for scalar implementation
#printf "N \t cycles base \t opt__1 \t opt__2 \t opt__3 \t opt__2sr \t opt__3 \t opt__3sr \t opt__4 \t opt__4sr\n" >> $textfile2
#
## call the functions for testing and timing, inputting N from the command line
#for (( N=N_MIN; N<=N_MAX; N*=2)); do
#  printf '\n' >> $textfile2
#  if [ $N -gt 1024 ]; then
#    REPS=30
#  fi
#  for flag in "${flags2[@]}"; do
#    # console inspection of progress
#    echo -n -e "N = $N, flag = $flag, REPS = $REPS"
#    $flag $N $seed $nfuncs2 $mode2 $REPS >> $textfile2
#    printf '\n'
#  done
#done
