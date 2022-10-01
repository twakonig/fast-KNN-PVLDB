#!/bin/bash

# delete output files if they already exist
rm -f out.txt

# create text files
touch out.txt

# try different input sizes
# N=(1000 2000 4000 8000 16000)
# M=(500 1000 2000 4000 8000)
# d=1000
K=1

N=(512, 1024)
M=(512, 1024)
d=256


# store output of C and Python implementation into text files
for i in "${!N[@]}"; do
  printf "${N[$i]} ${M[$i]} $d $K\n" >> out.txt
  printf "Alg1 output\n" >> out.txt
  cd ../../../../
  ./alg1.out ${N[$i]} ${M[$i]} $d $K >> alg1/opt/scalar/opt4/out.txt
  cd alg1/opt/scalar/opt4
  printf "\nAlg1 opt4 output\n" >> out.txt
  ./alg1opt4.out ${N[$i]} ${M[$i]} $d $K >> out.txt
  printf "\n" >> out.txt
done

python3 ./test_compare.py
