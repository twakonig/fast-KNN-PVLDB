#!/bin/bash

# delete output files if they already exist
rm -f out.txt

# create text files
touch out.txt

# try different input sizes
# N=(1000 2000 4000 8000 16000)
# M=(500 1000 2000 4000 8000)
# d=1000
K=3

N=(512, 1024)
M=(512, 1024)
d=(512, 1024)


# store output of C and Python implementation into text files
for i in "${!N[@]}"; do
  printf "${N[$i]} ${M[$i]} ${d[$i]} $K\n" >> out.txt
  printf "Alg1 output\n" >> out.txt
  cd ../../../../
  ./alg1.out ${N[$i]} ${M[$i]} ${d[$i]} $K >> alg1/opt/scalar/opt5/out.txt
  cd alg1/opt/scalar/opt5
  printf "\nAlg1 opt5 output\n" >> out.txt
  ./alg1opt5.out ${N[$i]} ${M[$i]} ${d[$i]} $K >> out.txt
  printf "\n" >> out.txt
done

python3 ./test_compare.py
