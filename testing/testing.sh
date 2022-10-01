#!/bin/bash

# delete output files if they already exist
rm -f alg1_out.txt
rm -f alg2_out.txt

# create text files
touch alg1_out.txt
touch alg2_out.txt


# try different input sizes
# N=(1000 2000 4000 8000 16000)
# M=(500 1000 2000 4000 8000)
# d=1000
K=1

N=(1024 200)
M=(1024 200)
d=60


# store output of C and Python implementation into text files
for i in "${!N[@]}"; do
  printf "${N[$i]} ${M[$i]} $d $K\n" >> alg1_out.txt
  printf "${N[$i]} ${M[$i]} $d $K\n" >> alg2_out.txt
  printf "C output\n" >> alg1_out.txt
  ../src/alg1.out ${N[$i]} ${M[$i]} $d $K >> alg1_out.txt
  printf "\nPython output\n" >> alg1_out.txt
  python ../python-baseline/alg1.py ${N[$i]} ${M[$i]} $d $K >> alg1_out.txt
  printf "C output\n" >> alg2_out.txt
  ../src/alg2.out ${N[$i]} ${M[$i]} $d $K >> alg2_out.txt
  printf "\nPython output\n" >> alg2_out.txt
  python ../python-baseline/alg2.py ${N[$i]} ${M[$i]} $d $K >> alg2_out.txt
done

python ./test_compare.py
