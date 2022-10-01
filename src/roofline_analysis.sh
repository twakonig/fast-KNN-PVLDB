#!/bin/bash
make
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./alg1/roofline_analysis/512 -- ./alg1.out 512 512 512 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./alg1/roofline_analysis/1024 -- ./alg1.out 1024 1024 1024 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./alg1/roofline_analysis/2048 -- ./alg1.out 2048 2048 2048 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./alg1/roofline_analysis/4096 -- ./alg1.out 4096 4096 4096 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./alg2/roofline_analysis/128 -- ./alg2.out 128 128 128 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./alg2/roofline_analysis/256 -- ./alg2.out 256 256 256 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./alg2/roofline_analysis/512 -- ./alg2.out 512 512 512 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./alg2/roofline_analysis/1024 -- ./alg2.out 1024 1024 1024 3 1
