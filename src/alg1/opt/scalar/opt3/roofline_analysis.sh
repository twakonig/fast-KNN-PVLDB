#!/bin/bash
make
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./roofline_analysis/512 -- ./alg1opt3.out 512 512 512 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./roofline_analysis/1024 -- ./alg1opt3.out 1024 1024 1024 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./roofline_analysis/2048 -- ./alg1opt3.out 2048 2048 2048 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./roofline_analysis/4096 -- ./alg1opt3.out 4096 4096 4096 3 1
