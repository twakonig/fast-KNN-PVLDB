#!/bin/bash
make
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./roofline_analysis/128 -- ./alg2opt1.out 128 128 128 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./roofline_analysis/256 -- ./alg2opt1.out 256 256 256 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./roofline_analysis/512 -- ./alg2opt1.out 512 512 512 3 1
advixe-cl -collect=roofline -stacks -enable-cache-simulation -project-dir=./roofline_analysis/1024 -- ./alg2opt1.out 1024 1024 1024 3 1
