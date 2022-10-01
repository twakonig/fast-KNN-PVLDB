## VECTORIZED OPTIMIZATIONS FOR ALG2
- run the shell script "compile_all" to compile the optimizations
contained in each directory
- run the shell script "benchmark_all" to first compile all the
optimizations contained in each directory then run the
corresponding script for benchmarking runtime
---
- opt1 - algorithm 2 with [optimized l2 norm (AVX)]
- opt2 - algorithm 2 with [optimized l2 norm (AVX) + optimized heap (iterative)]
- opt3 - algorithm 2 with [optimized l2 norm (AVX) + optimized heap (iterative) + optimized colmean (AVX)]
- opt4 - algorithm 2 with [optimized l2 norm (AVX) + optimized heap (iterative) + optimized colmean (AVX) + small optimizations for knn utility (manual inlining and subsequent removal of unnecssary array mallocs)]
---
