## VECTORIZED OPTIMIZATIONS FOR ALG1
- run the shell script "compile_all" to compile the optimizations
contained in each directory
- run the shell script "benchmark_all" to first compile all the
optimizations contained in each directory then run the
corresponding script for benchmarking runtime
---
- opt1 - algorithm 1 with [optimized l2 norm (AVX)]
- opt2 - algorithm 1 with [optimized l2 norm (AVX) + optimized sorting]
- opt3 - algortihm 1 with [optimized l2 norm (AVX) + optimized sorting + optimized get_true_knn]
- opt4 - algortihm 1 with [optimized l2 norm (AVX) + optimized sorting + optimized get_true_knn + optimized compute_single_unweighted_knn_class_shapley]
---
- opt5 - algortihm 1 with [blocking (AVX) in get_true_knn + optimized sorting + optimized compute_single_unweighted_knn_class_shapley]
