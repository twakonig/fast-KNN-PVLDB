import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def getN(data):
    n = (data[:, 0])
    m = len(n) - 1
    min_pow = int(math.log(n[0], 2))
    max_pow = int(math.log(n[m], 2))
    pows = np.arange(min_pow, max_pow+1)
    base = np.full(len(pows), 2)
    return np.power(base, pows)


def prepareData_py(data, N):
    cycles = data[:,1]
    cycles = cycles.reshape(len(N), 3)
    return np.median(cycles, axis=1)


def prepareData(data, N, runs_30):
    cycles = data[:,1]
    # differnetiate between 30 runs and 3 runs per input size
    split_id = 30 * runs_30
    cycles_30 = cycles[0:split_id].reshape(runs_30, 30)
    median1 = np.median(cycles_30, axis=1)
    cycles_3 = cycles[split_id::].reshape(len(N)-runs_30, 3)
    median2 = np.median(cycles_3, axis=1)
    return np.append(median1, median2)


def createSpeedupPlot(N, scalar, simd, flags):
    # generate plot
    plt.figure(figsize=(9, 8))

    plt.plot(N, scalar, marker='o', ms=7, c='darkgreen', linewidth=2.2, label='fastest scalar implementation')
    plt.plot(N, simd, marker='o', ms=7, c='darkblue', linewidth=2.2, linestyle='solid', label='fastest SIMD implementation')

    # move title and labeling
    ax = plt.gca()
    plt.xlabel("Input size N (=N_tst =d)")
    plt.ylabel('Speedup', rotation=0)
    ax.yaxis.set_label_coords(0.001, 1.03)
    ax.set_title("Intel Core i7-11700K @ 3.60 GHz \n L1: 48 KB, L2: 512 KB, L3 (shared): 16 MB \n "
                 "Compiler: GCC 7.5.0, OS: Ubuntu 18.04 \n  Flags:" + flags, loc='left', pad=50)
    plt.legend()
    plt.tight_layout()

    # axes adjustments
    # plt.ylim([0.07, 0.08])
    #plt.xlim([2048, 8192])
    #plt.yscale('log', base=10)
    #plt.yscale('linear')
    plt.xscale('log', base=2)
    # coordinates diplayed on axes
    plt.xticks(N)
    plt.yticks()

    # grid and background
    ax.set_facecolor((0.92, 0.92, 0.92))
    plt.grid(axis='y', color='white', linewidth=2.0)
    plt.savefig('./a1_overall_speedup.png')


# return array from 0 to index-1
def take_subset(array, start, end):
    return array[start:end]



# ------------------------------------LOAD DATA--------------------------------------------
#extract data: N = input dimension, t = runtime in cycles

# python
#python = pd.read_csv('./data/python/alg1_cycles_py.csv').values
# ID base
base_O3nov = pd.read_csv('./data/baselines/alg1_O3-novec_cycles.csv').values
base_O3v = pd.read_csv('./data/baselines/alg1_O3-vec_cycles.csv').values
base = pd.read_csv('./data/baselines/alg1_O3-vec_cycles.csv').values

# optimizations
# best scalar
scalar_opt5 = pd.read_csv('./data/alg1_opts/scalar/scalar_alg1opt5_O3_cycles.csv').values
# best simd
simd_opt4 = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt4_O3-vec_cycles.csv').values


# number of data points with 30 measurement runs (afterwards only 3 runs per input dimension)
base_runs_30 = 5
opt_runs_30 = 4
#N_py = getN(python)
N = getN(base_O3nov)

#t_python = prepareData_py(python, N_py)
t_base = prepareData(base, N, base_runs_30)
t_scalar = prepareData(scalar_opt5, N, opt_runs_30)
t_simd = prepareData(simd_opt4, N, opt_runs_30)

# # crop data if plot against python
# t_base = take_subset(t_base, 0, len(N_py))
# t_scalar = take_subset(t_scalar, 0, len(N_py))
# t_simd = take_subset(t_simd, 0, len(N_py))

speedup_scalar = t_base / t_scalar
speedup_simd = t_base / t_simd
#speedup_python = t_base / t_python

# plot the data
flags_scalar_O3 = '-fno-tree-vectorize -mfma -O3'
flags_scalar_O3_ffm = '-fno-tree-vectorize -mfma -O3 -ffast-math -march=native'
flags_vectorized_O3 = '-fno-tree-vectorize -mfma -O3 -mavx2'
flags_vectorized_O3_ffm = '-mfma -O3 -ffast-math -march=native -mavx2'
createSpeedupPlot(N, speedup_scalar, speedup_simd, '-O3 -ffast-math -march=native')
