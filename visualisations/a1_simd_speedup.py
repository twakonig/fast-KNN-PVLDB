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


def prepareData(data, N, runs_30):
    cycles = data[:,1]
    # differnetiate between 30 runs and 3 runs per input size
    split_id = 30 * runs_30
    cycles_30 = cycles[0:split_id].reshape(runs_30, 30)
    median1 = np.median(cycles_30, axis=1)
    cycles_3 = cycles[split_id::].reshape(len(N)-runs_30, 3)
    median2 = np.median(cycles_3, axis=1)
    return np.append(median1, median2)


def createSpeedupPlot(N, s1, s2, s3, s4, s5, flags, name):
    # generate plot
    plt.figure(figsize=(12, 12))

    plt.plot(N, s1, marker='o', ms=7, c='goldenrod', linewidth=2.2, label='l2-norm (SIMD)')
    plt.plot(N, s2, marker='o', ms=7, c='darkgreen', linewidth=2.2, linestyle='solid', label='l2-norm (SIMD) + quadsort')
    plt.plot(N, s3, marker='o', ms=7, c='darkblue', linewidth=2.2, linestyle='solid', label='l2-norm (SIMD) + quadsort + get_true_knn (unrolling & referencing) + compute_shapley')
    plt.plot(N, s5, marker='o', ms=7, c='brown', linewidth=2.2, linestyle='solid', label='quadsort + get_true_knn (SIMD blocking) + compute_shapley')

    # move title and labeling
    ax = plt.gca()
    plt.xlabel("Input size N", size=15)
    plt.ylabel('Speedup', rotation=0, size=15)
    ax.yaxis.set_label_coords(0.001, 1.03)
    plt.title("\n" "\n" "\n"
                 "GCC 7.5.0 with flags: " + flags, loc='left', pad=50, size=16)
    plt.legend(prop={'size': 13}, loc='upper left')
    plt.tight_layout()

    # axes adjustments
    plt.ylim([0, 85])
    #plt.xlim([2048, 8192])
    #plt.yscale('log', base=10)
    #plt.yscale('linear')
    plt.xscale('log', base=2)
    # coordinates diplayed on axes
    plt.xticks(N, fontsize=14)
    plt.yticks(fontsize=14)

    # grid and background
    ax.set_facecolor((0.92, 0.92, 0.92))
    plt.grid(axis='y', color='white', linewidth=2.0)
    plt.savefig('./plots/speedup/a1_simd_speedup_' + name + '.pdf')


# return array from 0 to index-1
def take_subset(array, start, end):
    return array[start:end]



# ------------------------------------LOAD DATA--------------------------------------------
#extract data: N = input dimension, t = runtime in cycles

# ID base
base_O3v = pd.read_csv('./data/baselines/alg1_O3-vec_cycles.csv').values

# optimizations
# cheanged to not vec
opt1_v = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt1_O3-vec_cycles.csv').values
opt2_v = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt2_O3-vec_cycles.csv').values
opt3_v = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt3_O3-vec_cycles.csv').values
opt4_v = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt4_O3-vec_cycles.csv').values
opt5_v = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt5_O3-vec_cycles.csv').values

# number of data points with 30 measurement runs (afterwards only 3 runs per input dimension)
base_runs_30 = 5
opt_runs_30 = 4
N = getN(base_O3v)

t_baseO3v = prepareData(base_O3v, N, base_runs_30)

t_o1v = prepareData(opt1_v, N, opt_runs_30)
t_o2v = prepareData(opt2_v, N, opt_runs_30)
t_o3v = prepareData(opt3_v, N, opt_runs_30)
t_o4v = prepareData(opt4_v, N, opt_runs_30)
t_o5v = prepareData(opt5_v, N, opt_runs_30)

speedup1v = t_baseO3v / t_o1v
speedup2v = t_baseO3v / t_o2v
speedup3v = t_baseO3v / t_o3v
speedup4v = t_baseO3v / t_o4v
speedup5v = t_baseO3v / t_o5v

flags_scalar_O3 = '-O3 -mfma -fno-ipa-cp-clone -fno-tree-vectorize'
flags_scalar_O3_ffm = '-O3 -mfma -ffast-math -march=native -fno-ipa-cp-clone -fno-tree-vectorize'
flags_vectorized_O3 = '-O3 -mfma -fno-ipa-cp-clone -mavx2'
flags_vectorized_O3_ffm = '-O3 -mfma -ffast-math -march=native -fno-ipa-cp-clone -mavx2'

# plot the data
createSpeedupPlot(N, speedup1v, speedup2v, speedup3v, speedup4v, speedup5v, flags_vectorized_O3, 'O3')
