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


def createRuntimePlot(N, base, baseO3v, y_1, y_2, y_3, y_4, y_5, flags, name):
    # generate plot
    plt.figure(figsize=(12, 12))

    plt.plot(N, base, marker='o', ms=7, c='darkred', linewidth=2.2, linestyle='dashed', label='C baseline: no flags')
    base_s = 'C baseline: -O3 -fno-tree-vectorize'
    base_v = 'C baseline: -O3 -ffast-math -march=native -mavx2'
    plt.plot(N, baseO3v, marker='o', ms=7, c='olive', linewidth=2.2, linestyle='dashed', label=base_s)
    plt.plot(N, y_1, marker='o', ms=7, c='orange', linewidth=2.2, label='l2-norm')
    plt.plot(N, y_2, marker='o', ms=7, c='darkblue', linewidth=2.2, linestyle='solid', label='l2-norm + quadsort')
    #plt.plot(N, y_3, marker='o', ms=7, c='darkgreen', linewidth=2.2, linestyle='dashdot', label='opt3')
    plt.plot(N, y_4, marker='o', ms=7, c='darkgreen', linewidth=2.2, linestyle='solid', label='l2-norm + quadsort + get_true_knn (unrolling & referencing) + compute_shapley')
    plt.plot(N, y_5, marker='o', ms=7, c='purple', linewidth=2.2, linestyle='solid', label='quadsort + get_true_knn (blocking) + compute_shapley')

    # move title and labeling
    ax = plt.gca()
    plt.xlabel("Input size N", size=15)
    plt.ylabel('Runtime [cycles]', rotation=0, size=15)
    ax.yaxis.set_label_coords(0.001, 1.03)
    plt.title("\n" "\n" "\n"
                 "GCC 7.5.0 with flags: " + flags, loc='left', pad=50, size=16)
    plt.legend(prop={'size': 12.45})
    plt.tight_layout()

    # axes adjustments
    #plt.ylim([10**4, 10**14])
    #plt.xlim([2048, 8192])
    #ax.set_ylim([10**4, 10**14])

    plt.yscale('log', base=10)
    #plt.yscale('linear')
    plt.xscale('log', base=2)
    # coordinates diplayed on axes
    extraticks=[10**5, 10**7, 10**9, 10**11, 10**13]
    plt.yticks(list(plt.yticks()[0]) + extraticks)
    plt.yticks(fontsize=14)
    plt.xticks(N, fontsize=14)
    #plt.yticks(np.linspace(pow(10, 10), pow(10, 12), num=10))


    # grid and background
    ax.set_facecolor((0.92, 0.92, 0.92))
    plt.grid(axis='y', color='white', linewidth=2.0)

    ax.set_ylim([10**5, 10**(13.5)])

    plt.savefig('./plots/runtime/a1_scalar_opts_' + name + '.pdf')
    #plt.savefig('./a1_vectorized_opts_' + name + '.pdf')



# return array from 0 to index-1
def take_subset(array, start, end):
    return array[start:end]

# ------------------------------------LOAD DATA--------------------------------------------
#extract data: N = input dimension, t = runtime in cycles

# ID base
base = pd.read_csv('./data/baselines/alg1_cycles.csv').values
base_O3nov = pd.read_csv('./data/baselines/alg1_O3-novec_cycles.csv').values
base_O3v = pd.read_csv('./data/baselines/alg1_O3-vec_cycles.csv').values

# optimizations
opt1 = pd.read_csv('./data/alg1_opts/scalar/scalar_alg1opt1_O3_cycles.csv').values
opt2 = pd.read_csv('./data/alg1_opts/scalar/scalar_alg1opt2_O3_cycles.csv').values
opt3 = pd.read_csv('./data/alg1_opts/scalar/scalar_alg1opt3_O3_cycles.csv').values
opt4 = pd.read_csv('./data/alg1_opts/scalar/scalar_alg1opt4_O3_cycles.csv').values
opt5 = pd.read_csv('./data/alg1_opts/scalar/scalar_alg1opt5_O3_cycles.csv').values

opt1_v = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt1_O3-vec_cycles.csv').values
opt2_v = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt2_O3-vec_cycles.csv').values
opt3_v = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt3_O3-vec_cycles.csv').values
opt4_v = pd.read_csv('./data/alg1_opts/vectorized/avx_alg1opt4_O3-vec_cycles.csv').values


# number of data points with 30 measurement runs (afterwards only 3 runs per input dimension)
runs_30 = 5
opt_runs_30 = 4
N = getN(base)

t_base = prepareData(base, N, runs_30)
t_baseO3 = prepareData(base_O3nov, N, runs_30)
t_baseO3v = prepareData(base_O3v, N, runs_30)

t_o1 = prepareData(opt1, N, opt_runs_30)
t_o2 = prepareData(opt2, N, opt_runs_30)
t_o3 = prepareData(opt3, N, opt_runs_30)
t_o4 = prepareData(opt4, N, opt_runs_30)
t_o5 = prepareData(opt5, N, opt_runs_30)

t_o1v = prepareData(opt1_v, N, opt_runs_30)
t_o2v = prepareData(opt2_v, N, opt_runs_30)
t_o3v = prepareData(opt3_v, N, opt_runs_30)
t_o4v = prepareData(opt4_v, N, opt_runs_30)

# plot subsets
# TO PLOT ONLY FOR BIG OR SMALL N
# id_start = 6
# id_pastEnd = 9
# N = take_subset(N, id_start, id_pastEnd)
# t_base = take_subset(t_base, id_start, id_pastEnd)
# t_baseO3 = take_subset(t_baseO3, id_start, id_pastEnd)
# t_o1 = take_subset(t_o1, id_start, id_pastEnd)
# t_o2 = take_subset(t_o2, id_start, id_pastEnd)
# t_o3 = take_subset(t_o3, id_start, id_pastEnd)
# t_o4 = take_subset(t_o4, id_start, id_pastEnd)
# t_o5 = take_subset(t_o5, id_start, id_pastEnd)

# plot subsets
# TO PLOT ONLY FOR BIG OR SMALL N
# id_start = 6
# id_pastEnd = 9
#N = take_subset(N, id_start, id_pastEnd)
#t_base = take_subset(t_base, id_start, id_pastEnd)
# t_baseO3v = take_subset(t_baseO3v, id_start, id_pastEnd)
# t_o1v = take_subset(t_o1v, id_start, id_pastEnd)
# t_o2v = take_subset(t_o2v, id_start, id_pastEnd)
# t_o3v = take_subset(t_o3v, id_start, id_pastEnd)
# t_o4v = take_subset(t_o4v, id_start, id_pastEnd)

# plot the data
flags_scalar_O3 = '-O3 -mfma -fno-ipa-cp-clone -fno-tree-vectorize'
flags_scalar_O3_ffm = '-O3 -mfma -ffast-math -march=native -fno-ipa-cp-clone -fno-tree-vectorize'
flags_vectorized_O3 = '-O3 -mfma -fno-tree-vectorize -fno-ipa-cp-clone -mavx2'
flags_vectorized_O3_ffm = '-O3 -mfma -ffast-math -march=native -fno-ipa-cp-clone -mavx2'
createRuntimePlot(N, t_base, t_baseO3, t_o1, t_o2, t_o3, t_o4, t_o5, flags_scalar_O3, 'O3')
