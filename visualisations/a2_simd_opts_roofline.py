import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#-------------------------------------------PROJECT---------------------------------------------
# machine (s...scalar, v...vector)
# UNITS: FLOPS/CYCLE, FLOPS/BYTE, BYTES/CYCLE, ...
beta = 6.025
pi_s = 2
pi_s_fma = 4
pi_v = 16
pi_v_fma = 32
ridge_s = pi_s / beta
ridge_s_fma = pi_s_fma / beta
ridge_v = pi_v / beta
ridge_v_fma = pi_v_fma / beta

# values on x-axis
# exp_factor = np.linspace(-6, 2, num=9)
# intensity = 2 ** exp_factor
# bandwidth bounds in plots
x_bandw_s = np.geomspace(0.01, ridge_s, num=4)
x_bandw_s_fma = np.geomspace(ridge_s, ridge_s_fma, num=4)
x_bandw_v = np.geomspace(ridge_s_fma, ridge_v, num=4)
x_bandw_v_fma = np.geomspace(ridge_v, ridge_v_fma, num=4)

bandw_s = beta * x_bandw_s
bandw_v = beta * x_bandw_v
bandw_s_fma = beta * x_bandw_s_fma
bandw_v_fma = beta * x_bandw_v_fma

# peak performance bounds in plots
peakperf_s = np.full((9, 1), pi_s)
peakperf_s_fma = np.full((9, 1), pi_s_fma)
peakperf_v = np.full((9, 1), pi_v)
peakperf_v_fma = np.full((9, 1), pi_v_fma)

x_peakperf_s = np.geomspace(ridge_s, 2**28, num=9)
x_peakperf_s_fma = np.geomspace(ridge_s_fma, 2**28, num=9)
x_peakperf_v = np.geomspace(ridge_v, 2**28, num=9)
x_peakperf_v_fma = np.geomspace(ridge_v_fma, 2**28, num=9)

#a2_scalar
# base_128
I_b_128 = 20.2520
P_b_128 = 0.2222
# optimizations_128
I_o1_128 = 12.4760
P_o1_128 = 2.2750
# I_o2_128 = 14303.1000
# P_o2_128 = 1.8472
I_o4_128 = 10318.2000
P_o4_128 = 2.9250

# base_256
I_b_256 = 761.92500
P_b_256 = 0.2167
# optimizations_256
I_o1_256 = 17.5290
P_o1_256 = 3.5111
# I_o2_256 = 680.69700
# P_o2_256 = 2.0028
I_o4_256 = 22879.4000
P_o4_256 = 4.3944

# base_512
I_b_512 = 53.9930
P_b_512 = 0.2111
# optimizations_512
I_o1_512 = 27.9554
P_o1_512 = 4.1639
# I_o2_512 = 22786500.0000
# P_o2_512 = 1.9583
I_o4_512 = 32951.4000
P_o4_512 = 5.1250

# base_1024
I_b_1024 = 92.7160
P_b_1024 = 0.2028
# optimizations_1024
I_o1_1024 = 46.7760
P_o1_1024 = 4.4806
# I_o2_1024 = 173730.0000
# P_o2_1024 = 1.9722
I_o4_1024 = 182463.0000
P_o4_1024 = 4.8778

# lines to connect points
line_base = ([I_b_128, I_b_256, I_b_512, I_b_1024], [P_b_128, P_b_256, P_b_512, P_b_1024])
line_opt1 = ([I_o1_128, I_o1_256, I_o1_512, I_o1_1024], [P_o1_128, P_o1_256, P_o1_512, P_o1_1024])
# line_opt2 = ([I_o2_128, I_o2_256, I_o2_512, I_o2_1024], [P_o2_128, P_o2_256, P_o2_512, P_o2_1024])
line_opt4 = ([I_o4_128, I_o4_256, I_o4_512, I_o4_1024], [P_o4_128, P_o4_256, P_o4_512, P_o4_1024])
#-----------------------------------------------------------------------------------------------

# generate plot
plt.figure(figsize=(12, 12))
#--------------------------------------ROOFLINE----------------------------------------------------------------
#diagonal parts (bandwidth)
plt.plot(x_bandw_s, bandw_s, c='black', linewidth=1.2)
plt.plot(x_bandw_s_fma, bandw_s_fma, c='black', linewidth=1.2)
plt.plot(x_bandw_v, bandw_v, c='black', linewidth=1.2)
plt.plot(x_bandw_v_fma, bandw_v_fma, c='black', linewidth=1.2)
#horizontal parts (peak performance)
plt.plot(x_peakperf_v_fma, peakperf_v_fma, c='dimgrey', linewidth=1.3)
plt.plot(x_peakperf_v, peakperf_v, c='dimgrey',linestyle='dashed', linewidth=1.3)
plt.plot(x_peakperf_s_fma, peakperf_s_fma, c='dimgrey', linewidth=1.3)
plt.plot(x_peakperf_s, peakperf_s, c='dimgrey',linestyle='dashed', linewidth=1.3)
# move title
ax = plt.gca()
ax.text(0.35/2.5, 4.0/2.5, 'DRAM bandwidth (21.7 GB/s)', fontsize=14, rotation=68)
ax.text(2**21 + 8.2, 2.2, 'scalar ADD peak', c='dimgrey', fontsize=14, rotation=0)
ax.text(2**21 + 12.2, 4.5, 'scalar peak', c='dimgrey', fontsize=14, rotation=0)
ax.text(2**21 + 6.4, 17.5, 'vector ADD SP peak', c='dimgrey', fontsize=14, rotation=0)
ax.text(2**21 + 9.4, 35.0, 'vector SP peak', c='dimgrey', fontsize=14, rotation=0)

# dots_128
plt.plot(I_b_128, P_b_128, c='red', marker='o', ms=7,label='C baseline')
plt.plot(I_o1_128, P_o1_128, c='limegreen',marker='o', ms=7,label='l2-norm (SIMD)')
# plt.plot(I_o2_128, P_o2_128, c='fuchsia', marker='o', ms=7,label='l2-norm (SIMD) + heap (iterative)')
plt.plot(I_o4_128, P_o4_128, c='royalblue', marker='o', ms=7,label='l2-norm (SIMD) + heap (iterative) + col_mean (SIMD) + knn_utility (inlining)')

# dots_256
plt.plot(I_b_256, P_b_256, c='red', marker='o', ms=7)
plt.plot(I_o1_256, P_o1_256, c='limegreen',marker='o', ms=7)
# plt.plot(I_o2_256, P_o2_256, c='fuchsia', marker='o', ms=7)
plt.plot(I_o4_256, P_o4_256, c='royalblue', marker='o', ms=7)

# dots_512
plt.plot(I_b_512, P_b_512, c='red', marker='o', ms=7)
plt.plot(I_o1_512, P_o1_512, c='limegreen',marker='o', ms=7)
# plt.plot(I_o2_512, P_o2_512, c='fuchsia', marker='o', ms=7)
plt.plot(I_o4_512, P_o4_512, c='royalblue', marker='o', ms=7)

# dots_1024
plt.plot(I_b_1024, P_b_1024, c='red', marker='o', ms=7)
plt.plot(I_o1_1024, P_o1_1024, c='limegreen',marker='o', ms=7)
# plt.plot(I_o2_1024, P_o2_1024, c='fuchsia', marker='o', ms=7)
plt.plot(I_o4_1024, P_o4_1024, c='royalblue', marker='o', ms=7)

# lines to connect points
plt.plot(line_base[0], line_base[1], c='red')
plt.plot(line_opt1[0], line_opt1[1], c='limegreen')
# plt.plot(line_opt2[0], line_opt2[1], c='fuchsia')
plt.plot(line_opt4[0], line_opt4[1], c='royalblue')
#-----------------------------------------------------------------------------------------------------------------------

plt.yscale("log", base=2)
plt.xscale("log", base=2)

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

# labeling of plot
plt.xlabel("Operational Intensity, I = W/Q [flops/byte]", size=15)
plt.ylabel('Performance, P = W/T [flops/cycle]', rotation=0, size=15)
ax.yaxis.set_label_coords(0.001, 1.02)
plt.title("\n" "\n" "\n"
             "GCC 7.5.0 with flags: " + '-O3 -mfma -ffast-math -march=native -mavx2', loc='left', pad=50, size=16)
plt.legend(loc='upper left', numpoints = 1, handlelength=1, prop={'size': 13})
plt.tight_layout()

# axes adjustments
plt.ylim([0.02, 100])
plt.xlim([0.05, 2**28])

# grid and background
ax.set_facecolor((0.92, 0.92, 0.92))
plt.grid(axis = 'y', color='white', linewidth=1.0)
plt.grid(axis = 'x', color='white', linewidth=1.0)

# one tick per power of two
ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, numticks=30))
ax.yaxis.set_major_locator(ticker.LogLocator(base=2.0, numticks=30))


plt.savefig('./plots/roofline/a2_simd_roofline.pdf')
