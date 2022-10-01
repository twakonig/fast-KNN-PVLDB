from alg2 import *
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    x_trn = np.arange(20).reshape(4, 5)
    y_trn = np.resize([1.0, 0.0], 4)
    x_tst = np.arange(10).reshape(2, 5)
    y_tst = np.resize([0.0, 1.0], 2)

    # we are using 1-nn classifier
    K = 1

    # compute ground truth with exact algorithm
    x_tst_knn_gt = get_true_KNN(x_trn, x_tst)
    sp_gt = compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_tst_knn_gt, y_tst, K)



    # # -------------------------------------------write matrix to console----------------------------------------------
    # sp_approx_130 = knn_mc_approximation(x_trn, y_trn, x_tst, y_tst, K, 130)
    # sp_approx_40 = knn_mc_approximation(x_trn, y_trn, x_tst, y_tst, K, 40)
    # print('--------------------shapley exact, ALG1--------------------------')
    # print(sp_gt)
    # print('--------------------shapley MC approx, ALG2, T = 130 --------------------------')
    # print(sp_approx_130)
    # print('--------------------shapley MC approx, ALG2, T = 40 --------------------------')
    # print(sp_approx_40)
    # # ----------------------------------------------------------------------------------------------------------------


    # -----------------------------------------------write matrix to file-----------------------------------------------
    # try different values for T (#permutations)
    T_val = [40, 80, 130]
    sp_approx_0 = knn_mc_approximation(x_trn, y_trn, x_tst, y_tst, K, T_val[0])
    sp_approx_1 = knn_mc_approximation(x_trn, y_trn, x_tst, y_tst, K, T_val[1])
    sp_approx_2 = knn_mc_approximation(x_trn, y_trn, x_tst, y_tst, K, T_val[2])

    # create file output
    header = ['row, col', 'alg1 (exact)', 'alg2 (T = ' + str(T_val[2]) + ')', '|delta|    ',
              'alg2 (T = ' + str(T_val[1]) + ')', '|delta|    ', 'alg2 (T = ' + str(T_val[0]) + ')', '|delta|    ']

    # TODO: compute error (delta of mean of colwise sums) and inspect
    file = open("compare_algs.txt", "w")
    file.write("{:} {: >22} {: >30} {: >25} {: >15} {: >15} {: >15} {: >15}".format(*header) + '\n')
    for i in range(sp_gt.shape[0]):
        for j in range(sp_gt.shape[1]):
            entries = [str(i) + ', ' + str(j), str("{:10.20f}".format(sp_gt[i][j])), str("{:10.20f}".format(sp_approx_2[i][j])),
                       str("{:10.4f}".format(abs(sp_gt[i][j] - sp_approx_2[i][j]))), str("{:10.4f}".format(sp_approx_1[i][j])),
                       str("{:10.4f}".format(abs(sp_gt[i][j] - sp_approx_1[i][j]))), str("{:10.4f}".format(sp_approx_0[i][j])),
                       str("{:10.4f}".format(abs(sp_gt[i][j] - sp_approx_0[i][j])))]
            file.write("{:} {: >30} {: >30} {: >17} {: >15} {: >15} {: >15} {: >15}".format(*entries) + '\n')
    file.close()
    # ----------------------------------------------------------------------------------------------------------------
