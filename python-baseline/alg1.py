from hwcounter import Timer
import numpy as np
import utils as ut
import sys

def get_true_KNN(x_trn, x_tst):
    N = x_trn.shape[0]
    N_tst = x_tst.shape[0]
    x_tst_knn_gt = np.zeros((N_tst, N), dtype=np.int64)
    #for i_tst in tqdm(range(N_tst)):
    for i_tst in range(N_tst):
        dist_gt = np.zeros(N, dtype=np.float32)
        for i_trn in range(N):
            dist_gt[i_trn] = np.linalg.norm(x_trn[i_trn, :] - x_tst[i_tst, :], 2)
        x_tst_knn_gt[i_tst, :] = np.argsort(dist_gt, kind='mergesort')
    return x_tst_knn_gt.astype(int)


def compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_tst_knn_gt, y_tst, K):
    N = x_trn.shape[0]

    N_tst = x_tst_knn_gt.shape[0]
    sp_gt = np.zeros((N_tst, N), dtype=np.float32)
    for j in range(N_tst):
        tmp = np.float32(y_trn[x_tst_knn_gt[j, -1]] == y_tst[j])
        sp_gt[j, x_tst_knn_gt[j, -1]] = tmp / N

        for i in np.arange(N - 2, -1, -1):
            tmp1 = sp_gt[j, x_tst_knn_gt[j, i + 1]]
            tmp2 = (int(y_trn[x_tst_knn_gt[j, i]] == y_tst[j])) - int(y_trn[x_tst_knn_gt[j, i + 1]] == y_tst[j])
            sp_gt[j, x_tst_knn_gt[j, i]] = tmp1 + tmp2 / K * np.float32(min([K, i + 1]) / (i + 1))
    return sp_gt

def main():
    N = int(sys.argv[1])
    M = int(sys.argv[2])
    d = int(sys.argv[3])
    K = int(sys.argv[4])

    X_trn = np.load('../data/features_training.npy', mmap_mode='r+')
    X_tst = np.load('../data/features_testing.npy', mmap_mode='r+')
    y_trn = np.load('../data/labels_training.npy', mmap_mode='r+')
    y_tst = np.load('../data/labels_testing.npy', mmap_mode='r+')

    X_trn = np.array(X_trn[:N, :d])
    X_tst = np.array(X_tst[:M, :d])
    y_trn = np.array(y_trn[:N])
    y_tst = np.array(y_tst[:M])

    with Timer() as t:
        X_tst_knn_gt = get_true_KNN(X_trn, X_tst)
        sp_gt = compute_single_unweighted_knn_class_shapley(X_trn, y_trn, X_tst_knn_gt, y_tst, K)
    # print(f'{N}, {t.cycles} ')
    mean = np.mean(sp_gt, axis=0)

    for i, mean_i in enumerate(mean):
        if i != len(mean) - 1:
            print("{:.8f}".format(mean_i), end=" ")
        else:
            print("{:.8f}".format(mean_i))

if __name__ == "__main__":
    main()
