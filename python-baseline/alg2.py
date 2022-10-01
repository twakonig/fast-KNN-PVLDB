from hwcounter import Timer
import numpy as np
from ctypes import CDLL
import sys

# fix seed for RNG
libc = CDLL("libc.so.6")
libc.srand(42)

# implementation of heap data structure
class Heap:
    def __init__(self,K,dist):
        self.K=K
        self.counter = -1
        self.heap = []
        self.changed = 0
        self.dist = dist
        self.heap_dist = {}

    def insert(self,a):
        dist_a = self.dist(a)
        # if a == 11:
        #     pdb.set_trace()
        if self.counter <= self.K-2:
            self.heap.append(a)
            self.heap_dist[a] = dist_a
            self.counter += 1
            self.up(self.counter)
            self.changed = 1
        else:
            if dist_a < self.dist(self.heap[0]):
                self.heap[0] = a
                self.heap_dist[a] = dist_a
                self.down(0)
                self.changed = 1
            else:
                self.changed = 0

    def up(self,index):
        if index == 0:
            return
        parent_index = (index-1)//2
        if self.dist(self.heap[index]) > self.dist(self.heap[parent_index]):
            self.heap[index],self.heap[parent_index] = self.heap[parent_index],self.heap[index]
            self.up(parent_index)
        return

    def down(self,index):
        if 2*index + 1 > self.counter:
            return
        if 2*index + 1 < self.counter:
            if self.dist(self.heap[2*index + 1]) < self.dist(self.heap[2*index + 2]):
                tar_index = 2*index + 2
            else:
                tar_index = 2*index + 1
        else:
            tar_index = 2*index + 1
        if self.dist(self.heap[index]) < self.dist(self.heap[tar_index]):
            self.heap[index],self.heap[tar_index] = self.heap[tar_index],self.heap[index]
            self.down(tar_index)
        return



def unweighted_knn_class_utility_heap(y_trn,y_tst,K):
    # x_tst is a single test point
    utility = np.sum((y_trn==y_tst))/K
    return utility

# permutation function, same as in the C implementatiom such that
# both C and Python produce same results with same number generator
# given the same seed
def shuffle(arr):
     i = len(arr) - 1
     while i >= 0:
         j = libc.rand() % (i + 1)
         tmp = arr[j]
         arr[j] = arr[i]
         arr[i] = tmp
         i -= 1

## single user
def knn_mc_approximation(x_trn,y_trn,x_tst,y_tst,K,T):
    '''
    :param x_trn: training data
    :param y_trn: training label
    :param x_tst: test data
    :param y_tst: test label
    :param utility: utility function that maps a set of training instances to its utility
    :param T: the number of permutations
    :return: estimate of shapley value
    '''

    n_trn = x_trn.shape[0]
    n_tst = x_tst.shape[0]
    sp_approx_all = np.zeros((n_tst, T, n_trn), dtype=np.float32)
    sp_approx =  np.zeros((n_tst, n_trn), dtype=np.float32)
    perm = np.arange(n_trn)
    for n_tst_i in range(n_tst):
        dist = lambda a: np.linalg.norm(x_trn[a,:]-x_tst[n_tst_i,:],ord=2)
        for t in range(T):
            value_now = np.zeros(n_trn)
            shuffle(perm)
            heap = Heap(K=K,dist=dist)
            for k in range(n_trn):
                heap.insert(perm[k])
                # if heap.changed == 1:
                #     trn_dist = np.array([heap.heap_dist[key] for key in heap.heap[:(heap.counter+1)]])
                #     value_now[k] = utility(y_trn[heap.heap[:(heap.counter+1)]], y_tst[n_tst_i], trn_dist)
                # else:
                #     value_now[k] = value_now[k-1]
                if heap.changed == 1:
                    value_now[k] = unweighted_knn_class_utility_heap(y_trn[heap.heap[:(heap.counter + 1)]],
                                                                     y_tst[n_tst_i], K)
                else:
                    value_now[k] = value_now[k-1]
            # compute the marginal contribution of k-th user's data
            sp_approx_all[n_tst_i, t, perm[0]] = value_now[0]
            sp_approx_all[n_tst_i, t, perm[1:]] = value_now[1:] - value_now[0:-1]
            #if t % 100 == 0:
                #print('%s out of %s' % (t, T))
        sp_approx[n_tst_i,:] = np.mean(sp_approx_all[n_tst_i, :, :], axis=0)
    return sp_approx

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
        sp_approx = knn_mc_approximation(X_trn, y_trn, X_tst, y_tst, K, 32)
    # print(f'{N}, {t.cycles} ')
    mean = np.mean(sp_approx, axis=0)

    for i, mean_i in enumerate(mean):
        if i != len(mean) - 1:
            print("{:.8f}".format(mean_i), end=" ")
        else:
            print("{:.8f}".format(mean_i))

if __name__ == "__main__":
    main()
