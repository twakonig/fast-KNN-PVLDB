import sys
import math

def nearest_next_multiple_of_8(num):
    return math.ceil(num/8.0) * 8

def compute_T(eps, delta, K):
    return ((1/((K**2)*(eps**2))) * math.log(2*K/delta))

if __name__ == "__main__":
    eps = float(sys.argv[1])
    delta = float(sys.argv[2])
    K = int(sys.argv[3])

    res = compute_T(eps, delta, K)
    print(nearest_next_multiple_of_8(res))
