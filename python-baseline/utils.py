import numpy as np

def partition(array, idx, low, high):
  # choose the rightmost element as pivot
  pivot = array[high]

  # pointer for greater element
  i = low - 1

  # traverse through all elements
  # compare each element with pivot
  for j in range(low, high):
    if array[j] <= pivot:
      # if element smaller than pivot is found
      # swap it with the greater element pointed by i
      i = i + 1

      # swapping element at i with element at j
      (array[i], array[j]) = (array[j], array[i])
      (idx[i], idx[j]) = (idx[j], idx[i])

  # swap the pivot element with the greater element specified by i
  (array[i + 1], array[high]) = (array[high], array[i + 1])
  (idx[i + 1], idx[high]) = (idx[high], idx[i + 1])

  # return the position from where partition is done
  return i + 1

# function to perform quicksort
def quickSort(array, idx, low, high):
  if low < high:

    # find pivot element such that
    # element smaller than pivot are on the left
    # element greater than pivot are on the right
    pi = partition(array, idx, low, high)

    # recursive call on the left of pivot
    quickSort(array, idx, low, pi - 1)

    # recursive call on the right of pivot
    quickSort(array, idx, pi + 1, high)

def argsort(array):
    size = len(array)
    tmp = np.copy(array)
    idx = np.arange(size)
    quickSort(tmp, idx, 0, size - 1)

    return idx


if __name__ == "__main__":
    data = np.array([5.0, 5.0, 5.0, 3.0, 3.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    idx = np.arange(len(data))
    print("Unsorted Array")
    print(data)
    print(idx)

    size = len(data)

    idx = argsort(data)

    print('Sorted Array in Ascending Order:')
    print(data)
    print(idx)
