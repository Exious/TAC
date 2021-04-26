import numpy as np


class Utility():
    def concatenateSequences(first, second):
        if first is not None:
            return np.concatenate((first, second), axis=0)
        return second

    def separateArray(arr, over):
        train = []
        test = []
        for i in range(arr.shape[0]):
            if (i+1) % over:
                train.append(arr[i])
            else:
                test.append(arr[i])

        return np.array(train), np.array(test)

    def mergeArrays(first, second, over):
        arr = []
        k = 0

        for i in range(second.shape[0]):
            for j in range(over-1):
                arr.append(first[i+j+k])
            arr.append(second[i])
            k += 1

        return np.array(arr)
