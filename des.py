import imp
import numpy as np
import random


def getKey():
    k = ""
    for i in range(64):
        k += random.choice(("0","1"))
    return k

def strToArr(s):
    return np.array([int(i) for i in s])


def rm_parity_bits(arr):
    index = [7, 15, 23, 31, 39, 47, 55, 63]
    return np.delete(arr, index)
    