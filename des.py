import numpy as np
import random

from sqlalchemy import column


def genPseudoRandomKey():
    "Returns a pseudo random 64 bit np.array"
    k = np.zeros([random.choice((0,1)) for _ in range(64)], dtype="uint8")
    return k.reshape((8,8))

class DES:
    def rm_parity_bits(arr):
        index = [7, 15, 23, 31, 39, 47, 55, 63]
        return np.delete(arr, index)

    def ip(x:np.array):
        x = x.reshape((64,))
        a = np.array((58, 50, 42, 34, 26, 18, 10, 2, \
                60, 52, 44, 36, 28, 20, 12, 4, \
                62, 54, 46, 38, 30, 22, 14, 6, \
                64, 56, 48, 40, 32, 24, 16, 8,\
                57, 49, 41, 33, 25, 17, 9, 1, \
                59, 51, 43, 35, 27, 19, 11, 3, \
                61, 53, 45, 37, 29, 21, 13, 5, \
                63, 55, 47, 39, 31, 23, 15, 7), dtype="uint8")
        y = np.zeros((64,) , dtype="uint8")
        for i in range(64):
           y[i] = x[a[i]-1]
        return y.reshape((8,8))

    def ip_1(x):
        x = x.reshape((64,))
        a = np.array([  40,  8, 48, 16, 56, 24, 64, 32,\
                        39, 7, 47, 15, 55, 23, 63, 31,\
                        38, 6, 46, 14, 54, 22, 62, 30,\
                        37, 5, 45, 13, 53, 21, 61, 29,\
                        36, 4, 44, 12, 52, 20, 60, 28,\
                        35, 3, 43, 11, 51, 19, 59, 27,\
                        34, 2, 42, 10, 50, 18, 58, 26,\
                        33, 1, 41,  9, 49, 17, 57, 25])
        y = np.zeros((64,) , dtype="uint8")
        for i in range(64):
           y[i] = x[a[i]-1]
        return y.reshape((8,8))
    
    def e(x):
        x = x.reshape((32,))
        a = np.array([[32, 1, 2, 3, 4, 5 ,\
                        4, 5, 6, 7 , 8, 9, \
                        8, 9, 10, 11, 12, 13, \
                        16, 17, 18, 19, 20, 21, \
                        20, 21, 22, 23, 24, 25, \
                        24, 25, 26, 27, 28 ,29, \
                        28, 29, 30, 31, 32, 1]])
        y = np.zeros((48,) , dtype="uint8")
        for i in range(48):
           y[i] = x[a[i]-1]
        return y.reshape((6,8))
    
    def xor_48(x,k):
        return np.bitwise_xor(x.reshape((48,)),k.reshape((48,))).reshape((6,8))
    
    def s1(x):
        a = np.array([[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7], \
                    [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8], \
                    [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0], \
                    [15, 12, 8, 2,4 ,9 ,1 ,7 , 5, 11, 3, 14, 10, 0, 6, 13]], dtype=np.uint8)
        row = np.packbits(np.array((0,0,0,0,0,0,x[0], x[-1]), dtype=np.uint8))[0]
        column = np.packbits(np.append(np.zeros((4,), dtype=np.uint8), x[1:5]))[0]
        return np.unpackbits(a[row][column])[-4:]
