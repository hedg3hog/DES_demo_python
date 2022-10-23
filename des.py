import numpy as np
import random

from regex import R

def genPseudoRandomKey():
    "Returns a pseudo random 64 bit np.array"
    k = np.array([random.choice((0,1)) for _ in range(64)], dtype="uint8")
    return k.reshape((8,8))

def ip(x:np.array) -> np.array:
    """returns 8x8 array"""
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

def ip_1(x) -> np.array:
    """returns 8x8 array"""
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

def e(x) -> np.array:
    """returns 8x6 array"""
    x = x.reshape((32,))
    a = np.array((32, 1, 2, 3, 4, 5 ,\
                    4, 5, 6, 7 , 8, 9, \
                    8, 9, 10, 11, 12, 13, \
                    12, 13, 14, 15, 16, 17,\
                    16, 17, 18, 19, 20, 21, \
                    20, 21, 22, 23, 24, 25, \
                    24, 25, 26, 27, 28 ,29, \
                    28, 29, 30, 31, 32, 1))
    #print(a.size)
    y = np.zeros((48,) , dtype="uint8")
    for i in range(48):
        #print("ai",(a[i]))
        y[i] = x[(a[i]-1)]
    return y.reshape((8,6))


def xor_48(x,k) -> np.array:
    """returns 2 dim array (8,6)"""
    return np.bitwise_xor(x.reshape((48,1)),k.reshape((48,1))).reshape((8,6))

def s1(x) -> np.array:
    """returns 1 dim array with size 4"""
    a = np.array([[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7], \
                [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8], \
                [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0], \
                [15, 12, 8, 2,4 ,9 ,1 ,7 , 5, 11, 3, 14, 10, 0, 6, 13]], dtype=np.uint8)
    row = np.packbits(np.array((0,0,0,0,0,0,x[0], x[-1]), dtype=np.uint8))[0]
    column = np.packbits(np.append(np.zeros((4,), dtype=np.uint8), x[1:5]))[0]
    return np.unpackbits(a[row][column])[-4:]

def s2(x) -> np.array:
    """returns 1 dim array with size 4"""
    a = np.array([[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10], \
                [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5], \
                [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15], \
                [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]], dtype=np.uint8)

    row = np.packbits(np.array((0,0,0,0,0,0,x[0], x[-1]), dtype=np.uint8))[0]
    column = np.packbits(np.append(np.zeros((4,), dtype=np.uint8), x[1:5]))[0]
    return np.unpackbits(a[row][column])[-4:]
    
def s3(x) -> np.array:
    """returns 1 dim array with size 4"""
    a = np.array([[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8], \
                [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1], \
                [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7], \
                [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]], dtype=np.uint8)
    row = np.packbits(np.array((0,0,0,0,0,0,x[0], x[-1]), dtype=np.uint8))[0]
    column = np.packbits(np.append(np.zeros((4,), dtype=np.uint8), x[1:5]))[0]
    return np.unpackbits(a[row][column])[-4:]

def s4(x) -> np.array:
    """returns 1 dim array with size 4"""
    a = np.array([[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15], \
                [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9], \
                [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4], \
                [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]], dtype=np.uint8)
    row = np.packbits(np.array((0,0,0,0,0,0,x[0], x[-1]), dtype=np.uint8))[0]
    column = np.packbits(np.append(np.zeros((4,), dtype=np.uint8), x[1:5]))[0]
    return np.unpackbits(a[row][column])[-4:]

def s5(x) -> np.array:
    """returns 1 dim array with size 4"""
    a = np.array([[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9], \
                [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6], \
                [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14], \
                [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]], dtype=np.uint8)
    row = np.packbits(np.array((0,0,0,0,0,0,x[0], x[-1]), dtype=np.uint8))[0]
    column = np.packbits(np.append(np.zeros((4,), dtype=np.uint8), x[1:5]))[0]
    return np.unpackbits(a[row][column])[-4:]

def s6(x) -> np.array:
    """returns 1 dim array with size 4"""
    a = np.array([[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11], \
                [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8], \
                [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6], \
                [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]], dtype=np.uint8)
    row = np.packbits(np.array((0,0,0,0,0,0,x[0], x[-1]), dtype=np.uint8))[0]
    column = np.packbits(np.append(np.zeros((4,), dtype=np.uint8), x[1:5]))[0]
    return np.unpackbits(a[row][column])[-4:]
    
def s7(x) -> np.array:
    """returns 1 dim array with size 4"""
    a = np.array([[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1], \
                [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6], \
                [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2], \
                [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]], dtype=np.uint8)
    row = np.packbits(np.array((0,0,0,0,0,0,x[0], x[-1]), dtype=np.uint8))[0]
    column = np.packbits(np.append(np.zeros((4,), dtype=np.uint8), x[1:5]))[0]
    return np.unpackbits(a[row][column])[-4:]

def s8(x) -> np.array:
    """returns 1 dim array with size 4"""
    a = np.array([[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7], \
                [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2], \
                [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8], \
                [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]], dtype=np.uint8)
    row = np.packbits(np.array((0,0,0,0,0,0,x[0], x[-1]), dtype=np.uint8))[0]
    column = np.packbits(np.append(np.zeros((4,), dtype=np.uint8), x[1:5]))[0]
    return np.unpackbits(a[row][column])[-4:]

def p(x) -> np.array:
    """takes and returns 32 bit array"""
    a = np.array([16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, \
                10, 2, 8, 24, 14, 32, 27, 3, 9, 19, 13,30, 6, 22, 11, 4, 25], dtype=np.uint8)
    y = np.zeros((32), dtype=np.uint8)
    for i in range(32):
        y[i] = x[a[i]-1]
    return y


def pc_1(k) -> np.array:
    """returns key in 1-dim array"""
    a = np.array((  57, 49, 41, 33, 25, 17, 9, 1,\
                    58, 50, 42, 34, 26, 19, 10, 2,\
                    59, 51, 43, 35, 27, 19, 11, 3,\
                    60, 52, 44, 36, 63, 55, 47, 39,\
                    31, 23, 15, 7, 62, 54, 46, 38,\
                    30, 22, 14, 6, 61, 53, 45, 37,\
                    29, 21, 13, 5, 28, 20, 12, 4), dtype=np.uint8)
    y = np.zeros((56), dtype=np.uint8)
    for i in range(56):
        y[i] = k[a[i]-1]
    return y

def pc_2(ki) -> np.array:
    """takes 1-dim np array, returns key in 1-dim array"""
    a = np.array((  14, 17, 11, 24, 1, 5, 3, 28,\
                    15, 6, 21, 10, 23, 19, 12, 4,\
                    26, 8, 16, 7, 27, 20, 13, 2,\
                    41, 52, 31, 37, 47, 55, 30, 40,\
                    51, 45, 33, 48, 44, 49, 39, 56,\
                    34, 53, 46, 42, 50, 36, 29, 32), dtype=np.uint8)
    y = np.zeros((48), dtype=np.uint8)
    for i in range(48):
        y[i] = ki[a[i]-1]
    return y


def f(r, k) -> np.array:
    """takes 32 bit key array and 8x6 data array, returns 32 bit array"""
    r = e(r)
    r = xor_48(r,k)
    r1 = s1(r[0])
    r2 = s2(r[1])
    r3 = s3(r[2])
    r4 = s4(r[3])
    r5 = s5(r[4])
    r6 = s6(r[5])
    r7 = s7(r[6])
    r8 = s8(r[7])
    r = np.concatenate((r1, r2, r3, r4, r5, r6, r7, r8))

    return p(r)

def gen_round_keys(k) -> np.array:
    """takes a 64 bit key, returns 16 46-bit round keys"""
    k = np.reshape(k, (64,))
    #print("key before pc-1\n", k)
    k = pc_1(k)
    #print("key after pc-1\n", k)
    c = k[:28]
    d = k[28:]
    #print(c.size, d.size)
    keys = np.zeros((16, 48), dtype=np.uint8)
    for i in range(16):
        if i+1 in (1,2,9,16):
            c = np.roll(c, -1)
            d = np.roll(d, -1)
        else:
            c = np.roll(c, -2)
            d = np.roll(d, -2)
        keys[i] = pc_2(np.concatenate((c,d)))
    return keys

def ascii_to_key(s:str) -> np.array:
    
    x = bin(int.from_bytes(s.encode(), "big"))[2:]
    if len(x) < 64:
        raise ValueError("String to short")
    k = [int(x[i]) for i in range(64)]

    return np.array(k, dtype=np.uint8).reshape((8,8))

def string_to_array(s:str) -> np.array:
    x = np.array([i for i in s])
    x = x.view(np.uint32)
    a = []
    for i in x:
        for c in np.binary_repr(i):
            a.append(c)
    
    while len(a) % 64 != 0:
        a.append(0)
    a = np.array(a, dtype=np.uint8)
    return a.reshape((int(a.size/64),8,8))

def enc_block64(x:np.array, k:np.array) -> np.array:
    x = ip(x)
    keys = gen_round_keys(k)
    l = x[:4].reshape((32))
    r = x[4:].reshape((32))
    for i in range(16):
        l_new = r
        r_new = np.bitwise_xor(l, f(r,keys[i]))
        l = l_new
        r = r_new
    l_end = r 
    r_end = l
    return ip_1(np.concatenate((l_end, r_end)))


def dec_block64(x:np.array, k:np.array) -> np.array:
    x = ip(x)
    keys = gen_round_keys(k)
    l = x[:4].reshape((32))
    r = x[4:].reshape((32))
    for i in range(15,-1,-1):
        l_new = r
        r_new = np.bitwise_xor(l, f(r,keys[i]))
        l = l_new
        r = r_new
    l_end = r 
    r_end = l
    return ip_1(np.concatenate((l_end, r_end)))

def from_file(filename:str) -> np.array:
    """returns a np.array of 8x8 arrays"""
    file = np.fromfile(filename, dtype="uint8")
    a = []
    for b in file:
        for bit in np.binary_repr(b, 8):
            a.append(bit)
    while len(a) % 64 != 0:
        a.append(0)
    a = np.array(a, dtype=np.uint8)
    return a.reshape((int(a.size/64),8,8))

def to_file(filename:str, data_array:np.array):
    """writes data to a file"""
    l = []
    for block in data_array:
        for byte in block:
            byte = np.array2string(byte, separator="")[1:-1]
            byte = int(byte, 2)
            l.append(byte)
    l = np.array(l, dtype=np.uint8)
    l.tofile(filename)

def encrypt(data, key)->np.array:
    """decryts data array of shape (n, 8, 8,) returns array in the same shape"""
    e = np.empty(np.shape(data), dtype=np.uint8)
    for i in range(len(data)):
       e[i] = enc_block64(data[i], key)
    return e

def decypt(data, key)->np.array:
    """decryts data array of shape (n, 8, 8,) returns array in the same shape"""
    d = np.empty(np.shape(data), dtype=np.uint8)
    for i in range(len(data)):
       d[i] = enc_block64(data[i], key)
    return d