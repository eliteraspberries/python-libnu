import ctypes

from . import libnu
from . import array


'''
float nu_sum(float [], size_t);
'''
nu_sum = libnu.nu_sum
nu_sum.restype = ctypes.c_float
nu_sum.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]


def sum(x):
    return nu_sum(array.addressof(x), x.size)
