import ctypes

from . import libnu
from . import array
from . import NuTupleFloat


'''
float nu_sum(float [], size_t);
'''
nu_sum = libnu.nu_sum
nu_sum.restype = ctypes.c_float
nu_sum.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

'''
nu_tuplefloat nu_meanvar(float [], size_t);
'''
nu_meanvar = libnu.nu_meanvar
nu_meanvar.restype = NuTupleFloat
nu_meanvar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]


def sum(x):
    return nu_sum(array.addressof(x), x.size)


def meanvar(x):
    x = nu_meanvar(array.addressof(x), x.size)
    return (x.a, x.b)


def mean(x):
    x, _ = meanvar(x)
    return x


def var(x):
    _, x = meanvar(x)
    return x
