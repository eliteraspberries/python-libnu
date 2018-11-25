import ctypes
import functools
import numpy

from . import libnu

numpy.empty = functools.partial(numpy.empty, dtype=numpy.float32)


def ctypeof(x):
    assert isinstance(x, numpy.ndarray)
    if x.dtype == numpy.float32:
        return ctypes.c_float


def addressof(x):
    assert isinstance(x, numpy.ndarray)
    return ctypes.cast(x.ctypes.data, ctypes.POINTER(ctypeof(x)))


'''
float nu_array_max(float [], size_t);
'''
nu_array_max = libnu.nu_array_max
nu_array_max.restype = ctypes.c_float
nu_array_max.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

'''
float nu_array_min(float [], size_t);
'''
nu_array_min = libnu.nu_array_min
nu_array_min.restype = ctypes.c_float
nu_array_min.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

'''
void nu_array_linspace(float [], float, float, size_t);
'''
nu_array_linspace = libnu.nu_array_linspace
nu_array_linspace.restype = None
nu_array_linspace.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_size_t,
]


def max(x):
    return nu_array_max(addressof(x), x.size)


def min(x):
    return nu_array_min(addressof(x), x.size)


def linspace(a, b, n):
    x = numpy.empty(n)
    nu_array_linspace(addressof(x), a, b, n)
    return x
