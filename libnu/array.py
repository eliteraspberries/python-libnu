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
void nu_array_add(float [], float [], float [], size_t);
'''
nu_array_add = libnu.nu_array_add
nu_array_add.restype = None
nu_array_add.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

'''
void nu_array_mul(float [], float [], float [], size_t);
'''
nu_array_mul = libnu.nu_array_mul
nu_array_mul.restype = None
nu_array_mul.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
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


def _arithmetic(cfunction):
    def wrapcfunction(function):
        @functools.wraps(function)
        def wrapfunction(x, y, out=None):
            n = x.size
            if out is None:
                out = numpy.empty(n)
            cfunction(addressof(out), addressof(x), addressof(y), n)
            return out
        return wrapfunction
    return wrapcfunction


@_arithmetic(nu_array_add)
def add(x, y, out=None):
    pass


@_arithmetic(nu_array_mul)
def multiply(x, y, out=None):
    pass


def linspace(a, b, n):
    x = numpy.empty(n)
    nu_array_linspace(addressof(x), a, b, n)
    return x
