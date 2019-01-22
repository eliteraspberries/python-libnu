import ctypes
import functools
import numpy

from . import libnu
from . import NuComplex


def ctypeof(x):
    assert isinstance(x, numpy.ndarray)
    if x.dtype == numpy.float32:
        return ctypes.c_float
    if x.dtype == numpy.complex64:
        return NuComplex
    return None


def addressof(x):
    assert isinstance(x, numpy.ndarray)
    return ctypes.cast(x.ctypes.data, ctypes.POINTER(ctypeof(x)))


pybuffer_from_memory = ctypes.pythonapi.PyBuffer_FromMemory
pybuffer_from_memory.restype = ctypes.py_object
pybuffer_from_memory.argtypes = [
    ctypes.c_void_p,
    ctypes.c_ssize_t,
]

'''
void *nu_array_alloc(size_t, size_t);
'''
nu_array_alloc = libnu.nu_array_alloc
nu_array_alloc.restype = ctypes.c_void_p
nu_array_alloc.argtypes = [
    ctypes.c_size_t,
    ctypes.c_size_t,
]

'''
void nu_array_free(void *);
'''
nu_array_free = libnu.nu_array_free
nu_array_free.restype = None
nu_array_free.argtypes = [
    ctypes.c_void_p,
]

'''
size_t nu_array_argmax(float [], size_t);
'''
nu_array_argmax = libnu.nu_array_argmax
nu_array_argmax.restype = ctypes.c_size_t
nu_array_argmax.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

'''
size_t nu_array_argmin(float [], size_t);
'''
nu_array_argmin = libnu.nu_array_argmin
nu_array_argmin.restype = ctypes.c_size_t
nu_array_argmin.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

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
void nu_array_cadd(nu_complex [], nu_complex [], nu_complex [], size_t);
'''
nu_array_cadd = libnu.nu_array_cadd
nu_array_cadd.restype = None
nu_array_cadd.argtypes = [
    ctypes.POINTER(NuComplex),
    ctypes.POINTER(NuComplex),
    ctypes.POINTER(NuComplex),
    ctypes.c_size_t,
]

'''
void nu_array_cmul(nu_complex [], nu_complex [], nu_complex [], size_t);
'''
nu_array_cmul = libnu.nu_array_cmul
nu_array_cmul.restype = None
nu_array_cmul.argtypes = [
    ctypes.POINTER(NuComplex),
    ctypes.POINTER(NuComplex),
    ctypes.POINTER(NuComplex),
    ctypes.c_size_t,
]

'''
void nu_array_conj(nu_complex [], nu_complex [], size_t);
'''
nu_array_conj = libnu.nu_array_conj
nu_array_conj.restype = None
nu_array_conj.argtypes = [
    ctypes.POINTER(NuComplex),
    ctypes.POINTER(NuComplex),
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

'''
void nu_array_cos(float [], float [], size_t);
'''
nu_array_cos = libnu.nu_array_cos
nu_array_cos.restype = None
nu_array_cos.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

'''
void nu_array_exp(float [], float [], size_t);
'''
nu_array_exp = libnu.nu_array_exp
nu_array_exp.restype = None
nu_array_exp.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

'''
void nu_array_log(float [], float [], size_t);
'''
nu_array_log = libnu.nu_array_log
nu_array_log.restype = None
nu_array_log.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

'''
void nu_array_sin(float [], float [], size_t);
'''
nu_array_sin = libnu.nu_array_sin
nu_array_sin.restype = None
nu_array_sin.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]


class Array(numpy.ndarray):

    def __init__(self, n, dtype):
        pass

    def __new__(cls, n, dtype):
        stride = numpy.dtype(dtype).itemsize
        ptr = nu_array_alloc(n, stride)
        buffer = pybuffer_from_memory(ptr, n * stride)
        array = super(Array, cls).__new__(cls, n, dtype=dtype, buffer=buffer)
        array.ptr = ptr
        return array

    def __del__(self):
        nu_array_free(self.ptr)


def empty(n, dtype):
    return Array(n, dtype)


def argmax(x):
    return nu_array_argmax(addressof(x), x.size)


def argmin(x):
    return nu_array_argmin(addressof(x), x.size)


def max(x):
    return nu_array_max(addressof(x), x.size)


def min(x):
    return nu_array_min(addressof(x), x.size)


def _binary(cfunction, dtype):
    def wrapcfunction(function):
        @functools.wraps(function)
        def wrapfunction(x, y, out=None):
            n = x.size
            if out is None:
                out = numpy.empty(n, dtype=dtype)
            cfunction(addressof(out), addressof(x), addressof(y), n)
            return out
        return wrapfunction
    return wrapcfunction


def _unary(cfunction, dtype):
    def wrapcfunction(function):
        @functools.wraps(function)
        def wrapfunction(x, out=None):
            n = x.size
            if out is None:
                out = numpy.empty(n, dtype=dtype)
            cfunction(addressof(out), addressof(x), n)
            return out
        return wrapfunction
    return wrapcfunction


@_binary(nu_array_add, numpy.float32)
def add(x, y, out=None):
    pass


@_binary(nu_array_mul, numpy.float32)
def multiply(x, y, out=None):
    pass


@_binary(nu_array_cadd, numpy.complex64)
def cadd(x, y, out=None):
    pass


@_binary(nu_array_cmul, numpy.complex64)
def cmul(x, y, out=None):
    pass


@_unary(nu_array_conj, numpy.complex64)
def conj(x, out=None):
    pass


@_unary(nu_array_cos, numpy.float32)
def cos(x, out=None):
    pass


@_unary(nu_array_exp, numpy.float32)
def exp(x, out=None):
    pass


@_unary(nu_array_log, numpy.float32)
def log(x, out=None):
    pass


@_unary(nu_array_sin, numpy.float32)
def sin(x, out=None):
    pass


def linspace(a, b, n):
    x = numpy.empty(n, dtype=numpy.float32)
    nu_array_linspace(addressof(x), a, b, n)
    return x
