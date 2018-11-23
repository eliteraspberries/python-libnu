import ctypes
import numpy


def ctypeof(x):
    assert isinstance(x, numpy.ndarray)
    if x.dtype == numpy.float32:
        return ctypes.c_float


def addressof(x):
    assert isinstance(x, numpy.ndarray)
    return ctypes.cast(x.ctypes.data, ctypes.POINTER(ctypeof(x)))
