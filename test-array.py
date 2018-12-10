#!/usr/bin/env python

import functools
import math
import numpy

import hypothesis
import hypothesis.extra.numpy
from hypothesis.strategies import complex_numbers, floats

import libnu.array

from test import eq

arrays = functools.partial(
    hypothesis.extra.numpy.arrays,
    shape=10,
    unique=True,
)
numpy.ones = functools.partial(numpy.ones, dtype=numpy.float32)
numpy.zeros = functools.partial(numpy.zeros, dtype=numpy.float32)
numpy.linspace = functools.partial(numpy.linspace, dtype=numpy.float32)


@hypothesis.given(arrays(dtype=numpy.float32, elements=floats(-1.0, 1.0)))
def test_maxmin(x):
    assert libnu.array.max(x) >= libnu.array.min(x)
    assert libnu.array.min(x) <= libnu.array.max(x)
    assert libnu.array.max(x) == x[libnu.array.argmax(x)]
    assert libnu.array.min(x) == x[libnu.array.argmin(x)]
    assert libnu.array.max(numpy.sin(x)) <= 1.0
    assert libnu.array.min(numpy.sin(x)) >= -1.0


@hypothesis.given(
    arrays(dtype=numpy.float32, elements=floats(-1.0, 1.0)),
    arrays(dtype=numpy.float32, elements=floats(-1.0, 1.0)),
)
def test_add(x, y):
    z = numpy.zeros(x.size)
    assert all(libnu.array.add(x, z) == x)
    assert all(libnu.array.add(x, y) == libnu.array.add(y, x))
    assert eq(libnu.array.add(x, y), x + y, 1e-8)


@hypothesis.given(
    arrays(dtype=numpy.float32, elements=floats(-1.0, 1.0)),
    arrays(dtype=numpy.float32, elements=floats(-1.0, 1.0)),
)
def test_multiply(x, y):
    z = numpy.ones(x.size)
    assert all(libnu.array.multiply(x, z) == x)
    assert all(libnu.array.multiply(x, y) == libnu.array.multiply(y, x))
    assert eq(libnu.array.multiply(x, y), x * y, 1e-8)


@hypothesis.given(
    arrays(dtype=numpy.complex64, elements=complex_numbers(0.0, 1.0)),
    arrays(dtype=numpy.complex64, elements=complex_numbers(0.0, 1.0)),
)
def test_cadd(x, y):
    z = numpy.zeros(x.size, dtype=numpy.complex64)
    assert all(libnu.array.cadd(x, z) == x)
    assert all(libnu.array.cadd(x, y) == libnu.array.cadd(y, x))


@hypothesis.given(
    arrays(dtype=numpy.complex64, elements=complex_numbers(0.0, 1.0)),
    arrays(dtype=numpy.complex64, elements=complex_numbers(0.0, 1.0)),
)
def test_cmul(x, y):
    z = numpy.ones(x.size, dtype=numpy.complex64)
    assert all(libnu.array.cmul(x, z) == x)
    assert all(libnu.array.cmul(x, y) == libnu.array.cmul(y, x))


@hypothesis.given(
    arrays(dtype=numpy.complex64, elements=complex_numbers(0.0, 1.0))
)
def test_conj(x):
    assert all(libnu.array.conj(libnu.array.conj(x)) == x)


@hypothesis.given(
    arrays(dtype=numpy.float32, elements=floats(0.0, 2.0 * math.pi))
)
def test_cossin(x):
    cosx = libnu.array.cos(x)
    sinx = libnu.array.sin(x)
    assert eq(cosx ** 2 + sinx ** 2, 1.0, 1e-4)
    assert eq(cosx, libnu.array.cos(x + 2.0 * math.pi), 1e-4)
    assert eq(sinx, libnu.array.sin(x + 2.0 * math.pi), 1e-4)
    assert eq(cosx, libnu.array.sin(x + math.pi / 2.0), 1e-4)


@hypothesis.given(
    arrays(dtype=numpy.float32, elements=floats(1e-8, math.e))
)
def test_explog(x):
    logx = libnu.array.log(x)
    expx = libnu.array.exp(x)
    assert eq(libnu.array.exp(logx), x, 1e-4)
    assert eq(libnu.array.log(expx), x, 1e-4)
    assert eq(libnu.array.exp(x + 2.0), expx * math.exp(2.0), 1e-4)
    assert eq(libnu.array.log(x * 2.0), logx + math.log(2.0), 1e-4)


@hypothesis.given(floats(0.0, 10.0), floats(20.0, 100.0))
def test_linspace(a, b):
    n = 10000
    x = libnu.array.linspace(a, b, n)
    d = x[1:] - x[:-1]
    assert all(d > 0.0)
    assert numpy.mean(d) <= (b - a) / (n - 1)
    assert numpy.mean(d) >= (b - a) / (n + 1)


if __name__ == '__main__':

    test_maxmin()
    test_add()
    test_multiply()
    test_cadd()
    test_cmul()
    test_conj()
    test_cossin()
    test_explog()
    test_linspace()
