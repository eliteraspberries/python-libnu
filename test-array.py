#!/usr/bin/env python

import functools
import numpy

import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies

import libnu.array

from test import eq

arrays = functools.partial(
    hypothesis.extra.numpy.arrays,
    dtype=numpy.float32,
    unique=True,
)
floats = hypothesis.strategies.floats
numpy.ones = functools.partial(numpy.ones, dtype=numpy.float32)
numpy.zeros = functools.partial(numpy.zeros, dtype=numpy.float32)
numpy.linspace = functools.partial(numpy.linspace, dtype=numpy.float32)


@hypothesis.given(arrays(shape=10, elements=floats(-1.0, 1.0)))
def test_maxmin(x):
    assert libnu.array.max(x) >= libnu.array.min(x)
    assert libnu.array.min(x) <= libnu.array.max(x)
    assert libnu.array.max(numpy.sin(x)) <= 1.0
    assert libnu.array.min(numpy.sin(x)) >= -1.0


@hypothesis.given(
    arrays(shape=10, elements=floats(-1.0, 1.0)),
    arrays(shape=10, elements=floats(-1.0, 1.0)),
)
def test_add(x, y):
    z = numpy.zeros(x.size)
    assert all(libnu.array.add(x, z) == x)
    assert all(libnu.array.add(x, y) == libnu.array.add(y, x))
    assert eq(libnu.array.add(x, y), x + y, 1e-8)


@hypothesis.given(
    arrays(shape=10, elements=floats(-1.0, 1.0)),
    arrays(shape=10, elements=floats(-1.0, 1.0)),
)
def test_multiply(x, y):
    z = numpy.ones(x.size)
    assert all(libnu.array.multiply(x, z) == x)
    assert all(libnu.array.multiply(x, y) == libnu.array.multiply(y, x))
    assert eq(libnu.array.multiply(x, y), x * y, 1e-8)


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
    test_linspace()
