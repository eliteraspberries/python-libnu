#!/usr/bin/env python

import functools
import numpy

import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies

import libnu.array

arrays = functools.partial(
    hypothesis.extra.numpy.arrays,
    dtype=numpy.float32,
    unique=True,
)
floats = hypothesis.strategies.floats
numpy.linspace = functools.partial(numpy.linspace, dtype=numpy.float32)


@hypothesis.given(arrays(shape=10, elements=floats(-1.0, 1.0)))
def test_maxmin(x):
    assert libnu.array.max(x) >= libnu.array.min(x)
    assert libnu.array.min(x) <= libnu.array.max(x)
    assert libnu.array.max(numpy.sin(x)) <= 1.0
    assert libnu.array.min(numpy.sin(x)) >= -1.0


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
    test_linspace()
