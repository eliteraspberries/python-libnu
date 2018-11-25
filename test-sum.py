#!/usr/bin/env python

import functools
import numpy

import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies

import libnu.sum

from test import eq

arrays = functools.partial(
    hypothesis.extra.numpy.arrays,
    dtype=numpy.float32,
    unique=True,
)
floats = hypothesis.strategies.floats(-1.0, 1.0)
numpy.zeros = functools.partial(numpy.zeros, dtype=numpy.float32)


@hypothesis.given(arrays(shape=10, elements=floats))
def test_sum(x):
    assert eq(libnu.sum.sum(x), numpy.sum(x), 1e-6)
    y = numpy.copy(x[::-1])
    assert eq(libnu.sum.sum(x), libnu.sum.sum(y), 1e-6)


@hypothesis.given(arrays(shape=10, elements=floats), floats)
def test_meanvar(x, a):
    mean, var = libnu.sum.meanvar(x)
    assert eq(mean, numpy.mean(x), 1e-6)
    assert eq(var, numpy.var(x), 1e-6)
    y = numpy.copy(x * a)
    assert eq(libnu.sum.mean(x) * a, libnu.sum.mean(y), 1e-6)
    assert eq(libnu.sum.var(x) * a * a, libnu.sum.var(y), 1e-6)


@hypothesis.given(arrays(shape=100, elements=floats))
def test_mean(x):
    assert libnu.sum.mean(x) >= numpy.min(x)
    assert libnu.sum.mean(x) <= numpy.max(x)


@hypothesis.given(arrays(shape=100, elements=floats))
def test_var(x):
    assert libnu.sum.var(x) >= 0.0
    assert libnu.sum.var(x) <= numpy.max(x) - numpy.min(x)


if __name__ == '__main__':

    test_sum()
    test_meanvar()
    test_mean()
    test_var()
