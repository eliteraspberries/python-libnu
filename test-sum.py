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


@hypothesis.given(arrays(shape=10, elements=floats))
def test_meanvar(x):
    mean, var = libnu.sum.meanvar(x)
    assert eq(mean, numpy.mean(x), 1e-6)
    assert eq(var, numpy.var(x), 1e-6)


if __name__ == '__main__':

    test_sum()
    test_meanvar()
