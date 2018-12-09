#!/usr/bin/env python

import math
import numpy

import hypothesis
import hypothesis.extra.numpy
from hypothesis.strategies import floats

import libnu.random

from test import eq, relerror

random = libnu.random.Random()

try:
    range = xrange
except:
    pass


@hypothesis.given(floats(0.0, 1.0), floats(1.0, 2.0))
def test_random(a, b):
    n = 100000
    x = random.random(n) * (b - a) + a
    assert eq(numpy.mean(x), (a + b) / 2.0, 1e-2)
    assert eq(numpy.var(x), (b - a) ** 2 / 12.0, 1e-2)


def ecdf(x, ps):
    return [float(x[numpy.abs(x) < p].size) / x.size for p in ps]


def test_random_cdf():
    x = random.random(1000000)
    cdf = [float(i) / 6.0 for i in range(1, 6)]
    edf = ecdf(x, cdf)
    error = [relerror(p, q) for p, q in zip(cdf, edf)]
    assert all([e < 1e-2 for e in error])


@hypothesis.given(floats(0.0, 1.0), floats(1.0, 2.0))
def test_normal(a, b):
    n = 100000
    x = random.normal(a, b, n)
    assert eq(numpy.mean(x), a, 1e-1)
    assert eq(numpy.var(x), b * b, 1e-1)


def test_normal_cdf():
    x = random.normal(0.0, 1.0, 1000000)
    cdf = [math.erf(i / math.sqrt(2.0)) for i in range(1, 6)]
    edf = ecdf(x, range(1, 6))
    error = [relerror(p, q) for p, q in zip(cdf, edf)]
    assert all([e < 1e-2 for e in error])


if __name__ == '__main__':

    test_random()
    test_random_cdf()
    test_normal()
    test_normal_cdf()
