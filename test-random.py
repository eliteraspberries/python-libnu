#!/usr/bin/env python

import numpy

import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies

import libnu.random

from test import eq

floats = hypothesis.strategies.floats
random = libnu.random.Random()


@hypothesis.given(floats(0.0, 1.0), floats(1.0, 2.0))
def test_random(a, b):
    n = 100000
    x = random.random(n) * (b - a) + a
    assert eq(numpy.mean(x), (a + b) / 2.0, 1e-2)
    assert eq(numpy.var(x), (b - a) ** 2 / 12.0, 1e-2)


@hypothesis.given(floats(0.0, 1.0), floats(1.0, 2.0))
def test_normal(a, b):
    n = 100000
    x = random.normal(a, b, n)
    assert eq(numpy.mean(x), a, 1e-1)
    assert eq(numpy.var(x), b * b, 1e-1)


if __name__ == '__main__':

    test_random()
    test_normal()
