#!/usr/bin/env python

import numpy

from test import eq
import libnu.random
numpy.random = libnu.random.Random()

try:
    range = xrange
except:
    pass


def test_random():
    n = 100000
    x = numpy.random.random(n)
    assert eq(numpy.mean(x), 1.0 / 2.0, 1e-3)
    assert eq(numpy.var(x), 1.0 / 12.0, 1e-3)


def test_normal():
    n = 100000
    x = numpy.random.normal(0.0, 1.0, n)
    assert eq(numpy.mean(x), 0.0, 1e-2)
    assert eq(numpy.var(x), 1.0, 1e-2)


if __name__ == '__main__':

    test_random()
    test_normal()
