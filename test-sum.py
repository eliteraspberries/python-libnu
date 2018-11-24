#!/usr/bin/env python

import functools
import numpy

import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

import libnu.sum

from test import eq

numpy.zeros = functools.partial(numpy.zeros, dtype=numpy.float32)


@hypothesis.given(arrays(numpy.float32, 10, elements=floats(0.0, 1.0)))
def test_sum(x):
    assert eq(libnu.sum.sum(x), numpy.sum(x), 1e-6)


if __name__ == '__main__':

    test_sum()
