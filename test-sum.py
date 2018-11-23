#!/usr/bin/env python

import functools
import numpy

import libnu.sum

numpy.zeros = functools.partial(numpy.zeros, dtype=numpy.float32)


def test_sum():
    x = numpy.zeros(1000)
    assert libnu.sum.sum(x) == 0.0


if __name__ == '__main__':

    test_sum()
