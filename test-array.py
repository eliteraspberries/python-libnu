#!/usr/bin/env python

import functools
import numpy

import libnu.array

numpy.linspace = functools.partial(numpy.linspace, dtype=numpy.float32)


def test_maxmin():
    x = numpy.sin(numpy.linspace(0.0, 2.0 * numpy.pi, 1000))
    assert libnu.array.max(x) <= 1.0
    assert libnu.array.min(x) >= -1.0


if __name__ == '__main__':

    test_maxmin()
