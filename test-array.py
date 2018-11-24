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
floats = hypothesis.strategies.floats(-1.0, 1.0)
numpy.linspace = functools.partial(numpy.linspace, dtype=numpy.float32)


@hypothesis.given(arrays(shape=10, elements=floats))
def test_maxmin(x):
    assert libnu.array.max(numpy.sin(x)) <= 1.0
    assert libnu.array.min(numpy.sin(x)) >= -1.0


if __name__ == '__main__':

    test_maxmin()
