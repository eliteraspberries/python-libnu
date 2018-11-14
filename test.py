#!/usr/bin/env python

import math
import numpy


def eq(a, b, e):
    if type(a) == numpy.ndarray:
        return all(a - b < e)
    return math.fabs(a - b) < e


if __name__ == '__main__':
    pass
