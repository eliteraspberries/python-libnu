#!/usr/bin/env python

import numpy


def abserror(a, b):
    return numpy.abs(a - b)


def relerror(a, b):
    return abserror(a, b) / max(numpy.abs(a), numpy.abs(b))


def eq(a, b, e):
    if type(a) == numpy.ndarray:
        return all(abserror(a, b) < e)
    return abserror(a, b) < e


if __name__ == '__main__':
    pass
