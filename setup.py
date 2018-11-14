#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import libnu


_classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: ISC License (ISCL)',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries',
]

_install_requires = [
    'numpy',
]

with open('README.rst', 'r') as f:
    _long_description = f.read()

_metadata = {
    'author':           libnu.__author__,
    'author_email':     libnu.__email__,
    'description':      libnu.__doc__,
    'license':          libnu.__license__,
    'version':          libnu.__version__,
}


if __name__ == '__main__':
    setup(
        classifiers=_classifiers,
        install_requires=_install_requires,
        long_description=_long_description,
        name='LibNu',
        packages=['libnu'],
        scripts=[],
        url='https://github.com/eliteraspberries/python-libnu',
        **_metadata
    )
