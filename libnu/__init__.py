'''ctypes wrapper for the Nu library'''


import ctypes
import distutils.sysconfig
import os


__author__ = 'Mansour Moufid'
__copyright__ = 'Copyright 2018, Mansour Moufid'
__license__ = 'ISC'
__version__ = '0.1'
__email__ = 'mansourmoufid@gmail.com'
__status__ = 'Development'

__all__ = [
    'array',
    'random',
    'sum',
]


libdirs = ['.']
libdir = distutils.sysconfig.get_config_var('LIBDIR')
if libdir is not None:
    libdirs.append(libdir)
names = ['libnu.dll', 'libnu.dylib', 'libnu.so']
for lib in [os.path.join(dir, name) for dir in libdirs for name in names]:
    if os.path.exists(lib):
        break
libnu = ctypes.cdll.LoadLibrary(lib)


class NuTupleFloat(ctypes.Structure):
    '''
    typedef struct {float a, b;} nu_tuplefloat;
    '''
    _fields_ = [
        ('a', ctypes.c_float),
        ('b', ctypes.c_float),
    ]


class NuComplex(ctypes.Structure):
    '''
    typedef struct {float r, i;} nu_complex;
    '''
    _fields_ = [
        ('r', ctypes.c_float),
        ('i', ctypes.c_float),
    ]
