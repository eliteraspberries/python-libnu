import ctypes
import numpy

from . import libnu
from . import array


class NuRandomState(ctypes.Structure):
    '''
    struct nu_random_state {
        uint64_t s0;
        uint64_t s1;
        uint64_t s2;
        uint64_t s3;
    };
    '''
    _fields_ = [
        ('s0', ctypes.c_uint64),
        ('s1', ctypes.c_uint64),
        ('s2', ctypes.c_uint64),
        ('s3', ctypes.c_uint64),
    ]


'''
void nu_random_seed256(struct nu_random_state *, uint64_t [4]);
'''
nu_random_seed256 = libnu.nu_random_seed256
nu_random_seed256.restype = None
nu_random_seed256.argtypes = [
    ctypes.POINTER(NuRandomState),
    ctypes.c_uint64 * 4,
]

'''
void nu_random_seed(struct nu_random_state *, uint64_t);
'''
nu_random_seed = libnu.nu_random_seed
nu_random_seed.restype = None
nu_random_seed.argtypes = [
    ctypes.POINTER(NuRandomState),
    ctypes.c_uint64,
]

'''
uint64_t nu_random(struct nu_random_state *);
'''
nu_random = libnu.nu_random
nu_random.restype = ctypes.c_uint64
nu_random.argtypes = [
    ctypes.POINTER(NuRandomState),
]

'''
float nu_random_float(struct nu_random_state *);
'''
nu_random_float = libnu.nu_random_float
nu_random_float.restype = ctypes.c_float
nu_random_float.argtypes = [
    ctypes.POINTER(NuRandomState),
]

'''
float nu_random_gauss(struct nu_random_state *);
'''
nu_random_gauss = libnu.nu_random_gauss
nu_random_gauss.restype = ctypes.c_float
nu_random_gauss.argtypes = [
    ctypes.POINTER(NuRandomState),
]

'''
void nu_random_array(struct nu_random_state *, uint64_t [], size_t);
'''
nu_random_array = libnu.nu_random_array
nu_random_array.restype = None
nu_random_array.argtypes = [
    ctypes.POINTER(NuRandomState),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_size_t,
]

'''
void nu_random_array_float(struct nu_random_state *, float [], size_t);
'''
nu_random_array_float = libnu.nu_random_array_float
nu_random_array_float.restype = None
nu_random_array_float.argtypes = [
    ctypes.POINTER(NuRandomState),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]

'''
void nu_random_array_gauss(struct nu_random_state *, float [], size_t);
'''
nu_random_array_gauss = libnu.nu_random_array_gauss
nu_random_array_gauss.restype = None
nu_random_array_gauss.argtypes = [
    ctypes.POINTER(NuRandomState),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]


class Random(object):

    def __init__(self, seed=None):
        self.state = NuRandomState()
        if seed is None:
            seed = 0
        self.seed(seed)

    def seed(self, x):
        xs = (ctypes.c_uint64 * 4)(
            x % (2 ** 64) // (2 ** 0),
            x % (2 ** 128) // (2 ** 64),
            x % (2 ** 192) // (2 ** 128),
            x % (2 ** 256) // (2 ** 192),
        )
        libnu.nu_random_seed256(ctypes.byref(self.state), xs)

    def random(self, size=None):
        if size is None:
            x = libnu.nu_random_float(ctypes.byref(self.state))
        else:
            x = numpy.empty(size, dtype=numpy.float32)
            libnu.nu_random_array_float(
                ctypes.byref(self.state),
                array.addressof(x),
                size,
            )
        return x

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            x = libnu.nu_random_gauss(ctypes.byref(self.state))
        else:
            x = numpy.empty(size, dtype=numpy.float32)
            libnu.nu_random_array_gauss(
                ctypes.byref(self.state),
                array.addressof(x),
                size,
            )
        return x * scale + loc
