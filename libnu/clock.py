import ctypes

from . import libnu


'''
int nu_clock_tick(uint64_t *);
'''
nu_clock_tick = libnu.nu_clock_tick
nu_clock_tick.restype = ctypes.c_int
nu_clock_tick.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),
]

'''
int nu_clock_tock(uint64_t *);
'''
nu_clock_tock = libnu.nu_clock_tock
nu_clock_tock.restype = ctypes.c_int
nu_clock_tock.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),
]


def tick():
    t = ctypes.c_uint64()
    if not nu_clock_tick(ctypes.byref(t)) == 0:
        raise RuntimeError
    return t.value
