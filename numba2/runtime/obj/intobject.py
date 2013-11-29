
# -*- coding: utf-8 -*-

"""
int/long implementation.
"""

from __future__ import print_function, division, absolute_import
import math
import ctypes

import numba2
from numba2 import jit, sjit, typeof
from numba2.runtime import formatting
from ..interfaces import Number

@sjit('Int[nbits, unsigned]')
class Int(Number):
    layout = []

    @jit('a -> int64')
    def __int__(self):
        return numba2.cast(self, numba2.int64)

    @jit
    def __str__(self):
        return int_format(self)

    __repr__ = __str__

    # --------------------

    @classmethod
    def toctypes(cls, val, ty):
        return cls.ctype(ty)(val)

    @classmethod
    def fromctypes(cls, val, ty):
        return val

    @classmethod
    def ctype(cls, ty):
        nbits, unsigned = ty.parameters
        return getattr(ctypes, 'c_%sint%d' % ('u' if unsigned else '', nbits))


#===------------------------------------------------------------------===
# Formatters
#===------------------------------------------------------------------===

@jit('int64 -> a')
def getformat(x):
    return "%lld"

@jit('uint64 -> a')
def getformat(x):
    return "%llu"

@jit('a : signed -> int64')
def upcast(x):
    return x

@jit('a : unsigned -> int64')
def upcast(x):
    return x

@jit
def ndigits(x):
    if x == 0:
        return 1
    else:
        n = int(math.log10(abs(x))) + 1
        if x < 0:
            # Include the sign
            n += 1

        return n

@jit
def int_format(x):
    """
    Format an integer:

        - upcast to a (u)int64
        - determine buffer size
        - use snprintf
    """
    x = upcast(x)
    buf = numba2.runtime.obj.core.newbuffer(numba2.char, ndigits(x) + 1)
    formatting.sprintf(buf, getformat(x), x)
    return numba2.String(buf)

#===------------------------------------------------------------------===
# typeof
#===------------------------------------------------------------------===

@typeof.case(int)
def typeof(pyval):
    if isinstance(pyval, bool):
        # TODO: Make this go away... Fix the pyoverload
        from .boolobject import Bool
        return Bool[()]
    return Int[32, False]

