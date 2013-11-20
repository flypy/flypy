# -*- coding: utf-8 -*-

"""
float/double implementation.
"""

from __future__ import print_function, division, absolute_import
import math
import ctypes

import numba2
from ... import sjit, jit, typeof
from numba2.runtime import formatting
from ..interfaces import Number

@sjit('Float[nbits]', Number)
class Float(Number):
    layout = [('x', 'Float[nbits]')]

    @jit
    def __str__(self):
        return float_format(self)

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
        [nbits] = ty.parameters
        return {32: ctypes.c_float, 64: ctypes.c_double}[nbits]


#===------------------------------------------------------------------===
# Formatters
#===------------------------------------------------------------------===

@jit('float64 -> float64')
def upcast(x):
    return x

@jit
def float_format(x):
    """
    Format an integer:

        - upcast to a double
        - use snprintf
        - resize buffer according to # of bytes written
    """
    return formatting.format_static("%f", upcast(x), 20)

#===------------------------------------------------------------------===
# typeof
#===------------------------------------------------------------------===

@typeof.case(float)
def typeof(pyval):
    return Float[64]