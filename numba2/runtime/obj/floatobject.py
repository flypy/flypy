# -*- coding: utf-8 -*-

"""
float/double implementation.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from ... import sjit, typeof
from ..interfaces import Number

@sjit('Float[nbits]', Number)
class Float(Number):
    layout = [('x', 'Float[nbits]')]

    @classmethod
    def toctypes(cls, val, ty):
        return cls.ctype(ty)(val)

    @classmethod
    def ctype(cls, ty):
        [nbits] = ty.parameters
        return {32: ctypes.c_float, 64: ctypes.c_double}[nbits]


@typeof.case(float)
def typeof(pyval):
    return Float[64]