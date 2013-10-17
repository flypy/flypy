# -*- coding: utf-8 -*-

"""
float/double implementation.
"""

from __future__ import print_function, division, absolute_import

from ... import sjit, typeof
from ..interfaces import Number

@sjit('Float[nbits]', Number)
class Float(Number):
    layout = [('x', 'Float[nbits]')]

    @staticmethod
    def toctypes(val, ty):
        import ctypes
        [nbits] = ty.parameters
        ctype = {32: ctypes.c_float, 64: ctypes.c_double}[nbits]
        return ctype(val)


@typeof.case(float)
def typeof(pyval):
    return Float[64]