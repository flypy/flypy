
# -*- coding: utf-8 -*-

"""
int/long implementation.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from numba2 import sjit, typeof
from ..interfaces import Number

@sjit('Int[nbits, unsigned]')
class Int(Number):
    layout = [('x', 'Int[nbits, unsigned]')]

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


@typeof.case(int)
def typeof(pyval):
    if isinstance(pyval, bool):
        # TODO: Make this go away... Fix the pyoverload
        from .boolobject import Bool
        return Bool[()]
    return Int[32, False]