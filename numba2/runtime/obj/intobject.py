
# -*- coding: utf-8 -*-

"""
int/long implementation.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from numba2 import jit, sjit, typeof
from numba2.runtime import formatting
from ..interfaces import Number

@sjit('Int[nbits, unsigned]')
class Int(Number):
    layout = [('x', 'Int[nbits, unsigned]')]

    @jit #('a -> String[]')
    def __str__(self):
        return formatting.int_format(self)

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


@typeof.case(int)
def typeof(pyval):
    if isinstance(pyval, bool):
        # TODO: Make this go away... Fix the pyoverload
        from .boolobject import Bool
        return Bool[()]
    return Int[32, False]