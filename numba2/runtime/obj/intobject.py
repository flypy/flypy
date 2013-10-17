
# -*- coding: utf-8 -*-

"""
int/long implementation.
"""

from __future__ import print_function, division, absolute_import

from numba2 import sjit, typeof
from ..interfaces import Number

@sjit('Int[nbits, unsigned]')
class Int(Number):
    layout = [('x', 'Int[nbits, unsigned]')]

    @staticmethod
    def toctypes(val, ty):
        import ctypes
        nbits, unsigned = ty.parameters
        ctype = getattr(ctypes, 'c_%sint%d' % ('u' if unsigned else '', nbits))
        return ctype(val)


@typeof.case(int)
def typeof(pyval):
    if isinstance(pyval, bool):
        # TODO: Make this go away... Fix the pyoverload
        from .boolobject import Bool
        return Bool[()]
    return Int[32, False]