
# -*- coding: utf-8 -*-

"""
int/long implementation.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, typeof
from ..interfaces import Number, implements

@implements('Int[nbits, unsigned]', Number)
class Int(object):
    layout = [('x', 'Int[nbits, unsigned]')]

@typeof.case(int)
def typeof(pyval):
    if isinstance(pyval, bool):
        # TODO: Make this go away... Fix the pyoverload
        from .boolobject import Bool
        return Bool[()]
    return Int[32, False]