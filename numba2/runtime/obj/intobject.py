# -*- coding: utf-8 -*-

"""
int/long implementation.
"""

from __future__ import print_function, division, absolute_import

from ... import jit, implements, typeof
from ..interfaces import Number

@implements('Int[nbits]', Number)
class Int(object):
    pass

@typeof.case((int, long))
def typeof(pyval):
    return Int[32]