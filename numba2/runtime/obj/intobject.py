# -*- coding: utf-8 -*-

"""
int/long implementation.
"""

from __future__ import print_function, division, absolute_import

from ... import jit, typeof
from ..interfaces import Number, implements

@implements('Int[nbits]', Number)
class Int(object):
    pass

@typeof.case(int)
def typeof(pyval):
    return Int[32]