# -*- coding: utf-8 -*-

"""
float/double implementation.
"""

from __future__ import print_function, division, absolute_import

from ... import jit, typeof
from ..interfaces import Number, implements

@implements('Float[nbits]', Number)
class Float(object):
    pass


@typeof.case(float)
def typeof(pyval):
    return Float[64]