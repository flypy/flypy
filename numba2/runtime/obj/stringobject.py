# -*- coding: utf-8 -*-

"""
String implementation.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, typeof
from .bufferobject import Buffer

@jit
class String(object):
    layout = [('chars', 'Buffer[char]')]


@typeof.case(str)
def typeof(pyval):
    return String[()]