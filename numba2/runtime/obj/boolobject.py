# -*- coding: utf-8 -*-

"""
bool implementation.
"""

from __future__ import print_function, division, absolute_import
from numba2 import jit, typeof

@jit
class Bool(object):
    layout = [('x', 'Bool')]

@typeof.case(bool)
def typeof(pyval):
    return Bool[()]