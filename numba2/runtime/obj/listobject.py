# -*- coding: utf-8 -*-

"""
List implementation.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, typeof
from .pointerobject import Pointer


@jit('List[a]')
class List(object):
    layout = [('buf', 'Pointer[a]')]

@jit
class EmptyList(List):
    layout = []


@typeof.case(list)
def typeof(pyval):
    if pyval:
        types = [typeof(x) for x in pyval]
        if len(set(types)) != 1:
            raise TypeError("Got multiple types for elements, %s" % set(types))
        return List[types[0]]
    else:
        return EmptyList[()]