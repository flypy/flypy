# -*- coding: utf-8 -*-

"""
Dummy None implementation.
"""

from __future__ import print_function, division, absolute_import
import types
from numba2 import overlay
from ... import jit, typeof

@jit
class NoneType(object):
    layout = []

    @jit('a -> bool')
    def __nonzero__(self):
        return False

    @jit('a -> a -> bool')
    def __eq__(self, other):
        return True

    @jit('a -> b -> bool')
    def __eq__(self, other):
        return other is None


NoneValue = NoneType()


@typeof.case(types.NoneType)
def typeof(pyval):
     return NoneType[()]

overlay(None, NoneValue)