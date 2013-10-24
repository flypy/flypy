# -*- coding: utf-8 -*-

"""
Number interfaces.
"""

from __future__ import print_function, division, absolute_import
from numba2 import jit

__all__ = ['Type']

@jit('Type[a]')
class Type(object):
    layout = []

    @staticmethod
    def toobject(obj, type):
        return type.parameters[0]


@jit('Constructor[a]')
class Constructor(object):
    layout = [] #('ctor', 'a')]

    @jit('Constructor[a] -> Type[b] -> Type[a[b]]', opaque=True)
    def __getitem__(self, item):
        raise NotImplementedError
