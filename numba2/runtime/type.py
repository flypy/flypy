# -*- coding: utf-8 -*-

"""
Number interfaces.
"""

from __future__ import print_function, division, absolute_import
import ctypes
from numba2 import jit
from numba2.runtime.classes import dummy_layout

__all__ = ['Type']

@dummy_layout
@jit('Type[a]')
class Type(object):
    layout = [('params', 'a')]

    @jit
    def __init__(self, params):
        self.params = params

    @staticmethod
    def toobject(obj, type):
        return type.parameters[0]

    @classmethod
    def toctypes(cls, val, ty):
        return ctypes.pointer(ctypes.c_int(0))

    @classmethod
    def ctype(cls, ty):
        return ctypes.POINTER(ctypes.c_int)


@jit('Constructor[a]')
class Constructor(object):
    layout = [('ctor', 'a')]

    @jit('Constructor[a] -> Type[b] -> Type[a[b]]', opaque=True)
    def __getitem__(self, item):
        return self.ctor[item]
