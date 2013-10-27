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

    # __________________________________________________________________

    @staticmethod
    def fromobject(ptr, type):
        return Pointer(make_ctypes_ptr(ptr, type))

    @staticmethod
    def toobject(obj, type):
        return obj.ptr

    @classmethod
    def toctypes(cls, val, ty):
        return make_ctypes_ptr(val.p, ty)

    @classmethod
    def ctype(cls, ty):
        [base] = ty.params
        return ctypes.POINTER(ctype(base))


@typeof.case(str)
def typeof(pyval):
    return String[()]