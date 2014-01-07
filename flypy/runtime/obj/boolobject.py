# -*- coding: utf-8 -*-

"""
bool implementation.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from flypy import sjit, jit, typeof

@sjit
class Bool(object):
    layout = []

    # ----------------------

    @jit('a -> a')
    def __nonzero__(self):
        return self

    @jit
    def __str__(self):
        if self:
            return "True"
        else:
            return "False"

    __repr__ = __str__

    # ----------------------

    @classmethod
    def toctypes(cls, val, ty):
        return cls.ctype(ty)(val)

    @classmethod
    def fromctypes(cls, val, ty):
        return val

    @classmethod
    def ctype(cls, ty):
        return ctypes.c_bool

    @classmethod
    def toobject(cls, value, ty):
        assert value in (0, 1, True, False) or isinstance(value, Bool), value
        return bool(value)


@typeof.case(bool)
def typeof(pyval):
    return Bool[()]