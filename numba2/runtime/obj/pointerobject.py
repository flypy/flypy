# -*- coding: utf-8 -*-

"""
Pointer implementation.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from ... import jit, typeof
from numba2.runtime.conversion import ctype

@jit('Pointer[a]')
class Pointer(object):
    layout = [] # [('p', 'Pointer[a]')]

    def __init__(self, p):
        self.p = p

    # __________________________________________________________________

    @jit('a -> int32 -> a', opaque=True)
    def __add__(self, index):
        return self.p + index

    @jit('Pointer[a] -> a', opaque=True)
    def deref(self):
        return self.p[0]

    @jit('Pointer[a] -> a -> void', opaque=True)
    def store(self, value):
        self.p[0] = value

    # __________________________________________________________________

    @jit('Pointer[a] -> int32 -> a')
    def __getitem__(self, index):
        return (self + index).deref()

    @jit('Pointer[a] -> int32 -> a -> void')
    def __setitem__(self, idx, value):
        (self + idx).store(value)

    @jit('a -> int64 -> a')
    def __sub__(self, index):
        return self + -index

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
        [base] = ty.parameters
        return ctypes.POINTER(ctype(base))


def make_ctypes_ptr(ptr, type):
    from numba2.cffi_support import is_cffi, ffi
    from numba2.ctypes_support import is_ctypes_pointer_type

    cty = ctype(type)

    if is_cffi(ptr):
        addr = ffi.cast('uintptr_t', ptr)
        ctypes_ptr = ctypes.c_void_p(int(addr))
        ptr = ctypes.cast(ctypes_ptr, cty)
    else:
        ptr = ctypes.cast(ptr, cty)

    return ptr


# TODO: ...
# @typeof.case(ctypes.c_long)
# def typeof(pyval):
#     return Int[32]