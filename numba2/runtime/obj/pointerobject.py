# -*- coding: utf-8 -*-

"""
Pointer implementation.
"""

from __future__ import print_function, division, absolute_import
import ctypes

import numba2
from numba2 import sjit, jit
from numba2.compiler import representation_type
from numba2.conversion import ctype
from ..lowlevel_impls import add_impl_cls

from pykit import types as ptypes

#===------------------------------------------------------------------===
# Pointer
#===------------------------------------------------------------------===

@sjit('Pointer[a]')
class Pointer(object):
    layout = [] # [('p', 'Pointer[a]')]

    def __init__(self, p):
        self.p = p

    # __________________________________________________________________

    @jit('a -> int64 -> a', opaque=True)
    def __add__(self, index):
        return self.p + index

    @jit('Pointer[a] -> a', opaque=True)
    def deref(self):
        return self.p[0]

    @jit('Pointer[a] -> a -> void', opaque=True)
    def store(self, value):
        self.p[0] = value

    # __________________________________________________________________

    @jit('a -> Pointer[b] -> bool')
    def __eq__(self, other):
        val1 = numba2.cast(self, numba2.int64)
        val2 = numba2.cast(other, numba2.int64)
        return val1 == val2

    @jit('a -> NULL -> bool')
    def __eq__(self, other):
        val1 = numba2.cast(self, numba2.int64)
        val2 = numba2.cast(0, numba2.int64)
        return val1 == val2

    # __________________________________________________________________

    @jit('Pointer[a] -> int64 -> a')
    def __getitem__(self, index):
        return (self + index).deref()

    @jit('Pointer[a] -> int64 -> a -> void')
    def __setitem__(self, idx, value):
        (self + idx).store(value)

    @jit('a -> int64 -> a')
    def __sub__(self, index):
        return self + -index

    # __________________________________________________________________

    @staticmethod
    def fromobject(ptr, type):
        return Pointer(make_ctypes_ptr(ptr, type))

    @classmethod
    def toctypes(cls, val, ty):
        if isinstance(val, Pointer):
            val = val.p
        return make_ctypes_ptr(val, ty)

    @classmethod
    def fromctypes(cls, val, ty):
        if isinstance(val, (int, long)):
            cty = ctype(ty)
            return cty(val)
        return val

    @classmethod
    def ctype(cls, ty):
        [base] = ty.parameters
        return ctypes.POINTER(ctype(base))

#===------------------------------------------------------------------===
# Low-level Implementation
#===------------------------------------------------------------------===

def pointer_add(builder, argtypes, ptr, addend):
    builder.ret(builder.ptradd(ptr, addend))

def pointer_load(builder, argtypes, ptr):
    builder.ret(builder.ptrload(ptr))

def pointer_store(builder, argtypes, ptr, value):
    builder.ptrstore(value, ptr)
    builder.ret(None)

# Determine low-level return types

def _getitem_type(argtypes):
    base = argtypes[0].parameters[0]
    return representation_type(base)

# Implement

add_impl_cls(Pointer, "__add__", pointer_add)
add_impl_cls(Pointer, "deref", pointer_load, restype_func=_getitem_type)
add_impl_cls(Pointer, "store", pointer_store, restype=ptypes.Void)

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

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

