# -*- coding: utf-8 -*-

"""
Pointer implementation.
"""

from __future__ import print_function, division, absolute_import
import ctypes

import flypy.types
from flypy import sjit, jit, ijit, cjit
from flypy.compiler import representation_type
from flypy.conversion import ctype
from flypy.runtime import formatting
from .richcompare import RichComparisonMixin
from ..lowlevel_impls import add_impl_cls

from pykit import types as ptypes

jit = cjit

#===------------------------------------------------------------------===
# Pointer
#===------------------------------------------------------------------===

@sjit('Pointer[a]')
class Pointer(RichComparisonMixin):
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

    @jit('Pointer[a] -> Pointer[b] -> bool')
    def __eq__(self, other):
        return self.ptrtoint() == other.ptrtoint()

    @jit('a -> NullType[] -> bool')
    def __eq__(self, other):
        return self.ptrtoint() == 0

    @jit('Pointer[a] -> Pointer[b] -> bool')
    def __lt__(self, other):
        return self.ptrtoint() < other.ptrtoint()

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

    @jit('Pointer[a] -> Pointer[a] -> int64')
    def __sub__(self, index):
        return self.ptrtoint() - index.ptrtoint()

    @jit('a -> bool')
    def __nonzero__(self):
        return self != flypy.NULL

    __bool__ = __nonzero__

    @jit('a -> int64')
    def ptrtoint(self):
        return flypy.cast(self, flypy.types.int64)

    @jit
    def __str__(self):
        # maxlen = int(math.log10(2**64)) + 1 = 20 in a 64-bit address space
        return formatting.format_static("%p", self, 20 + 1)

    __repr__ = __str__

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
        if isinstance(val, int):
            cty = ctype(ty)
            return cty(val)
        return val

    @classmethod
    def ctype(cls, ty):
        [base] = ty.parameters
        return ctypes.POINTER(ctype(base))

#===------------------------------------------------------------------===
# NULL
#===------------------------------------------------------------------===

@sjit
class NullType(object):
    layout = []

    @jit('a -> bool')
    def __nonzero__(self):
        return False

    @jit('a -> Pointer[b] -> bool')
    def __eq__(self, other):
        return other.ptrtoint() == 0

    #@jit('a -> b -> bool')
    #def __eq__(self, other):
    #    return False

NULL = NullType()

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
    from flypy.support.cffi_support import is_cffi, ffi
    from flypy.support.ctypes_support import is_ctypes_pointer_type

    cty = ctype(type)

    if is_cffi(ptr):
        addr = ffi.cast('uintptr_t', ptr)
        ctypes_ptr = ctypes.c_void_p(int(addr))
        ptr = ctypes.cast(ctypes_ptr, cty)
    else:
        ptr = ctypes.cast(ptr, cty)

    return ptr

def address(ptr):
    return ctypes.cast(ptr, ctypes.c_void_p).value
