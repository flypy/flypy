# -*- coding: utf-8 -*-

"""
Support for "LLVM style" Array.
This is a value-types that can live in a virtual register, not just in memory. Of course Array can be load/store
to stack/heap memory, but Array has no inherent memory allocation (and therefore pointer).
"""

from __future__ import print_function, division, absolute_import
import ctypes

from numba2 import jit
from numba2.compiler import lltype
from numba2.conversion import ctype
from numba2.runtime.lowlevel_impls import add_impl_cls
from numba2.runtime.obj.sliceobject import Slice

@jit('Array[base, count]')
class Array(object):
    layout = []

    def __init__(self, items):
        #assert len(items) == 'count' TODO: ??
        self.items = items

    @jit('Array[base, count] -> int64 -> base', opaque=True)
    def __getitem__(self, idx, opaque=True):
        return self.items[idx]

    @jit('Array[base, count] -> Iterator[base]')
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # __setitem__ can not be implemented since Array is immutable
    @jit('Array[base, count] -> int64 -> base -> Array[base, count]', opaque=True)
    def set(self, idx, value):
        items = list(self.items)
        items[idx] = value
        return Array(items)

    @jit('Array[base, count] -> int64', opaque=True)
    def __len__(self):
        return len(self.items)

    # -- Numba <-> Python -- #
    @staticmethod
    def fromobject(items, type):
        return Array(make_ctypes_array(items, type))

    @classmethod
    def toctypes(cls, val, ty):
        # TODO:
        if isinstance(val, Array):
            val = val.items
        return make_ctypes_arrayr(val, ty)

    @classmethod
    def fromctypes(cls, val, ty):
        # TODO:
        if hasattr(val, '_type_'):
            return Array(list(val))
        return val

    @classmethod
    def ctype(cls, ty):
        base, count = ty.parameters
        return ctype(base) * count

#===------------------------------------------------------------------===
# Low-level Implementations
#===------------------------------------------------------------------===

# Methods

def implement_getitem(builder, argtypes, arr, idx):
    e = builder.get(arr, idx)
    return builder.ret(e)

def implement_set(builder, argtypes, arr, idx, item):
    e = builder.set(arr, item, idx)
    return builder.ret(e)

def implement_len(builder, argtypes, arr):
    count = argtypes[0].parameters[1]
    return builder.ret(ir.Const(count, ptypes.Int64))

add_impl_cls(Array, "__getitem__", implement_getitem)
add_impl_cls(Array, "set", implement_set)
add_impl_cls(Array, "__len__", implement_len)

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

def make_ctypes_array(items, type):
    # TODO:
    from numba2.support.cffi_support import is_cffi, ffi
    from numba2.support.ctypes_support import is_ctypes_pointer_type

    cty = ctype(type)
    if is_cffi(ptr):
        addr = ffi.cast('uintptr_t', ptr)
        ctypes_ptr = ctypes.c_void_p(int(addr))
        ptr = ctypes.cast(ctypes_ptr, cty)
    else:
        ptr = ctypes.cast(ptr, cty)

    return ptr
