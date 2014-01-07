# -*- coding: utf-8 -*-

"""
Support for "LLVM style" Array.
This is a value-types that can live in a virtual register, not just in memory. Of course Array can be load/store
to stack/heap memory, but Array has no inherent memory allocation (and therefore pointer).
"""

from __future__ import print_function, division, absolute_import
import ctypes

from numba2 import jit, typeof
from numba2.compiler import lltype
from numba2.conversion import ctype
from numba2.runtime.lowlevel_impls import add_impl_cls, add_impl
from numba2.runtime.obj.sliceobject import Slice
from numba2.runtime.interfaces import Sequence
from numba2.runtime.obj.typeobject import Type
from numba2.runtime.obj.listobject import List

@jit('Array[base, count]')
class Array(object):
    layout = []

    # create from ctypes array
    def __init__(self, arr):
        self.arr = arr

    @jit('Array[base, count] -> int64 -> base', opaque=True)
    def __getitem__(self, idx, opaque=True):
        return self.arr[idx]

    @jit('Array[base, count] -> Iterator[base]')
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # __setitem__ can not be implemented since Array is immutable
    @jit('Array[base, count] -> int64 -> base -> Array[base, count]', opaque=True)
    def set(self, idx, value):
        # copy ctypes array
        arr = type(self.arr)()
        pointer(arr)[0] = self.arr

        # set value
        arr[idx] = value
        return Array(arr)

    @jit('Array[base, count] -> int64', opaque=True)
    def __len__(self):
        return self.arr._length_

    # -- Numba <-> Python -- #
    @staticmethod
    def fromobject(arr, type):
        return Array(make_ctypes_array(arr, type))

    @classmethod
    def toctypes(cls, val, ty):
        if isinstance(val, Array):
            arr = val.arr
        return make_ctypes_array(arr, ty)

    @classmethod
    def fromctypes(cls, val, ty):
        if isinstance(val, ctypes.Array):
            cty = ctype(ty)
            return cty(*val)
        return val

    @classmethod
    def ctype(cls, ty):
        base, count = ty.parameters
        return ctype(base) * count

# Utils

@jit('Type[base] -> int64 -> Array[base, count]')
def newarray(basetype, size):
    arr = (ctype(basetype) * size)()
    return Array(arr)

# @jit('Sequence[a] -> Type[a] -> Buffer[a]') # TODO: <--
def fromseq(seq, basetype):
    # TODO: create on construct when CALL_FUNCTION_VAR is supported
    n = len(seq)
    arr = newarray(basetype, n)
    for i, item in enumerate(seq):
        arr[i] = item
    return arr

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

def make_ctypes_array(arr, type):
    cty = ctype(type)
    arr = ctypes.cast(arr, cty)
    return arr
