# -*- coding: utf-8 -*-

"""
Support for LLVM-style array.
These are value-types that can live in a virtual register, not on the stack or heap. Of course Array can be load/store
to stack/heap memory, but Array has not inherent memory allocation and therefore pointer of any kind.
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

    def __init__(self):
        pass

    @jit('Array[base, count] -> int64 -> base')
    def __getitem__(self, idx):
        pass

    @jit('Array[base, count] -> int64 -> base -> void')
    def __setitem__(self, idx, item):
        pass

    # -- Numba <-> Python -- #

    @classmethod
    def ctype(cls, ty):
        base, count = ty.parameters
        return ctype(base) * count

#===------------------------------------------------------------------===
# Low-level Implementations
#===------------------------------------------------------------------===

# Methods

# def implement_array_getitem(builder, argtypes, arr, idx):
#     e = builder.get(arr, idx)
#     builder.ret(e)
#
# def implement_array_setitem(builder, argtypes, arr, idx, item):
#     e = builder.set(arr, item, idx)
#     builder.ret(e)

# Implement

#add_impl_cls(Array, "__getitem__", implement_array_getitem)
#add_impl_cls(Array, "__setitem__", implement_array_setitem)
