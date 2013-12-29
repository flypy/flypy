# -*- coding: utf-8 -*-

"""
Support for small stack-allocated C-style 1D arrays.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from numba2 import jit, cast, Pointer
from numba2.compiler import lltype
from numba2.conversion import ctype
from numba2.runtime.lowlevel_impls import add_impl_cls

@jit('Array[base, count]')
class Array(object):
    layout = []

    @jit('Array[base, count] -> base')
    def __getitem__(self, idx):
        # TODO: boundscheck
        return self.pointer()[idx]

    @jit('Array[base, count] -> base')
    def __setitem__(self, idx, item):
        # TODO: boundscheck
        self.pointer()[idx] = item

    # __________________________________________________________________

    @jit('Array[base, count] -> Pointer[base]', opaque=True)
    def pointer(self):
        raise NotImplementedError("Pointer method of Array in pure python")

    # __________________________________________________________________

    @classmethod
    def ctype(cls, ty):
        base, count = ty.parameters
        return ctype(base) * count

#===------------------------------------------------------------------===
# Low-level Implementations
#===------------------------------------------------------------------===

def implement_array_to_pointer_cast(builder, argtypes, array):
    base, count = argtypes[0]
    builder.ret(builder.bitcast(array, lltype(base)))

# Implement

add_impl_cls(Array, "pointer", implement_array_to_pointer_cast)