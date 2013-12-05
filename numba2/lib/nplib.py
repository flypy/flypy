# -*- coding: utf-8 -*-

"""
Basic NumPy support.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, overlay, sizeof

from numba2.runtime import cast
from numba2.runtime.gc import boehm as gc
from numba2.runtime.obj.core import (Type, StaticTuple, EmptyTuple,
                                     head, tail, Pointer)
from numba2.runtime.hacks import choose
from .arrays.arrayobject import Array, Dimension, EmptyDim

import numpy as np

#===------------------------------------------------------------------===
# Constructors
#===------------------------------------------------------------------===

@jit('Array[dtype1, dims] -> x -> Array[dtype3, dims]')
def empty_like(array, dtype=None):
    # TODO: Use normal malloc() for array data for primitives
    dtype = choose(array.dtype, dtype)
    shape = array.getshape()
    return empty(shape, dtype)

@jit('shape -> Type[dtype] -> Array[dtype, dims]')
def empty(shape, dtype):
    items = product(shape)
    p = gc.gc_alloc(items, dtype)
    data = cast(p, Pointer[dtype])
    dims = c_layout_from_shape(shape, dtype)
    return Array(data, dims, dtype)

#===------------------------------------------------------------------===
# Helpers
#===------------------------------------------------------------------===

@jit('StaticTuple[a, b] -> dtype -> Dimension[base]')
def c_layout_from_shape(shape, dtype):
    dim = c_layout_from_shape(tail(shape), dtype)
    extent = head(shape)
    stride = dim.stride * extent
    return Dimension(dim, extent, stride)

@jit('StaticTuple[a, EmptyTuple[]] -> dtype -> Dimension[base]')
def c_layout_from_shape(shape, dtype):
    extent = head(shape)
    stride = sizeof(dtype)
    return Dimension(EmptyDim(), extent, stride) # TODO: ContigDim

@jit
def product(it):
    prod = 1
    for x in it:
        prod *= x
    return prod

#===------------------------------------------------------------------===
# Overlays
#===------------------------------------------------------------------===

overlay(np.empty, empty)
overlay(np.empty_like, empty_like)