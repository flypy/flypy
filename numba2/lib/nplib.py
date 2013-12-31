# -*- coding: utf-8 -*-

"""
Basic NumPy support.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, cjit, overlay

from numba2.runtime import cast
from numba2.runtime.gc import boehm as gc
from numba2.runtime.obj.core import (Type, StaticTuple, EmptyTuple,
                                     head, tail, Pointer)
from numba2.runtime.hacks import choose
from .arrays.ndarrayobject import NDArray, Dimension, EmptyDim

import numpy as np

#===------------------------------------------------------------------===
# Constructors
#===------------------------------------------------------------------===

@jit('NDArray[dtype1, dims] -> x -> NDArray[dtype3, dims]')
def empty_like(array, dtype=None):
    # TODO: Use normal malloc() for array data for primitives
    dtype = choose(array.dtype, dtype)
    shape = array.getshape()
    return empty(shape, dtype)

@jit
def zeros_like(array, dtype=None):
    result = empty_like(array, dtype)
    result[:] = 0
    return result

@jit
def ones_like(array, dtype=None):
    result = empty_like(array, dtype)
    result[:] = 1
    return result

@jit('shape -> Type[dtype] -> NDArray[dtype, dims]')
def empty(shape, dtype):
    items = product(shape)
    p = gc.gc_alloc(items, dtype)
    data = cast(p, Pointer[dtype])
    dims = c_layout_from_shape(shape, dtype)
    return NDArray(data, dims, dtype)

@jit('shape -> Type[dtype] -> NDArray[dtype, dims]')
def zeros(shape, dtype):
    result = empty(shape, dtype)
    result[:] = 0
    return result

@jit('shape -> Type[dtype] -> NDArray[dtype, dims]')
def ones(shape, dtype):
    result = empty(shape, dtype)
    result[:] = 0
    return result

#===------------------------------------------------------------------===
# Helpers
#===------------------------------------------------------------------===

@jit('StaticTuple[a, b] -> dtype -> Dimension[base]')
def c_layout_from_shape(shape, dtype):
    """Construct dimensions for an array of shape `shape` with a C layout"""
    dim = c_layout_from_shape(tail(shape), dtype)
    extent = head(shape)
    stride = dim.stride * dim.extent
    return Dimension(dim, extent, stride)

@jit('StaticTuple[a, EmptyTuple[]] -> dtype -> Dimension[base]')
def c_layout_from_shape(shape, dtype):
    extent = head(shape)
    return Dimension(EmptyDim(), extent, 1) # TODO: ContigDim

@jit
def product(it):
    """Take the product of the iterable"""
    prod = 1
    for x in it:
        prod *= x
    return prod

#===------------------------------------------------------------------===
# Overlays
#===------------------------------------------------------------------===

overlay(np.empty, empty)
overlay(np.empty_like, empty_like)
overlay(np.zeros_like, zeros_like)
overlay(np.ones_like,  ones_like)