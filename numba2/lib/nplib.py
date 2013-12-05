# -*- coding: utf-8 -*-

"""
Basic NumPy support.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit

from numba2.runtime.gc import boehm as gc
from numba2.runtime.obj.core import Type, StaticTuple, head, tail
from numba2.runtime.hacks import choose
from .arrays import Array

@jit('Array[dims, dtype1] -> dtype2 -> Array[dims, dtype3]')
def empty_like(array, dtype=None):
    # TODO: Use normal malloc() for array data for primitives
    dtype = choose(array.dtype, dtype)
    items = product(array.getshape())
    data = gc.gc_alloc(items, dtype)
    return Array(data, array.dims, dtype)
