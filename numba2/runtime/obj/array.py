# -*- coding: utf-8 -*-

"""
Arrays and NumPy conversion.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, char

import numpy as np

@jit('Array[type T, Int n]')
class Array(object):

    layout = [('data', char.pointer()),
              ('shape', 'Tuple[Int, n]'),
              ('strides', 'Tuple[Int, n]'),
              ('keep_alive', 'Object')]

    def __init__(self, data, shape, strides, keep_alive=None):
        # NOTE: we should generate constructors if not present from the layout struct
        self.data = data
        self.shape = shape
        self.strides = strides
        self.keep_alive = keep_alive

    @jit('Array[T, n] -> a -> b')
    def __getitem__(self, indices):
        ptr = self.data
        for i, index in unroll(enumerate(indices)):
            ptr += index * self.shape[i]
        return ptr[0]

    @signature('Rep[Array[T, ?]] -> Rep[Tuple[Slice | Int, n]] -> Rep[T]')
    def __getslice__(self, s_indices):
        # Note: This could be an opague method and emit a compiler operator, which later resolves
        # ndim = ...
        # new_data = ...
        # new_shape = ...
        # new_strides = ...
        return quote[Array(escape[new_data], escape[new_shape],
                           escape[new_strides], escape[self].keep_alive)]



@overload(np.ndarray)
def typeof(array):
    # if array.flags['C_CONTIGUOUS']:
    #     return ContigArray[typeof(array.dtype)]
    # else:
    return Array[typeof(array.dtype), array.ndim]

@overload(np.ndarray, Array)
@signature('Object -> Array[T, n]')
def convert(ndarray):
    data = convert(ndarray.ctypes.data, 'Pointer[T]')
    shape = convert(ndarray.shape, 'Tuple[Int, n]')
    strides = convert(ndarray.strides, 'Tuple[Int, n]')
    return Array(data, shape, strides, ndarray)