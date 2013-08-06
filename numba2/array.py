"""
Implement arrays and conversion between objects and arrays.

Required compiler support:

    1) Parametric types, supporting types and values. These are all resolved
       at compile time.
    2) Overriding and adding builtin conversion rules (overloading 'convert')
    3) Overriding how objects are typed through 'typeof'

Performance (not strictly necessary but highly recommended:

    1) loop unrolling (an alternative would be to use MSP)
    2) inlining
    3) control over stack/heap allocation
    4) Staged code and type inference of return staged fragments
"""

import ast
import numpy as np
from core import structclass, Struct, signature, char, void

@structclass('Array[type T, Int n]')
class Array(object):

    layout = Struct([('data', char.pointer()),
                     ('shape', 'Tuple[Int, n]'),
                     ('strides', 'Tuple[Int, n]'),
                     ('keep_alive', 'Object')])

    def __init__(self, data, shape, strides, keep_alive=None):
        # NOTE: we should generate constructors if not present from the layout struct
        self.data = data
        self.shape = shape
        self.strides = strides
        self.keep_alive = keep_alive

    # NOTE: we consider slicing static syntax!
    @signature('Array[T, n] -> Tuple[Int, n] -> T')
    def __getindex__(self, indices):
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