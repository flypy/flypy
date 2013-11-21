# -*- coding: utf-8 -*-

"""
Arrays and NumPy conversion.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2 import jit, sjit, typeof
from numba2.support import numpy_support
from numba2.conversion import fromobject, toobject
from .core import Type, Pointer, StaticTuple, address, Buffer
from .extended import Object
from .bufferobject import fromseq
from .tupleobject import head, tail, EmptyTuple
from ..lib import libcpy

import numpy as np

#===------------------------------------------------------------------===
# NumPy-like ndarray
#===------------------------------------------------------------------===

@jit('Array[a, n]')
class Array(object):
    """
    N-dimensional NumPy-like array object.
    """

    layout = [
        ('data', 'Pointer[a]'),
        ('shape', 'Buffer[int64]'),
        ('strides', 'Buffer[int64]'),
        #('keep_alive', 'Object'),
    ]

    # ---------------------------------------

    @jit('Array[a, n] -> StaticTuple[a, b] -> a')
    def __getitem__(self, indices):
        ptr = _array_getptr(self.data, indices, self.shape, self.strides,
                            len(indices))
        return ptr[0]

    @jit('Array[a, n] -> int64 -> a')
    def __getitem__(self, item):
        return self[(item,)]

    @jit #('Array[a, n] -> Iterable[a]')
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @jit('a -> int64')
    def __len__(self):
        return self.shape[0]

    # ---------------------------------------

    @classmethod
    def fromobject(cls, ndarray, ty):
        if isinstance(ndarray, np.ndarray):
            return fromnumpy(ndarray)
        else:
            raise NotImplementedError("Array.fromobject(%s, %s)" % (ndarray, ty))

    @classmethod
    def toobject(cls, obj, ty):
        dtype, n = ty.parameters
        return tonumpy(obj, obj.shape, dtype)

#===------------------------------------------------------------------===
# Indexing
#===------------------------------------------------------------------===

@sjit('DimIndexer[a]')
class DimIndexer(object):
    """
    Dimension indexer, knows how to navigate through an array dimension.
    """

    layout = [('p', 'Pointer[a]'), ('extent', 'int64'), ('stride', 'int64')]

    @jit('DimIndexer[a] -> int64 -> Pointer[a]')
    def advance(self, item):
        return self.p + item * self.stride

# TODO: Indexers for bounds checking and wraparound


@jit('Pointer[t] -> a -> b -> b -> int64 -> Pointer[t]')
def _array_getptr(p, indices, shape, strides, dim):
    """
    Navigate data pointer `p` to point to the item according to the `indices`
    and `strides`.
    """
    # TODO: Pass in indexer class, e.g. WrapAroundIndexer, BoundsCheckIndexer, etc
    indexer = DimIndexer(p, strides[dim], shape[dim])
    result = indexer.advance(head(indices))
    return _array_getptr(result, tail(indices), shape, strides, dim + 1)

@jit('Pointer[t] -> EmptyTuple[] -> b -> b -> int64 -> Pointer[t]')
def _array_getptr(p, indices, shape, strides, dim):
    return p


#===------------------------------------------------------------------===
# Conversion
#===------------------------------------------------------------------===

def fromnumpy(ndarray):
    """Build an Array from a numpy ndarray"""
    # Compute steps
    itemsize = ndarray.dtype.itemsize
    steps = tuple(stride // itemsize for stride in ndarray.strides)

    # Check that strides are divisible by itemsize. We need to do this to
    # allow computation on Pointer[T] instead of Pointer[char], which we
    # need since we cannot access 'T' in __getitem__ to cast the Pointer[char]
    # back to Pointer[T]
    if any(stride % itemsize for stride in ndarray.strides):
        raise ValueError("Cannot handle non-element size strides "
                         "(e.g. views in record arrays)")

    # Build array object
    data = fromobject(ndarray.ctypes.data, Pointer[numba2.int8])

    # Type we use for shape/strides. We use Buffer since we can't spell
    # "a tuple of size n" very well yet
    shape = fromseq(ndarray.shape, numba2.int64)
    strides = fromseq(steps, numba2.int64)
    #keepalive = fromobject(ndarray, Object[()])

    return Array(data, shape, strides) #, keepalive)

def tonumpy(arr, shape, dtype):
    """Build an Array from a numpy ndarray"""
    size = np.prod(shape)
    total_size = size * numba2.sizeof_type(dtype)
    np_dtype = numpy_support.to_dtype(dtype)

    # Build NumPy array
    data = toobject(arr.data, Pointer[dtype])
    ndarray = libcpy.dummy_array(address(data), total_size)

    # Cast and reshape
    result = ndarray.view(np_dtype).reshape(shape)

    return result

@typeof.case(np.ndarray)
def typeof(array):
    # if array.flags['C_CONTIGUOUS']:
    #     return ContigArray[typeof(array.dtype)]
    # else:
    dtype = numpy_support.from_dtype(array.dtype)
    return Array[dtype, array.ndim]

#@pyoverload(np.ndarray, Array)
#def convert(ndarray):
#    data = convert(ndarray.ctypes.data, 'Pointer[T]')
#    shape = convert(ndarray.shape, 'Tuple[Int, n]')
#    strides = convert(ndarray.strides, 'Tuple[Int, n]')
#    return Array(data, shape, strides, ndarray)