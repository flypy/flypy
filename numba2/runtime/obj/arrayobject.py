# -*- coding: utf-8 -*-

"""
Arrays and NumPy conversion.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2 import jit, sjit, typeof
from numba2.conversion import fromobject, toobject
from . import Type, Pointer, Object, Buffer
from .tupleobject import head, tail

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
        ('keep_alive', 'Object'),
    ]

    # ---------------------------------------

    @jit('Array[a, n] -> b -> a')
    def __getitem__(self, indices):
        #ptr = self.data
        #for i, index in unroll(enumerate(indices)):
        #    ptr += index * self.shape[i]
        ptr = _array_getptr(self.data, indices, self.shape, self.strides,
                            len(indices))
        p = numba2.cast(ptr, Pointer[numba2.float64]) # TODO: Cast to Pointer[a] !
        return p[0]

    # ---------------------------------------

    @classmethod
    def fromobject(cls, ndarray, ty):
        if isinstance(ndarray, np.ndarray):
            return fromnumpy(ndarray)
        else:
            raise NotImplementedError("Array.fromobject(%s, %s)" % (ndarray, ty))

    #@classmethod
    #def toobject(cls, obj, ty):
    #    raise NotImplementedError("Array -> NumPy")


#===------------------------------------------------------------------===
# Indexing
#===------------------------------------------------------------------===

@sjit('DimIndexer[a]')
class DimIndexer(object):
    """
    Dimension indexer, knows how to navigate through an array dimension.
    """

    layout = [('p', 'Pointer[a]'), ('extent', 'int64'), ('stride', 'int64')]

    @jit('DimIndexer[a] -> a')
    def advance(self, item):
        return self.p + item * self.stride

# TODO: Indexers for bounds checking and wraparound


@jit('Pointer[int8] -> a -> a -> a -> int64 -> Pointer[int8]')
def _array_getptr(p, indices, shape, strides, dim):
    """
    Navigate data pointer `p` to point to the item according to the `indices`
    and `strides`.
    """
    if dim == len(shape):
        return p

    # TODO: Pass in indexer class, e.g. WrapAroundIndexer, BoundsCheckIndexer, etc
    indexer = DimIndexer(p, strides[dim], shape[dim])
    result = indexer.advance(head(indices))
    return _array_getptr(result, tail(indices), shape, strides, dim + 1)

#===------------------------------------------------------------------===
# Conversion
#===------------------------------------------------------------------===

def fromnumpy(ndarray):
    """Build an Array from a numpy ndarray"""
    # Type we use for shape/strides. We use Buffer since we can't spell
    # "a tuple of size n" very well yet
    shapetype = Buffer[numba2.int64]

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
    shape = fromobject(ndarray.shape, shapetype)
    strides = fromobject(steps, shapetype)
    keepalive = fromobject(ndarray, Object)

    return Array(data, shape, strides, keepalive)

@typeof.case(np.ndarray)
def typeof(array):
    # if array.flags['C_CONTIGUOUS']:
    #     return ContigArray[typeof(array.dtype)]
    # else:
    return Array[typeof(array.dtype), array.ndim]

#@pyoverload(np.ndarray, Array)
#def convert(ndarray):
#    data = convert(ndarray.ctypes.data, 'Pointer[T]')
#    shape = convert(ndarray.shape, 'Tuple[Int, n]')
#    strides = convert(ndarray.strides, 'Tuple[Int, n]')
#    return Array(data, shape, strides, ndarray)