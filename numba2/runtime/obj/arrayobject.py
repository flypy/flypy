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

@jit('Array[a, dims]')
class Array(object):
    """
    N-dimensional NumPy-like array object.
    """

    layout = [
        ('data', 'Pointer[a]'),
        ('dims', 'dims')
    ]

    # ---------------------------------------

    @jit('Array[dtype, dims] -> StaticTuple[a, b] -> r')
    def __getitem__(self, indices):
        result = self.dims.index(self.data, indices)
        return _unpack(result)

    @jit('Array[dtype, dims] -> int64 -> dtype')
    def __getitem__(self, item):
        return self[(item,)]

    @jit('Array[dtype, dims] -> StaticTuple[a, b] -> dtype -> void')
    def __setitem__(self, indices, value):
        result = self.dims.index(self.data, indices)
        fill(result, value)

    @jit('Array[dtype, dims] -> int64 -> dtype -> void')
    def __setitem__(self, item, value):
        self[(item,)] = value

    @jit #('Array[a, n] -> Iterable[a]')
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @jit('a -> int64')
    def __len__(self):
        return self.dims.extent

    # ---------------------------------------

    @classmethod
    def fromobject(cls, ndarray, ty):
        if isinstance(ndarray, np.ndarray):
            return fromnumpy(ndarray)
        else:
            raise NotImplementedError("Array.fromobject(%s, %s)" % (ndarray, ty))

    @classmethod
    def toobject(cls, obj, ty):
        dtype, dimtype = ty.parameters
        return tonumpy(obj, dtype)

#===------------------------------------------------------------------===
# Indexing
#===------------------------------------------------------------------===

@sjit('Dimension[base]')
class Dimension(object):
    """
    Dimension indexer, knows how to navigate through an array dimension.
    """

    layout = [('base', 'base'), ('extent', 'int64'), ('stride', 'int64')]

    @jit('Dimension[base] -> Pointer[a] -> StaticTuple[x, y] -> r')
    def index(self, p, indices):
        idx = head(indices)
        return self.base.index(p + idx * self.stride, tail(indices))

    @jit('Dimension[base] -> Pointer[a] -> EmptyTuple[] -> Array[a, Dimension[base]]')
    def index(self, p, indices):
        return Array(p, self)

@sjit
class EmptyDim(object):
    layout = []

    @jit('empty -> Pointer[a] -> EmptyTuple[] -> Array[a, empty]')
    def index(self, p, indices):
        return Array(p, self)


# TODO: Dimensions for bounds checking and wraparound

#===------------------------------------------------------------------===
# getitem/setitem
#===------------------------------------------------------------------===

# Unpack the result for getitem

@jit('Array[a, EmptyDim[]] -> a')
def _unpack(array):
    return array.data[0]

@jit('Array[a, dims] -> Array[a, dims]')
def _unpack(array):
    return array

# Fill the array with `value`

@jit('Array[a, EmptyDim[]] -> a -> void')
def fill(array, value):
    array.data[0] = value

@jit('Array[a, dims] -> a -> void')
def fill(array, value):
    for i in range(len(array)):
        fill(array[i], value)

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
    #shape = fromseq(ndarray.shape, numba2.int64)
    #strides = fromseq(steps, numba2.int64)
    #keepalive = fromobject(ndarray, Object[()])

    dims = EmptyDim()
    for extent, stride in reversed(zip(ndarray.shape, steps)):
        dims = Dimension(dims, extent, stride)

    return Array(data, dims)


def _getshape(dims):
    if isinstance(dims, Dimension):
        return (dims.extent,) + _getshape(dims.base)
    else:
        return ()

def tonumpy(arr, dtype):
    """Build an Array from a numpy ndarray"""
    shape = _getshape(arr.dims)

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

    dims = EmptyDim[()]
    for i in range(array.ndim):
        dims = Dimension[dims]

    return Array[dtype, dims]

#@pyoverload(np.ndarray, Array)
#def convert(ndarray):
#    data = convert(ndarray.ctypes.data, 'Pointer[T]')
#    shape = convert(ndarray.shape, 'Tuple[Int, n]')
#    strides = convert(ndarray.strides, 'Tuple[Int, n]')
#    return Array(data, shape, strides, ndarray)