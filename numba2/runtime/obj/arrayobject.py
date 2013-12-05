# -*- coding: utf-8 -*-

"""
Arrays and NumPy conversion.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2 import jit, sjit, ijit, typeof
from numba2.support import numpy_support
from numba2.conversion import fromobject, toobject
from .core import Type, Pointer, StaticTuple, address, Buffer, Slice, NoneType
from .extended import Object
from .bufferobject import fromseq
from .tupleobject import head, tail, EmptyTuple
from ..lib import libcpy

import numpy as np

#===------------------------------------------------------------------===
# NumPy-like ndarray
#===------------------------------------------------------------------===

@sjit('Array[a, dims]')
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
        result = _unpack(result)
        return result

    @jit('Array[dtype, dims] -> int64 -> r')
    def __getitem__(self, item):
        return self[(item,)]

    @jit('Array[dtype, dims] -> Slice[start, stop, step] -> r')
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

## ---- begin hack --- ##

# NOTE: This works around the type inferencer not recognizing 'is not None'
#       as type checks.

# TODO: Integrate SCCP pass in type inferencer

@jit('a -> NoneType[] -> a')
def choose(a, b):
    return a

@jit('a -> b -> b')
def choose(a, b):
    return b

## ---- end hack --- ##

@sjit('Dimension[base]')
class Dimension(object):
    """
    Dimension indexer, knows how to navigate through an array dimension.
    Each dimension applies a single index and stride, and dispatches to the
    next dimension.
    """

    layout = [('base', 'base'), ('extent', 'int64'), ('stride', 'int64')]

    @jit('s -> Pointer[a] -> StaticTuple[x : integral, y] -> r')
    def index(self, p, indices):
        idx = head(indices)
        #if idx < 0 or idx > self.extent:
        #    print("Index out of bounds!")
        #    return self.base.index(p, tail(indices))
        return self.base.index(p + idx * self.stride, tail(indices))

    @jit('s -> Pointer[a] -> StaticTuple[Slice[start, stop, step], y] -> r')
    def index(self, p, indices):
        # TODO: wraparound
        s = head(indices)

        data = p
        extent = self.extent
        stride = self.stride

        start = choose(0, s.start)
        stop = choose(extent, s.stop)
        step = choose(1, s.step)

        #-- Wrap around --#
        if start < 0:
            start += extent
            if start < 0:
                start = 0
        if start > extent:
            start = extent - 1

        if stop < 0:
            stop += extent
            if stop < -1:
                stop = -1
        if stop > extent:
            stop = extent

        extent = len(xrange(start, stop, step))

        # Process start
        if s.start is not None:
            data += start * stride

        # Process step
        if s.step is not None:
            stride *= step

        array = self.base.index(data, tail(indices))
        dims = Dimension(array.dims, extent, stride)
        return Array(array.data, dims)

    @jit('s -> Pointer[a] -> EmptyTuple[] -> r')
    def index(self, p, indices):
        return Array(p, self)

@sjit('DimensionContig[base]')
class DimensionContig(Dimension):
    """
    Note: stride attribute is left here for uniform layout.
    """
    layout = [('base', 'base'), ('extent', 'int64'), ('stride', 'int64')]

    @jit('s -> Pointer[a] -> StaticTuple[x : integral, y] -> r')
    def index(self, p, indices):
        idx = head(indices)
        return self.base.index(p + idx, tail(indices))

    @jit('s -> Pointer[a] -> StaticTuple[Slice[start, stop, step], y] -> r')
    def index(self, p, indices):
        # TODO: wraparound
        s = head(indices)

        data = p
        extent = self.extent
        stride = self.stride

        start = choose(0, s.start)
        stop = choose(extent, s.stop)
        step = choose(1, s.step)

        #-- Wrap around --#
        if start < 0:
            start += extent
            if start < 0:
                start = 0
        if start > extent:
            start = extent - 1

        if stop < 0:
            stop += extent
            if stop < -1:
                stop = -1
        if stop > extent:
            stop = extent

        extent = len(xrange(start, stop, step))

        # Process start
        if s.start is not None:
            data += start * stride

        # Process step
        if s.step is not None:
            stride *= step

        array = self.base.index(data, tail(indices))
        dims = DimensionContig(array.dims, extent, stride)
        return Array(array.data, dims)


@sjit
class EmptyDim(object):
    layout = []

    @jit('empty -> Pointer[a] -> EmptyTuple[] -> Array[a, empty]')
    def index(self, p, indices):
        return Array(p, self)

@sjit('BoundsCheck[base]')
class BoundsCheck(object):
    """
    Check bounds for the Dimension that we wrap.
    """

    layout = [('base', 'base')]

    @jit
    def index(self, p, indices):
        idx = head(indices)
        if 0 <= idx < self.base.extent:
            return self.base.index(p, indices)

        # TODO: Exceptions
        print("Index out of bounds: index", end="")
        print(idx, end=", extent ")
        print(self.base.extent)

    #@jit('BoundsCheck[EmptyTuple[]] -> a -> b -> c')
    #def index(self, p, indices):
    #    return self.base.index(p, indices)

    # TODO: Support properties to allow composing BoundsCheck/WrapAround

    @property
    def extent(self):
        return self.base.extent

    @property
    def stride(self):
        return self.base.stride

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

@jit('Array[dtype, EmptyDim[]] -> a -> void')
def fill(array, value):
    """Fill a 0D array"""
    array.data[0] = value

@jit('Array[dtype, dims] -> a -> void')
def fill(array, value):
    """Fill an ND-array with N > 0"""
    for i in range(len(array)):
        subarray = array.dims.index(array.data, (i,))
        fill(subarray, value)

#===------------------------------------------------------------------===
# Conversion
#===------------------------------------------------------------------===

def fromnumpy(ndarray, boundscheck=False):
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
        dimcls = DimensionContig if stride == 1 else Dimension
        dims = dimcls(dims, extent, stride)

        if boundscheck:
            dims = BoundsCheck(dims)

    return Array(data, dims)

def tonumpy(arr, dtype):
    """Build an Array from a numpy ndarray"""
    itemsize = numba2.sizeof_type(dtype)
    steps = np.array(_getsteps(arr.dims))
    shape = np.array(_getshape(arr.dims))
    strides = steps * itemsize

    np_dtype = numpy_support.to_dtype(dtype)

    # Build NumPy array
    data = toobject(arr.data, Pointer[dtype])
    ndarray = libcpy.create_array(address(data), shape, strides, np_dtype)
    return ndarray

@typeof.case(np.ndarray)
def typeof(array):
    # if array.flags['C_CONTIGUOUS']:
    #     return ContigArray[typeof(array.dtype)]
    # else:
    dtype = numpy_support.from_dtype(array.dtype)

    dims = EmptyDim[()]

    for i, stride in zip(range(array.ndim), reversed(array.strides)):
        dimcls = DimensionContig if stride == array.itemsize else Dimension
        dims = dimcls[dims]

    return Array[dtype, dims]

#@pyoverload(np.ndarray, Array)
#def convert(ndarray):
#    data = convert(ndarray.ctypes.data, 'Pointer[T]')
#    shape = convert(ndarray.shape, 'Tuple[Int, n]')
#    strides = convert(ndarray.strides, 'Tuple[Int, n]')
#    return Array(data, shape, strides, ndarray)

# ------------- Helpers ------------- #

def _getshape(dims):
    if isinstance(dims, Dimension):
        return (dims.extent,) + _getshape(dims.base)
    else:
        assert isinstance(dims, EmptyDim)
        return ()

def _getsteps(dims):
    if isinstance(dims, Dimension):
        return (dims.stride,) + _getsteps(dims.base)
    else:
        assert isinstance(dims, EmptyDim)
        return ()