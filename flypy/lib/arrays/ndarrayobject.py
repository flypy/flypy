# -*- coding: utf-8 -*-

"""
Arrays and NumPy conversion.
"""

from __future__ import print_function, division, absolute_import

import flypy.types
from flypy import jit, sjit, ijit, typeof, cjit
from flypy.support import numpy_support
from flypy.conversion import fromobject, toobject
from flypy.runtime.obj.core import (Type, Pointer, StaticTuple, address,
                                     Buffer, Slice, NoneType,
                                     fromseq, head, tail, EmptyTuple)
from flypy.runtime.lib import libcpy
from flypy.runtime.hacks import choose

import numpy as np

jit = cjit

#===------------------------------------------------------------------===
# NumPy-like ndarray
#===------------------------------------------------------------------===

@sjit('NDArray[a, dims]')
class NDArray(object):
    """
    N-dimensional NumPy-like array object.
    """

    layout = [
        ('data', 'Pointer[a]'),
        ('dims', 'dims'),
        ('dtype', 'Type[a]'),
    ]

    # ---------------------------------------

    @jit('NDArray[dtype, dims] -> StaticTuple[a, b] -> r')
    def __getitem__(self, indices):
        result = _unpack(self._index(indices))
        return result

    @jit('NDArray[dtype, dims] -> int64 -> r')
    def __getitem__(self, item):
        return self[(item,)]

    @jit('NDArray[dtype, dims] -> Slice[start, stop, step] -> r')
    def __getitem__(self, item):
        return self[(item,)]

    @jit('NDArray[dtype, dims] -> StaticTuple[a, b] -> dtype -> void')
    def __setitem__(self, indices, value):
        result = self._index(indices)
        fill(result, value)

    @jit('NDArray[dtype, dims] -> int64 -> dtype -> void')
    def __setitem__(self, item, value):
        self[(item,)] = value

    @jit('NDArray[dtype, dims] -> Slice[start, stop, step] -> dtype -> void')
    def __setitem__(self, item, value):
        self[(item,)] = value

    @jit #('NDArray[a, n] -> Iterable[a]')
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @jit('a -> int64')
    def __len__(self):
        return self.dims.extent

    @jit
    def getshape(self):
        # TODO: properties
        return getshape(self.dims)

    @jit
    def getsteps(self):
        # TODO: properties
        return getsteps(self.dims)

    # -- Private -- #

    @jit('NDArray[dtype, dims] -> StaticTuple[a, b] -> r')
    def _index(self, indices):
        return self.dims.index(self.data, indices, self.dtype)

    # -- flypy <-> Python -- #

    @classmethod
    def fromobject(cls, ndarray, ty):
        if isinstance(ndarray, np.ndarray):
            return fromnumpy(ndarray, ty)
        else:
            raise NotImplementedError("NDArray.fromobject(%s, %s)" % (ndarray, ty))

    @classmethod
    def toobject(cls, obj, ty):
        dtype, dimtype = ty.parameters
        return tonumpy(obj, dtype)

# ------------- Helpers ------------- #

@jit('Dimension[base] -> a')
def getshape(dim):
    return StaticTuple(dim.extent, getshape(dim.base))

@jit('EmptyDim[] -> a')
def getshape(dim):
    return EmptyTuple()


@jit('Dimension[base] -> a')
def getsteps(dim):
    return StaticTuple(dim.stride, getsteps(dim.base))

@jit('EmptyDim[] -> a')
def getsteps(dim):
    return EmptyTuple()

#===------------------------------------------------------------------===
# Indexing
#===------------------------------------------------------------------===

@jit('dim -> Pointer[a] -> StaticTuple[Slice[start, stop, step], y] -> Type[t] -> r')
def slice_dim(dim, p, indices, dtype):
    # TODO: wraparound
    s = head(indices)

    data = p
    extent = dim.extent
    stride = dim.stride

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

    array = dim.base.index(data, tail(indices), dtype)
    # TODO: Without 'step', ContigDim should remain a ContigDim
    dims = Dimension(array.dims, extent, stride)
    return NDArray(array.data, dims, dtype)


@sjit('Dimension[base]')
class Dimension(object):
    """
    Dimension indexer, knows how to navigate through an array dimension.
    Each dimension applies a single index and stride, and dispatches to the
    next dimension.
    """

    layout = [('base', 'base'), ('extent', 'int64'), ('stride', 'int64')]

    @jit('Dimension[base] -> Pointer[a] -> '
         'StaticTuple[x : integral, y] -> Type[a] -> r')
    def index(self, p, indices, dtype):
        idx = head(indices)
        #if idx < 0 or idx > self.extent:
        #    print("Index out of bounds!")
        #    return self.base.index(p, tail(indices))
        return self.base.index(p + idx * self.stride, tail(indices), dtype)

    @jit('Dimension[base] -> Pointer[a] -> '
         'StaticTuple[Slice[start, stop, step], y] -> Type[a] -> r')
    def index(self, p, indices, dtype):
        # TODO: wraparound
        return slice_dim(self, p, indices, dtype)

    @jit('s -> Pointer[a] -> EmptyTuple[] -> Type[a] -> r')
    def index(self, p, indices, dtype):
        return NDArray(p, self, dtype)


#@sjit('DimensionContig[base]')
#class DimensionContig(Dimension):
#    """
#    Note: stride attribute is left here for uniform layout.
#    """
#    layout = [('base', 'base'), ('extent', 'int64'), ('stride', 'int64')]
#
#    @jit('s -> Pointer[a] -> StaticTuple[x : integral, y] -> r')
#    def index(self, p, indices):
#        idx = head(indices)
#        return self.base.index(p + idx, tail(indices))
#
#    @jit('Dimension[base] -> Pointer[a] -> EmptyTuple[] -> Type[a] -> r')
#    def index(self, p, indices, dtype):
#        return NDArray(p, self, dtype)
#
#    @jit('s -> Pointer[a] -> StaticTuple[Slice[start, stop, step], y] -> r')
#    def index(self, p, indices):
#        return slice_dim(Dimension(self.base, self.extent, self.stride),
#                         p, indices)


@sjit
class EmptyDim(object):
    layout = []

    @jit('empty -> Pointer[a] -> EmptyTuple[] -> Type[a] -> NDArray[a, empty]')
    def index(self, p, indices, dtype):
        return NDArray(p, self, dtype)


@sjit('BoundsCheck[base]')
class BoundsCheck(object):
    """
    Check bounds for the Dimension that we wrap.
    """

    layout = [('base', 'base')]

    @jit
    def index(self, p, indices, dtype):
        idx = head(indices)
        if 0 <= idx < self.base.extent:
            return self.base.index(p, indices, dtype)

        # TODO: Exceptions
        print("Index out of bounds: index", end="")
        print(idx, end=", extent ")
        print(self.base.extent)

    #@jit('BoundsCheck[EmptyTuple[]] -> a -> b -> c')
    #def index(self, p, indices):
    #    return self.base.index(p, indices, dtype)

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

@jit('NDArray[a, EmptyDim[]] -> a')
def _unpack(array):
    return array.data[0]

@jit('NDArray[a, dims] -> NDArray[a, dims]')
def _unpack(array):
    return array

# Fill the array with `value`

@jit('NDArray[dtype, EmptyDim[]] -> a -> void')
def fill(array, value):
    """Fill a 0D array"""
    array.data[0] = value

@jit('NDArray[dtype, dims] -> a -> void')
def fill(array, value):
    """Fill an ND-array with N > 0"""
    for i in range(len(array)):
        subarray = array.dims.index(array.data, (i,), array.dtype)
        fill(subarray, value)

#===------------------------------------------------------------------===
# Conversion
#===------------------------------------------------------------------===

def fromnumpy(ndarray, ty, boundscheck=False):
    """Build an NDArray from a numpy ndarray"""
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
    data = fromobject(ndarray.ctypes.data, Pointer[flypy.types.int8])

    dims = EmptyDim()
    for extent, stride in reversed(zip(ndarray.shape, steps)):
        dimcls = Dimension
        #dimcls = DimensionContig if stride == 1 else Dimension
        dims = dimcls(dims, extent, stride)

        if boundscheck:
            dims = BoundsCheck(dims)

    base_type = ty.parameters[0]
    return NDArray(data, dims, base_type)

def tonumpy(arr, dtype):
    """Build a NumPy array from an NDArray"""
    itemsize = flypy.types.sizeof_type(dtype)
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
    #     return ContigNDArray[typeof(array.dtype)]
    # else:
    dtype = numpy_support.from_dtype(array.dtype)

    dims = EmptyDim[()]

    for i, stride in zip(range(array.ndim), reversed(array.strides)):
        # TODO:
        #dimcls = DimensionContig if stride == array.itemsize else Dimension
        dimcls = Dimension
        dims = dimcls[dims]

    return NDArray[dtype, dims]

#@pyoverload(np.ndarray, NDArray)
#def convert(ndarray):
#    data = convert(ndarray.ctypes.data, 'Pointer[T]')
#    shape = convert(ndarray.shape, 'Tuple[Int, n]')
#    strides = convert(ndarray.strides, 'Tuple[Int, n]')
#    return NDArray(data, shape, strides, ndarray)

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