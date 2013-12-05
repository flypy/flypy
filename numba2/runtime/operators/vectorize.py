# -*- coding: utf-8 -*-

"""
numba.vectorize().
"""

from __future__ import print_function, division, absolute_import

from numba2 import sjit, jit, typeof, parse
from numba2.runtime.obj.arrayobject import Array, Dimension, EmptyDim

def vectorize(py_func, signatures, **kwds):
    signature = parse(signatures[0])
    nargs = len(signature.argtypes)
    py_func = make_ndmap(py_func, nargs)

    func = jit(py_func, signatures[0], **kwds)
    for signature in signatures[1:]:
        func.overload(py_func, signature, **kwds)

def make_ndmap(py_func, nargs):
    raise NotImplementedError("This will be messy")

#===------------------------------------------------------------------===
# broadcasting
#===------------------------------------------------------------------===

@jit('Array[dtype1, dims1] -> Array[dtype2, dims2] -> r')
def broadcast(a, b):
    """Broadcast two arrays"""
    return _broadcast(a.dims, b.dims, a, b)

@jit('Dimension[base] -> Dimension[base] -> a -> b -> r')
def _broadcast(a, b, array1, array2):
    """Broadcast two given dimensions"""
    # Equivalent dimensions, done
    return (array1, array2)

@jit('Dimension[base1] -> Dimension[base2] -> a -> b -> r')
def _broadcast(a, b, array1, array2):
    # Reduce structure
    return _broadcast(a.base, b.base, array1, array2)

@jit('Dimension[base] -> EmptyDim[] -> a -> b -> r')
def _broadcast(a, b, array1, array2):
    # LHS has more dims, patch RHS with extra dimensions
    dims2 = raise_level(array2.dims, a)
    return (array1, Array(array2.data, dims2, array2.dtype))

@jit('EmptyDim[] -> Dimension[base] -> a -> b ->r')
def _broadcast(a, b, array1, array2):
    # RHS has more dims, patch LHS with extra dimensions
    dims1 = raise_level(array1.dims, b)
    return (Array(array1.data, dims1, array1.dtype), array2)

# -- broadcast helper -- #

@jit('Dimension[base1] -> Dimension[base2] -> Dimension[base3]')
def raise_level(dims, missing):
    """
    Raise the level of `dims` by prepending broadcasting dimensions as
    dictated by `missing`.
    """
    base = raise_level(dims, missing.base)
    return Dimension(base, 1, 0)

@jit('Dimension[base1] -> EmptyDim[] -> Dimension[base3]')
def raise_level(dims, missing):
    return dims

#===------------------------------------------------------------------===
# test
#===------------------------------------------------------------------===

@jit('Array[dtype1, dims1] -> Array[dtype2, dims2] -> r')
def add(a, b):
    return broadcast(a, b)