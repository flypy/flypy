# -*- coding: utf-8 -*-

"""
numba.vectorize().
"""

from __future__ import print_function, division, absolute_import
from itertools import starmap

from numba2 import sjit, jit, typeof, conversion, parse
from numba2.runtime.obj.arrayobject import Dimension, EmptyDim

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

@jit('Array[dtype1, dims] -> Array[dtype2, dims]')
def broadcast(a, b):
    """Broadcast two arrays"""
    return _broadcast(a.dims, b.dims, a.dims, b.dims)

@jit('Dimension[base] -> Dimension[base] -> a -> b -> r')
def _broadcast(a, b, adims, bdims):
    """Broadcast two given dimensions"""
    # Equivalent dimensions, done
    return (a, b)

@jit('Dimension[base1] -> Dimension[base1] -> a -> b -> r')
def _broadcast(a, b, adims, bdims):
    # Reduce structure
    return _broadcast(a.base, b.base, adims, bdims)

@jit('Dimension[base] -> EmptyDim[] -> a -> b -> r')
def _broadcast(a, b, adims, bdims):
    # LHS has more dims, patch RHS with extra dimensions
    return (adims, _patch(bdims, a))

@jit('EmptyDim[] -> Dimension[base] -> a -> b ->r')
def _broadcast(a, b, adims, bdims):
    # RHS has more dims, patch LHS with extra dimensions
    return (_patch(adims, b), b)


@jit('Dimension[base1] -> Dimension[base2] -> Dimension[base3]')
def _patch(dims, missing):
    """
    Patch `dims` with broadcasting leading dimensions dictated by `missing`
    """
    base = _patch(dims, missing.base)
    return Dimension(base, 1, 0)

@jit('Dimension[base1] -> EmptyDim[] -> Dimension[base3]')
def _patch(dims, missing):
    return dims
