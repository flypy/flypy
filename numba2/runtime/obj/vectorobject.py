# -*- coding: utf-8 -*-

"""
vector implementation.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2 import sjit, jit
from numba2.compiler import representation_type, lltype
from numba2.conversion import ctype
from numba2.runtime import formatting
from ..interfaces import Number
from ..lowlevel_impls import add_impl_cls

from pykit import types as ptypes, ir


@sjit('Vector[base, count]')
class Vector(Number):
    layout = []

    def __init__(self, items):
        # NOTE: This constructor is not used from jitted code
        self.items = items

    @jit('Vector[base, count] -> int64 -> base', opaque=True)
    def __getitem__(self, index):
        return self.items[index]

    @jit('Vector[base, count] -> int64 -> Vector[base, count]', opaque=True)
    def insert(self, idx, value):
        items = list(self.items)
        items[idx] = value
        return Vector(items)

    # __setitem__ can not be implemented since insertelement creates a
    # new vector

    @jit('Vector[base, count] -> int64', opaque=True)
    def __len__(self):
        return len(self.items)

    # --------------------

    @classmethod
    def toctypes(cls, val, ty):
        raise NotImplementedError("Cannot pass in vector to function from Python")

    @classmethod
    def fromctypes(cls, val, ty):
        raise NotImplementedError("Cannot return vector from function to Python")

    @classmethod
    def ctype(cls, ty):
        raise NotImplementedError("Cannot convert vector type to ctypes")

#===------------------------------------------------------------------===
# Low-level implementations
#===------------------------------------------------------------------===

# Constructors

@jit('Array[base, count] -> Vector[base, count]', opaque=True)
def vector_pack(builder, argtypes, array):
    base, count = argtypes[0]
    vector_type = ptypes.Vector(base, count)
    ptr_type    = ptypes.Pointer(vector_type)
    p = builder.bitcast(array, ptr_type)
    builder.ret(builder.load(p))

#@jit('base -> Vector[base, count]', opaque=True)
#def implement_fill(builder, argtypes, value):
#    pack(builder, argtypes, *([value] * argtypes[0].parameters[1]))

# low level operations

def vector_getitem(builder, argtypes, vector, idx):
    builder.ret(builder.extractelement(vector, idx))

def vector_insert(builder, argtypes, vector, value, idx):
    builder.ret(builder.insertelement(vector, value, idx))

def vector_len(builder, argtypes, vector):
    count = argtypes[0].parameters[1]
    builder.ret(ir.Const(count, ptypes.Int64))

# implementations

add_impl_cls(Vector, "__getitem__", vector_getitem,
             restype_func=lambda argtypes: lltype(argtypes[0].parameters[0]))
add_impl_cls(Vector, "insert", vector_insert,
             restype_func=lambda argtypes: lltype(argtypes[0]))
add_impl_cls(Vector, "__len__", vector_len, restype=ptypes.UInt64)
