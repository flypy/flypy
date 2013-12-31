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
from numba2.lib.arrayobject import Array
from numba2.runtime.interfaces import Number
from numba2.runtime.lowlevel_impls import add_impl_cls, add_impl
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

    @jit('Vector[base, count] -> base -> int64 -> Vector[base, count]', opaque=True)
    def set(self, value, idx):
        # __setitem__ can not be implemented since Vector is immutable
        items = list(self.items)
        items[idx] = value
        return Vector(items)

    @jit('Vector[base, count] -> Pointer[Array[base, count]] -> Void', opaque=True)
    def to_array(self, parray):
        pass

    @jit('Vector[base, count] -> Int[bits, True]', opaque=True)
    def to_int(self, parray):
        pass

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

@jit('Pointer[Array[base, count]] -> Vector[base, count]', opaque=True)
def vector_from_array(parray):
    pass

@jit('Pointer[Int[bits, True]] -> Type[base] -> Vector[base, count]', opaque=True)
def vector_from_int(pint, base_t):
    pass

def implement_vector_load_from_array(builder, argtypes, parray):
    base, count = argtypes[0]
    vector_type = ptypes.Vector(base, count)
    ptr_type    = ptypes.Pointer(vector_type)
    pv = builder.bitcast(parray, ptr_type)
    v = builder.ptrload(p)
    builder.ret(v)

def implement_vector_load_from_int(builder, argtypes, pint, base_t):
    (bits,), count = argtypes[0]
    assert bits % base_t.bits == 0
    vector_type = ptypes.Vector(base, bits / base_t.bits)
    ptr_type    = ptypes.Pointer(vector_type)
    pv = builder.bitcast(pint, ptr_type)
    v = builder.ptrload(p)
    builder.ret(v)

def restype_vector_load_from_array(argtypes):
    base, count = argtypes[0]
    return ptypes.Vector(base, count)

def restype_vector_load_from_int(argtypes):
    (bits,), count = argtypes[0]
    assert bits % base_t.bits == 0
    vector_type = ptypes.Vector(base, bits / base_t.bits)

add_impl(vector_from_array, "vector_from_array", implement_vector_load_from_array, restype_func=restype_vector_load_from_array)
add_impl(vector_from_int, "vector_from_int", implement_vector_load_from_int, restype_func=restype_vector_load_from_int)

# low level operations

def vector_getitem(builder, argtypes, vector, idx):
    builder.ret(builder.get(vector, idx))

def vector_set(builder, argtypes, vector, value, idx):
    builder.ret(builder.set(vector, value, idx))

def vector_len(builder, argtypes, vector):
    count = argtypes[0].parameters[1]
    builder.ret(ir.Const(count, ptypes.Int64))

def vector_to_array(builder, argtypes, vector, parray):
    pvector_t = ptypes.Pointer(argtypes[0])
    pvector = builder.bitcast(pvector_t, parray)
    builder.ptrstore(vector, pvector)

def vector_to_int(builder, argtypes, vector, pint):
    (bits,), count = argtypes[0]
    (targetBits,) = argtypes[1]
    assert bits * count == targetBits
    pvector_t = ptypes.Pointer(argtypes[0])
    pvector = builder.bitcast(pvector_t, pint)
    builder.ptrstore(vector, pvector)

# implementations

add_impl_cls(Vector, "__getitem__", vector_getitem, restype_func=lambda argtypes: lltype(argtypes[0].parameters[0]))
add_impl_cls(Vector, "set", vector_set, restype_func=lambda argtypes: lltype(argtypes[0]))
add_impl_cls(Vector, "__len__", vector_len, restype=ptypes.UInt64)
add_impl_cls(Vector, "to_array", vector_to_array, restype=ptypes.Void)
add_impl_cls(Vector, "to_int", vector_to_int, restype=ptypes.Void)
