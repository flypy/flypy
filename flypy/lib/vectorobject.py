# -*- coding: utf-8 -*-

"""
vector implementation.
"""

from __future__ import print_function, division, absolute_import

import flypy
from flypy import sjit, jit
from flypy.compiler import representation_type, lltype
from flypy.conversion import ctype
from flypy.runtime import formatting
from flypy.lib.arrayobject import Array
from flypy.runtime.interfaces import Number
from flypy.runtime.lowlevel_impls import add_impl_cls, add_impl
from pykit import types as ptypes, ir
from pykit.ir.value import Undef

@sjit('Vector[base, count]')
class Vector(Number):
    layout = []

    def __init__(self, items):
        self.items = items

    def wrap(self, items):
        return Vector(items)

    def unwrap(self):
        return self.items

    @jit('Vector[base, count] -> int64 -> base', opaque=True)
    def __getitem__(self, index):
        return self.items[index]

    @jit('Vector[base, count] -> Iterator[base]')
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __setitem__(self, index, value):
        raise TypeError("can not be implemented since Vector is immutable")

    @jit('Vector[base, count] -> int64 -> base -> Vector[base, count]', opaque=True)
    def set(self, idx, value):
        items = list(self.items)
        items[idx] = value
        return Vector(items)

    @jit('Vector[base, count] -> Pointer[Array[base, count]] -> void', opaque=True)
    def to_array(self, parray):
        p[0][:] = self.items

    @jit('Vector[base, count] -> Int[bits, True]', opaque=True)
    def to_int(self, parray):
        # TODO:
        pass

    @jit('Vector[base, count] -> int64', opaque=True)
    def __len__(self):
        return len(self.items)

    # --------------------
    @staticmethod
    def fromobject(items, type):
        return Vector(make_ctypes_vector(items, type))

    @classmethod
    def toctypes(cls, val, ty):
        # TODO:
        if isinstance(val, Vector):
            val = val.items
        return make_ctypes_vector(val, ty)

    @classmethod
    def fromctypes(cls, val, ty):
        # TODO:
        if hasattr(val, '_type_'):
            return Vector(list(val))
        return val

    @classmethod
    def ctype(cls, ty):
        base, count = ty.parameters
        return ctype(base) * count


#===------------------------------------------------------------------===
# Low-level implementations
#===------------------------------------------------------------------===

# Constructors

@jit('Pointer[Array[base, count]] -> Vector[base, count]', opaque=True)
def from_array(parray):
    return Vector(parray[0].items)

@jit('Int[bits, True] -> Type[base] -> Vector[base, count]', opaque=True)
def from_int(pint, base_t):
    # TODO:
    pass

def implement_from_array(builder, argtypes, parray):
    base, count = argtypes[0]
    vector_type = ptypes.Vector(base, count)
    ptr_type    = ptypes.Pointer(vector_type)
    pv = builder.bitcast(parray, ptr_type)
    v = builder.ptrload(p)
    return builder.ret(v)

def restype_from_array(argtypes):
    base, count = argtypes[0]
    return ptypes.Vector(base, count)

def restype_from_int(argtypes):
    (bits,), count = argtypes[0]
    base_t = argtypes[1]
    assert bits % base_t.bits == 0
    return ptypes.Vector(base_t, bits / base_t.bits)

def implement_from_int(builder, argtypes, i, base_t):
    vector_type = restype_from_int(argtypes)
    v = builder.bitcast(i, vector_type)
    return builder.ret(v)

add_impl(from_array, "from_array", implement_from_array, restype_func=restype_from_array)
add_impl(from_int, "from_int", implement_from_int, restype_func=restype_from_int)

# low level operations

def implement_len(builder, argtypes, vector):
    count = argtypes[0].parameters[1]
    return builder.ret(ir.Const(count, ptypes.Int64))

def implement_to_array(builder, argtypes, vector, parray):
    pvector_t = ptypes.Pointer(argtypes[0])
    pvector = builder.bitcast(pvector_t, parray)
    builder.ptrstore(vector, pvector)

def restype_to_int(argtypes):
    (bits,), count = argtypes[0]
    i_t = ptypes.Int(targetBits, True)
    return i_t

def implement_to_int(builder, argtypes, vector):
    i_t = restype_to_int(argtypes)
    i = builder.bitcast(i_t, vector)
    return builder.ret(i)

# implementations

add_impl_cls(Vector, "__len__", implement_len, restype=ptypes.UInt64)
add_impl_cls(Vector, "to_array", implement_to_array, restype=ptypes.Void)
add_impl_cls(Vector, "to_int", implement_to_int, restype_func=restype_to_int)

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

def make_ctypes_vector(items, type):
    # TODO:
    from flypy.support.cffi_support import is_cffi, ffi
    from flypy.support.ctypes_support import is_ctypes_pointer_type

    cty = ctype(type)
    if is_cffi(ptr):
        addr = ffi.cast('uintptr_t', ptr)
        ctypes_ptr = ctypes.c_void_p(int(addr))
        ptr = ctypes.cast(ctypes_ptr, cty)
    else:
        ptr = ctypes.cast(ptr, cty)

    return ptr
