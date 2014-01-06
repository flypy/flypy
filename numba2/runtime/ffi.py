# -*- coding: utf-8 -*-

"""
Foreign function interface functionality.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2 import jit, cjit
from .casting import cast
from .obj.core import Type, Pointer, Void
from .lib import libc
from .lowlevel_impls import add_impl
from numba2.compiler import lltype

from pykit import ir
from pykit import types as ptypes

void = Void[()]

__all__ = ['malloc', 'memcmp', 'sizeof']

#===------------------------------------------------------------------===
# Implementations
#===------------------------------------------------------------------===

@cjit('int64 -> Type[a] -> Pointer[a]')
def malloc(items, type):
    p = libc.malloc(items * sizeof(type)) # TODO: errcheck
    return cast(p, Pointer[type])

@cjit('Pointer[a] -> int64 -> void')
def realloc(p, n):
    p = cast(p, Pointer[void])
    libc.realloc(p, n) # TODO: errcheck

@cjit('Pointer[a] -> void')
def free(p):
    libc.free(p)

@cjit('Pointer[a] -> Pointer[b] -> int64 -> bool')
def memcmp(a, b, size):
    p1 = cast(a, Pointer[void])
    p2 = cast(b, Pointer[void])
    return libc.memcmp(p1, p2, size) == 0

@cjit('a -> int64', opaque=True)
def sizeof(obj):
    raise NotImplementedError("Not implemented at the python level")

@cjit('Type[a] -> int64', opaque=True)
def sizeof(obj):
    raise NotImplementedError("Not implemented at the python level")

#===------------------------------------------------------------------===
# Low-level implementations
#===------------------------------------------------------------------===

def implement_sizeof(builder, argtypes, obj):
    [argtype] = argtypes
    if argtype.impl == Type:
        [argtype] = argtype.parameters # Unpack 'a' from 'Type[a]'
        size = numba2.sizeof_type(argtype)
        result = ir.Const(size, ptypes.Int64)
    else:
        result = builder.sizeof(ptypes.Int64, obj)
    return builder.ret(result)

add_impl(sizeof, "sizeof", implement_sizeof, ptypes.Int64)

#======= UNDEF ==================================================

@jit('Type[base] -> base', opaque=True)
def undef(type):
    raise NotImplementedError("Not implemented at the python level")

def implement_undef(builder, argtypes, obj):
    restype = restype_undef(argtypes)
    result = ir.Undef(restype)
    return builder.ret(result)

def restype_undef(argtypes):
    (type,) = argtypes[0]
    assert type.impl == Type
    (restype,) = type.parameters
    t = lltype(restype)
    return t

add_impl(undef, "undef", implement_undef, restype_func=restype_undef)
