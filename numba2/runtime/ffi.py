# -*- coding: utf-8 -*-

"""
Foreign function interface functionality.
"""

from __future__ import print_function, division, absolute_import
import ctypes

import numba2
from numba2 import jit, overlay
from .obj import Type
from .casting import cast
from .obj import Type, Pointer, Void
from .lib import libc
from .lowlevel_impls import add_impl

from pykit import ir
from pykit import types as ptypes

void = Void[()]

__all__ = ['malloc', 'memcmp', 'sizeof']

#===------------------------------------------------------------------===
# Implementations
#===------------------------------------------------------------------===

@jit('int64 -> Type[a] -> Pointer[a]')
def malloc(items, type):
    p = libc.malloc(items * sizeof(type)) # TODO: errcheck
    return cast(p, Pointer[type])

@jit('Pointer[a] -> int64 -> void')
def realloc(p, n):
    p = cast(p, Pointer[void])
    libc.realloc(p, n) # TODO: errcheck

@jit('Pointer[a] -> void')
def free(p):
    libc.free(p)

@jit('Pointer[a] -> Pointer[b] -> int64 -> bool')
def memcmp(a, b, size):
    p1 = cast(a, Pointer[void])
    p2 = cast(b, Pointer[void])
    return libc.memcmp(p1, p2, size) == 0

@jit('a -> int64', opaque=True)
def sizeof(obj):
    raise NotImplementedError("Not implemented at the python level")

@jit('Type[a] -> int64', opaque=True)
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
