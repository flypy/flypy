# -*- coding: utf-8 -*-

"""
Foreign function interface functionality.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from numba2 import jit
from numba2 import jit, overlay
from .obj import Type
from .conversion import ctype
from .casting import cast
from .obj import Type, Pointer, Void

from .lowlevel_impls import add_impl

from pykit import ir
from pykit import types as ptypes

import cffi

void = Void[()]

__all__ = ['malloc', 'sizeof']

#===------------------------------------------------------------------===
# Decls
#===------------------------------------------------------------------===

ffi = cffi.FFI()
ffi.cdef("""
void *malloc(size_t size);
int memcmp(void *s1, void *s2, size_t n);
int printf(char *s, ...);
""")
libc = ffi.dlopen(None)

#===------------------------------------------------------------------===
# Implementations
#===------------------------------------------------------------------===

@jit('int64 -> Type[a] -> Pointer[a]')
def malloc(items, type):
    p = libc.malloc(items * sizeof(type))
    return cast(p, Pointer[type])

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
        cty = ctype(argtype)
        size = ctypes.sizeof(cty)
        result = ir.Const(size, ptypes.Int64)
    else:
        result = builder.sizeof(ptypes.Int64, obj)
    return builder.ret(result)

add_impl(sizeof, "sizeof", implement_sizeof, ptypes.Int64)
