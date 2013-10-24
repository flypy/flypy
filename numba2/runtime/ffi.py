# -*- coding: utf-8 -*-

"""
Foreign function interface functionality.
"""

from __future__ import print_function, division, absolute_import
from numba2 import jit, typeof, overlay, sizeof
from numba2.types import Type

import cffi

ffi = cffi.FFI()
ffi.cdef("void *malloc(size_t size);")
libc = ffi.dlopen(None)

#===------------------------------------------------------------------===
# Implementations
#===------------------------------------------------------------------===

@jit('int64 -> Type[a] -> Pointer[a]')
def malloc(items, type):
    return libc.malloc(items * sizeof(type))
