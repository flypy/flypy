# -*- coding: utf-8 -*-

"""
Buffer objects.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit
import numba2
from .core import Type, Pointer

@jit('Buffer[base]')
class Buffer(object):
    layout = [('p', 'Pointer[base]'), ('size', 'int64'),
              #('free', 'Function[Pointer[void], void]')
    ]

    @jit('Buffer[a] -> Pointer[a] -> int64 -> void') # Function[Pointer[a], void]
    def __init__(self, p, size): #, free):
        self.p = p
        self.size = size
        #self.free = free

    @jit('a -> a -> bool')
    def __eq__(self, other):
        if self.p == other.p:
            return True
        elif self.size != other.size:
            return False
        else:
            return numba2.runtime.ffi.memcmp(self.p, other.p, self.size)

    @jit('a -> b -> bool')
    def __eq__(self, other):
        return False

    @jit('a -> int64 -> base')
    def __getitem__(self, item):
        return self.p[item]

    @jit('a -> int64 -> base -> void')
    def __setitem__(self, item, value):
        self.p[item] = value

    @jit('a -> int64')
    def __len__(self):
        return self.size

    #@jit
    #def __del__(self):
    #    self.free(self.p)

    # ----------------------------------

    @jit('a -> int64 -> void')
    def resize(self, n):
        numba2.runtime.ffi.realloc(self.p, n)
        self.size = n

    @jit('Buffer[a] -> Pointer[a]')
    def pointer(self): # TODO: Properties
        return self.p

#===------------------------------------------------------------------===
# Buffer utils
#===------------------------------------------------------------------===

@jit('Type[a] -> int64 -> Buffer[a]')
def newbuffer(basetype, size):
    p = numba2.runtime.ffi.malloc(size, basetype)
    return Buffer(p, size)

# @jit('Sequence[a] -> Type[a] -> Buffer[a]') # TODO: <--
def fromseq(seq, basetype):
    n = len(seq)
    buf = newbuffer(basetype, n)
    for i, item in enumerate(seq):
        buf[i] = item
    return buf

@jit('Buffer[a] -> Buffer[a] -> int64 -> void')
def copyto(src, dst, offset):
    p_src = src.pointer()
    p_dst = dst.pointer() + offset

    # TODO: bounds-check

    for i in range(len(src)):
        p_dst[i] = p_src[i]