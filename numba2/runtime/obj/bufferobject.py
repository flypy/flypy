# -*- coding: utf-8 -*-

"""
Buffer objects.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2 import jit
from numba2.runtime import ffi
from . import Type, Pointer

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
            return ffi.memcmp(self.p, other.p, self.size)

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

    @jit
    def resize(self):
        raise NotImplementedError

    @jit('Buffer[a] -> Pointer[a]')
    def pointer(self): # TODO: Properties
        return self.p

#===------------------------------------------------------------------===
# Buffer creation
#===------------------------------------------------------------------===

@jit('Type[a] -> int64 -> Buffer[a]')
def newbuffer(basetype, size):
    p = ffi.malloc(size, basetype)
    return Buffer(p, size)

# @jit('Sequence[a] -> Type[a] -> Buffer[a]') # TODO: <--
def fromseq(seq, basetype):
    n = len(seq)
    buf = newbuffer(basetype, n)
    for i, item in enumerate(seq):
        buf[i] = item
    return buf