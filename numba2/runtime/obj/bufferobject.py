# -*- coding: utf-8 -*-

"""
Buffer objects.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, typeof
from . import Pointer

@jit('Buffer[base]')
class Buffer(object):
    layout = [('p', 'Pointer[base]')]

    @jit('Buffer[a] -> Pointer[a] -> int32 -> Function[Pointer[a] -> void] -> void')
    def __init__(self, p, size, free):
        self.p = p
        self.size = size
        self.free = free

    @jit('a -> int64 -> base')
    def __getitem__(self, item):
        return self.p[item]

    @jit
    def __del__(self):
        self.free(self.p)

    @jit
    def resize(self):
        raise NotImplementedError

@jit
def newbuffer(size):
    pass