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

    @jit('a -> int64 -> base')
    def __getitem__(self, item):
        return self.p[item]

    @jit
    def resize(self):
        raise NotImplementedError
