# -*- coding: utf-8 -*-

"""
String implementation.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, typeof
from . import librt as lib
from .bufferobject import Buffer
from .pointerobject import Pointer

@jit
class String(object):
    layout = [('buf', 'Buffer[char]')]

    @jit('a -> a -> bool')
    def __eq__(self, other):
        return self.buf == other.buf

    @jit('a -> b -> bool')
    def __eq__(self, other):
        return False

    @jit('a -> int64 -> a')
    def __getitem__(self, idx):
        #c = self.buf[idx]
        p = self.buf.p + idx
        # TODO: Keep original string alive!
        return String(Buffer(p, 1))

    @jit('a -> a')
    def __str__(self):
        return self

    @jit('a -> int64')
    def __len__(self):
        return len(self.buf) - 1

    # __________________________________________________________________

    @staticmethod
    def fromobject(strobj, type):
        assert isinstance(strobj, str)
        p = lib.asstring(strobj)
        buf = Buffer(Pointer(p), len(strobj) + 1)
        return String(buf)

    @staticmethod
    def toobject(obj, type):
        buf = obj.buf
        return lib.fromstring(buf.p, len(obj))

    # __________________________________________________________________


@typeof.case(str)
def typeof(pyval):
    return String[()]