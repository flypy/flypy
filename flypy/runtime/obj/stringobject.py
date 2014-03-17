# -*- coding: utf-8 -*-

"""
String implementation.
"""

from __future__ import print_function, division, absolute_import

import flypy
from flypy import sjit, jit, typeof
from .bufferobject import Buffer, newbuffer, copyto
from .pointerobject import Pointer

@sjit
class String(object):
    layout = [('buf', 'Buffer[char]')]

    @jit('a -> a -> bool')
    def __eq__(self, other):
        return self.buf == other.buf

    @jit('a -> b -> bool')
    def __eq__(self, other):
        return False

    # TODO: Fix the below
    #@jit('a -> int64 -> a')
    #def __getitem__(self, idx):
    #    #c = self.buf[idx]
    #    p = self.buf.p + idx
    #    # TODO: Keep original string alive!
    #    return String(Buffer(p, 1)) # <- this is not \0 terminated

    @jit('a -> a')
    def __str__(self):
        return self

    @jit('a -> int64')
    def __len__(self):
        return len(self.buf) - 1

    @jit('a -> a -> a')
    def __add__(self, other):
        n = len(self) + len(other) + 1
        buf = newbuffer(flypy.char, n)

        copyto(self.buf, buf, 0)
        copyto(other.buf, buf, len(self))

        return String(buf)

    @jit('a -> bool')
    def __nonzero__(self):
        return bool(len(self))

    __bool__ = __nonzero__

    # __________________________________________________________________

    @staticmethod
    def fromobject(strobj, type):
        assert isinstance(strobj, str)
        p = flypy.runtime.lib.librt.asstring(strobj)
        buf = Buffer(Pointer(p), len(strobj) + 1)
        return String(buf)

    @staticmethod
    def toobject(obj, type):
        buf = obj.buf
        return flypy.runtime.lib.librt.fromstring(buf.p, len(obj))

    # __________________________________________________________________


#===------------------------------------------------------------------===
# String <-> char *
#===------------------------------------------------------------------===

@jit('Pointer[char] -> String[]')
def from_cstring(p):
    return String(Buffer(p, flypy.runtime.lib.strlen(p)))

@jit('String[] -> Pointer[char]')
def as_cstring(s):
    return s.buf.pointer()

#===------------------------------------------------------------------===
# typeof
#===------------------------------------------------------------------===

@typeof.case(str)
def typeof(pyval):
    return String[()]