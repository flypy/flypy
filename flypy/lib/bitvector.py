# -*- coding: utf-8 -*-

"""
Vector of bits.
"""

from __future__ import print_function, division, absolute_import

import math

from flypy import jit, sjit, cast
from flypy.runtime.obj.core import newbuffer, Buffer
from flypy import types

@sjit
class BitVector(object):
    """
    Vector of bits. Bits can be marked and unmarked for desired positions,
    and checked for.
    """

    layout = [('buf', 'Buffer[int8]')]

    @jit
    def __init__(self, n):
        self.buf = newbuffer(types.int8, int(math.ceil(n / 8.0)))
        self.clear()

    @jit
    def mark(self, pos):
        """Mark the bit at `pos`"""
        byte = 1 << self._bitpos(pos)
        byte = cast(byte, types.int8)
        self.buf[self._bytepos(pos)] |= byte

    @jit
    def unmark(self, pos):
        """Unmark the bit at `pos`"""
        byte = ~(1 << self._bitpos(pos))
        byte = cast(byte, types.int8)
        self.buf[self._bytepos(pos)] &= byte

    @jit
    def check(self, pos):
        """Check whether the bit at `pos` is set"""
        return bool(self.buf[self._bytepos(pos)] & (1 << self._bitpos(pos)))

    @jit
    def clear(self):
        """Clear the bit vector"""
        self.buf[:] = cast(0, types.int8)

    __contains__ = check

    @jit
    def _bytepos(self, pos):
        return pos >> 3

    @jit
    def _bitpos(self, pos):
        return pos & 7