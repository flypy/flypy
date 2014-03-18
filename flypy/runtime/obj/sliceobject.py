# -*- coding: utf-8 -*-

"""
Slice implementation.
"""

from __future__ import print_function, division, absolute_import
from itertools import starmap

from flypy import sjit, jit, typeof, conversion
import flypy.runtime

@sjit('Slice[start, stop, step]')
class Slice(object):
    layout = [('start', 'start'), ('stop', 'stop'), ('step', 'step')]

    @jit('a -> a -> bool')
    def __eq__(self, other):
        return (self.start == other.start and
                self.stop == other.stop and
                self.step == other.step)

    @jit('a -> b -> bool')
    def __eq__(self, other):
        return False

    # ---------------------- #

    @staticmethod
    def fromobject(s, type, keepalive):
        keepalive = []
        args = zip((s.start, s.stop, s.step), type.parameters)
        return Slice(*[conversion.fromobject(val, typ, keepalive)
                            for val, typ in args])

    @staticmethod
    def toobject(s, type):
        args = zip((s.start, s.stop, s.step), type.parameters)
        return slice(*starmap(conversion.toobject, args))


@jit('Slice[start, stop, step] -> int64 -> r')
def normalize(s, length):
    start = flypy.runtime.choose(0, s.start)
    stop = flypy.runtime.choose(length, s.stop)
    step = flypy.runtime.choose(1, s.step)

    #-- Wrap around --#
    if start < 0:
        start += length
        if start < 0:
            start = 0
    if start > length:
        start = length - 1

    if stop < 0:
        stop += length
        if stop < -1:
            stop = -1
    if stop > length:
        stop = length

    return int(start), int(stop), int(step)


@typeof.case(slice)
def typeof(s):
    types = tuple(map(typeof, [s.start, s.stop, s.step]))
    return Slice[types]
