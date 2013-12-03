# -*- coding: utf-8 -*-

"""
Slice implementation.
"""

from __future__ import print_function, division, absolute_import
from itertools import starmap

from numba2 import sjit, jit, typeof, conversion

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
    def fromobject(s, type):
        args = zip((s.start, s.stop, s.step), type.parameters)
        return Slice(*starmap(conversion.fromobject, args))

    @staticmethod
    def toobject(s, type):
        args = zip((s.start, s.stop, s.step), type.parameters)
        return slice(*starmap(conversion.toobject, args))


@typeof.case(slice)
def typeof(s):
    types = tuple(map(typeof, [s.start, s.stop, s.step]))
    return Slice[types]
