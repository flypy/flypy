# -*- coding: utf-8 -*-

"""
Pointer implementation.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from ... import jit, typeof

@jit('Pointer[B]')
class Pointer(object):
    layout = [('p', 'Pointer[B]')]

    @jit('Pointer[A] -> int64 -> A', opaque=True)
    def __getitem__(self, index):
        return self.p[index]

    @jit('a -> int64 -> a', opaque=True)
    def __add__(self, index):
        return self.p + index

    @jit('a -> int64 -> a', opaque=True)
    def __sub__(self, index):
        return self.p + index


# TODO: ...
# @typeof.case(ctypes.c_long)
# def typeof(pyval):
#     return Int[32]