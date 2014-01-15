# -*- coding: utf-8 -*-

"""
Complex object implementation.
"""

from __future__ import print_function, division, absolute_import

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

import flypy.types
from flypy import jit, sjit, typeof, overlay

#===------------------------------------------------------------------===
# Pointer
#===------------------------------------------------------------------===

@sjit('Complex[base]')
class Complex(object):
    layout = [('real', 'base'), ('imag', 'base')]

    @jit('a -> a -> a')
    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    @jit('a -> a -> a')
    def __sub__(self, other):
        return Complex(self.real - other.real, self.imag - other.imag)

    @jit('a -> a -> a')
    def __mul__(self, other):
        real = (self.real * other.real) - (self.imag * other.imag)
        imag = (self.imag * other.real) + (self.real * other.imag)
        return Complex(real, imag)

    @jit('a -> a -> a')
    def __div__(self, other):
        real = (self.real * other.real) - (self.imag * other.imag)
        imag = (self.imag * other.real) + (self.real * other.imag)
        return Complex(real, imag)

    @jit('a -> a -> bool')
    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    @jit('a -> b -> bool')
    def __eq__(self, other):
        return False

    @jit('a -> bool')
    def __nonzero__(self):
        return bool(self.real) or bool(self.imag)

    @jit
    def __str__(self):
        # TODO: __mod__
        return str(self.real) + "+" + str(self.imag) + str("j")

    __repr__ = __str__

    # __________________________________________________________________

    @staticmethod
    def fromobject(c, type):
        return Complex(c.real, c.imag)

    @staticmethod
    def toobject(c, type):
        return builtins.complex(c.real, c.imag)


@typeof.case(builtins.complex)
def typeof(pyval):
    return Complex[flypy.types.float64]
