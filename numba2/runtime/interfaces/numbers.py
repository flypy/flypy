# -*- coding: utf-8 -*-

"""
Number interfaces.
"""

from __future__ import print_function, division, absolute_import
from ... import abstract

__all__ = ['Number', 'Real', 'Complex', 'Rational', 'Irrational',
           'Integer', 'Floating']

@abstract
class Number(object):
    """Interface for all numbers"""

    #===------------------------------------------------------------------===
    # Arith
    #===------------------------------------------------------------------===

    @abstract('a -> a -> a')
    def __add__(self, other):
        return self + other

    @abstract('a -> a -> a')
    def __mul__(self, other):
        return self * other

    @abstract('a -> a -> a')
    def __sub__(self, other):
        return self - other

    @abstract('a -> a -> a')
    def __div__(self, other):
        return self / other

    @abstract('a -> a -> a')
    def __truediv__(self, other):
        return self / other

    @abstract('a -> a -> a')
    def __floordiv__(self, other):
        return self // other

    @abstract('a -> a -> a')
    def __mod__(self, other):
        return self % other

    @abstract('a -> a')
    def __invert__(self):
        return ~self

    @abstract('a -> a')
    def __abs__(self):
        if self < 0:
            return -self
        return self

    #===------------------------------------------------------------------===
    # Compare
    #===------------------------------------------------------------------===

    @abstract('a -> a -> bool')
    def __eq__(self, other):
        return self == other

    @abstract('a -> a -> bool')
    def __ne__(self, other):
        return self != other

    @abstract('a -> a -> bool')
    def __lt__(self, other):
        return self < other

    @abstract('a -> a -> bool')
    def __le__(self, other):
        return self <= other

    @abstract('a -> a -> bool')
    def __gt__(self, other):
        return self > other

    @abstract('a -> a -> bool')
    def __ge__(self, other):
        return self >= other

    #===------------------------------------------------------------------===
    # Bitwise
    #===------------------------------------------------------------------===

    @abstract('a -> a -> a')
    def __and__(self, other):
        return self & other

    @abstract('a -> a -> a')
    def __or__(self, other):
        return self | other

    @abstract('a -> a -> a')
    def __xor__(self, other):
        return self ^ other

    @abstract('a -> a -> a')
    def __lshift__(self, other):
        return self << other

    @abstract('a -> a -> a')
    def __rshift__(self, other):
        return self >> other


@abstract
class Real(Number):
    """Real numbers"""

@abstract
class Complex(Number):
    """Complex numbers"""

@abstract
class Rational(Real):
    """Rational numbers"""

@abstract
class Irrational(Real):
    """Irrational numbers"""

@abstract
class Integer(Real):
    """Integers"""

@abstract
class Floating(Real):
    """Floating point numbers"""