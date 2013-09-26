# -*- coding: utf-8 -*-

"""
Number interfaces.
"""

from __future__ import print_function, division, absolute_import
from functools import wraps
from ... import abstract, jit

__all__ = ['Number', 'Real', 'Complex', 'Rational', 'Irrational',
           'Integer', 'Floating']

def ojit(signature):
    """
    Simple helper that unboxes arguments and boxes results of opaque methods.
    These methods have external implementations assigned!
    """
    def decorator(f):
        @wraps(f)
        def wrapper(self, *args):
            args = [self.unwrap()] + [x.unwrap() for x in args]
            return self.wrap(f(*args))
        return jit(signature, opaque=True)(wrapper)
    return decorator


@abstract
class Number(object):
    """Interface for all numbers"""

    layout = [('x', 'Number')]
    immutable = ('x',)

    def __init__(self, x):
        self.x = x

    def wrap(self, x):
        return Number(x)

    def unwrap(self):
        return self.x

    #===------------------------------------------------------------------===
    # Arith
    #===------------------------------------------------------------------===

    @ojit('a -> a -> a')
    def __add__(self, other):
        return self + other

    @ojit('a -> a -> a')
    def __mul__(self, other):
        return self * other

    @ojit('a -> a -> a')
    def __sub__(self, other):
        return self - other

    @ojit('a -> a -> a')
    def __div__(self, other):
        return self / other

    @ojit('a -> a -> a')
    def __truediv__(self, other):
        return self / other

    @ojit('a -> a -> a')
    def __floordiv__(self, other):
        return self // other

    @ojit('a -> a -> a')
    def __mod__(self, other):
        return self % other

    @ojit('a -> a')
    def __invert__(self):
        return ~self

    @jit('a -> a')
    def __abs__(self):
        if self < 0:
            return -self
        return self

    #===------------------------------------------------------------------===
    # Compare
    #===------------------------------------------------------------------===

    @ojit('a -> a -> bool')
    def __eq__(self, other):
        return self == other

    @ojit('a -> a -> bool')
    def __ne__(self, other):
        return self != other

    @ojit('a -> a -> bool')
    def __lt__(self, other):
        return self < other

    @ojit('a -> a -> bool')
    def __le__(self, other):
        return self <= other

    @ojit('a -> a -> bool')
    def __gt__(self, other):
        return self > other

    @ojit('a -> a -> bool')
    def __ge__(self, other):
        return self >= other

    #===------------------------------------------------------------------===
    # Bitwise
    #===------------------------------------------------------------------===

    @ojit('a -> a -> a')
    def __and__(self, other):
        return self & other

    @ojit('a -> a -> a')
    def __or__(self, other):
        return self | other

    @ojit('a -> a -> a')
    def __xor__(self, other):
        return self ^ other

    @ojit('a -> a -> a')
    def __lshift__(self, other):
        return self << other

    @ojit('a -> a -> a')
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


#===------------------------------------------------------------------===
# Implementations...
#===------------------------------------------------------------------===

import textwrap

from numba2.compiler import opaque
from pykit import from_c

def impl_add(py_func, argtypes):
    mod = from_c(textwrap.dedent("""
    #include <pykit_ir.h>
    Int32 add(Int32 a, Int32 b) {
        return a + b;
    }
    """))
    return mod.get_function('add')

def impl_lt(py_func, argtypes):
    mod = from_c(textwrap.dedent("""
    #include <pykit_ir.h>
    Bool lt(Int32 a, Int32 b) {
        return a < b;
    }
    """))
    return mod.get_function('lt')


opaque.implement_opaque(Number.__add__, impl_add)
opaque.implement_opaque(Number.__lt__, impl_lt)
