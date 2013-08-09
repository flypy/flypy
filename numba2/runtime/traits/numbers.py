# -*- coding: utf-8 -*-

"""
Number traits.
"""

from __future__ import print_function, division, absolute_import
from .. import trait

@trait
class Number(object):
    """Trait for all numbers"""

@trait
class Real(Number):
    """Real numbers"""

@trait
class Complex(Number):
    """Complex numbers"""

@trait
class Rational(Real):
    """Rational numbers"""

@trait
class Irrational(Real):
    """Irrational numbers"""

@trait
class Integer(Real):
    """Integers"""

@trait
class Floating(Real):
    """Floating point numbers"""