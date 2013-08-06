# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from . import overload
from . import types

class Typevar(object):
    """Type variable"""

    def __init__(self, name=None):
        self.name = gensym(name)


class Constraint(object):
    """Typing constraint"""

    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right


@overloadable
def typeof(pyval):
    """Python value -> Type"""

@overload('a -> b')
def convert(value):
    """Convert a value of type 'a' to a new type indicated by the return type"""
    return value

@overload(types.Type, types.Type)
def promote(type1, type2):
    """Promote two types to a common type"""
    return Union([type1, type2])


T, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10 = ['T%d' % i for i in range(11)]