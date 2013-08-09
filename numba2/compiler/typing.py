# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import itertools

from .overload import overload, overloadable
from .types import Type, Union

from pykit.utils import make_temper

# ______________________________________________________________________

_temp = make_temper()
typevar_names = u'αβγδεζηθικλμνξοπρςστυφχψω'

def make_stream(seq=typevar_names):
    for x in seq:
        yield _temp(x)

gensym = make_stream(typevar_names).next

# ______________________________________________________________________

class Typevar(object):
    """Type variable"""

    def __init__(self, name=None):
        self.name = _temp(name)

    def __repr__(self):
        return ('T(%s)' % (unicode(self),)).encode('utf-8')

    def __unicode__(self):
        return self.name


class Constraint(object):
    """Typing constraint"""

    def __init__(self, op, args):
        self.op = op
        self.args = args


@overloadable
def typeof(pyval):
    """Python value -> Type"""

@overload('ν -> Type[τ] -> τ')
def convert(value, type):
    """Convert a value of type 'a' to the given type"""
    return value

@overload(Type, Type)
def promote(type1, type2):
    """Promote two types to a common type"""
    return Union([type1, type2])

class TypedefRegistry(object):
    def __init__(self):
        self.typedefs = {} # builtin -> numba function

    def typedef(self, pyfunc, numbafunc):
        assert pyfunc not in self.typedefs
        self.typedefs[pyfunc] = numbafunc


typedef_registry = TypedefRegistry()
typedef = typedef_registry.typedef

T, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10 = [
        Typevar(typevar_names[i]) for i in range(12)]