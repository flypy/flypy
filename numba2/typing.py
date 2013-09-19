# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from . import pyoverload
from .compiler.overload import overload

from blaze.datashape import TypeVar, TypeConstructor, dshape

#===------------------------------------------------------------------===
# Parsing
#===------------------------------------------------------------------===

def parse(s):
    return dshape(s)

#===------------------------------------------------------------------===
# Stopgaps...
#===------------------------------------------------------------------===

units = {}

def declare_unit(type, unit_type):
    """
    Declare `type` to be a unit type.

    NOTE: stopgap until we have an overloaded `promote` and `coerce` function.
    """
    units[type] = unit_type

#===------------------------------------------------------------------===
# Runtime
#===------------------------------------------------------------------===

class MetaType(type):
    """
    Type of types.
    """

    def __init__(self, name, bases, dct):
        self.fields = {}

    def __getitem__(cls, key):
        if not isinstance(key, tuple):
            key = (key,)
        constructor = type(cls.type)
        result = constructor(*key)
        result.fields = cls.fields
        return result


@pyoverload
def typeof(pyval):
    """Python value -> Type"""
    raise NotImplementedError("typeof(%s, %s)" % (pyval, type(pyval)))

@overload('ν -> Type[τ] -> τ')
def convert(value, type):
    """Convert a value of type 'a' to the given type"""
    return value

# @overload('Type[α] -> Type[β] -> Type[γ]')
def promote(type1, type2):
    """Promote two types to a common type"""
    # return Sum([type1, type2])
    if type1 == Opaque():
        return type2
    elif type2 == Opaque():
        return type1
    else:
        assert type1 == type2
        return type1

class TypedefRegistry(object):
    def __init__(self):
        self.typedefs = {} # builtin -> numba function

    def typedef(self, pyfunc, numbafunc):
        assert pyfunc not in self.typedefs
        self.typedefs[pyfunc] = numbafunc


typedef_registry = TypedefRegistry()
typedef = typedef_registry.typedef
