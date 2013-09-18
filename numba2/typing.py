# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from functools import partial

from . import pyoverload
from .compiler.overload import overload
from .runtime.obj.intobject import Int

from pykit import types
from pykit.utils import make_temper, pattern
from blaze.datashape import *

#===------------------------------------------------------------------===
# Runtime
#===------------------------------------------------------------------===

units = {}

def declare_unit(type, unit_type):
    """
    Declare `type` to be a unit type.

    NOTE: stopgap until we have an overloaded `promote` and `coerce` function.
    """
    units[type] = unit_type


class MetaType(type):
    """
    Type of types.
    """

    def __init__(self, name, bases, dct):
        params = dct['parameters']
        flags = [{'coercible': False} for i in range(len(params))]
        self.type = TypeConstructor(name, len(params), flags)

    def __getitem__(cls, key):
        if not isinstance(key, tuple):
            key = (key,)
        return self.type(*key)

#===------------------------------------------------------------------===
#
#===------------------------------------------------------------------===

@pyoverload
def typeof(pyval):
    """Python value -> Type"""
    raise NotImplementedError("typeof(%s)" % (pyval,))

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

T, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10 = [
        Typevar(typevar_names[i]) for i in range(12)]
