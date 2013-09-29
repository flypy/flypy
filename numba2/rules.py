# -*- coding: utf-8 -*-

"""
User-defined typing rules:

    typeof(const) -> type
    convert(value, type) -> value
    promote(type1, type2) -> type3
"""

from __future__ import print_function, division, absolute_import
from . import pyoverload

#===------------------------------------------------------------------===
# User-defined typing rules
#===------------------------------------------------------------------===

@pyoverload
def typeof(pyval):
    """Python value -> Type"""
    from .runtime.type import Type

    if is_numba_type(pyval):
        return Type[pyval.type]

    raise NotImplementedError("typeof(%s, %s)" % (pyval, type(pyval)))

#@overload('ν -> Type[τ] -> τ')
def convert(value, type):
    """Convert a value of type 'a' to the given type"""
    return value

#@overload('Type[α] -> Type[β] -> Type[γ]')
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


def is_numba_type(x):
   return isinstance(x, type) and hasattr(x, 'fields') and hasattr(x, 'layout')