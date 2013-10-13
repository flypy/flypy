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
    elif is_numba_type(type(pyval)):
        return pyval.type

    raise NotImplementedError("typeof(%s, %s)" % (pyval, type(pyval)))

#@overload('ν -> Type[τ] -> τ')
def convert(value, type):
    """Convert a value of type 'a' to the given type"""
    return value

#@overload('Type[α] -> Type[β] -> Type[γ]')
def promote(type1, type2):
    """Promote two types to a common type"""
    if type1 == type2:
        return type1
    else:
        raise TypeError("Cannot promote %s and %s" % (type1, type2))


def is_numba_type(x):
    return getattr(x, '_is_numba_class', False)