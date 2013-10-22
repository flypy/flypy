# -*- coding: utf-8 -*-

"""
User-defined typing rules:

    typeof(const) -> type
    convert(value, type) -> value
    promote(type1, type2) -> type3
"""

from __future__ import print_function, division, absolute_import
from . import pyoverload
from numba2.typing import unify, free

from pykit.utils.ctypes import (is_ctypes_value, is_ctypes_function,
                                from_ctypes_type, from_ctypes_value)

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
        return infer_constant(pyval)
    elif is_ctypes_value(pyval) and is_ctypes_function(pyval):
        funcptr = from_ctypes_value(pyval) # pykit.ir.value.Pointer
        return funcptr.type
    elif is_ctypes_value(pyval):
        return from_ctypes_type(type(pyval))

    raise NotImplementedError("typeof(%s, %s)" % (pyval, type(pyval)))


def infer_constant(value):
    """
    Infer the type of a user-defined type instance.

    E.g. Foo(10) -> Foo[int32]
    """
    if not is_numba_type(type(value)):
        return typeof(value)

    classtype = type(value).type

    if not classtype.parameters:
        return classtype

    concrete_layout = {}
    for argname in classtype.resolved_layout:
        concrete_layout[argname] = infer_constant(getattr(value, argname))
    return infer_type_from_layout(classtype, concrete_layout.items())


def infer_type_from_layout(classtype, concrete_layout):
    """
    Infer class type from concrete layout.

    E.g. Foo[x, y], [('x', 'int32'), ('y', 'float32')] => Foo[int32, float32]
    """
    cls = classtype.impl
    argnames, argtypes = zip(*concrete_layout)

    # Build constraint list for unification
    constraints = [(argtype, classtype.resolved_layout[argname])
                       for argtype, argname in zip(argtypes, argnames)
                           if argname in cls.layout]
    # Add the constructor type with itself, this will flow in resolved variables
    # from the arguments
    constraints.append((classtype, classtype))
    result, remaining = unify(constraints)

    result_type = result[-1]
    if free(result_type):
        raise TypeError(
            "Result classtype stil has free variables: %s" % (result_type,))

    return result_type


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