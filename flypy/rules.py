# -*- coding: utf-8 -*-

"""
User-defined typing rules:

    typeof(const) -> type
    convert(value, type) -> value
    promote(type1, type2) -> type3
"""

from __future__ import print_function, division, absolute_import
from . import pyoverload
from flypy.typing import unify, free, to_blaze, resolve_type

import datashape as ds

#===------------------------------------------------------------------===
# User-defined typing rules
#===------------------------------------------------------------------===

@pyoverload
def typeof(pyval):
    """Python value -> Type"""
    from .runtime.obj.core import Type, Constructor
    from flypy import coretypes
    from flypy.support import ctypes_support, cffi_support

    if is_flypy_type(pyval):
        if pyval.type.parameters:
            return Constructor[pyval.type]
        return Type[pyval.type]
    elif isinstance(pyval, coretypes.Mono):
        return Type[pyval]
    elif is_flypy_type(type(pyval)):
        return infer_constant(pyval)
    elif cffi_support.is_cffi(pyval):
        cffi_type = cffi_support.ffi.typeof(pyval)
        return cffi_support.map_type(cffi_type)
    elif (ctypes_support.is_ctypes_value(pyval) and
          ctypes_support.is_ctypes_function(pyval)):
        return ctypes_support.from_ctypes_type(type(pyval), pyval)
    elif ctypes_support.is_ctypes_value(pyval):
        return ctypes_support.from_ctypes_type(type(pyval))
    elif ctypes_support.is_ctypes_struct_type(pyval):
        return ctypes_support.from_ctypes_type(pyval)

    raise NotImplementedError("typeof(%s, %s)" % (pyval, type(pyval)))


def infer_constant(value):
    """
    Infer the type of a user-defined type instance.

    E.g. Foo(10) -> Foo[int32]
    """
    if not is_flypy_type(type(value)):
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
    result = unify(constraints)
    return result[-1]


def convert(value, type):
    """Convert a value of type 'a' to the given type"""
    return value


def promote(type1, type2):
    """Promote two types to a common type"""
    from flypy.compiler.typing import inference

    if type1 == type2:
        return type1
    elif (type(type1), type(type2)) == (inference.Method, inference.Method):
        # promote Method types
        # TODO: Bit of a hack, do this better
        func1, obj1 = type1.parameters
        func2, obj2 = type2.parameters
        result = promote(obj1, obj2)
        if result == obj1:
            return type1
        elif result == obj2:
            return type2
        else:
            raise TypeError("Cannot promote methods %s and %s" % (type1, type2))
    else:
        t1, t2 = to_blaze(type1), to_blaze(type2)
        result = ds.promote(t1, t2)
        return resolve_type(result)


def typejoin(type1, type2):
    """
    Join two types to a type encompassing both. We promote them, join them to
    common supertype or use a variant.
    """
    from .types import void

    if type1 == type2:
        return type1
    elif type1 == void:
        return type2
    elif type2 == void:
        return type1
    else:
        return promote(type1, type2)


def is_flypy_type(x):
    return getattr(x, '_is_flypy_class', False)


def typematch(ty, impl):
    """
    See whether type instance `ty` is a type for value instances of `impl`.

        >>> @jit('Foo[a, b]')
        ... class Foo(object):
        ...     pass
        ...
        >>> typematch(Foo[int32, 2], Foo)
        True
    """
    return isinstance(ty, type(impl.type))
