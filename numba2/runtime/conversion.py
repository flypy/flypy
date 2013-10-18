# -*- coding: utf-8 -*-

"""
Convert between objects and numba representations.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from numba2 import typing

#===------------------------------------------------------------------===
# Object Conversion
#===------------------------------------------------------------------===

def fromobject(value, type):
    """
    Convert a Python value to a numba representation according to `type`
    (e.g. list -> List)
    """
    cls = type.impl
    if hasattr(cls, 'fromobject') and not isinstance(value, cls):
        return cls.fromobject(value, type)
    return value


def toobject(value, type):
    """
    Convert a Numba value to a Python representation (e.g. List -> list)
    """
    cls = type.impl
    if hasattr(cls, 'toobject'):
        return cls.toobject(value, type)
    return value


def toctypes(value, type, keepalive, memo=None):
    """Return (ctypes_value, keep_alive)"""
    if memo is None:
        memo = {}

    cls = type.impl
    if hasattr(cls, 'toctypes'):
        return cls.toctypes(value, type)
    elif stack_allocate(type):
        cty = ctype(type, memo)
        return cty(*[getattr(value, name) for name, _ in cty._fields_])
    else:
        cty = ctype(type, memo)
        cty = cty._type_
        result = cty(*[getattr(value, name) for name, _ in cty._fields_])
        keepalive.append(result)
        return ctypes.pointer(result)


def ctype(type, memo=None):
    # -------------------------------------------------
    # Setup cache

    if memo is None:
        memo = {}
    if type in memo:
        return memo[type]

    # -------------------------------------------------
    # Create ctypes type

    cls = type.impl
    if hasattr(cls, 'ctype'):
        result = cls.ctype(type)
    else:
        # -------------------------------------------------
        # Determine field ctypes

        names, types = zip(*dict(type.layout).items()) or [(), ()]
        types = [typing.resolve_simple(type, t) for t in types]
        types = [ctype(ty, memo) for ty in types]
        if not types:
            types = [ctypes.c_int32]

        # -------------------------------------------------
        # Build struct

        class result(ctypes.Structure):
            _fields_ = zip(names, types)

            def __repr__(self):
                return "{ %s }" % (", ".join("%s:%s" % (name, getattr(self, name))
                                                 for name in names))

        result.__name__ = 'CTypes' + type.__class__.__name__

        # -------------------------------------------------
        # Handle stack allocation

        if not stack_allocate(type):
            result = ctypes.POINTER(result)

    # -------------------------------------------------
    # Cache result
    memo[type] = result
    return result


def stack_allocate(type):
    """
    Determine whether values of this type should be stack-allocated and partake
    directly as values under composition.
    """
    return type.impl.stackallocate


def make_coercers(type):
    """
    Build coercion functions that reconstruct the values.
    """

    cls = type.impl
    pycls = lambda *args: cls(*args)
    layout = cls.layout

    @jit('%s -> Type[Object] -> Object' % (type,))
    def topy(obj, _):
        args = []
        for name, type in unroll(layout):
            args.append(coerce(getattr(obj, name), Object))
        return pycls(*args)

    @jit('Object -> Type[%s] -> %s' % (type, type))
    def frompy(obj, _):
        args = []
        for name, type in unroll(layout):
            args.append(coerce(getattr(obj, name), type))
        return cls(*args)

    return topy, frompy

#===------------------------------------------------------------------===
# General Type Conversion
#===------------------------------------------------------------------===

# TODO:
