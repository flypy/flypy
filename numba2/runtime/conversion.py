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


def toctypes(value, type, keepalive, valmemo=None, typememo=None):
    """Return (ctypes_value, keep_alive)"""
    from numba2.types import int8

    if valmemo is None:
        valmemo = {}
        typememo = {}
    if id(value) in valmemo:
        return valmemo[id(value)]

    cls = type.impl
    if hasattr(cls, 'toctypes'):
        result = cls.toctypes(value, type)
    else:
        cty = ctype(type, typememo)
        if not stack_allocate(type):
            cty = cty._type_ # Get the base type

        # Resolve types
        layout = type.resolved_layout
        types = [layout[name] for name, _ in cty._fields_] or [int8]

        # Resolve values
        values = [getattr(value, name) for name, _ in cty._fields_]
        values = [toctypes(v, t, keepalive, valmemo, typememo)
                      for v, t in zip(values, types)]

        # Construct value from ctypes struct
        result = cty(*values)

        if not stack_allocate(type):
            keepalive.append(result)
            result = ctypes.pointer(result)

    valmemo[id(value)] = result
    return result


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

        names, types = zip(*type.resolved_layout.items()) or [(), ()]
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
