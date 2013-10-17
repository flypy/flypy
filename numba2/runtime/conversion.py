# -*- coding: utf-8 -*-

"""
Convert between objects and numba representations.
"""

from __future__ import print_function, division, absolute_import

#===------------------------------------------------------------------===
# Object Conversion
#===------------------------------------------------------------------===

def fromobject(value, type):
    """
    Convert a Python value to a numba representation according to `type`
    (e.g. list -> List)
    """
    cls = type.impl
    if hasattr(cls, 'fromobject'):
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

def toctypes(value, type, seen=None, keepalive=None):
    """Return (ctypes_value, keep_alive)"""
    if keepalive is None:
        keepalive = []

    cls = type.impl
    if hasattr(cls, 'toctypes'):
        return cls.toctypes(value, type)

    from numba2.compiler.representation import build_ctypes_representation
    value, _ = build_ctypes_representation(type, value, seen, keepalive)
    return value

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
