# -*- coding: utf-8 -*-

"""
Convert between objects and numba representations.
"""

from __future__ import print_function, division, absolute_import
import ctypes

import numba2 as nb
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
    """
    Convert a numba object given as a Python value to a low-level ctypes
    representation.

    Returns (ctypes_value, keep_alive)
    """
    from numba2.types import int8

    if hasattr(type, 'type'):
        type = type.type

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

        if hasattr(value, 'contents'):
            value = value.contents

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

def fromctypes(value, ty, memo=None):
    """
    Construct a numba object from a ctypes representation.
    """
    from numba2.ctypes_support import is_ctypes_pointer_type, CTypesStruct

    if hasattr(ty, 'type'):
        ty = ty.type

    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]

    cls = ty.impl
    if hasattr(cls, 'fromctypes'):
        result = cls.fromctypes(value, ty)
    else:
        cls = ty.impl
        layout = ty.resolved_layout
        values = {}

        if is_ctypes_pointer_type(type(value)):
            # TODO: stack jit
            # Recover original names from the type
            cty = ctype(ty)
            value = ctypes.cast(value, cty)

        for name, ty in ty.resolved_layout.iteritems():
            if is_ctypes_pointer_type(type(value)):
                value = value[0]
            cval = getattr(value, name)
            pyval = fromctypes(cval, ty, memo)
            values[name] = pyval

        result = cls(**values)

    memo[id(value)] = result
    return result

def ctype(type, memo=None):
    """
    Return the low-level ctypes type representation for a numba type instance.
    """

    # -------------------------------------------------
    # Setup cache

    if hasattr(type, 'type'):
        type = type.type

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

def c_primitive(type):
    return type.impl in (nb.Bool, nb.Int, nb.Float, nb.Pointer, nb.Void,
                         nb.Function, nb.ForeignFunction)

def stack_allocate(type):
    """
    Determine whether values of this type should be stack-allocated and partake
    directly as values under composition.
    """
    return True
    #return type.impl.stackallocate

def byref(type):
    return stack_allocate(type) and not c_primitive(type)

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
